"""Configurable triplet refinement, optionally starting from a saved TPE fit.

--fit-config, --bounds, --initial, --fix and --vary share their meaning with
all triplet fitters. D defaults to 500--3000 MHz. Bounds and fixed/free flags
are inherited from TPE's saved fit configuration and may be explicitly changed.
COBYLA uses named scaled coordinates, a population simplex and enumeration of
requested population bases. All-fixed models are evaluated without optimization.
SciPy is the default backend; --backend mars uses MaRS/Nevergrad Cobyla.
See src/fit_utils/README.md for units, persistence and acceptance requirements.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

if __package__:
    from .tpe_search import (FIELD_SCALES, INVALID_LOSS, PARAMETER_BOUNDS, InvalidSimulationError,
                            TripletSpectraSimulator, build_sample_and_context, device, dtype,
                            load_spectrum, normalized_mse, normalize_populations, save_results,
                            validate_parameters, validate_spectrum)
    from .fit_parameters import (NUMERIC_NAMES, POP_NAMES, POPULATION_BASES, FitConfig, add_fit_arguments,
                                 config_from_args, read_config)
else:
    from tpe_search import (FIELD_SCALES, INVALID_LOSS, PARAMETER_BOUNDS, InvalidSimulationError,
                           TripletSpectraSimulator, build_sample_and_context, device, dtype,
                           load_spectrum, normalized_mse, normalize_populations, save_results,
                           validate_parameters, validate_spectrum)
    from fit_parameters import (NUMERIC_NAMES, POP_NAMES, POPULATION_BASES, FitConfig, add_fit_arguments,
                                config_from_args, read_config)


def load_tpe_results(path):
    """Load full physical parameters, including strings and fixed values."""
    params = {}
    numeric = set(NUMERIC_NAMES) | {"harmonic", "mesh", "loss_mse", "scale", "n_evaluations",
                                   "initial_loss_mse", "converged", "seed", "n_free_model"}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"parameter", "value"} <= set(reader.fieldnames or ()):
            raise ValueError("Parameter CSV must have parameter and value columns")
        for row in reader:
            name = row["parameter"]
            if name in params:
                raise ValueError(f"Duplicate parameter {name!r}")
            if name in ("basis", "fit_config"):
                params[name] = row["value"]
            elif name in numeric:
                params[name] = float(row["value"])
                if not np.isfinite(params[name]):
                    raise ValueError(f"Parameter {name} must be finite")
    required = {"g_iso", "D", "E", "ham_strain", "gauss", *POP_NAMES}
    if required-params.keys():
        raise ValueError(f"Missing parameters: {sorted(required-params.keys())}")
    params.setdefault("basis", "xyz")
    params["pop"] = [params[k] for k in POP_NAMES]
    return validate_parameters(params)


def inherited_config(csv_path, params):
    """Never silently lose a referenced configuration when moving a saved fit."""
    if "fit_config" not in params:
        return None
    source = (Path(csv_path).resolve().parent / params["fit_config"]).resolve()
    return read_config(source)


# Original single-fit objective interface remains available to Python callers.
_exp_fields = _exp_intensity = _freq = _temperature = None
_basis, _harmonic, _mesh = "xyz", 0, 20


def set_global_data(fields, intensity, freq, temperature, basis="xyz", harmonic=0, mesh=20):
    global _exp_fields, _exp_intensity, _freq, _temperature, _basis, _harmonic, _mesh
    fields, intensity = validate_spectrum(fields, intensity)
    TripletSpectraSimulator(freq, temperature, harmonic, mesh)
    if basis not in POPULATION_BASES:
        raise ValueError(f"basis must be one of {POPULATION_BASES}")
    _exp_fields, _exp_intensity = fields.copy(), intensity.copy()
    _freq, _temperature, _basis, _harmonic, _mesh = freq, temperature, basis, harmonic, mesh


def _parameters_from_vector(values, basis):
    vector = np.asarray(values, dtype=float)
    names = ("g_iso", "D", "E", "ham_strain", "gauss", *POP_NAMES)
    if vector.shape != (len(names),) or not np.isfinite(vector).all():
        raise ValueError("Expected eight finite physical parameters")
    return validate_parameters(dict(zip(names, vector.tolist()), basis=basis))


def _evaluate(params, fields, intensity, freq, temperature, harmonic, mesh):
    simulator = TripletSpectraSimulator(params.get("freq", freq), params.get("temperature", temperature), harmonic, mesh)
    try:
        spectrum = simulator(torch.as_tensor(fields, dtype=dtype, device=device), params)
        return normalized_mse(spectrum, intensity), spectrum
    except InvalidSimulationError:
        return INVALID_LOSS, None


def cobyla_objective(params_vec):
    if _exp_fields is None:
        raise RuntimeError("Call set_global_data before cobyla_objective")
    try:
        params = _parameters_from_vector(params_vec, _basis)
    except ValueError:
        return INVALID_LOSS
    return _evaluate(params, _exp_fields, _exp_intensity, _freq, _temperature, _harmonic, _mesh)[0]


def _mars_optimizer(evaluate, x0, budget, fields, intensity, seed, method):
    """Use MaRS' runner with the same validated objective; no dashboard/server."""
    from mars.optimization import ParamSpec, ParameterSpace, SpectrumFitter
    from mars.optimization.objectives import MSEObjective
    class SeededSpace(ParameterSpace):
        def instrument_nevergrad(self):
            instrument = super().instrument_nevergrad()
            instrument.random_state.seed(seed)
            return instrument
    names = [f"coordinate_{i}" for i in range(len(x0))]
    space = SeededSpace([ParamSpec(name=k, bounds=(0., 1.), default=float(v)) for k, v in zip(names, x0)])
    class Runner(SpectrumFitter):
        def _set_experimental(self, x_exp, y_exp, device, dtype):
            x = torch.as_tensor(x_exp, dtype=dtype, device=device)
            y = torch.as_tensor(y_exp, dtype=dtype, device=device)
            return x, y/y.abs().max(), False
        def _loss_from_params(self, params, **kwargs):
            return torch.as_tensor(evaluate([params[k] for k in names]), device=device, dtype=dtype)
    def unused_simulator(*args, **kwargs):
        raise RuntimeError("The configured MaRS runner must evaluate its validated loss callback")
    fitter = Runner(x_exp=torch.as_tensor(fields, dtype=dtype, device=device),
                    y_exp=torch.as_tensor(intensity, dtype=dtype, device=device), param_space=space,
                    spectra_simulator=unused_simulator,
                    objective=MSEObjective(), norm_mode="max", device=device, dtype=dtype)
    if method == "TPE":
        import optuna
        return fitter.fit(backend="optuna", n_trials=budget, sampler=optuna.samplers.TPESampler(seed=seed),
                          run_dashboard=False, show_progress=False, return_best_spectrum=False)
    return fitter.fit(backend="ng", budget=budget, optimizer="Cobyla", seed=None,
                      show_progress=False, return_best_spectrum=False)


def _start_config(initial, config, freq, temperature, harmonic, mesh):
    initial = validate_parameters(initial)
    if config is None:
        definitions = {k: {"value": initial[k]} for k in NUMERIC_NAMES+("basis",) if k in initial}
        definitions.update(freq={"value": freq}, temperature={"value": temperature})
        config = FitConfig(definitions, {"harmonic": harmonic, "mesh": mesh})
    definitions = config.to_dict()
    for name, spec in definitions["parameters"].items():
        if spec["vary"] and name in initial:
            if name != "basis":
                lo, hi = spec["bounds"]
                tol = 1e-12*max(1,abs(lo),abs(hi))
                if not lo-tol <= initial[name] <= hi+tol:
                    raise ValueError(f"Initial {name} is outside selected bounds; supply --bounds/--fit-config")
            spec["value"] = initial[name]
    return FitConfig.from_dict(definitions)


def run_cobyla(fields, intensity, initial, *, freq, temperature, budget=300, harmonic=0, mesh=20,
               config=None, backend="scipy", seed=42, _method="Cobyla"):
    """Optimize only enabled coordinates; return the best feasible model seen.

    Fixed flags/values come from config. Initial values update its varying
    parameters, not fixed ones. Categorical bases share the total budget.
    _method='TPE' is used by the MaRS two-stage entry point only.
    """
    fields, intensity = validate_spectrum(fields, intensity)
    if backend not in ("scipy", "mars"):
        raise ValueError("backend must be scipy or mars")
    config = _start_config(initial, config, freq, temperature, harmonic, mesh)
    TripletSpectraSimulator(config.values()["freq"], config.values()["temperature"], **config.settings)
    bases = config.basis_choices
    dimension = len(config.free_names)
    minimum = dimension+2 if dimension and _method == "Cobyla" else 1
    if type(budget) is not int or budget < minimum*len(bases):
        raise ValueError(f"budget must be >= {minimum*len(bases)} for this free mask and basis choices")
    x0 = np.asarray(config.start_vector(), dtype=float)
    best = {"loss": INVALID_LOSS, "params": None, "spectrum": None}
    cache, messages, convergence = {}, [], []
    initial_loss = None
    for index, basis in enumerate(bases):
        allocation = budget//len(bases)+int(index < budget%len(bases))
        branch_calls = 0
        def evaluate(vector):
            nonlocal branch_calls
            x = np.asarray(vector, dtype=float)
            key = (basis, tuple(x))
            if key in cache:
                return cache[key]
            if branch_calls >= allocation:
                return INVALID_LOSS
            branch_calls += 1
            loss = INVALID_LOSS
            if x.shape == x0.shape and np.isfinite(x).all() and np.all(x >= -1e-12) and np.all(x <= 1+1e-12):
                try:
                    params = validate_parameters(config.decode(np.clip(x, 0, 1), basis))
                except ValueError:
                    pass
                else:
                    loss, spectrum = _evaluate(params, fields, intensity, params["freq"], params["temperature"], **config.settings)
                    if loss < best["loss"]:
                        best.update(loss=loss, params=params, spectrum=spectrum.detach().clone())
            cache[key] = loss
            return loss
        first = evaluate(x0)
        if initial_loss is None:
            initial_loss = first
        if not dimension:
            convergence.append(True)
            messages.append("All numeric parameters fixed; evaluated this basis once")
        elif backend == "scipy":
            from scipy.optimize import minimize
            result = minimize(evaluate, x0=x0, method="COBYLA",
                              constraints=[{"type": "ineq", "fun": lambda x: np.asarray(config.margins(x, basis))}],
                              options={"maxiter": allocation, "rhobeg": .1, "tol": 1e-6, "catol": 1e-9})
            convergence.append(bool(result.success))
            messages.append(str(result.message))
        else:
            _mars_optimizer(evaluate, x0, allocation, fields, intensity, seed+index, _method)
            convergence.append(False)
            messages.append(f"MaRS {_method} budget completed; best feasible evaluated point retained")
    if best["params"] is None:
        raise RuntimeError("No feasible point produced a valid nonzero spectrum; no fit is saved")
    best.update(initial_loss=initial_loss, n_evaluations=len(cache), converged=all(convergence),
                message="; ".join(messages), config=config, backend=backend)
    return best


def main(argv=None):
    parser = argparse.ArgumentParser(description="Configurable COBYLA refinement for a triplet EPR slice")
    parser.add_argument("--input", help="Experimental two-row CSV, NPZ or (N,2) NPY")
    parser.add_argument("--field-unit", choices=tuple(FIELD_SCALES), default=None)
    parser.add_argument("--tpe-results", help="Parameter CSV; its saved fit config is inherited")
    parser.add_argument("--freq", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--harmonic", type=int, choices=(0, 1), default=None)
    parser.add_argument("--mesh", type=int, default=None)
    parser.add_argument("--budget", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42, help="MaRS backend seed; SciPy is deterministic")
    parser.add_argument("--backend", choices=("scipy", "mars"), default="scipy")
    parser.add_argument("--output-dir", default="results/tables")
    for name in ("g-iso", "D", "E", "ham-strain", "gauss", "lorentz", "pop-x", "pop-y", "pop-z"):
        parser.add_argument("--"+name, type=float, default=None)
    parser.add_argument("--basis", choices=POPULATION_BASES, default=None)
    add_fit_arguments(parser)
    args = parser.parse_args(argv)
    loaded = load_tpe_results(args.tpe_results) if args.tpe_results else None
    inherited = inherited_config(args.tpe_results, loaded) if loaded else None
    config = config_from_args(args, inherited=inherited, initial=loaded)
    if args.print_fit_config:
        print(json.dumps(config.to_dict(), indent=2, allow_nan=False))
        return 0
    if not args.input:
        parser.error("--input is required unless --print-fit-config is used")
    fields, intensity = load_spectrum(args.input, args.field_unit)
    values = config.values()
    result = run_cobyla(fields, intensity, values, freq=values["freq"], temperature=values["temperature"],
                        budget=args.budget, config=config, backend=args.backend, seed=args.seed)
    params = result["params"]
    csv_path, npz_path = save_results(args.output_dir, "cobyla", params, result["loss"], fields,
        result["spectrum"], intensity, freq=params["freq"], temperature=params["temperature"],
        **result["config"].settings, config=result["config"],
        extra={"initial_loss_mse": result["initial_loss"], "n_evaluations": result["n_evaluations"],
               "converged": int(result["converged"])})
    print(f"Initial loss: {result['initial_loss']:.8g}; best feasible loss: {result['loss']:.8g}")
    print(f"Parameters saved to {csv_path}")
    print(f"Best spectrum saved to {npz_path}")
    return 0


if __name__ == "__main__":
    main()
