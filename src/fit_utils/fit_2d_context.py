"""High-J 2D TPE / MaRS COBYLA with a frozen 1D Hamiltonian and linewidths.

Only populations, triplet losses, intratriplet relaxation and the optional
rise time may vary. See README_2D.md before fitting. This module never updates
source data, the fit table, the diary or task status. CLI results are unchecked.
"""

import argparse
import copy
import csv
import hashlib
import json
import math
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.fit_utils.context_parameters import ContextFitConfig, POP_NAMES, TIME_NAMES, POPULATION_BASES, validate_population_basis, add_fit_arguments, config_from_args, number
    from src.mars_cleaning.data import DatasetLoader, EPRDataset
    from src.mars_simulation.engine import SimulationEngine
else:
    from .context_parameters import ContextFitConfig, POP_NAMES, TIME_NAMES, POPULATION_BASES, validate_population_basis, add_fit_arguments, config_from_args, number
    from ..mars_cleaning.data import DatasetLoader, EPRDataset
    from ..mars_simulation.engine import SimulationEngine


INVALID_LOSS = 1e30


def _json(path):
    path = Path(path)
    if path.stat().st_size > 1_000_000:
        raise ValueError(f"JSON exceeds 1 MB: {path}")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _fingerprint(path):
    path = Path(path).resolve()
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest}


def _snapshot(value):
    if not isinstance(value, dict) or set(value) != {"model", "hamiltonian", "settings"}:
        raise ValueError("spin_model must contain model, hamiltonian and settings")
    if value["model"] != "high_j":
        raise ValueError("These context fitters support high_j (S=1) only")
    required_settings = {"freq", "temperature", "mesh", "frame", "device", "dtype", "harmonic"}
    if not isinstance(value["settings"], dict) or not required_settings <= set(value["settings"]):
        raise ValueError("Use the complete saved 1D spin_model: freq, temperature, mesh, frame, "
                         "device, dtype and harmonic must be recorded")
    return SimulationEngine().make_spin_model("high_j", value["hamiltonian"], value["settings"])


def load_fixed_model(path):
    """Read a full spin_model JSON or the current 1D fitters' parameter CSV.

    CSV is the vertical parameter,value,unit format, in SI. Hamiltonian strain
    stays fixed; the population basis is inherited independently of loss/T1.
    Returns (validated snapshot, seed populations, source fingerprints).
    """
    path = Path(path).resolve()
    sources = [_fingerprint(path)]
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or not {"parameter", "value"} <= set(reader.fieldnames):
                raise ValueError("Use the vertical 1D parameter,value,unit CSV, not the summary fit table")
            params = {}
            for row in reader:
                key = row["parameter"]
                if key in params or not key:
                    raise ValueError("Duplicate or empty 1D parameter name")
                expected_unit = {"D": "Hz", "E": "Hz", "ham_strain": "Hz", "freq": "Hz",
                                 "gauss": "T", "lorentz": "T", "temperature": "K"}.get(key)
                if expected_unit and row.get("unit", "") not in ("", expected_unit):
                    raise ValueError(f"1D CSV {key} must be in {expected_unit}, as in the current fitter exports")
                params[key] = row["value"]
        settings = {}
        if params.get("fit_config"):
            contract_path = path.parent/params["fit_config"]
            contract = _json(contract_path)
            sources.append(_fingerprint(contract_path))
            settings = contract.get("settings", {})
    elif path.suffix.lower() == ".json":
        value = _json(path)
        if {"model", "hamiltonian", "settings"} <= set(value) and not set(value)-{
                "model", "hamiltonian", "settings", "population_basis"}:
            basis = validate_population_basis(value.get("population_basis", "xyz"))
            return _snapshot({k: value[k] for k in ("model", "hamiltonian", "settings")}), {"population_basis": basis}, sources
        if "spin_model" in value:
            snapshot = _snapshot(value["spin_model"])
            populations = value.get("input_populations", value.get("metadata", {}).get("input_populations", {}))
            seed = {k: populations[k] for k in POP_NAMES if k in populations}
            metadata = value.get("metadata", {})
            basis = value.get("population_basis", metadata.get("population_basis", "xyz"))
            if basis in ("molecular_xyz", "singlet_molecular_xyz"):
                basis = "xyz"  # Names used by older simulation metadata.
            seed["population_basis"] = validate_population_basis(basis)
            return snapshot, seed, sources
        params = value.get("parameters", value)
        settings = {}
        if value.get("fit_config"):
            contract_path = path.parent/value["fit_config"]
            settings = _json(contract_path).get("settings", {})
            sources.append(_fingerprint(contract_path))
    else:
        raise ValueError("--from-1d requires a full spin_model JSON or a 1D parameter CSV/JSON")
    required = ("g_iso", "D", "E", "ham_strain", "gauss", "lorentz", "basis", "freq", "temperature")
    missing = set(required)-set(params)
    if missing:
        raise ValueError(f"Incomplete 1D artifact: missing {sorted(missing)}; use its full parameter CSV or spin_model")
    basis = validate_population_basis(params["basis"])
    h = {target: float(params[source]) for target, source in (
        ("g", "g_iso"), ("D", "D"), ("E", "E"), ("gauss", "gauss"), ("lorentz", "lorentz"),
        ("ham_strain", "ham_strain"))}
    config = {k: float(params[k]) for k in ("freq", "temperature")}
    for k in ("mesh", "harmonic"):
        raw = params.get(k, settings.get(k))
        if raw is None or not math.isfinite(float(raw)) or not float(raw).is_integer():
            raise ValueError(f"1D artifact must record its integer {k}; use the full parameter CSV/contract")
        config[k] = int(float(raw))
    # These are the actual fixed conventions of the three 1D fitters.
    config.update(dtype="float64", device="auto", frame=[0., 0., 0.])
    snapshot = SimulationEngine().make_spin_model("high_j", h, config)
    seed = {"population_basis": basis}
    for name in POP_NAMES:
        if name in params or "pop_"+name[-1] in params:
            seed[name] = float(params.get(name, params.get("pop_"+name[-1])))
    return snapshot, seed, sources


class ContextFitProblem:
    """One measured 2D map, fixed spin model, explicit time origin and loss.

    Experimental times and time_zero_s/fit_window_s use seconds. The complete
    post-origin grid is simulated and maxabs-normalized before windowing. A
    positive calibration scale is fixed once (default: experimental maxabs on
    that full grid). The loss is MSE divided by scale**2, equivalent to the
    existing noise-K objective with the same grid/window and no alignment.
    """

    def __init__(self, fields_T, times_s, intensity, spin_model, config=None, *,
                 time_zero_s, fit_window_s=None, scale=None, engine=None):
        data = EPRDataset.from_arrays(fields_T, times_s, intensity)
        self.spin_model = _snapshot(spin_model)
        self.config = ContextFitConfig() if config is None else config
        if not isinstance(self.config, ContextFitConfig):
            raise ValueError("config must be a ContextFitConfig")
        self.time_zero_s = number(time_zero_s, "time_zero_s")
        shifted = data.times_s-self.time_zero_s
        time_tol = 32*np.finfo(float).eps*max(float(np.max(np.abs(data.times_s))), abs(self.time_zero_s), 1e-15)
        shifted[np.abs(shifted) <= time_tol] = 0.
        post = shifted >= 0
        if np.count_nonzero(post) < 2:
            raise ValueError("Need at least two measured times at/after the explicit time origin")
        self.fields_T = data.fields_T
        self.times_s = shifted[post]
        self.experimental_times_s = data.times_s[post]
        self.measured = data.signal[post].copy()
        self.mask = np.ones(self.times_s.size, dtype=bool)
        if fit_window_s is not None:
            if not isinstance(fit_window_s, (tuple, list)) or len(fit_window_s) != 2:
                raise ValueError("fit_window_s must be [low, high] on the experimental axis")
            lo, hi = [number(v, "fit_window_s") for v in fit_window_s]
            if lo < self.time_zero_s-time_tol or lo >= hi:
                raise ValueError("Fit window must start at/after the time origin and have low < high")
            self.mask = (self.experimental_times_s >= lo-time_tol) & (self.experimental_times_s <= hi+time_tol)
        if np.count_nonzero(self.mask) < 2:
            raise ValueError("Fit window must contain at least two post-origin measured times")
        self.fit_window_s = [float(self.experimental_times_s[self.mask][0]),
                             float(self.experimental_times_s[self.mask][-1])]
        peak = float(np.max(np.abs(self.measured)))
        if peak == 0 or not np.any(self.measured[self.mask] != 0):
            raise ValueError("Experimental map/fit window must be nonzero")
        self.scale = peak if scale is None else number(scale, "scale")
        if self.scale <= 0 or not np.isfinite(self.measured/self.scale).all():
            raise ValueError("scale must be positive and yield finite scaled data")
        self.engine = SimulationEngine() if engine is None else engine
        if self.fields_T.size*self.times_s.size > self.engine.max_points:
            raise ValueError(f"Map exceeds max_points={self.engine.max_points}; select a documented grid first")

    def with_config(self, config):
        result = copy.copy(self)
        result.config = config
        return result

    def evaluate(self, params):
        self.config.check_point(params)
        result = self.engine.simulate("high_j", "2d", spin_model=copy.deepcopy(self.spin_model),
            fields_T=self.fields_T, times_s=self.times_s,
            context={k: params[k] for k in POP_NAMES+TIME_NAMES[:-1]},
            population_mode="raw", normalization="maxabs", rise_time_us=params["rise_time_us"],
            population_basis=self.config.population_basis)
        spectrum = np.asarray(result["spectrum"])
        if np.iscomplexobj(spectrum) or spectrum.shape != self.measured.shape or not np.isfinite(spectrum).all():
            raise RuntimeError("2D simulator returned a nonfinite, complex or incompatible map")
        if not np.any(spectrum[self.mask] != 0):
            return INVALID_LOSS, None
        residual = self.measured[self.mask]/self.scale-spectrum[self.mask]
        loss = float(np.mean(residual**2))
        if not math.isfinite(loss) or loss >= INVALID_LOSS:
            raise RuntimeError("Nonfinite/overflowing 2D loss; check data calibration")
        return loss, spectrum.copy()


class _Objective:
    def __init__(self, problem, budget):
        self.problem, self.budget = problem, budget
        self.calls, self.invalid = 0, 0
        self.cache = OrderedDict()
        self.best = None

    def parameters(self, params):
        key = tuple(params[k] for k in self.problem.config.parameters)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        if self.calls >= self.budget:
            return INVALID_LOSS
        self.calls += 1
        try:
            self.problem.config.check_point(params)
        except ValueError:
            loss, spectrum = INVALID_LOSS, None
        else:
            # Backend failures are errors, never successful fits or penalty-only trials.
            loss, spectrum = self.problem.evaluate(params)
        if spectrum is None:
            self.invalid += 1
        elif self.best is None or loss < self.best["loss"]:
            self.best = {"params": copy.deepcopy(params), "loss": loss, "spectrum": spectrum}
        self.cache[key] = loss
        if len(self.cache) > 256:
            self.cache.popitem(last=False)
        return loss

    def vector(self, vector):
        try:
            params = self.problem.config.decode(vector)
        except ValueError:
            if self.calls < self.budget:
                self.calls += 1
                self.invalid += 1
            return INVALID_LOSS
        return self.parameters(params)

    def result(self, method, initial_loss):
        if self.best is None:
            raise RuntimeError("No valid nonzero 2D candidate; no fit is saved")
        return {**self.best, "config": self.problem.config, "method": method,
                "initial_loss": initial_loss, "n_evaluations": self.calls,
                "n_invalid": self.invalid, "n_free_model": self.problem.config.n_free_model}


def run_tpe(problem, *, n_trials=300, seed=42):
    """Optuna TPE; the seeded initial model is included in the trial budget."""
    if type(n_trials) is not int or n_trials < 1 or type(seed) is not int:
        raise ValueError("n_trials must be positive and seed must be an integer")
    objective = _Objective(problem, n_trials)
    study = None
    if not problem.config.free_names:
        initial_loss = objective.parameters(problem.config.values())
    else:
        import optuna
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.enqueue_trial({k: problem.config.values()[k] for k in problem.config.free_names})
        def evaluate(trial):
            params = problem.config.suggest(trial)
            trial.set_user_attr("full_context_parameters", params)
            return objective.parameters(params)
        study.optimize(evaluate, n_trials=n_trials, show_progress_bar=False)
        initial_loss = study.trials[0].value
    return {**objective.result("tpe", initial_loss), "study": study}


def _mars_cobyla(objective, problem, budget, seed):
    import torch
    from mars.optimization import ParamSpec, ParameterSpace, Spectrum2DFitter
    from mars.optimization.objectives import MSEObjective
    class SeededSpace(ParameterSpace):
        def instrument_nevergrad(self):
            instrument = super().instrument_nevergrad()
            instrument.random_state.seed(seed)
            return instrument
    names = [f"coordinate_{i}" for i in range(len(problem.config.free_names))]
    space = SeededSpace([ParamSpec(name=k, bounds=(0., 1.), default=float(v))
                         for k, v in zip(names, problem.config.start_vector())])
    class Runner(Spectrum2DFitter):
        def _loss_from_params(self, params, **kwargs):
            return torch.as_tensor(objective.vector([params[k] for k in names]), dtype=torch.float64)
    def unused_simulator(*args, **kwargs):
        raise RuntimeError("Configured MaRS runner must use its validated objective callback")
    fitter = Runner(x1_exp=problem.fields_T, x2_exp=problem.times_s[problem.mask],
        y_exp=problem.measured[problem.mask]/problem.scale, param_space=space,
        spectra_simulator=unused_simulator, objective=MSEObjective(), norm_mode="none",
        device=torch.device("cpu"), dtype=torch.float64)
    fitter.fit(backend="ng", optimizer="Cobyla", budget=budget, seed=None,
               show_progress=False, return_best_spectrum=False)


def run_cobyla(problem, *, budget=300, seed=42):
    """MaRS Spectrum2DFitter / Nevergrad Cobyla, retaining the best feasible model."""
    minimum = len(problem.config.free_names)+2 if problem.config.free_names else 1
    if type(budget) is not int or budget < minimum or type(seed) is not int:
        raise ValueError(f"budget must be an integer >= {minimum}; seed must be an integer")
    objective = _Objective(problem, budget)
    initial_loss = objective.parameters(problem.config.values())
    if problem.config.free_names:
        _mars_cobyla(objective, problem, budget, seed)
    return objective.result("cobyla", initial_loss)


def run_two_stage(problem, *, n_trials=300, budget=300, seed=42):
    """TPE -> MaRS Cobyla, without changing bounds or the free/fixed mask."""
    global_result = run_tpe(problem, n_trials=n_trials, seed=seed)
    local_config = ContextFitConfig.from_dict(problem.config.to_dict(global_result["params"]))
    result = run_cobyla(problem.with_config(local_config), budget=budget, seed=seed)
    result.update(method="two-stage", tpe_loss=global_result["loss"],
                  tpe_evaluations=global_result["n_evaluations"],
                  tpe_invalid=global_result["n_invalid"], tpe_config=problem.config.to_dict())
    return result


def _parameter_path(name):
    return "simulation.rise_time_us" if name == "rise_time_us" else "simulation.context."+name


def refinement_snapshot(problem, result):
    """JSON for check_refit_2d.py; no Hamiltonian/width variables are permitted."""
    cfg, values = result["config"], result["params"]
    cfg.check_point(values)
    variables = [{"path": _parameter_path(k), "bounds": cfg.parameters[k]["bounds"]}
                 for k in cfg.free_names]
    relations = []
    if cfg.population_pivot is not None:
        relations.append({"target": _parameter_path(cfg.population_pivot), "constant": 1.,
                          "terms": {_parameter_path(k): -1. for k in POP_NAMES if k != cfg.population_pivot}})
    return {"model": "high_j", "dimension": "2d", "objective": "noise_k",
            "simulation": {"spin_model": copy.deepcopy(problem.spin_model),
                "context": {k: values[k] for k in POP_NAMES+TIME_NAMES[:-1]},
                "rise_time_us": values["rise_time_us"], "population_mode": "raw", "normalization": "maxabs",
                "population_basis": cfg.population_basis},
            "calibration": {"scale": problem.scale, "offset": 0.},
            "variables": variables, "relations": relations,
            "constraints": ([{"path": _parameter_path(cfg.population_pivot),
                              "bounds": cfg.parameters[cfg.population_pivot]["bounds"]}]
                            if cfg.population_pivot is not None else [])}


def artifact_paths(output_dir, name):
    if not isinstance(name, str) or not re.fullmatch(r"[\w-]{1,100}", name):
        raise ValueError("--name must contain 1..100 letters, digits, _ or - (use a new fit_id)")
    out = Path(output_dir).resolve()
    paths = {key: out/f"{name}_{suffix}" for key, suffix in (
        ("parameters", "params.json"), ("fit_config", "fit_config.json"),
        ("spectrum", "spectrum.npz"), ("report", "report.json"))}
    if any(p.exists() for p in paths.values()):
        raise FileExistsError("A result with this name exists; choose a new --name/fit_id")
    return paths


def save_results(problem, result, output_dir, name, *, source=None, fixed_model_sources=None):
    """Save new files only; never mark scientific acceptance or edit summary tables."""
    paths = artifact_paths(output_dir, name)
    snapshot = refinement_snapshot(problem, result)
    loss, normalized = problem.evaluate(result["params"])
    if normalized is None or not math.isclose(loss, result["loss"], rel_tol=1e-9, abs_tol=1e-14):
        raise ValueError("Best parameters do not reproduce the reported loss")
    predicted = normalized*problem.scale
    if not np.isfinite(predicted).all() or not np.any(predicted != 0):
        raise ValueError("Exported 2D fit must be finite and nonzero")
    metadata = {"axis_order": ["time", "field"], "field_unit": "T", "time_unit": "s",
                "time_zero_s": problem.time_zero_s, "fit_window_s": problem.fit_window_s,
                "normalization": "maxabs", "scale": problem.scale, "population_mode": "raw",
                "population_basis": problem.config.population_basis, "relaxation_basis": "eigen", "decay_basis": "xyz"}
    report = {"status": "optimized_unchecked", "name": name, "method": result["method"],
        "backend_cobyla": "MaRS Spectrum2DFitter / Nevergrad Cobyla",
        "loss_mse_scaled": loss, "initial_loss_mse_scaled": result["initial_loss"],
        "n_evaluations": result["n_evaluations"], "n_invalid": result["n_invalid"],
        "n_free_model": result["config"].n_free_model,
        "free_coordinates": list(result["config"].free_names),
        "dependent_population": result["config"].population_pivot,
        "best_parameters": result["params"], "fixed_spin_model": problem.spin_model,
        "source": source or {}, "fixed_model_sources": fixed_model_sources or [],
        **metadata, **{key: path.name for key, path in paths.items()}}
    if "tpe_loss" in result:
        report.update(tpe_loss_mse_scaled=result["tpe_loss"], tpe_evaluations=result["tpe_evaluations"],
                      tpe_invalid=result["tpe_invalid"], tpe_config=result["tpe_config"])
    documents = {"parameters": snapshot, "fit_config": result["config"].to_dict(), "report": report}
    rendered = {key: json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n"
                for key, value in documents.items()}
    paths["report"].parent.mkdir(parents=True, exist_ok=True)
    for key in ("parameters", "fit_config"):
        with paths[key].open("x", encoding="utf-8") as handle:
            handle.write(rendered[key])
    with paths["spectrum"].open("xb") as handle:
        np.savez_compressed(handle, fields_T=problem.fields_T, times_s=problem.times_s,
            spectrum=predicted, normalized_spectrum=normalized, experimental=problem.measured,
            experimental_times_s=problem.experimental_times_s, fit_mask=problem.mask,
            metadata_json=np.asarray(json.dumps(metadata, allow_nan=False)))
    with paths["report"].open("x", encoding="utf-8") as handle:
        handle.write(rendered["report"])
    return paths


def load_previous(path):
    path = Path(path).resolve()
    report = _json(path)
    if report.get("relaxation_basis") != "eigen" or report.get("decay_basis") != "xyz":
        raise ValueError("Previous report does not record the current T1=eigen/loss=xyz contract; "
                         "start a new fit from --from-1d rather than changing old kinetics silently")
    config = _json(path.parent/report["fit_config"])
    cfg = ContextFitConfig.from_dict(config)
    cfg.check_point(report["best_parameters"])
    return report, cfg.to_dict(report["best_parameters"])


def main(argv=None, *, default_method="two-stage"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("tpe", "cobyla", "two-stage"), default=default_method)
    parser.add_argument("--basis", "--population-basis", dest="basis", choices=POPULATION_BASES,
                        help="Initial population basis; defaults to the recorded 1D/previous 2D basis")
    parser.add_argument("--input", type=Path, help="2D CSV/NPZ/plain NumPy PKL")
    parser.add_argument("--from-1d", "--spin-model", dest="from_1d", type=Path,
                        help="Fixed full spin_model JSON or current 1D parameter CSV")
    parser.add_argument("--previous-fit", "--tpe-results", dest="previous_fit", type=Path,
                        help="Previous 2D report JSON: inherit best point, contract and setup")
    parser.add_argument("--time-zero-ns", type=float, help="Explicit state-preparation time on experimental axis (ns)")
    parser.add_argument("--fit-window-ns", type=float, nargs=2, metavar=("START", "STOP"))
    parser.add_argument("--scale", type=float, help="Fixed positive amplitude; default post-origin experimental maxabs")
    parser.add_argument("--field-unit", choices=("T", "mT", "G"), default=None)
    parser.add_argument("--time-unit", choices=("s", "ms", "us", "ns", "ps"), default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--signal-key", default=None)
    parser.add_argument("--layout", choices=("time_field", "field_time", "auto"), default=None)
    parser.add_argument("--n-trials", type=int, default=300)
    parser.add_argument("--budget", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/fits/2d")
    parser.add_argument("--name", help="New unique fit_id used for artifact names")
    add_fit_arguments(parser)
    args = parser.parse_args(argv)
    try:
        previous, inherited = load_previous(args.previous_fit) if args.previous_fit else ({}, None)
        if args.from_1d:
            snapshot, seeds, sources = load_fixed_model(args.from_1d)
            if previous and snapshot != previous["fixed_spin_model"]:
                raise ValueError("--from-1d disagrees with the previous fit's frozen spin model")
        elif previous:
            snapshot, seeds = _snapshot(previous["fixed_spin_model"]), {}
            sources = previous["fixed_model_sources"]
        else:
            snapshot, seeds, sources = None, {}, []
        cfg = config_from_args(args, inherited=inherited, initial=None if previous else seeds)
        if args.print_fit_config:
            print(json.dumps(cfg.to_dict(), indent=2, allow_nan=False))
            return 0
        if snapshot is None or not args.name:
            raise ValueError("--from-1d (or --previous-fit) and a new --name are required")
        artifact_paths(args.output_dir, args.name)  # Refuse overwrites before any fitting.
        source = copy.deepcopy(previous.get("source", {}).get("loader", {}))
        if args.input:
            source["path"] = str(args.input.resolve())
        if not source.get("path"):
            raise ValueError("--input is required for the first 2D fit")
        for key in ("dataset", "signal_key", "layout", "time_unit", "field_unit"):
            if getattr(args, key) is not None:
                source[key] = getattr(args, key)
        if Path(source["path"]).suffix.lower() == ".csv":
            source.setdefault("field_unit", "G")
            source.setdefault("time_unit", "ns")
        if previous and _fingerprint(source["path"])["sha256"] != previous["source"]["sha256"]:
            raise ValueError("Previous fit's input bytes changed; start a new fit from --from-1d")
        data = DatasetLoader().load(source)
        origin = args.time_zero_ns/1e9 if args.time_zero_ns is not None else previous.get("time_zero_s")
        if origin is None:
            raise ValueError("--time-zero-ns is required: the laser time is not inferred from the 0-500 ns noise window")
        window = ([v/1e9 for v in args.fit_window_ns] if args.fit_window_ns is not None
                  else previous.get("fit_window_s"))
        scale = args.scale if args.scale is not None else previous.get("scale")
        problem = ContextFitProblem(data.fields_T, data.times_s, data.signal, snapshot, cfg,
                                    time_zero_s=origin, fit_window_s=window, scale=scale)
        if args.method == "tpe":
            result = run_tpe(problem, n_trials=args.n_trials, seed=args.seed)
        elif args.method == "cobyla":
            result = run_cobyla(problem, budget=args.budget, seed=args.seed)
        else:
            result = run_two_stage(problem, n_trials=args.n_trials, budget=args.budget, seed=args.seed)
        paths = save_results(problem, result, args.output_dir, args.name,
            source={**_fingerprint(source["path"]), "loader": source}, fixed_model_sources=sources)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps({"status": "optimized_unchecked", "loss_mse_scaled": result["loss"],
                      "n_free_model": cfg.n_free_model,
                      "artifacts": {k: str(p) for k, p in paths.items()}}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
