"""
Configurable global parameter search for 1D EPR slices using Optuna TPE.

Use --fit-config JSON, repeatable --bounds NAME LOW HIGH, --fix NAME VALUE,
--vary NAME and --initial NAME VALUE. All numerical parameters use SI units;
D defaults to 500--3000 MHz. Disabled parameters retain their fixed values.
Populations remain nonnegative, normalized and within their individual bounds.
The resolved bounds, fixed/free flags and values are saved beside the fit.
See src/fit_utils/README.md for all three entry points and configuration rules.

Usage example::

    python src/fit_utils/tpe_search.py \
        --input results/spectra/slices/slice_001.npy \
        --freq 9.8e9 \
        --temperature 10.0 \
        --n-trials 500 \
        --seed 42 \
        --output-dir results/tables

Input format:
    CSV: two rows (field in G by default, intensity); NPY: (N, 2);
    NPZ: ``field`` and ``intensity``. NPY/NPZ fields default to tesla.
    Use --field-unit to specify another unit explicitly. No unit guessing.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

if __package__:
    from .fit_parameters import (DEFAULT_PARAMETERS, NUMERIC_NAMES, PARAMETER_UNITS,
                                 POP_NAMES, POPULATION_BASES, FitConfig, add_fit_arguments, config_from_args)
else:
    from fit_parameters import (DEFAULT_PARAMETERS, NUMERIC_NAMES, PARAMETER_UNITS,
                                POP_NAMES, POPULATION_BASES, FitConfig, add_fit_arguments, config_from_args)

# ---------------------------------------------------------------------------
# Optuna is imported by main; the shared simulator also works without Optuna.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Mars imports from the active mars-epr environment.
# ---------------------------------------------------------------------------
from mars import spin_model
from mars import spectra_manager
from mars.population.contexts import Context

# ---------------------------------------------------------------------------
# Device configuration – use CUDA if available, otherwise CPU.
# ---------------------------------------------------------------------------
dtype = torch.float64
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Compatibility export; optimizers use the resolved FitConfig, never dict slices.
PARAMETER_BOUNDS = {name: tuple(DEFAULT_PARAMETERS[name]["bounds"]) for name in
                    ("g_iso", "D", "E", "ham_strain", "gauss", *POP_NAMES)}
FIELD_SCALES = {"T": 1.0, "mT": 1e-3, "G": 1e-4, "uT": 1e-6}
# Unit-maxabs MSE is at most 4 for valid spectra.
INVALID_LOSS = 1e6


class InvalidSimulationError(ValueError):
    """The model returned no usable spectrum for a candidate."""


def normalize_populations(values):
    pop = np.asarray(values, dtype=float)
    if pop.shape != (3,) or not np.isfinite(pop).all() or np.any(pop < 0) or not np.any(pop > 0):
        raise ValueError("Populations must be three finite nonnegative values with positive sum")
    pop = pop / pop.max()
    return (pop / pop.sum()).tolist()


def validate_parameters(params):
    result = dict(params)
    result.setdefault("lorentz", 0.0)
    if "pop" not in result:
        result["pop"] = [result[k] for k in POP_NAMES]
    for key in ("g_iso", "D", "E", "ham_strain", "gauss", "lorentz"):
        result[key] = float(result[key])
        if not np.isfinite(result[key]):
            raise ValueError(f"{key} must be finite")
    if result["g_iso"] <= 0 or any(result[k] < 0 for k in ("ham_strain", "gauss", "lorentz")):
        raise ValueError("g_iso must be positive; broadening parameters must be nonnegative")
    if not any(result[k] > 0 for k in ("ham_strain", "gauss", "lorentz")):
        raise ValueError("At least one broadening parameter must be positive")
    if abs(result["E"]) > abs(result["D"])/3 + 1e-8:
        raise ValueError("The chosen ZFS convention requires |E| <= |D|/3")
    if result.get("basis") not in POPULATION_BASES:
        raise ValueError(f"basis must be one of {POPULATION_BASES}")
    pop = np.asarray(result["pop"], dtype=float)
    if pop.shape != (3,) or not np.isfinite(pop).all() or np.any(pop < 0):
        raise ValueError("Invalid populations")
    result["pop"] = pop.tolist() if np.isclose(pop.sum(), 1, rtol=0, atol=1e-10) else normalize_populations(pop)
    result.update(zip(("pop_x", "pop_y", "pop_z"), result["pop"]))
    return result


def _vector(values, name):
    raw = np.asarray(values)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real")
    if raw.ndim == 2 and 1 in raw.shape:
        raw = raw.reshape(-1)
    if raw.ndim != 1 or raw.size < 2:
        raise ValueError(f"{name} must be a vector with at least two points")
    array = np.asarray(raw, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def validate_spectrum(fields, intensity):
    fields, intensity = _vector(fields, "field"), _vector(intensity, "intensity")
    if fields.shape != intensity.shape:
        raise ValueError("Field and intensity shapes differ")
    if np.any(fields < 0) or np.any(np.diff(fields) <= 0):
        raise ValueError("Fields must be nonnegative and strictly increasing")
    if not np.any(intensity != 0):
        raise ValueError("Experimental spectrum is entirely zero")
    return fields, intensity


def normalized_mse(simulated, experimental):
    """MSE of independently maxabs-normalized real vectors; polarity is retained."""
    exp = torch.as_tensor(experimental, device=device)
    if exp.ndim == 2 and 1 in exp.shape:
        exp = exp.reshape(-1)
    if exp.is_complex() or exp.ndim != 1 or exp.numel() < 2:
        raise ValueError("Experimental spectrum must be a real vector with at least two points")
    exp = exp.to(dtype=dtype)
    sim = torch.as_tensor(simulated, device=device)
    if sim.ndim == 2 and 1 in sim.shape:
        sim = sim.reshape(-1)
    if sim.is_complex() or sim.shape != exp.shape or not bool(torch.isfinite(sim).all()):
        raise InvalidSimulationError("Simulation must be a finite real vector matching the data")
    sim = sim.to(dtype=dtype)
    if not bool(torch.isfinite(exp).all()) or not bool(exp.abs().max() > 0):
        raise ValueError("Experimental spectrum must be finite and nonzero")
    sim_max = sim.abs().max()
    if not bool(sim_max > 0):
        raise InvalidSimulationError("Simulation is entirely zero")
    return float(torch.mean((sim / sim_max - exp / exp.abs().max()) ** 2).item())


def save_results(out_dir, prefix, params, loss, fields, simulated, experimental, *,
                 freq, temperature, harmonic, mesh, extra=None, config=None, filenames=None):
    """Write numeric NPZ arrays and unambiguous, reproducible parameter CSV."""
    params = validate_parameters(params)
    fields, experimental = validate_spectrum(fields, experimental)
    loss_check = normalized_mse(simulated, experimental)
    if not np.isfinite(loss) or not np.isclose(loss, loss_check, rtol=1e-7, atol=1e-12):
        raise ValueError("Saved loss does not match the simulated spectrum")
    simulated = torch.as_tensor(simulated).detach().to(dtype=dtype).cpu().numpy().reshape(-1)
    # Positive scale puts the saved simulation on the experimental amplitude scale.
    scale = float(np.max(np.abs(experimental)) / np.max(np.abs(simulated)))
    if not np.isfinite(scale):
        raise ValueError("Amplitude scale is not finite")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = filenames or (("tpe_best_params.csv" if prefix == "tpe" else "cobyla_fit_results.csv"),
                          f"{prefix}_best_spectrum.npz", f"{prefix}_fit_config.json")
    csv_name, spectrum_name, config_name = names
    csv_path = out_dir / csv_name
    values = {k: params[k] for k in NUMERIC_NAMES if k in params}
    values.update(basis=params["basis"], freq=freq, temperature=temperature,
                  harmonic=harmonic, mesh=mesh, loss_mse=loss_check, scale=scale)
    values.update(extra or {})
    if config is not None:
        values["fit_config"] = config_name
        values["n_free_model"] = config.n_free_model
        full_values = dict(params, freq=freq, temperature=temperature)
        config.check_point(full_values)
        (out_dir / config_name).write_text(json.dumps(config.to_dict(), indent=2, allow_nan=False)+"\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["parameter", "value", "unit"])
        writer.writerows((k, v, PARAMETER_UNITS.get(k, "")) for k, v in values.items())
    npz_path = out_dir / spectrum_name
    np.savez(npz_path, field=fields, fields_T=fields, intensity=simulated * scale,
             simulated=simulated, experimental=experimental,
             simulated_normalized=simulated / np.max(np.abs(simulated)),
             experimental_normalized=experimental / np.max(np.abs(experimental)),
             field_unit="T", loss_mse=loss_check, scale=scale)
    return csv_path, npz_path

# ---------------------------------------------------------------------------
# Helper: build a MultiOrientedSample and associated Context with the desired
# population basis.
# ---------------------------------------------------------------------------

def build_sample_and_context(params: dict) -> tuple[spin_model.MultiOrientedSample, Context]:
    """Create a sample and a Context that holds the initial populations.

    Parameters
    ----------
    params:
        Dictionary with keys:
        - ``g_iso`` – isotropic g‑factor (float)
        - ``D``, ``E`` – ZFS parameters (Hz)
        - ``ham_strain`` – strain (Hz)
        - ``gauss`` – Gaussian broadening (Tesla)
        - ``lorentz`` – Lorentzian broadening (Tesla; default 0)
        - ``pop`` – list/array of three population numbers (must sum to 1)
        - ``basis`` – population basis: xyz, product, eigen, zfs, multiplet or zeeman
    """
    params = validate_parameters(params)
    # Isotropic g‑tensor – same value on all three axes
    g_tensor = spin_model.Interaction([params["g_iso"]] * 3, dtype=dtype, device=device)

    # Zero‑field splitting interaction
    zfs = spin_model.DEInteraction([params["D"], params["E"]], dtype=dtype, device=device)

    # Base spin system (triplet, S=1)
    base_spin_system = spin_model.SpinSystem(
        electrons=[1.0],
        g_tensors=[g_tensor],
        electron_electron=[(0, 0, zfs)],
        dtype=dtype,
        device=device,
    )

    # Use the selected Gaussian, Lorentzian and Hamiltonian broadening.
    sample = spin_model.MultiOrientedSample(
        base_spin_system=base_spin_system,
        ham_strain=params["ham_strain"],
        gauss=params["gauss"],
        lorentz=params["lorentz"],
        mesh=(params.get("mesh", 20), params.get("mesh", 20)),
        dtype=dtype,
        device=device,
    )

    context = Context(
        sample=sample,
        basis=params["basis"],
        init_populations=params["pop"],
        device=device,
        dtype=dtype,
    )
    return sample, context

# ---------------------------------------------------------------------------
# Simulator – wraps the Mars ``StationarySpectra`` creator.  The Context is used
# to obtain the transformed populations before the spectrum is evaluated.
# ---------------------------------------------------------------------------

class TripletSpectraSimulator:
    """Simulator for a triplet EPR spectrum with variable parameters.

    The ``__call__`` method receives a field tensor and a parameter dictionary
    and returns the simulated intensity array.
    """

    def __init__(self, freq: float, temperature: float, harmonic: int = 0, mesh: int = 20):
        if not np.isfinite(freq) or freq <= 0 or not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("Frequency and temperature must be finite and positive")
        # Very coarse meshes can suppress the powder spectrum in the installed MaRS.
        if harmonic not in (0, 1) or not isinstance(mesh, int) or isinstance(mesh, bool) or mesh < 5:
            raise ValueError("harmonic must be 0 or 1; mesh must be an integer >= 5")
        self.freq = freq
        self.temperature = temperature
        self.harmonic = harmonic
        self.mesh = mesh

    def __call__(self, fields: torch.Tensor, params: dict) -> torch.Tensor:
        fields = torch.as_tensor(_vector(torch.as_tensor(fields).detach().cpu().numpy(), "field"),
                                 dtype=dtype, device=device)
        if bool(torch.any(fields < 0)) or not bool(torch.all(torch.diff(fields) > 0)):
            raise ValueError("Fields must be nonnegative and strictly increasing")
        sample, context = build_sample_and_context(dict(params, mesh=self.mesh))
        creator = spectra_manager.StationarySpectra(
            freq=self.freq,
            sample=sample,
            temperature=self.temperature,
            context=context,
            harmonic=self.harmonic,
            device=device,
            dtype=dtype,
        )
        with torch.inference_mode():
            spectrum = creator(sample, fields)
        if spectrum.shape != fields.shape or not bool(torch.isfinite(spectrum).all()):
            raise InvalidSimulationError("MaRS returned an unexpected shape or nonfinite spectrum")
        return spectrum

# ---------------------------------------------------------------------------
# Data loading utilities.
# ---------------------------------------------------------------------------

def load_spectrum(path: str, field_unit: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load two-row CSV (default G), (N,2) NPY or NPZ (default T), without pickle."""
    p = Path(path)
    suffix = p.suffix.lower()
    unit = "G" if suffix == ".csv" else "T"
    if suffix == ".csv":
        data = np.loadtxt(p, delimiter=",", comments="#", encoding="utf-8-sig", ndmin=2)
        if data.shape[0] != 2:
            raise ValueError("CSV must contain exactly two data rows: field, intensity")
        fields, intensity = data
    elif suffix == ".npy":
        data = np.load(p, allow_pickle=False)
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError("NPY must contain exactly two columns: field, intensity")
        fields, intensity = data[:, 0], data[:, 1]
    elif suffix == ".npz":
        with np.load(p, allow_pickle=False) as data:
            fields, intensity = data["field"].copy(), data["intensity"].copy()
            if "field_unit" in data:
                unit = str(data["field_unit"].item())
    else:
        raise ValueError(f"Unsupported file format: {p.suffix}")
    unit = field_unit if field_unit is not None else unit
    if unit not in FIELD_SCALES:
        raise ValueError(f"Unsupported field unit {unit!r}; choose from {tuple(FIELD_SCALES)}")
    return validate_spectrum(_vector(fields, "field") * FIELD_SCALES[unit], intensity)

# ---------------------------------------------------------------------------
# Optuna objective – mean‑squared error between normalized spectra.
# ---------------------------------------------------------------------------

def objective(trial, fields_exp, intensity_exp, freq, temperature, harmonic=0, mesh=20, config=None):
    """Maxabs MSE over the supplied physical bounds and fixed/free mask."""
    fields_exp, intensity_exp = validate_spectrum(fields_exp, intensity_exp)
    if config is None:
        config = FitConfig({"freq": {"value": freq}, "temperature": {"value": temperature}},
                           {"harmonic": harmonic, "mesh": mesh})
    try:
        params = config.suggest(trial)
    except ValueError as exc:
        trial.set_user_attr("invalid_reason", str(exc))
        return INVALID_LOSS
    # Optuna best_params omits fixed and dependent parameters; retain the full model.
    params = validate_parameters(params)
    trial.set_user_attr("full_parameters", params)
    simulator = TripletSpectraSimulator(params["freq"], params["temperature"], **config.settings)
    try:
        return normalized_mse(simulator(torch.as_tensor(fields_exp, dtype=dtype, device=device), params), intensity_exp)
    except InvalidSimulationError as exc:
        trial.set_user_attr("invalid_reason", str(exc))
        return INVALID_LOSS


def run_tpe(fields, intensity, config=None, *, n_trials=500, seed=42):
    """Run TPE without writing files; return a complete best model and its spectrum."""
    fields, intensity = validate_spectrum(fields, intensity)
    config = FitConfig() if config is None else config
    if type(n_trials) is not int or n_trials < 1:
        raise ValueError("n_trials must be a positive integer")
    study = None
    if config.n_free_model == 0:
        params = validate_parameters(config.values())
        simulator = TripletSpectraSimulator(params["freq"], params["temperature"], **config.settings)
        spectrum = simulator(torch.as_tensor(fields, dtype=dtype, device=device), params)
        try:
            loss = normalized_mse(spectrum, intensity)
        except InvalidSimulationError as exc:
            raise RuntimeError("No valid nonzero spectrum; no fit is saved") from exc
        n_evaluations = 1
    else:
        import optuna
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=seed), direction="minimize")
        values = config.values()
        study.optimize(lambda trial: objective(trial, fields, intensity, values["freq"], values["temperature"],
                                               config=config), n_trials=n_trials)
        loss = study.best_value
        if not np.isfinite(loss) or loss >= INVALID_LOSS:
            raise RuntimeError("No trial produced a valid nonzero spectrum; no fit is saved")
        params = validate_parameters(study.best_trial.user_attrs["full_parameters"])
        config.check_point(params)
        simulator = TripletSpectraSimulator(params["freq"], params["temperature"], **config.settings)
        spectrum = simulator(torch.as_tensor(fields, dtype=dtype, device=device), params)
        n_evaluations = len(study.trials)
    return {"params": params, "loss": loss, "spectrum": spectrum, "study": study,
            "n_evaluations": n_evaluations, "config": config}


# ---------------------------------------------------------------------------
# Main entry point.
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Configurable global TPE search for a triplet EPR slice")
    parser.add_argument("--input", help="Experimental two-row CSV, (N,2) NPY or NPZ")
    parser.add_argument("--field-unit", choices=tuple(FIELD_SCALES), default=None,
                        help="Input unit (CSV default G; NPY/NPZ default T)")
    parser.add_argument("--freq", type=float, default=None, help="Microwave frequency (Hz; default 9.8e9)")
    parser.add_argument("--temperature", type=float, default=None, help="Temperature (K; default 10)")
    parser.add_argument("--n-trials", type=int, default=500, help="Number of Optuna trials")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--basis", choices=POPULATION_BASES, default=None)
    parser.add_argument("--output-dir", default="results/tables")
    parser.add_argument("--harmonic", type=int, choices=(0, 1), default=None)
    parser.add_argument("--mesh", type=int, default=None)
    add_fit_arguments(parser)
    args = parser.parse_args(argv)
    config = config_from_args(args)
    if args.print_fit_config:
        print(json.dumps(config.to_dict(), indent=2, allow_nan=False))
        return 0
    if not args.input:
        parser.error("--input is required unless --print-fit-config is used")
    fields, intensity = load_spectrum(args.input, args.field_unit)
    print(f"Loaded {len(fields)} points; running {args.n_trials} TPE trials ({config.n_free_model} free model coordinates)")
    result = run_tpe(fields, intensity, config, n_trials=args.n_trials, seed=args.seed)
    params = result["params"]
    csv_path, npz_path = save_results(args.output_dir, "tpe", params, result["loss"], fields,
        result["spectrum"], intensity, freq=params["freq"], temperature=params["temperature"],
        **config.settings, extra={"n_evaluations": result["n_evaluations"]}, config=config)
    print(f"Parameters saved to {csv_path}")
    print(f"Best spectrum saved to {npz_path}")
    return 0


if __name__ == "__main__":
    main()
