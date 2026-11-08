#!/usr/bin/env python3
"""Two-stage 1D triplet fitting through MaRS: Optuna TPE then Nevergrad Cobyla.

All parameter bounds, values and fixed/free flags use the shared FitConfig:
--fit-config JSON, --bounds NAME LOW HIGH, --initial NAME VALUE,
--fix NAME VALUE and --vary NAME. D defaults to 500--3000 MHz (values in Hz).
Both stages retain the selected bounds; there is no implicit bounds reduction.
Fixed/dependent parameters are restored in the full model. Optuna Dashboard is
always disabled. See src/fit_utils/README.md for configuration and output fields.
"""
import argparse
import json
from pathlib import Path
import re
import sys

# Direct script invocation and `python -m src.agent_utils.fit_1d_slice` both work.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.fit_utils.fit_parameters import ALIASES, POPULATION_BASES, add_fit_arguments, config_from_args
from src.fit_utils import tpe_search as shared
from src.fit_utils.cobyla_fit import run_cobyla


def load_slice(path):
    """Compatibility helper returning CSV field in G and intensity."""
    field, intensity = shared.load_spectrum(path)
    return field * 1e4, intensity


def build_sample(params):
    mapped = {ALIASES.get(k, k): v for k, v in params.items()}
    return shared.build_sample_and_context(mapped)[0]


class TripletSpectraSimulator(shared.TripletSpectraSimulator):
    """Accept the original p_x/p_y/p_z aliases as well as shared pop_x/y/z."""
    def __call__(self, fields, params):
        return super().__call__(fields, {ALIASES.get(k, k): v for k, v in params.items()})


def run_fit(fields, intensity, config, *, n_trials=300, budget=300, seed=42):
    """Use MaRS for both stages, with the same physical configuration and loss."""
    initial = config.values()
    first = run_cobyla(fields, intensity, initial, config=config,
                       freq=initial["freq"], temperature=initial["temperature"],
                       budget=n_trials, backend="mars", seed=seed, _method="TPE")
    second = run_cobyla(fields, intensity, first["params"], config=first["config"],
                        freq=first["params"]["freq"], temperature=first["params"]["temperature"],
                        budget=budget, backend="mars", seed=seed)
    second["tpe_loss"] = first["loss"]
    second["tpe_evaluations"] = first["n_evaluations"]
    return second


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice", "--input", dest="slice", help="Two-row CSV, (N,2) NPY or NPZ slice")
    parser.add_argument("--field-unit", choices=tuple(shared.FIELD_SCALES), default=None)
    parser.add_argument("--freq", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None, help="K (legacy entry point default 80)")
    parser.add_argument("--harmonic", type=int, choices=(0, 1), default=None)
    parser.add_argument("--mesh", type=int, default=None)
    parser.add_argument("--n-trials", type=int, default=300, help="Total MaRS TPE evaluation budget")
    parser.add_argument("--budget", type=int, default=300, help="Total MaRS Cobyla evaluation budget")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--basis", choices=POPULATION_BASES, default=None)
    parser.add_argument("--output-dir", default="results/fits/1d")
    parser.add_argument("--name", default=None)
    add_fit_arguments(parser)
    args = parser.parse_args(argv)
    config = config_from_args(args, defaults={"temperature": 80.0})
    if args.print_fit_config:
        print(json.dumps(config.to_dict(), indent=2, allow_nan=False))
        return 0
    if not args.slice:
        parser.error("--slice is required unless --print-fit-config is used")
    fields, intensity = shared.load_spectrum(args.slice, args.field_unit)
    name = args.name or Path(args.slice).stem
    if not re.fullmatch(r"[\w-]{1,100}", name):
        parser.error("--name must contain 1..100 letters, digits, _ or -")
    print(f"Loaded {len(fields)} points; {config.n_free_model} free model coordinates")
    result = run_fit(fields, intensity, config, n_trials=args.n_trials, budget=args.budget, seed=args.seed)
    params = result["params"]
    names = (f"{name}_params.csv", f"{name}_spectrum.npz", f"{name}_fit_config.json")
    csv_path, npz_path = shared.save_results(args.output_dir, name, params, result["loss"], fields,
        result["spectrum"], intensity, freq=params["freq"], temperature=params["temperature"],
        **result["config"].settings, config=result["config"], filenames=names,
        extra={"initial_loss_mse": result["initial_loss"], "n_evaluations": result["n_evaluations"]})
    out = Path(args.output_dir)
    parameters = {k: v for k, v in params.items() if k != "pop"}
    parameters.update({alias: params[canonical] for alias, canonical in (("p_x", "pop_x"), ("p_y", "pop_y"), ("p_z", "pop_z"))})
    parameters["loss_mse"] = result["loss"]
    (out / f"{name}_params.json").write_text(json.dumps(parameters, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    report = {"status": "optimized_unchecked", "name": name, "slice": str(args.slice),
              "backend": "MaRS SpectrumFitter: Optuna TPE / Nevergrad Cobyla",
              "loss_mse": result["loss"], "tpe_loss_mse": result["tpe_loss"],
              "n_evaluations_tpe": result["tpe_evaluations"], "n_evaluations_cobyla": result["n_evaluations"],
              "n_free_model": result["config"].n_free_model,
              "free_coordinates": list(result["config"].free_names),
              "dependent_population": result["config"].population_pivot,
              "fit_config": str(out / names[2]), "parameters": parameters,
              "artifacts": {"csv": str(csv_path), "npz": str(npz_path)}}
    (out / f"{name}_report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(f"TPE loss: {result['tpe_loss']:.8g}; refined loss: {result['loss']:.8g}")
    print(f"Parameters saved to {csv_path}; spectrum saved to {npz_path}")
    return 0


if __name__ == "__main__":
    main()
