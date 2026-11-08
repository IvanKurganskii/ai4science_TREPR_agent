"""Read-only Pearson gate and paired Q comparison under joint D/E perturbations.

The saved model is regenerated through the independent MaRS adapter. All model
inputs, calibration and the original normalization divisor stay fixed, except
for D and E. No optimizer or project fitting implementation is imported.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

import fit_noise as noise
from fit_refinement import Problem, finite, fit_arrays, run_child


PERTURBATIONS = ({"D": -2.0, "E": -1.0},)
ROUNDING_EPS = float(64 * np.finfo(float).eps)


def fingerprint(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def pearson(measured, simulated):
    """Signed, centered correlation; a constant array has no correlation."""
    y = np.asarray(measured, dtype=float).ravel()
    f = np.asarray(simulated, dtype=float).ravel()
    if y.shape != f.shape or y.size < 3 or not np.isfinite(y).all() or not np.isfinite(f).all():
        raise noise.CheckError("Pearson requires matching finite arrays with at least three points")
    y, f = y-y.mean(), f-f.mean()
    y_norm, f_norm = np.linalg.norm(y), np.linalg.norm(f)
    if y_norm == 0 or f_norm == 0:
        return None
    value = float(np.dot(y/y_norm, f/f_norm))
    if not np.isfinite(value):
        raise noise.CheckError("Nonfinite Pearson correlation")
    return float(np.clip(value, -1., 1.))


def score(measured, simulated, sigma, n_parameters):
    y, f = np.asarray(measured, dtype=float), np.asarray(simulated, dtype=float)
    if y.shape != f.shape or not np.isfinite(y).all() or not np.isfinite(f).all():
        raise noise.CheckError("Invalid comparison arrays")
    if sigma <= 0 or not np.isfinite(sigma) or not 0 <= n_parameters < y.size:
        raise noise.CheckError("Q requires positive noise and N > p >= 0")
    residual = y-f
    rss = float(np.sum(residual**2))
    k = rss / ((y.size-n_parameters)*sigma**2)
    if not np.isfinite(k):
        raise noise.CheckError("Nonfinite Q")
    return {"pearson_r": pearson(y, f), "Q": float(np.sqrt(k)), "K": k,
            "rmse": float(np.sqrt(rss/y.size)), "rss": rss,
            "residual_mean": float(np.mean(residual)), "n_points": int(y.size),
            "n_parameters": int(n_parameters), "noise_sigma": float(sigma)}


def hamiltonian(state, dimension):
    simulation = state["simulation"]
    return simulation["spin_model"]["hamiltonian"] if dimension == "2d" else simulation["hamiltonian"]


def variant(state, dimension, factors):
    """Deep copy; do not reapply fitting relations/bounds to a control model."""
    result = copy.deepcopy(state)
    h = hamiltonian(result, dimension)
    d, e = finite(h["D"], "D"), finite(h["E"], "E")
    if d == 0 or not 0 <= e/d < 1/3:
        raise noise.CheckError("Require D != 0 and 0 <= E/D < 1/3")
    h["D"], h["E"] = factors["D"]*d, factors["E"]*e
    return result


def raw_simulation(engine, problem, state, fields, times):
    simulation = copy.deepcopy(state["simulation"])
    simulation["normalization"] = "none"
    result = engine.simulate(problem.model, problem.dimension, fields_T=fields,
                             **({} if times is None else {"times_s": times}), **simulation)
    expected = (len(fields),) if times is None else (len(times), len(fields))
    raw = noise.array(result["spectrum"], "regenerated spectrum", len(expected))
    if raw.shape != expected:
        raise noise.CheckError("Regenerated model does not match the full saved grid")
    return raw, result.get("metadata", {})


def calibrated(raw, state, divisor, mask):
    result = raw/divisor
    result = result*finite(state["calibration"]["scale"], "calibration.scale")
    result = result+finite(state["calibration"]["offset"], "calibration.offset")
    return result if mask is None else result[mask]


def compare_models(problem, engine, measured, saved, fields, times, mask, sigma, args):
    state = problem.state
    # Validate the original signed convention even when all D/E are fixed.
    variant(state, problem.dimension, {"D": 1., "E": 1.})
    raw, metadata = raw_simulation(engine, problem, state, fields, times)
    mode = state["simulation"].get("normalization", "none")
    if mode not in ("none", "maxabs"):
        raise noise.CheckError("Snapshot normalization must be none or maxabs")
    peak = float(np.max(np.abs(raw)))
    if peak == 0:
        raise noise.CheckError("Regenerated original spectrum is zero")
    divisor = peak if mode == "maxabs" else 1.
    regenerated = calibrated(raw, state, divisor, mask)
    reproduction = float(np.max(np.abs(regenerated-saved))/max(float(np.max(np.abs(saved))), np.finfo(float).tiny))
    if reproduction > args.spectrum_rtol:
        raise noise.CheckError("Snapshot does not reproduce the saved spectrum within --spectrum-rtol")
    # Compare exactly the submitted array, after confirming its provenance.
    original = score(measured, saved, sigma, args.n_parameters)
    controls = []
    h_initial = hamiltonian(state, problem.dimension)
    for factors in PERTURBATIONS:
        changed = variant(state, problem.dimension, factors)
        raw_control, _ = raw_simulation(engine, problem, changed, fields, times)
        predicted = calibrated(raw_control, state, divisor, mask)
        metrics = score(measured, predicted, sigma, args.n_parameters)
        tolerance = ROUNDING_EPS*max(1., original["Q"], metrics["Q"])
        h_control = hamiltonian(changed, problem.dimension)
        controls.append({"factors": dict(factors), "D_Hz": h_control["D"], "E_Hz": h_control["E"],
                         "metrics": metrics, "Q_original_minus_control": original["Q"]-metrics["Q"],
                         "rounding_tolerance": tolerance,
                         "original_not_worse": original["Q"] <= metrics["Q"]+tolerance})
    return {"original": original, "controls": controls,
            "D_Hz": h_initial["D"], "E_Hz": h_initial["E"],
            "spectrum_relative_error": reproduction,
            "normalization": {"snapshot_mode": mode, "frozen_divisor": divisor,
                              "calibration": copy.deepcopy(state["calibration"])},
            "backend_metadata": metadata,
            "comparison_passed": all(item["original_not_worse"] for item in controls)}


def worker(args, dimension, engine_factory=None):
    args.root = Path(args.root).resolve()
    parameter_path = noise.input_path(args.parameters, args.root)
    if parameter_path.stat().st_size > 100000:
        raise noise.CheckError("Parameter snapshot exceeds 100000 bytes")
    parameter_hash = fingerprint(parameter_path)
    baseline, _ = noise.check(args, dimension)
    measured, saved, fields, times, mask = fit_arrays(args, dimension)
    original = score(measured, saved, baseline["metrics"]["noise_sigma"], args.n_parameters)
    correlation = original["pearson_r"]
    if correlation is None:
        raise noise.CheckError("Pearson is undefined for a constant experiment or saved model")
    result = {"status": "failed", "dimension": dimension,
              "min_pearson": args.min_pearson, "pearson_passed": correlation >= args.min_pearson,
              "original": original, "perturbations": list(PERTURBATIONS), "controls": [],
              "comparison_passed": False, "max_k": args.max_k,
              "noise_passed": args.max_k is None or original["K"] <= args.max_k,
              "inputs_sha256": {**baseline["sha256"],
                                parameter_path.relative_to(args.root).as_posix(): parameter_hash},
              "noise_window_ns": args.noise_window_ns, "projection": baseline["projection"],
              "warnings": ["One fixed control does not establish a global optimum or parameter uniqueness",
                           "The same signed population numbers and basis are retained for every control",
                           "Control D/E may lie outside the original search bounds; no clipping or skipping is allowed"]}
    if not result["pearson_passed"]:
        result["reason"] = "pearson_below_threshold"
        result["controls_skipped"] = "Pearson gate failed; no MaRS simulations were run"
    else:
        problem = Problem(json.loads(parameter_path.read_text(encoding="utf-8-sig")), dimension)
        if args.n_parameters < len(problem.variables):
            raise noise.CheckError("n-parameters cannot be smaller than the snapshot variable count")
        if finite(problem.state["calibration"]["scale"], "calibration.scale") == 0:
            raise noise.CheckError("Zero calibration scale")
        if engine_factory is None:
            from check_support.engine import SimulationEngine
            engine_factory = SimulationEngine
        comparison = compare_models(problem, engine_factory(), measured, saved, fields, times, mask,
                                    original["noise_sigma"], args)
        result.update(comparison)
        result["status"] = "passed" if result["comparison_passed"] and result["noise_passed"] else "failed"
        result["reason"] = ("all_controls_not_better" if result["status"] == "passed" else
                            "control_better_than_original" if not result["comparison_passed"] else "max_k_exceeded")
    for relative, digest in result["inputs_sha256"].items():
        if fingerprint(args.root/relative) != digest:
            raise noise.CheckError("An input changed while the check was running")
    return result


def parser(dimension):
    cli = noise.parser(dimension)
    cli.description = __doc__
    cli.add_argument("--parameters", required=True, help="Complete JSON snapshot in the check_refit format")
    cli.add_argument("--min-pearson", default=.32, type=float, help="Predeclared lower bound for signed Pearson r (default: 0.32)")
    cli.add_argument("--spectrum-rtol", type=float, default=1e-5, help="Saved spectrum reproduction tolerance")
    cli.add_argument("--time-limit-seconds", type=float, default=300., help="Hard limit including backend imports")
    return cli


def validate_args(args):
    if not np.isfinite(args.min_pearson) or not -1 <= args.min_pearson <= 1:
        raise noise.CheckError("min-pearson must be finite and in [-1, 1]")
    if args.alignment != "none":
        raise noise.CheckError("Only --alignment none is allowed: all scales and offsets must stay fixed")
    if args.n_parameters < 0:
        raise noise.CheckError("n-parameters must be nonnegative")
    for key in ("time_limit_seconds", "spectrum_rtol"):
        if not np.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            raise noise.CheckError(f"{key} must be finite and positive")
    if args.max_k is not None and (not np.isfinite(args.max_k) or args.max_k <= 0):
        raise noise.CheckError("max-k must be finite and positive")
    for key in ("slice_scale", "slice_offset", "fit_time_offset_ns"):
        if hasattr(args, key) and not np.isfinite(getattr(args, key)):
            raise noise.CheckError(f"{key} must be finite")
    if getattr(args, "slice_scale", 1.) == 0:
        raise noise.CheckError("slice-scale must be nonzero")
    if not args.report:
        raise noise.CheckError("Supply --report with a NEW JSON path inside --root")


def report_path(args):
    root = Path(args.root).resolve()
    value = Path(args.report)
    path = (value if value.is_absolute() else root/value).resolve()
    if not path.is_relative_to(root) or not path.parent.is_dir():
        raise noise.CheckError("Report must have an existing parent inside --root")
    if path.exists():
        raise noise.CheckError("Report already exists; choose a new path")
    return path


def main(dimension, argv=None, *, in_worker=False):
    tokens = sys.argv[1:] if argv is None else argv
    args = parser(dimension).parse_args(tokens)
    try:
        validate_args(args)
        path = report_path(args)
        if in_worker:
            with open(os.devnull, "w") as quiet, contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                result = worker(args, dimension)
        else:
            result = run_child([sys.executable, "-B", str(Path(__file__).resolve()),
                                "--worker", dimension, *tokens], args.time_limit_seconds)
            with path.open("x", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2)+"\n")
            result = {key: value for key, value in result.items()
                      if key not in ("inputs_sha256", "backend_metadata", "warnings", "projection")}
            result["report"] = path.relative_to(Path(args.root).resolve()).as_posix()
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0 if result["status"] == "passed" else 1 if result["status"] == "failed" else 2
    except Exception as error:
        print(json.dumps({"status": "invalid", "error": str(error)[:700]}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker" and sys.argv[2] in ("1d", "2d"):
        raise SystemExit(main(sys.argv[2], sys.argv[3:], in_worker=True))
    raise SystemExit("Use check_fit_perturbation_1d.py or check_fit_perturbation_2d.py")
