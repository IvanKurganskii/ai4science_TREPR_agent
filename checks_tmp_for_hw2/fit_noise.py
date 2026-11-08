"""Read-only post-fit checks against a declared pre-laser noise interval.

K = (RSS / (N - p)) / noise_variance. K is descriptive: filtering, nonlinear
fitting and correlated samples prevent assigning an automatic F-test p-value.
See FIT_CHECKS.md for the input contract and interpretation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Keep parsing independent of any implementation writable by the checked agent.
from check_support.data import DatasetLoader, FIELD_SCALES, TIME_SCALES


class CheckError(ValueError):
    """Input or provenance is insufficient for a defensible comparison."""


@dataclass
class Spectrum:
    fields: np.ndarray
    values: np.ndarray
    times: np.ndarray | None = None
    comments: tuple[str, ...] = ()
    signal_key: str = ""


def input_path(value, root):
    path = Path(value)
    path = (path if path.is_absolute() else root / path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise CheckError(f"Input must be an existing file inside the project: {value}")
    return path


def comments(path):
    if path.suffix.lower() != ".csv":
        return ()
    result = []
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            if not line.lstrip().startswith("#"):
                break
            if len(line) > 20000 or len(result) >= 100:
                raise CheckError("CSV metadata is too large")
            result.append(line.strip())
    return tuple(result)


def array(value, name, ndim):
    raw = np.asarray(value)
    if np.iscomplexobj(raw) or raw.dtype.kind not in "iuf":
        raise CheckError(f"{name} must be real numeric data")
    result = np.asarray(raw, dtype=float)
    if result.ndim != ndim or not result.size or not np.isfinite(result).all():
        raise CheckError(f"{name}: invalid dimensions, empty array, NaN or Inf")
    return result


def axis(value, name):
    result = array(value, name, 1)
    if result.size < 2 or np.any(np.diff(result) <= 0):
        raise CheckError(f"{name} must contain at least two strictly increasing coordinates")
    return result


def read_map(path, args, *, measured=True):
    options = {"path": str(path)}
    if path.suffix.lower() == ".csv":
        options.update(field_unit=args.csv_field_unit, time_unit=args.csv_time_unit)
    if measured and args.dataset:
        options["dataset"] = args.dataset
    dataset = DatasetLoader(args.root).load(options)
    return Spectrum(dataset.fields_T, dataset.signal, dataset.times_s, comments(path), "signal")


def read_slice(path, field_unit):
    if path.suffix.lower() != ".csv":
        raise CheckError("A measured 1D slice must be a two-row CSV with its original comments")
    rows = np.loadtxt(path, delimiter=",", comments="#", encoding="utf-8-sig", ndmin=2)
    if rows.ndim != 2 or rows.shape[0] != 2:
        raise CheckError("Slice CSV must contain exactly two numeric rows: field, intensity")
    fields = axis(rows[0] * FIELD_SCALES[field_unit], "slice field")
    return Spectrum(fields, array(rows[1], "slice intensity", 1), comments=comments(path))


def read_prediction(path, args, dimension):
    if path.suffix.lower() == ".csv":
        if dimension == "1d":
            return read_slice(path, args.fit_field_unit or args.csv_field_unit)
        return read_map(path, args, measured=False)
    if path.suffix.lower() != ".npz":
        raise CheckError("Save the fitted spectrum as numeric NPZ or CSV; pickled NPY is not accepted")
    with np.load(path, allow_pickle=False) as archive:
        field_key = next((key for key in ("fields_T", "field") if key in archive), None)
        if field_key is None:
            raise CheckError("Fit NPZ needs fields_T (MaRS) or field (COBYLA), in tesla")
        fields = axis(archive[field_key], "fit field")
        key = args.fit_key or next((key for key in ("simulated", "fitted", "spectrum") if key in archive), None)
        if key is None or key not in archive:
            raise CheckError("Fit NPZ needs simulated, fitted or spectrum; otherwise supply --fit-key")
        values = array(archive[key], "fitted intensity", 1 if dimension == "1d" else 2)
        times = axis(archive["times_s"], "fit time") if dimension == "2d" and "times_s" in archive else None
        if dimension == "2d" and times is None:
            raise CheckError("2D fit NPZ needs times_s, in seconds")
    expected = (len(fields),) if times is None else (len(times), len(fields))
    if values.shape != expected:
        raise CheckError("Fit axes do not match the intensity array; expected (time, field) for 2D")
    return Spectrum(fields, values, times, signal_key=key)


def match_indices(source, wanted, name):
    """Exact grid correspondence up to roundoff; never interpolate or extrapolate."""
    positions = np.searchsorted(source, wanted)
    candidates = np.clip(positions, 0, len(source)-1)
    previous = np.clip(positions-1, 0, len(source)-1)
    use_previous = np.abs(source[previous]-wanted) < np.abs(source[candidates]-wanted)
    candidates = np.where(use_previous, previous, candidates)
    tolerance = max(float(np.min(np.diff(source))) * 1e-6, np.finfo(float).eps * max(1e-12, float(np.max(np.abs(source)))) * 8)
    if np.any(np.abs(source[candidates]-wanted) > tolerance) or len(np.unique(candidates)) != len(candidates):
        raise CheckError(f"{name} grids do not match; check units and the saved fit coordinates")
    return candidates


def header_value(lines, label):
    values = [line.split(":", 1)[1].strip() for line in lines
              if line.lower().startswith("# " + label.lower() + ":")]
    if len(set(values)) > 1:
        raise CheckError(f"Conflicting CSV metadata: {label}")
    return values[0] if values else None


def parent_2d(data, args):
    if args.source_2d:
        return input_path(args.source_2d, args.root)
    processed = header_value(data.comments, "Processed spectrum")
    if processed:
        # The existing legacy header stores a basename, not a raw data alias.
        if Path(processed).name == processed:
            return input_path(Path("results/spectra/processed") / processed, args.root)
        return input_path(processed, args.root)
    raise CheckError("Cannot resolve the processed 2D parent; supply --source-2d explicitly")


def slice_window(data, args):
    if args.slice_window_ns is not None:
        return np.asarray(args.slice_window_ns, dtype=float) * 1e-9
    text = header_value(data.comments, "Integration time range")
    if text is None:
        raise CheckError("Slice has no Integration time range; supply --slice-window-ns")
    match = re.fullmatch(r"\[\s*([-+\deE.]+)\s*,\s*([-+\deE.]+)\s*\]\s*(s|ms|us|ns|ps)", text)
    if match is None:
        raise CheckError("Invalid Integration time range metadata; supply --slice-window-ns")
    return np.asarray([float(match[1]), float(match[2])]) * TIME_SCALES[match[3]]


def selection(times, bounds, name, minimum=1):
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] > bounds[1]:
        raise CheckError(f"{name}: invalid interval")
    tolerance = float(np.min(np.diff(times))) * 1e-6
    selected = np.flatnonzero((times >= bounds[0]-tolerance) & (times <= bounds[1]+tolerance))
    if selected.size < minimum:
        raise CheckError(f"{name}: requires at least {minimum} samples")
    return selected


def weights_for_slice(times, operation, integration_unit):
    count = len(times)
    if operation == "point":
        if count != 1:
            raise CheckError("point requires exactly one source time sample")
        return np.ones(1)
    if count < 2:
        raise CheckError("Averaging/integration requires at least two source time samples")
    if operation == "mean":
        return np.full(count, 1 / count)
    if operation == "sum":
        return np.ones(count)
    if not integration_unit:
        raise CheckError("integral requires --integration-unit to preserve the slice amplitude units")
    spacing = np.diff(times) / TIME_SCALES[integration_unit]
    result = np.zeros(count)
    result[:-1] += spacing / 2
    result[1:] += spacing / 2
    return result


def projected_noise(parent, field_indices, selected_times, weights, noise_indices):
    """Use disjoint pretrigger blocks with the same window as the real slice.

    Applying weights before detrending retains within-window noise correlations;
    unlike sigma/sqrt(n), this does not assume independent temporal samples.
    """
    width = len(weights)
    if width > 1:
        reference = parent.times[selected_times] - parent.times[selected_times[0]]
    blocks = []
    centers = []
    for start in range(0, len(noise_indices)-width+1, width):
        indices = noise_indices[start:start+width]
        if np.any(np.diff(indices) != 1):
            raise CheckError("Noise window must contain consecutive time samples")
        if width > 1 and not np.allclose(parent.times[indices]-parent.times[indices[0]], reference,
                                         rtol=1e-7, atol=1e-15):
            raise CheckError("Pretrigger grid cannot reproduce the slice integration window")
        blocks.append(weights @ parent.values[np.ix_(indices, field_indices)])
        centers.append(float(np.mean(parent.times[indices])))
    if len(blocks) < 3:
        raise CheckError("Need at least three complete pretrigger blocks for this slice window")
    return np.asarray(centers), np.asarray(blocks)


def detrend_noise(times, values):
    if len(times) < 3 or np.ptp(times) <= 0:
        raise CheckError("At least three distinct pretrigger times are required")
    x = (times - np.mean(times)) / np.ptp(times)
    design = np.column_stack((np.ones(len(x)), x))
    coefficients, _, rank, _ = np.linalg.lstsq(design, values, rcond=None)
    if rank != 2:
        raise CheckError("Pretrigger linear regression is rank deficient")
    residual = values - design @ coefficients
    if np.max(np.abs(residual)) <= 64*np.finfo(float).eps*np.max(np.abs(values)):
        raise CheckError("Pretrigger variance is zero within numerical precision")
    variance = float(np.sum(residual**2) / ((len(times)-2) * values.shape[1]))
    if not np.isfinite(variance) or variance <= 0:
        raise CheckError("Pretrigger noise variance is zero or invalid")
    return residual, variance


def lag_correlation(values, axis_number):
    if values.shape[axis_number] < 3:
        return None
    left = np.take(values, range(values.shape[axis_number]-1), axis=axis_number).ravel()
    right = np.take(values, range(1, values.shape[axis_number]), axis=axis_number).ravel()
    left, right = left - np.mean(left), right - np.mean(right)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 0 else None


def metrics(measured, simulated, noise_variance, args):
    measured, simulated = np.asarray(measured), np.asarray(simulated)
    if not np.any(measured != 0) or not np.any(simulated != 0):
        raise CheckError("Measured or fitted spectrum is entirely zero")
    m, s = measured.ravel(), simulated.ravel()
    scale, offset, extra_parameters = 1.0, 0.0, 0
    if args.alignment != "none":
        design = s[:, None] if args.alignment == "scale" else np.column_stack((s, np.ones(s.size)))
        solution, _, rank, _ = np.linalg.lstsq(design, m, rcond=None)
        if rank != design.shape[1]:
            raise CheckError("Amplitude alignment is rank deficient")
        scale = float(solution[0])
        offset = float(solution[1]) if len(solution) > 1 else 0.0
        extra_parameters = design.shape[1]
    p = args.n_parameters + extra_parameters
    if p >= m.size:
        raise CheckError("N must exceed the number of independently fitted parameters")
    residual = measured - (scale * simulated + offset)
    rss = float(np.sum(residual**2))
    variance = rss / (m.size - p)
    k = variance / noise_variance
    if not np.isfinite(k):
        raise CheckError("Nonfinite residual/noise ratio")
    result = {"n_points": int(m.size), "n_parameters": int(p), "residual_df_nominal": int(m.size-p),
              "rss": rss, "rmse": float(np.sqrt(rss/m.size)), "residual_mean": float(np.mean(residual)),
              "noise_sigma": float(np.sqrt(noise_variance)), "k_variance_ratio": k,
              "rms_noise_ratio": float(np.sqrt(k)), "alignment": {"mode": args.alignment, "scale": scale,
              "offset": offset, "added_parameters": extra_parameters},
              "residual_lag1_field": lag_correlation(np.atleast_2d(residual), 1)}
    if residual.ndim == 2:
        result["residual_lag1_time"] = lag_correlation(residual, 0)
    return result


def check(args, dimension):
    args.root = Path(args.root).resolve()
    data_path = input_path(args.slice if dimension == "1d" else args.data, args.root)
    fit_path = input_path(args.fit, args.root)
    prediction = read_prediction(fit_path, args, dimension)
    warnings = ["K is descriptive; no automatic F-test or proof of noise independence is claimed"]
    noise_bounds = np.asarray(args.noise_window_ns, dtype=float) * 1e-9
    if not np.isfinite(noise_bounds).all() or noise_bounds[0] >= noise_bounds[1]:
        raise CheckError("--noise-window-ns requires finite START < STOP, entirely before the laser")
    if dimension == "1d":
        data = read_slice(data_path, args.csv_field_unit)
        source_path = parent_2d(data, args)
        parent = read_map(source_path, args)
        field_indices = match_indices(parent.fields, data.fields, "source/slice field")
        match_indices(data.fields, prediction.fields, "slice/fit field")
        if len(data.fields) != len(prediction.fields):
            raise CheckError("The fitted 1D spectrum must cover the full measured slice")
        window = slice_window(data, args)
        selected_times = selection(parent.times, window, "slice time window")
        if window[0] <= noise_bounds[1]:
            raise CheckError("Slice and pretrigger intervals overlap")
        operation = args.slice_operation or ("point" if window[0] == window[1] else None)
        if operation is None:
            raise CheckError("A nonzero slice window requires --slice-operation mean, sum or integral")
        weights = weights_for_slice(parent.times[selected_times], operation, args.integration_unit)
        reconstructed = (weights @ parent.values[np.ix_(selected_times, field_indices)]) * args.slice_scale + args.slice_offset
        amplitude = max(float(np.max(np.abs(reconstructed))), float(np.max(np.abs(data.values))), np.finfo(float).tiny)
        if not np.allclose(reconstructed, data.values, rtol=1e-6, atol=amplitude*1e-10):
            raise CheckError("Slice does not match its 2D parent/window/operation; supply the true parent and explicit slice scaling")
        noise_indices = selection(parent.times, noise_bounds, "pretrigger window", 3)
        noise_times, noise_values = projected_noise(parent, field_indices, selected_times, weights, noise_indices)
        noise_values = noise_values * args.slice_scale
        measured, simulated = data.values, prediction.values
        projection = {"operation": operation, "samples_per_slice": len(weights),
                      "noise_blocks_used": len(noise_times), "window_ns": (window*1e9).tolist(),
                      "slice_scale": args.slice_scale, "slice_offset": args.slice_offset}
        raw_alias = header_value(data.comments, "Source")
        if raw_alias:
            alias_path = (args.root / raw_alias).resolve()
            if not alias_path.is_relative_to(args.root) or not alias_path.is_file():
                warnings.append("Legacy Source path is unavailable; the processed parent was verified by reconstructing the slice")
    else:
        source_path = data_path
        parent = read_map(data_path, args)
        field_indices = match_indices(parent.fields, prediction.fields, "data/fit field")
        times = prediction.times + args.fit_time_offset_ns*1e-9
        time_indices = match_indices(parent.times, times, "data/fit time")
        mask = times > noise_bounds[1]
        if args.fit_window_ns is not None:
            lo, hi = np.asarray(args.fit_window_ns) * 1e-9
            if not np.isfinite([lo, hi]).all() or lo > hi:
                raise CheckError("Invalid --fit-window-ns")
            mask &= (times >= lo) & (times <= hi)
        if not mask.any():
            raise CheckError("No fit samples remain after excluding the pretrigger interval")
        measured = parent.values[np.ix_(time_indices[mask], field_indices)]
        simulated = prediction.values[mask]
        noise_indices = selection(parent.times, noise_bounds, "pretrigger window", 3)
        noise_times, noise_values = parent.times[noise_indices], parent.values[np.ix_(noise_indices, field_indices)]
        projection = {"operation": "2d", "shape": list(measured.shape),
                      "fit_window_ns": [float(times[mask][0]*1e9), float(times[mask][-1]*1e9)],
                      "fit_time_offset_ns": args.fit_time_offset_ns}
    noise_residual, noise_variance = detrend_noise(noise_times, noise_values)
    result = metrics(measured, simulated, noise_variance, args)
    result.update({"noise_rows": len(noise_times), "noise_fields": int(noise_values.shape[1]),
                   "noise_df_nominal": int((len(noise_times)-2)*noise_values.shape[1]),
                   "noise_lag1_time": lag_correlation(noise_residual, 0),
                   "noise_lag1_field": lag_correlation(noise_residual, 1)})
    hashes = {}
    for path in dict.fromkeys((data_path, source_path, fit_path)):
        with path.open("rb") as handle:
            hashes[path.relative_to(args.root).as_posix()] = hashlib.file_digest(handle, "sha256").hexdigest()
    if args.max_k is None:
        status, code = "diagnostic", 2
        warnings.append("No acceptance threshold: record a justified --max-k before fitting")
    else:
        status = "passed" if result["k_variance_ratio"] <= args.max_k else "failed"
        code = 0 if status == "passed" else 1
    return {"status": status, "dimension": dimension, "max_k": args.max_k,
            "noise_window_ns": args.noise_window_ns, "source_2d": source_path.relative_to(args.root).as_posix(),
            "projection": projection, "metrics": result, "sha256": hashes, "warnings": warnings}, code


class JSONParser(argparse.ArgumentParser):
    def error(self, message):
        self.exit(2, json.dumps({"status": "invalid", "error": message[:500]}, ensure_ascii=False)+"\n")


def parser(dimension):
    cli = JSONParser(description=__doc__)
    cli.add_argument("--root", type=Path, default=Path.cwd(), help="Project root; defaults to cwd, pass explicitly when invoking externally")
    cli.add_argument("--slice" if dimension == "1d" else "--data", required=True)
    cli.add_argument("--fit", required=True, help="Saved prediction NPZ/CSV, not a table of parameters")
    cli.add_argument("--noise-window-ns", required=True, type=float, nargs=2, metavar=("START", "STOP"),
                     help="User-confirmed pre-laser interval; current measurements permit 0 500")
    cli.add_argument("--n-parameters", required=True, type=int, help="Independent fitted parameters, including nuisance parameters")
    cli.add_argument("--max-k", type=float, help="Predeclared upper bound for K; omission gives diagnostic status/exit 2")
    cli.add_argument("--alignment", choices=("none", "scale", "affine"), default="none")
    cli.add_argument("--csv-field-unit", choices=tuple(FIELD_SCALES), default="G")
    cli.add_argument("--csv-time-unit", choices=tuple(TIME_SCALES), default="ns")
    cli.add_argument("--fit-field-unit", choices=tuple(FIELD_SCALES), help="1D prediction CSV field units")
    cli.add_argument("--fit-key", help="Prediction intensity key in an NPZ")
    cli.add_argument("--dataset", help="Measured dataset name for a multi-dataset PKL")
    cli.add_argument("--report", help="Optional NEW JSON report inside the project; existing files are never overwritten")
    if dimension == "1d":
        cli.add_argument("--source-2d", help="Override the processed parent if the slice metadata is incomplete")
        cli.add_argument("--slice-window-ns", type=float, nargs=2)
        cli.add_argument("--slice-operation", choices=("point", "mean", "sum", "integral"))
        cli.add_argument("--integration-unit", choices=tuple(TIME_SCALES))
        cli.add_argument("--slice-scale", type=float, default=1.0)
        cli.add_argument("--slice-offset", type=float, default=0.0)
    else:
        cli.add_argument("--fit-window-ns", type=float, nargs=2)
        cli.add_argument("--fit-time-offset-ns", type=float, default=0.0)
    return cli


def main(dimension, argv=None):
    args = parser(dimension).parse_args(argv)
    try:
        if args.n_parameters < 0:
            raise CheckError("--n-parameters must be nonnegative")
        if args.max_k is not None and (not np.isfinite(args.max_k) or args.max_k <= 0):
            raise CheckError("--max-k must be finite and positive")
        for key in ("slice_scale", "slice_offset", "fit_time_offset_ns"):
            if hasattr(args, key) and not np.isfinite(getattr(args, key)):
                raise CheckError(f"{key} must be finite")
        if getattr(args, "slice_scale", 1) == 0:
            raise CheckError("--slice-scale must be nonzero")
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            report, code = check(args, dimension)
        if args.report:
            report_path = Path(args.report)
            report_path = (report_path if report_path.is_absolute() else args.root / report_path).resolve()
            if not report_path.is_relative_to(args.root):
                raise CheckError("Report must stay inside the project")
            with report_path.open("x", encoding="utf-8") as handle:
                handle.write(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2)+"\n")
        print(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2))
        return code
    except (ValueError, OSError, KeyError, FloatingPointError) as error:
        print(json.dumps({"status": "invalid", "error": str(error)[:700]}, ensure_ascii=False))
        return 2

