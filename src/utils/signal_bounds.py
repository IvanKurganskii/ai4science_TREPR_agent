"""Read-only threshold detection of signal boundaries in 1D traces/2D maps.

Noise is sampled at each edge (10 points by default), centered on its mean,
and summarized by max absolute deviation. Consecutive exceedances locate
boundaries. 2D data use per-channel baselines and an RMS/maxabs profile, so
absorption and emission cannot cancel. See SIGNAL_BOUNDS.md for the CLI,
background assumptions, statuses and the distinction between boundaries/Z peaks.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


FIELD_SCALES = {"T": 1., "mT": 1e-3, "G": 1e-4}
TIME_SCALES = {"s": 1., "ms": 1e-3, "us": 1e-6, "ns": 1e-9, "ps": 1e-12}


def _number(value, name, *, minimum=0., strict=False):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result) or (result <= minimum if strict else result < minimum):
        raise ValueError(f"{name} must be finite and {'>' if strict else '>='} {minimum}")
    return result


def _integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _array(value, name):
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real")
    result = np.asarray(raw, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains NaN/Inf")
    return result


def _profile(residual, aggregation):
    absolute = np.abs(residual)
    peak = np.max(absolute, axis=1)
    if aggregation == "maxabs":
        return peak
    scaled = np.divide(absolute, peak[:, None], out=np.zeros_like(absolute), where=peak[:, None] > 0)
    return peak*np.sqrt(np.mean(scaled**2, axis=1))


def _runs(mask):
    changes = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def _reference(values, window, aggregation, factor, floor, precision):
    block = values[window]
    mean = np.mean(block, axis=0)
    profile = _profile(values-mean, aggregation)
    noise = float(np.max(profile[window]))
    threshold = float(factor*max(noise, floor, precision))
    if not math.isfinite(threshold):
        raise ValueError("Noise threshold overflows; rescale intensities/factor")
    public = {"baseline_mean": float(np.mean(mean)), "baseline_min": float(np.min(mean)),
        "baseline_max": float(np.max(mean)), "raw_abs_max": float(np.max(np.abs(block))),
        "centered_abs_max": float(np.max(np.abs(block-mean))),
        "noise_profile_max": noise, "noise_profile_mean": float(np.mean(profile[window])),
        "threshold": threshold, "zero_noise_estimate": noise == 0.}
    first, stop, _ = window.indices(len(values))
    public["reference_indices"] = [first, stop-1]
    return {"mean": mean, "profile": profile, "public": public}


def _scan(axis, reference, side, noise_points, min_run, margin):
    above = reference["profile"] > reference["public"]["threshold"]
    if side == "left":
        above[:noise_points] = False
    else:
        above[-noise_points:] = False
    runs = [(int(a), int(b)) for a, b in _runs(above) if b-a >= min_run]
    result = {"status": "not_found", "index": None, "coordinate": None,
              "padded_index": None, "padded_coordinate": None, "noise": reference["public"]}
    if not runs:
        result["reason"] = "no_sustained_threshold_exceedance"
        return result
    first, stop = runs[0] if side == "left" else runs[-1]
    index = first if side == "left" else stop-1
    padded = max(0, index-margin) if side == "left" else min(len(axis)-1, index+margin)
    result.update(status="found", index=index, coordinate=float(axis[index]),
        padded_index=padded, padded_coordinate=float(axis[padded]),
        confirmed_at_index=first+min_run-1 if side == "left" else stop-min_run,
        above_threshold_run=[first, stop-1], padding_clipped=abs(padded-index) < margin,
        boundary_profile_value=float(reference["profile"][index]))
    return result


def _peak_candidates(axis, values, references, sides, min_run):
    """First/last prominent stable feature; physical assignment remains external."""
    from scipy.signal import find_peaks
    result = {"status": "not_found", "confirmed_z_peaks": False,
              "left": None, "right": None, "separation": None, "midpoint": None}
    found = [side["index"] for side in sides.values() if side["status"] == "found"]
    if not found:
        return result
    lo = sides.get("left", {}).get("index")
    hi = sides.get("right", {}).get("index")
    lo = 0 if lo is None else lo
    hi = len(axis)-1 if hi is None else hi
    for side, item in sides.items():
        if item["status"] != "found":
            continue
        reference = references[side]
        profile, threshold = reference["profile"], reference["public"]["threshold"]
        peaks, _ = find_peaks(profile, height=threshold, prominence=threshold, distance=min_run)
        valid = np.zeros(len(axis), dtype=bool)
        for first, stop in _runs(profile > threshold):
            if stop-first >= min_run:
                valid[first:stop] = True
        peaks = [int(i) for i in peaks if lo <= i <= hi and valid[i]]
        if not peaks:
            continue
        index = peaks[0] if side == "left" else peaks[-1]
        signed = float(values[index, 0]-reference["mean"][0]) if values.shape[1] == 1 else None
        result[side] = {"index": index, "coordinate": float(axis[index]),
            "profile_height": float(profile[index]), "centered_intensity": signed,
            "polarity": ("a" if signed > 0 else "e" if signed < 0 else "0") if signed is not None else None}
    left, right = result["left"], result["right"]
    if left is not None and right is not None and left["index"] < right["index"]:
        result.update(status="candidates", separation=right["coordinate"]-left["coordinate"],
                      midpoint=left["coordinate"]+(right["coordinate"]-left["coordinate"])/2)
    elif left is not None or right is not None:
        result["status"] = "insufficient_distinct_peaks"
    return result


def find_signal_bounds(axis, intensity, *, sample_axis=0, direction="both", noise_points=10,
                       factor=3., min_run=3, margin=3, aggregation="rms", noise_floor=0.,
                       peak_candidates=False, axis_kind="generic", axis_unit="a.u."):
    """Return compact JSON-compatible diagnostics; input arrays are never changed.

    axis must increase strictly. For 2D intensity, sample_axis identifies the
    search dimension; the other dimension is collapsed only after subtracting
    each channel's edge mean. Backward detection checks the end background
    against the assumed initial background when two disjoint references exist.
    Indices are zero-based in these input axes; margin affects padded bounds only.
    """
    x, raw = _array(axis, "axis"), _array(intensity, "intensity")
    noise_points = _integer(noise_points, "noise_points", 3)
    min_run = _integer(min_run, "min_run", 1)
    margin = _integer(margin, "margin", 0)
    factor = _number(factor, "factor", minimum=1., strict=True)
    noise_floor = _number(noise_floor, "noise_floor")
    if direction not in ("both", "forward", "backward") or aggregation not in ("rms", "maxabs"):
        raise ValueError("direction must be both/forward/backward; aggregation must be rms/maxabs")
    if x.ndim != 1 or raw.ndim not in (1, 2):
        raise ValueError("axis must be a vector; intensity must be 1D or 2D")
    if x.size < noise_points+min_run or not np.all(np.diff(x) > 0) or not math.isfinite(float(x[-1]-x[0])):
        raise ValueError("Axis must be finite, strictly increasing, and contain noise_points+min_run samples")
    if axis_kind not in ("generic", "field", "time"):
        raise ValueError("axis_kind must be generic, field or time")
    if axis_kind == "field" and np.any(x < 0):
        raise ValueError("Magnetic field must be nonnegative")
    sample_axis = _integer(sample_axis, "sample_axis", 0)
    if sample_axis >= raw.ndim:
        raise ValueError("sample_axis is outside the intensity dimensions")
    values = raw[:, None] if raw.ndim == 1 else np.moveaxis(raw, sample_axis, 0)
    if values.shape[0] != x.size or values.shape[1] < 1:
        raise ValueError("Intensity dimensions do not match axis")
    if type(peak_candidates) is not bool:
        raise ValueError("peak_candidates must be boolean")
    disjoint = x.size >= 2*noise_points+min_run
    if direction != "forward" and not disjoint:
        raise ValueError("Backward/both search needs two disjoint noise windows plus min_run samples")
    precision = float(64*np.finfo(float).eps*float(np.max(np.abs(values))))
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            references = {"left": _reference(values, slice(0, noise_points), aggregation, factor, noise_floor, precision)}
            if disjoint:
                references["right"] = _reference(values, slice(-noise_points, None), aggregation, factor, noise_floor, precision)
    except FloatingPointError as exc:
        raise ValueError("Intensity magnitudes overflow; rescale the data") from exc
    warnings = []
    comparison = {"status": "not_checked", "reason": "only_one_reference_window"}
    if "right" in references:
        left, right = references["left"], references["right"]
        delta = float(_profile((right["mean"]-left["mean"])[None, :], aggregation)[0])
        baseline_ok = delta <= left["public"]["threshold"]
        noise_ok = right["public"]["noise_profile_max"] <= left["public"]["threshold"]
        comparison = {"status": "compatible" if baseline_ok and noise_ok else "incompatible",
            "baseline_difference_profile": delta, "initial_threshold": left["public"]["threshold"],
            "baseline_compatible": baseline_ok, "noise_compatible": noise_ok}
        if comparison["status"] == "incompatible":
            warnings.append("End background differs from start: signal, baseline drift or changing noise; end is unestablished")
    names = ("left", "right") if direction == "both" else ("left",) if direction == "forward" else ("right",)
    sides = {}
    for name in names:
        reference = references[name]
        side = _scan(x, reference, name, noise_points, min_run, margin)
        if name == "right" and comparison["status"] == "incompatible":
            side = {"status": "not_found", "reason": "end_reference_incompatible_with_initial_background",
                    "index": None, "coordinate": None, "padded_index": None, "padded_coordinate": None,
                    "noise": reference["public"]}
        sides[name] = side
    count = sum(side["status"] == "found" for side in sides.values())
    status = "found" if count == len(names) else "partial" if count else "not_found"
    span = midpoint = None
    if len(sides) == 2 and count == 2:
        if sides["left"]["index"] > sides["right"]["index"]:
            status = "inconsistent"
            warnings.append("Independent left/right searches returned reversed boundaries")
        else:
            a, b = sides["left"]["coordinate"], sides["right"]["coordinate"]
            span, midpoint = b-a, a+(b-a)/2
    if any(r["public"]["zero_noise_estimate"] for r in references.values()):
        warnings.append("A reference has zero measured noise; numerical/user floor controls its threshold")
    report = {"status": status, "axis_kind": axis_kind, "axis_unit": axis_unit,
        "index_origin": 0, "n_samples": int(x.size), "n_channels": int(values.shape[1]),
        "configuration": {"direction": direction, "noise_points": noise_points, "factor": factor,
            "min_run": min_run, "margin": margin, "aggregation": "absolute" if raw.ndim == 1 else aggregation,
            "noise_floor": noise_floor, "numerical_floor": precision},
        "reference_assumption": "Initial window is signal-free; end reference must agree with it",
        "edge_comparison": comparison, "boundaries": sides,
        "threshold_span": span, "threshold_midpoint": midpoint, "warnings": warnings}
    if peak_candidates:
        report["edge_peak_candidates"] = _peak_candidates(x, values, references, sides, min_run)
    json.dumps(report, allow_nan=False)
    return report


def _unit_scale(unit, axis_kind):
    scales = FIELD_SCALES if axis_kind == "field" else TIME_SCALES
    if unit not in scales:
        raise ValueError(f"Unsupported {axis_kind} unit {unit!r}; choose {tuple(scales)}")
    return scales[unit]


def load_trace(path, *, axis_kind="field", field_unit=None, time_unit=None, signal_key=None):
    """Two-row CSV, numeric (N,2) NPY, or NPZ; SI coordinates, no pickle."""
    path = Path(path)
    requested = field_unit if axis_kind == "field" else time_unit
    unit = ("G" if axis_kind == "field" else "ns") if path.suffix.lower() == ".csv" else ("T" if axis_kind == "field" else "s")
    native = None
    if path.suffix.lower() == ".csv":
        rows = np.loadtxt(path, delimiter=",", comments="#", encoding="utf-8-sig", ndmin=2)
        if rows.shape[0] != 2:
            raise ValueError("1D CSV must contain exactly two numeric rows: coordinate and intensity")
        x, y = rows
    elif path.suffix.lower() == ".npy":
        rows = np.load(path, allow_pickle=False)
        if rows.ndim != 2 or rows.shape[1] != 2:
            raise ValueError("1D NPY must have shape (N,2)")
        x, y = rows[:, 0], rows[:, 1]
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            keys = ("fields_T", "field") if axis_kind == "field" else ("times_s", "time", "times")
            axis_key = next((k for k in keys if k in data), None)
            key = signal_key or next((k for k in ("intensity", "spectrum", "signal", "experimental", "simulated") if k in data), None)
            if axis_key is None or key not in data:
                raise ValueError("NPZ needs a coordinate vector and a numeric intensity vector; use --signal-key")
            x, y = data[axis_key].copy(), data[key].copy()
            if axis_key in ("fields_T", "times_s"):
                native = unit = "T" if axis_kind == "field" else "s"
            meta_key = "field_unit" if axis_kind == "field" else "time_unit"
            if meta_key in data:
                declared = str(data[meta_key].item())
                if native and declared != native:
                    raise ValueError("NPZ unit metadata contradicts its explicit SI axis key")
                native = unit = declared
    else:
        raise ValueError("1D input must be CSV, NPY or NPZ")
    if requested is not None and native is not None and requested != native:
        raise ValueError("Unit override contradicts the recorded NPZ axis unit")
    x = _array(x, "coordinate")*_unit_scale(requested or unit, axis_kind)
    y = _array(y, "intensity")
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("1D input must contain vectors; extract a slice first. "
                         "For map bounds use signal_bounds.py --kind 2d")
    return x, y, {"input_unit": requested or unit, "index_reference": "input coordinate order"}


def _window(axis, bounds, name, scale=1.):
    if bounds is None:
        return np.ones(len(axis), dtype=bool)
    lo, hi = [float(v)*scale for v in bounds]
    if not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi:
        raise ValueError(f"{name} requires finite START < STOP")
    mask = (axis >= lo) & (axis <= hi)
    if not np.any(mask):
        raise ValueError(f"{name} contains no measured points")
    return mask


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--input", type=Path, required=True)
    cli.add_argument("--kind", choices=("1d", "2d"), default="1d")
    cli.add_argument("--axis", choices=("field", "time"), default="field")
    cli.add_argument("--direction", choices=("both", "forward", "backward"), default="both")
    cli.add_argument("--noise-points", type=int, default=10)
    cli.add_argument("--factor", type=float, default=3.)
    cli.add_argument("--min-run", type=int, default=3)
    cli.add_argument("--margin", type=int, default=3)
    cli.add_argument("--aggregation", choices=("rms", "maxabs"), default="rms")
    cli.add_argument("--noise-floor", type=float, default=0.)
    cli.add_argument("--peak-candidates", action="store_true", help="Return first/last prominent stable features; these are not confirmed Z peaks")
    cli.add_argument("--field-unit", choices=tuple(FIELD_SCALES))
    cli.add_argument("--time-unit", choices=tuple(TIME_SCALES))
    cli.add_argument("--signal-key")
    cli.add_argument("--dataset")
    cli.add_argument("--layout", choices=("time_field", "field_time", "auto"), default="time_field")
    cli.add_argument("--time-window-ns", type=float, nargs=2, help="For a field search in a 2D map: select time channels")
    cli.add_argument("--field-window-T", type=float, nargs=2, help="For a time search in a 2D map: select field channels")
    cli.add_argument("--report", type=Path, help="Optional new JSON file; existing files are never replaced")
    args = cli.parse_args(argv)
    try:
        path = args.input.resolve()
        if args.kind == "1d":
            if args.time_window_ns is not None or args.field_window_T is not None or args.dataset:
                raise ValueError("Channel windows and --dataset require --kind 2d")
            x, y, info = load_trace(path, axis_kind=args.axis, field_unit=args.field_unit,
                time_unit=args.time_unit, signal_key=args.signal_key)
        else:
            if __package__ in (None, ""):
                import sys
                sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
                from src.mars_cleaning.data import DatasetLoader
            else:
                from ..mars_cleaning.data import DatasetLoader
            options = {"path": str(path), "layout": args.layout}
            for key in ("dataset", "signal_key", "field_unit", "time_unit"):
                if getattr(args, key) is not None:
                    options[key] = getattr(args, key)
            if path.suffix.lower() == ".csv":
                options.setdefault("field_unit", "G"); options.setdefault("time_unit", "ns")
            data = DatasetLoader().load(options)
            if args.axis == "field":
                if args.field_window_T is not None:
                    raise ValueError("--field-window-T selects channels only for --axis time")
                mask = _window(data.times_s, args.time_window_ns, "time window", 1e-9)
                x, y = data.fields_T, data.signal[mask].T
            else:
                if args.time_window_ns is not None:
                    raise ValueError("--time-window-ns selects channels only for --axis field")
                mask = _window(data.fields_T, args.field_window_T, "field window")
                x, y = data.times_s, data.signal[:, mask]
            info = {"dataset": data.name, "index_reference": "canonical increasing axis after DatasetLoader import",
                    "selected_channels": int(np.count_nonzero(mask))}
        report = find_signal_bounds(x, y, direction=args.direction, noise_points=args.noise_points,
            factor=args.factor, min_run=args.min_run, margin=args.margin, aggregation=args.aggregation,
            noise_floor=args.noise_floor, peak_candidates=args.peak_candidates,
            axis_kind=args.axis, axis_unit="T" if args.axis == "field" else "s")
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        report["source"] = {"path": str(path), "sha256": digest, "kind": args.kind, **info}
        payload = json.dumps(report, ensure_ascii=False, allow_nan=False)+"\n"
        if args.report:
            with args.report.open("x", encoding="utf-8") as handle:
                handle.write(payload)
        print(payload, end="")
        return 0 if report["status"] == "found" else 1
    except (OSError, ValueError, TypeError, KeyError, FloatingPointError) as exc:
        print(json.dumps({"status": "invalid", "error": str(exc)}, ensure_ascii=False, allow_nan=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
