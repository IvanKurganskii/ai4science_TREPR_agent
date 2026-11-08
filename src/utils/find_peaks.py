"""Find maxima/minima in a real 1D spectrum, without physical assignment.

CSV/NPY/NPZ input and SI coordinate conventions match signal_bounds.py.
The first noise_points samples supply a constant baseline and max absolute
noise deviation. Filters are configurable; output contains positions,
intensities, prominence and width, never X/Y/Z or a/e labels. Input is read-only.
See FIND_PEAKS.md for commands, range selection and compact JSON output.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    from signal_bounds import FIELD_SCALES, TIME_SCALES, _array, _integer, _number, load_trace
else:
    from .signal_bounds import FIELD_SCALES, TIME_SCALES, _array, _integer, _number, load_trace


def find_extrema(axis, intensity, *, noise_points=10, factor=3., min_prominence=None,
                 min_abs_height=None, min_width=3., distance=3, extrema="both",
                 coordinate_range=None, noise_floor=0., axis_kind="field", axis_unit="T"):
    """Return all retained extrema, sorted by increasing coordinate.

    Maxima are found on centered intensity, minima on its negative. There is
    no sign restriction: a minimum above baseline is still a minimum. Widths
    are measured at half prominence, not assumed physical linewidths.
    Range selects peak coordinates; original neighbors are retained when
    detecting extrema and computing prominence/width, avoiding artificial
    peaks at cut edges. Distance applies separately to maxima and minima.
    """
    from scipy.signal import find_peaks as scipy_find_peaks

    x, y = _array(axis, "axis"), _array(intensity, "intensity")
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or x.size < 3:
        raise ValueError("axis and intensity must be equal-length real vectors with at least 3 points")
    if not np.all(np.diff(x) > 0) or not math.isfinite(float(x[-1]-x[0])):
        raise ValueError("axis must increase strictly and have a finite span")
    if axis_kind not in ("field", "time", "generic"):
        raise ValueError("axis_kind must be field, time or generic")
    if axis_kind == "field" and np.any(x < 0):
        raise ValueError("Magnetic field must be nonnegative")
    noise_points = _integer(noise_points, "noise_points", 3)
    if noise_points > x.size:
        raise ValueError("noise_points exceeds the number of measured samples")
    factor = _number(factor, "factor", minimum=1., strict=True)
    noise_floor = _number(noise_floor, "noise_floor")
    min_width = _number(min_width, "min_width")
    distance = _integer(distance, "distance", 1)
    if extrema not in ("both", "maxima", "minima"):
        raise ValueError("extrema must be both, maxima or minima")
    limits = [float(x[0]), float(x[-1])]
    if coordinate_range is not None:
        if not isinstance(coordinate_range, (list, tuple, np.ndarray)) or len(coordinate_range) != 2:
            raise ValueError("coordinate_range must be [START, STOP]")
        limits = [float(v) for v in coordinate_range]
        if not all(math.isfinite(v) for v in limits) or limits[0] >= limits[1]:
            raise ValueError("coordinate_range needs finite START < STOP")
        if not np.any((x >= limits[0]) & (x <= limits[1])):
            raise ValueError("coordinate_range contains no measured points")
    try:
        with np.errstate(over="raise", invalid="raise"):
            baseline = float(np.mean(y[:noise_points]))
            centered = y-baseline
            noise_max = float(np.max(np.abs(centered[:noise_points])))
            precision = float(64*np.finfo(float).eps*float(np.max(np.abs(y))))
    except FloatingPointError as exc:
        raise ValueError("Intensity magnitudes overflow; rescale the data") from exc
    automatic = float(factor*max(noise_max, noise_floor, precision))
    if not math.isfinite(automatic):
        raise ValueError("Noise threshold overflows; reduce factor/rescale intensities")
    prominence = automatic if min_prominence is None else _number(min_prominence, "min_prominence")
    abs_height = automatic if min_abs_height is None else _number(min_abs_height, "min_abs_height")
    points = []
    sample_indices = np.arange(x.size)
    kinds = (("maximum", 1.), ("minimum", -1.))
    for kind, sign in kinds:
        if extrema != "both" and extrema != ("maxima" if kind == "maximum" else "minima"):
            continue
        indices, properties = scipy_find_peaks(sign*centered, prominence=prominence,
            width=min_width, distance=distance, plateau_size=(None, None), rel_height=.5)
        for offset, raw_index in enumerate(indices):
            index = int(raw_index)
            if not limits[0] <= x[index] <= limits[1] or abs(centered[index]) < abs_height:
                continue
            left = float(np.interp(properties["left_ips"][offset], sample_indices, x))
            right = float(np.interp(properties["right_ips"][offset], sample_indices, x))
            item = {"index": index, "coordinate": float(x[index]), "kind": kind,
                "intensity": float(y[index]), "relative_intensity": float(centered[index]),
                "prominence": float(properties["prominences"][offset]),
                "width_points": float(properties["widths"][offset]), "width_axis": right-left}
            if not all(math.isfinite(value) for value in item.values() if isinstance(value, float)):
                raise ValueError("Peak properties overflow; rescale the data")
            first, last = int(properties["left_edges"][offset]), int(properties["right_edges"][offset])
            if first != last:
                item["plateau_indices"] = [first, last]
            points.append(item)
    points.sort(key=lambda p: p["index"])
    maxima = sum(p["kind"] == "maximum" for p in points)
    warnings = []
    if noise_max == 0.:
        warnings.append("Reference has zero measured noise; automatic thresholds use the numerical/user floor")
    report = {"status": "found" if points else "no_peaks", "axis_kind": axis_kind, "axis_unit": axis_unit,
        "index_origin": 0, "n_samples": int(x.size), "n_peaks": len(points),
        "n_maxima": maxima, "n_minima": len(points)-maxima, "output_complete": True,
        "configuration": {"extrema": extrema, "range": limits, "noise_points": noise_points, "factor": factor,
            "min_prominence": prominence, "min_abs_height": abs_height, "min_width_points": min_width,
            "distance_points": distance, "noise_floor": noise_floor,
            "automatic_prominence": min_prominence is None, "automatic_abs_height": min_abs_height is None},
        "noise": {"baseline_mean": baseline, "reference_indices": [0, noise_points-1],
            "raw_abs_max": float(np.max(np.abs(y[:noise_points]))), "centered_abs_max": noise_max,
            "numerical_floor": precision, "assumption": "Initial samples contain signal-free background"},
        "warnings": warnings, "peaks": points}
    json.dumps(report, allow_nan=False)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--axis", choices=("field", "time"), default="field")
    parser.add_argument("--field-unit", choices=tuple(FIELD_SCALES))
    parser.add_argument("--time-unit", choices=tuple(TIME_SCALES))
    parser.add_argument("--signal-key")
    parser.add_argument("--range", dest="coordinate_range", type=float, nargs=2,
                        help="Select peak coordinates, in SI (T or s); baseline still uses the original prefix")
    parser.add_argument("--extrema", choices=("both", "maxima", "minima"), default="both")
    parser.add_argument("--noise-points", type=int, default=10)
    parser.add_argument("--factor", type=float, default=3.)
    parser.add_argument("--prominence", dest="min_prominence", type=float, default=None,
                        help="Minimum prominence in intensity units; default factor*noise")
    parser.add_argument("--min-abs-height", type=float, default=None,
                        help="Minimum absolute deviation from baseline; default factor*noise")
    parser.add_argument("--min-width", type=float, default=3., help="Minimum half-prominence width, in samples")
    parser.add_argument("--distance", type=int, default=3, help="Minimum same-kind peak distance, in samples")
    parser.add_argument("--noise-floor", type=float, default=0.)
    parser.add_argument("--report", type=Path, help="Optional new full JSON file; never overwrite")
    parser.add_argument("--max-output-peaks", type=int, default=20, help="CLI preview limit; full report/API keep all peaks")
    args = parser.parse_args(argv)
    try:
        limit = _integer(args.max_output_peaks, "max_output_peaks", 0)
        path = args.input.resolve()
        x, y, info = load_trace(path, axis_kind=args.axis, field_unit=args.field_unit,
                               time_unit=args.time_unit, signal_key=args.signal_key)
        report = find_extrema(x, y, noise_points=args.noise_points, factor=args.factor,
            min_prominence=args.min_prominence, min_abs_height=args.min_abs_height, min_width=args.min_width,
            distance=args.distance, extrema=args.extrema, coordinate_range=args.coordinate_range,
            noise_floor=args.noise_floor, axis_kind=args.axis, axis_unit="T" if args.axis == "field" else "s")
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        report["source"] = {"path": str(path), "sha256": digest, **info}
        full_path = str(args.report.resolve()) if args.report else None
        report["full_report"] = full_path
        if args.report:
            with args.report.open("x", encoding="utf-8") as handle:
                handle.write(json.dumps(report, ensure_ascii=False, allow_nan=False)+"\n")
        preview = dict(report)
        if len(report["peaks"]) > limit:
            preview.update(peaks=report["peaks"][:limit], output_complete=False)
            preview["warnings"] = report["warnings"]+[
                "CLI peak list truncated; n_peaks is the total. Read full_report or increase --max-output-peaks"]
        preview["n_output_peaks"] = len(preview["peaks"])
        print(json.dumps(preview, ensure_ascii=False, allow_nan=False))
        return 0 if report["status"] == "found" else 1
    except (OSError, ValueError, TypeError, KeyError, ImportError, FloatingPointError) as exc:
        print(json.dumps({"status": "invalid", "error": str(exc)}, ensure_ascii=False, allow_nan=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
