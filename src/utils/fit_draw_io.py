"""Read-only adapters for current fit NPZ exports and common SVG utilities."""
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np


def finite_array(value, name, ndim=None):
    raw = np.asarray(value)
    if np.iscomplexobj(raw) or raw.dtype.kind not in "biuf":
        raise ValueError(f"{name} must contain real numeric values")
    result = np.asarray(raw, dtype=float)
    if not np.isfinite(result).all() or (ndim is not None and result.ndim != ndim):
        raise ValueError(f"{name} has nonfinite values or wrong dimensionality")
    return result


def axis(value, name, nonnegative=False):
    result = finite_array(value, name, 1)
    if len(result) < 2 or not np.all(np.diff(result) > 0):
        raise ValueError(f"{name} must increase strictly and contain at least two samples")
    if nonnegative and np.any(result < 0):
        raise ValueError(f"{name} must be nonnegative")
    return result


def fingerprint(path):
    path = Path(path).resolve()
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest}


def scalar(value, name, minimum=None):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return number


def read_fit(path, dimension):
    """Read scaled fit + experiment, never the unscaled 'simulated' array.

    Current 1D exports have fields_T/intensity/experimental; 2D exports have
    fields_T/times_s/spectrum/experimental and shape (time, field). A 2D
    fit_mask limits the displayed comparison window, preserving absolute
    experimental times (model-relative times are never silently substituted).
    """
    if dimension not in ("1d", "2d"):
        raise ValueError("dimension must be 1d or 2d")
    path = Path(path).resolve()
    if path.suffix.lower() != ".npz":
        raise ValueError("Use the saved fitter NPZ containing both scaled fit and experimental arrays")
    with np.load(path, allow_pickle=False) as archive:
        bkey = "fields_T" if "fields_T" in archive else "field"
        if bkey not in archive or "experimental" not in archive:
            raise ValueError("Fit NPZ needs fields_T (T) and experimental")
        if "field_unit" in archive and str(archive["field_unit"].item()) != "T":
            raise ValueError("Fit NPZ must use field in T")
        b = axis(archive[bkey], "fields_T", True)
        key = ("intensity" if "intensity" in archive else "spectrum") if dimension == "1d" else "spectrum"
        if key not in archive:
            raise ValueError("Missing scaled fit: intensity (1D) or spectrum (2D); raw simulated is not interchangeable")
        fit = finite_array(archive[key], key, 1 if dimension == "1d" else 2)
        exp = finite_array(archive["experimental"], "experimental", fit.ndim)
        if exp.shape != fit.shape or fit.shape[-1] != len(b):
            raise ValueError("Fit and experiment must share the saved field/time grid and shape")
        if not np.any(fit != 0) or not np.any(exp != 0):
            raise ValueError("Fit and experiment must both be nonzero")
        result = {"fields_T": b, "fit": fit, "experimental": exp,
                  "source": fingerprint(path), "fit_key": key}
        if dimension == "2d":
            meta = json.loads(str(archive["metadata_json"].item())) if "metadata_json" in archive else {}
            if not isinstance(meta, dict) or meta.get("axis_order", ["time", "field"]) != ["time", "field"]:
                raise ValueError("2D fit must declare time,field axis order")
            if meta.get("field_unit", "T") != "T" or meta.get("time_unit", "s") != "s":
                raise ValueError("2D fit metadata must use T and s")
            t_model = axis(archive["times_s"], "times_s")
            origin = scalar(meta["time_zero_s"], "time_zero_s") if "time_zero_s" in meta else None
            if "experimental_times_s" in archive:
                t = axis(archive["experimental_times_s"], "experimental_times_s")
                if t.shape != t_model.shape:
                    raise ValueError("Experimental and model time axes have different shapes")
                offset = t-t_model
                if not np.allclose(offset, offset[0], rtol=0., atol=1e-15):
                    raise ValueError("Time axes must differ only by a constant origin")
                if origin is not None and not np.allclose(offset, origin, rtol=0., atol=1e-15):
                    raise ValueError("Time axes contradict metadata.time_zero_s")
                origin = float(offset[0])
            else:
                origin = 0. if origin is None else origin
                t = t_model+origin
            if fit.shape[0] != len(t):
                raise ValueError("Time axis does not match the map")
            mask = np.asarray(archive["fit_mask"]) if "fit_mask" in archive else np.ones(len(t), dtype=bool)
            if mask.shape != t.shape or mask.dtype.kind != "b" or np.count_nonzero(mask) < 2:
                raise ValueError("fit_mask must be boolean with at least two retained times")
            indices = np.flatnonzero(mask)
            if np.any(np.diff(indices) != 1):
                raise ValueError("fit_mask must select a contiguous comparison interval")
            result.update(times_s=t[mask], fit=fit[mask], experimental=exp[mask], time_zero_s=origin,
                          full_time_window_s=[float(t[0]), float(t[-1])],
                          comparison_window_s=[float(t[mask][0]), float(t[mask][-1])])
    return result


def sample_axis(coordinates, values, positions, *, sample_axis=0, half_width=0.):
    """Sample by linear interpolation or a declared unweighted window mean.

    All windows must be fully covered and contain measured points; never
    extrapolate, clamp coordinates, normalize individual curves or mutate input.
    The sampled axis is replaced by a final position axis in the output.
    """
    x = axis(coordinates, "sampling axis")
    y = finite_array(values, "sampled data")
    if sample_axis not in range(y.ndim) or y.shape[sample_axis] != len(x):
        raise ValueError("sample_axis does not match the data")
    y = np.moveaxis(y, sample_axis, -1)
    p = finite_array(positions, "positions", 1)
    width = scalar(half_width, "half_width", 0.)
    if not len(p) or np.any(p-width < x[0]) or np.any(p+width > x[-1]):
        raise ValueError("Every requested point/window must lie inside the measured comparison grid")
    sampled, selections = [], []
    for coordinate in p:
        if width:
            indices = np.flatnonzero((x >= coordinate-width) & (x <= coordinate+width))
            if not len(indices):
                raise ValueError("Sampling window contains no measured points")
            values_at = np.mean(y[..., indices], axis=-1)
            mode = "window_mean"
        else:
            right = int(np.searchsorted(x, coordinate))
            left = right if x[right] == coordinate else right-1
            weight = 0. if left == right else (coordinate-x[left])/(x[right]-x[left])
            values_at = (1.-weight)*y[..., left]+weight*y[..., right]
            indices, mode = np.unique([left, right]), "linear_interpolation"
        sampled.append(values_at)
        selections.append({"coordinate": float(coordinate), "mode": mode, "half_width": width,
                           "sample_indices": [int(indices[0]), int(indices[-1])], "n_samples": len(indices)})
    result = np.stack(sampled, axis=-1)
    if not np.isfinite(result).all():
        raise ValueError("Sampling overflow; rescale the input")
    return result, selections


def trace_summary(values):
    y = finite_array(values, "trace", 1)
    return {"n_points": len(y), "min": float(np.min(y)), "max": float(np.max(y)),
            "abs_max": float(np.max(np.abs(y)))}


def pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_pair(ax, x, experimental, fit, panel_id, title, xlabel):
    ax.plot(x, experimental, color="0.4", linewidth=1., label="Experiment")[0].set_gid(panel_id+"_experiment")
    ax.plot(x, fit, color="#c23728", linewidth=1.3, label="Fit")[0].set_gid(panel_id+"_fit")
    ax.set(title=title, xlabel=xlabel, ylabel="Intensity (a.u.)")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    ax.set_gid(panel_id)
    return {"id": panel_id, "experiment": trace_summary(experimental), "fit": trace_summary(fit)}


def output_path(directory, spectrum_id):
    if not isinstance(spectrum_id, str) or not re.fullmatch(r"[\w.-]{1,150}", spectrum_id):
        raise ValueError("spectrum_id must contain 1..150 letters, digits, _, . or -")
    return Path(directory).resolve()/f"fitdraw_{spectrum_id}.svg"


def save_svg(fig, path, payload):
    """Embed current-fit provenance and identifiable data lines in a new SVG."""
    import matplotlib as mpl
    plt = pyplot()
    try:
        if not isinstance(payload["fit_id"], str) or not payload["fit_id"].strip():
            raise ValueError("fit_id is required")
        with mpl.rc_context({"svg.fonttype": "none", "path.simplify": False}):
            with Path(path).open("x", encoding="utf-8") as handle:
                fig.savefig(handle, format="svg", metadata={"Title": payload["spectrum_id"], "Date": None,
                    "Description": json.dumps(payload, ensure_ascii=True, allow_nan=False), "Creator": "AI4Science fit drawing"})
    finally:
        plt.close(fig)
    return {"status": "drawn", "figure": fingerprint(path), "dimension": payload["dimension"],
            "spectrum_id": payload["spectrum_id"], "fit_id": payload["fit_id"], "n_panels": len(payload["panels"])}


def add_draw_arguments(parser):
    parser.add_argument("--fit", type=Path, required=True, help="Saved scaled fit NPZ with experimental arrays")
    parser.add_argument("--spectrum-id", "--name", required=True)
    parser.add_argument("--fit-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="Existing directory for this fit version")


def cli_result(function, argv=None):
    try:
        result = function(argv)
    except (OSError, ValueError, TypeError, KeyError, ImportError, OverflowError) as exc:
        result = {"status": "invalid", "error": str(exc)}
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0 if result["status"] in ("drawn", "sampled", "calculated") else 2
