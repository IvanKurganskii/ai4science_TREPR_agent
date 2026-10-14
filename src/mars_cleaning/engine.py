"""Baseline/background correction and MaRS denoising in canonical SI units."""

import copy
import hashlib
import inspect
import warnings

from ._runtime import numpy, numerical_threads
from .data import fields


def number(value, name, *, nonnegative=False):
    np = numpy()
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return float(value)


def integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def interval(value, name, *, optional_ends=False):
    if value is None:
        return None
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must contain two bounds")
    result = [None if optional_ends and v is None else number(v, name) for v in value]
    if result[0] is not None and result[1] is not None and result[0] >= result[1]:
        raise ValueError(f"{name} requires lower < upper")
    return result


def baseline_options(options):
    defaults = {"poly_order": 1, "subtract_average_background": True, "correct_baseline": True,
                "baseline_mode": "scaled_reference", "fit_intensity_shift": True,
                "field_bounds_T": None, "time_bounds_s": None}
    options = {**defaults, **fields(options, defaults, name="baseline")}
    integer(options["poly_order"], "poly_order")
    for key in ("subtract_average_background", "correct_baseline", "fit_intensity_shift"):
        if not isinstance(options[key], bool):
            raise ValueError(f"{key} must be boolean")
    if options["baseline_mode"] not in ("scaled_reference", "independent"):
        raise ValueError("Unknown baseline_mode")
    options["field_bounds_T"] = interval(options["field_bounds_T"], "field_bounds_T")
    options["time_bounds_s"] = interval(options["time_bounds_s"], "time_bounds_s", optional_ends=True)
    if options["correct_baseline"] and options["field_bounds_T"] is None:
        raise ValueError("Polynomial correction requires the useful-signal field_bounds_T")
    if options["subtract_average_background"] and options["time_bounds_s"] is None:
        raise ValueError("Background subtraction requires the useful-signal time_bounds_s")
    return options


def denoising_options(options):
    options = fields(options, ("method", "svd_rank", "sigma_t", "sigma_b", "wavelet",
                               "wavelet_level", "wavelet_threshold", "median_kernel"), ("method",), "denoising")
    method = options["method"]
    relevant = {
        "svd": {"svd_rank": 10}, "gaussian": {"sigma_t": 2.0, "sigma_b": 1.0},
        "wavelet": {"wavelet": "db4", "wavelet_level": 4, "wavelet_threshold": None},
        "median": {"median_kernel": 3},
        "combined": {"svd_rank": 10, "sigma_t": 2.0, "sigma_b": 1.0},
    }
    if method not in relevant:
        raise ValueError("method must be svd, gaussian, wavelet, median or combined")
    fields(options, ("method", *relevant[method]), name=f"{method} options")
    result = {"method": method, **relevant[method], **options}
    if "svd_rank" in result:
        integer(result["svd_rank"], "svd_rank", 1)
    for key in ("sigma_t", "sigma_b", "wavelet_threshold"):
        if key in result and result[key] is not None:
            result[key] = number(result[key], key, nonnegative=True)
    if "wavelet_level" in result:
        integer(result["wavelet_level"], "wavelet_level", 1)
        import pywt
        if not isinstance(result["wavelet"], str):
            raise ValueError("wavelet must be a name")
        pywt.Wavelet(result["wavelet"])
    if "median_kernel" in result:
        integer(result["median_kernel"], "median_kernel", 1)
        if result["median_kernel"] % 2 == 0:
            raise ValueError("median_kernel must be odd; no silent rounding is performed")
    return result


def _fingerprint(function):
    with open(inspect.getfile(function), "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class CleaningEngine:
    """Numerical API returning datasets and diagnostic arrays without disk output."""
    def __init__(self, threads=2):
        self.threads = integer(threads, "threads", 1)

    def correct_baseline(self, dataset, options):
        options = baseline_options(options)
        from .baseline import BaselineCorrector2D
        constructor = {k: options[k] for k in ("poly_order", "subtract_average_background", "correct_baseline",
                                              "baseline_mode", "fit_intensity_shift")}
        corrector = BaselineCorrector2D(**constructor, layout="time_field")
        # The original notebook fits in G/ns. Preserve its numerical conditioning,
        # including adjacent-slice initial guesses in independent mode.
        b_bounds = None if options["field_bounds_T"] is None else [v*1e4 for v in options["field_bounds_T"]]
        t_bounds = None if options["time_bounds_s"] is None else [None if v is None else v*1e9 for v in options["time_bounds_s"]]
        with warnings.catch_warnings(record=True) as caught, numerical_threads(self.threads):
            warnings.simplefilter("always")
            cleaned = corrector.fit_transform(dataset.fields_T*1e4, dataset.times_s*1e9,
                dataset.signal, b_bounds, t_bounds)
        output = dataset.copy(cleaned)
        info = {"stage": "baseline", "options": options, "reference_index": corrector.reference_index_,
                "reference_time_s": float(dataset.times_s[corrector.reference_index_]),
                "internal_fit_units": {"field": "G", "time": "ns"},
                "backend": "Notebook BaselineCorrector2D", "backend_sha256": _fingerprint(BaselineCorrector2D),
                "warnings": list(dict.fromkeys(str(w.message) for w in caught))}
        output.metadata["history"].append(info)
        output.metadata["baseline_correction"] = copy.deepcopy(info)
        diagnostic = {"background_profile": corrector.background_profile_,
                      "baseline_surface": corrector.baseline_surface_,
                      "reference_baseline": corrector.reference_baseline_,
                      "baseline_scale_factors": corrector.scale_factors_,
                      "baseline_intensity_shifts": corrector.intensity_shifts_}
        if corrector.polynomial_coefficients_ is not None:
            diagnostic["polynomial_coefficients_G"] = corrector.polynomial_coefficients_
        return output, diagnostic

    def denoise(self, dataset, options):
        options = denoising_options(options)
        from mars.spectra_processing import filter_noise_2d
        np = numpy()
        with warnings.catch_warnings(record=True) as caught, numerical_threads(self.threads):
            warnings.simplefilter("always")
            # MaRS expects field first, whereas the notebook and this API use time first.
            cleaned = filter_noise_2d(fields=dataset.fields_T, time=dataset.times_s,
                                     result=dataset.signal.T, **options).T
        if cleaned.shape != dataset.signal.shape or not np.isfinite(cleaned).all():
            raise RuntimeError("Denoising returned an invalid or nonfinite map")
        output = dataset.copy(cleaned)
        info = {"stage": "denoising", "options": options, "backend": "mars.filter_noise_2d",
                "backend_sha256": _fingerprint(filter_noise_2d),
                "warnings": list(dict.fromkeys(str(w.message) for w in caught))}
        if options["method"] in ("svd", "combined"):
            info["effective_svd_rank"] = min(options["svd_rank"], min(dataset.signal.shape))
        if options["method"] == "combined":
            info["effective_gaussian_sigmas_samples"] = {"time": options["sigma_t"]*0.5, "field": options["sigma_b"]*0.5}
        for axis, name in ((dataset.fields_T, "field"), (dataset.times_s, "time")):
            steps = np.diff(axis)
            if not np.allclose(steps, steps[0], rtol=1e-6, atol=abs(steps[0])*1e-9):
                if options["method"] in ("gaussian", "combined", "median", "wavelet"):
                    info["warnings"].append(f"{name} grid is nonuniform; filtering uses sample indices without resampling")
        output.metadata["history"].append(info)
        output.metadata["denoising"] = copy.deepcopy(info)
        return output

    def clean(self, dataset, *, baseline=None, denoising=None):
        if baseline is None and denoising is None:
            raise ValueError("Supply baseline and/or denoising options")
        # Validate every requested stage before the first operation.
        if baseline is not None:
            baseline_options(baseline)
        if denoising is not None:
            denoising_options(denoising)
        current, diagnostics = dataset.copy(), {}
        if baseline is not None:
            current, diagnostics = self.correct_baseline(current, baseline)
            diagnostics["baseline_corrected_signal"] = current.signal.copy()
        if denoising is not None:
            current = self.denoise(current, denoising)
        return current, diagnostics

    def singular_values(self, dataset, count=20):
        integer(count, "singular_values", 1)
        np = numpy()
        with warnings.catch_warnings(record=True) as caught, numerical_threads(self.threads):
            warnings.simplefilter("always")
            singular = np.linalg.svd(dataset.signal, compute_uv=False, full_matrices=False)
        total = float(np.sum(singular**2))
        fractions = np.cumsum(singular**2)/total if total > 0 else np.zeros_like(singular)
        return {"singular_values": singular[:count].tolist(),
                "cumulative_energy_fraction": fractions[:count].tolist(),
                "total_components": int(singular.size),
                "warnings": list(dict.fromkeys(str(w.message) for w in caught))}
