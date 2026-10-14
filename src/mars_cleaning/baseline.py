"""Baseline subtraction for experimental 2D time-resolved EPR spectra.

The expected logical data layout is ``(time, field)``.  Arrays stored as
``(field, time)`` are also accepted and are returned in their original layout.

The correction has two independently optional stages:

1. Average the field profiles outside the useful time interval and subtract
   that time-independent profile from every time slice.
2. Correct a field-dependent polynomial baseline either by fitting one
   reference slice and adjusting only its scale and intensity offset at every
   time, or by fitting every time slice independently.

Only pickle files from trusted sources should be used to construct the input
arrays; this module itself does not read or write data files.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
from scipy.optimize import least_squares

from mars.spectra_processing import correct_baseline_polynomial


Layout = Literal["auto", "time_field", "field_time"]
BaselineMode = Literal["scaled_reference", "independent"]
TimeBounds = Optional[tuple[Optional[float], Optional[float]]]


class BaselineCorrector2D:
    """Correct the baseline of a complete 2D TREPR spectrum.

    Parameters
    ----------
    poly_order
        Order of the polynomial fitted along the magnetic-field axis.
    subtract_average_background
        If ``True`` (default), average all time slices outside
        ``time_bounds`` and subtract this profile from every time slice.
    correct_baseline
        If ``True`` (default), perform the field-dependent polynomial
        correction after the optional average-background subtraction.
    baseline_mode
        ``"scaled_reference"`` (default) fits the maximum-amplitude time
        slice once.  Every other slice is corrected with the same polynomial
        shape after least-squares adjustment of its scale and, optionally, an
        additive intensity offset.

        ``"independent"`` fits a separate polynomial to every time slice.
        The maximum-amplitude slice is fitted first. Fits then propagate
        outward in both time directions, with each fit initialized from the
        coefficients of its previously fitted adjacent slice.
    fit_intensity_shift
        Include an additive intensity offset when scaling the reference
        baseline. This is enabled by default. In independent mode, the
        polynomial constant term already supplies an independent offset.
    layout
        Input layout. ``"auto"`` infers it by comparing the spectrum shape
        with the lengths of the time and field axes. For a square, ambiguous
        matrix, specify ``"time_field"`` or ``"field_time"`` explicitly.

    Notes
    -----
    ``field_bounds`` and ``time_bounds`` describe the region containing the
    useful signal. Baseline/background points are selected *outside* these
    inclusive bounds. Either endpoint of ``time_bounds`` may be ``None`` when
    useful signal reaches that edge of the recorded time range. Thus,
    ``(None, upper)`` uses only later times for the average background, while
    ``(lower, None)`` uses only earlier times.

    Fitted results are available after :meth:`fit_transform` in attributes
    ending in ``_``. All stored 2D intermediate arrays use ``(time, field)``
    layout, irrespective of the input orientation.
    """

    def __init__(
        self,
        poly_order: int = 1,
        subtract_average_background: bool = True,
        correct_baseline: bool = True,
        baseline_mode: BaselineMode = "scaled_reference",
        fit_intensity_shift: bool = True,
        layout: Layout = "auto",
    ) -> None:
        if not isinstance(poly_order, (int, np.integer)) or poly_order < 0:
            raise ValueError("poly_order must be a non-negative integer")
        if baseline_mode not in ("scaled_reference", "independent"):
            raise ValueError(
                "baseline_mode must be 'scaled_reference' or 'independent'"
            )
        if layout not in ("auto", "time_field", "field_time"):
            raise ValueError("layout must be 'auto', 'time_field', or 'field_time'")

        self.poly_order = int(poly_order)
        self.subtract_average_background = bool(subtract_average_background)
        self.correct_baseline = bool(correct_baseline)
        self.baseline_mode = baseline_mode
        self.fit_intensity_shift = bool(fit_intensity_shift)
        self.layout = layout

    def fit_transform(
        self,
        fields: np.ndarray,
        times: np.ndarray,
        spectrum: np.ndarray,
        field_bounds: Optional[tuple[float, float]],
        time_bounds: TimeBounds,
    ) -> np.ndarray:
        """Fit and subtract the requested 2D baseline components.

        Parameters
        ----------
        fields
            One-dimensional magnetic-field coordinates.
        times
            One-dimensional time coordinates.
        spectrum
            Real-valued 2D spectral intensity array.
        field_bounds
            Inclusive ``(lower, upper)`` useful-signal interval along field.
            Points outside it are used for polynomial fitting. May be ``None``
            only when ``correct_baseline=False``.
        time_bounds
            Inclusive ``(lower, upper)`` useful-signal interval along time.
            Either endpoint may be ``None`` if useful signal continues to that
            edge of the acquisition: ``(None, upper)`` averages background only
            after ``upper``, and ``(lower, None)`` averages it only before
            ``lower``. Slices outside the resulting interval form the average
            background profile, and the maximum-amplitude reference is selected
            inside it. Passing ``None`` (or ``(None, None)`` when average
            subtraction is disabled) selects the reference from all slices.

        Returns
        -------
        numpy.ndarray
            Corrected spectrum with the same shape and layout as ``spectrum``.
        """
        field_axis = self._validate_axis(fields, "fields")
        time_axis = self._validate_axis(times, "times")
        data, input_layout = self._as_time_field(
            spectrum, len(time_axis), len(field_axis)
        )

        if np.iscomplexobj(data):
            raise TypeError(
                "Complex spectra are not supported; phase the data or correct "
                "the real and imaginary channels separately"
            )
        data = np.asarray(data, dtype=float)
        if not np.all(np.isfinite(data)):
            raise ValueError("spectrum contains NaN or infinite values")

        useful_time_mask = self._inside_bounds_mask(
            time_axis,
            time_bounds,
            "time_bounds",
            required=self.subtract_average_background,
        )

        if self.subtract_average_background:
            background_time_mask = ~useful_time_mask
            if not np.any(background_time_mask):
                raise ValueError(
                    "time_bounds leave no outside slices for average-background subtraction"
                )
            background_profile = data[background_time_mask].mean(axis=0)
        else:
            background_profile = np.zeros(data.shape[1], dtype=float)

        working = data - background_profile[None, :]
        reference_candidates = (
            np.flatnonzero(useful_time_mask)
            if time_bounds is not None
            else np.arange(len(time_axis))
        )
        if reference_candidates.size == 0:
            raise ValueError("time_bounds contain no measured time points")
        candidate_amplitudes = np.max(
            np.abs(working[reference_candidates]), axis=1
        )
        reference_index = int(reference_candidates[np.argmax(candidate_amplitudes)])

        baseline_surface = np.zeros_like(working)
        scale_factors = np.zeros(len(time_axis), dtype=float)
        intensity_shifts = np.zeros(len(time_axis), dtype=float)
        polynomial_coefficients = None
        reference_baseline = np.zeros(data.shape[1], dtype=float)

        if self.correct_baseline:
            field_baseline_mask = self._outside_bounds_mask(
                field_axis, field_bounds, "field_bounds"
            )
            required_points = self.poly_order + 1
            if np.count_nonzero(field_baseline_mask) < required_points:
                raise ValueError(
                    f"field_bounds leave fewer than {required_points} points "
                    f"required for a degree-{self.poly_order} polynomial"
                )

            _, reference_baseline = correct_baseline_polynomial(
                field_axis,
                working[reference_index],
                field_baseline_mask,
                self.poly_order,
            )
            reference_baseline = np.asarray(reference_baseline, dtype=float)

            if self.baseline_mode == "scaled_reference":
                baseline_surface, scale_factors, intensity_shifts = (
                    self._fit_scaled_reference(
                        working,
                        reference_baseline,
                        field_baseline_mask,
                        reference_index,
                    )
                )
            else:
                baseline_surface, polynomial_coefficients = (
                    self._fit_independent_baselines(
                        field_axis,
                        working,
                        field_baseline_mask,
                        reference_index,
                    )
                )

        corrected = working - baseline_surface

        self.input_layout_ = input_layout
        self.background_profile_ = background_profile
        self.reference_index_ = reference_index
        self.reference_time_ = float(time_axis[reference_index])
        self.reference_baseline_ = reference_baseline
        self.baseline_surface_ = baseline_surface
        self.scale_factors_ = scale_factors
        self.intensity_shifts_ = intensity_shifts
        self.polynomial_coefficients_ = polynomial_coefficients
        self.corrected_time_field_ = corrected

        return corrected.copy() if input_layout == "time_field" else corrected.T.copy()

    __call__ = fit_transform

    def _fit_scaled_reference(
        self,
        data: np.ndarray,
        reference_baseline: np.ndarray,
        mask: np.ndarray,
        reference_index: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        masked_template = reference_baseline[mask]
        if np.linalg.norm(masked_template) <= np.finfo(float).eps:
            raise ValueError(
                "The fitted reference baseline is zero in the selected baseline regions"
            )

        n_times = data.shape[0]
        scales = np.empty(n_times, dtype=float)
        shifts = np.zeros(n_times, dtype=float)
        surface = np.empty_like(data)

        for index, trace in enumerate(data):
            if index == reference_index:
                scale = 1.0
                shift = 0.0
            elif self.fit_intensity_shift:
                design = np.column_stack(
                    (masked_template, np.ones(masked_template.size, dtype=float))
                )
                scale, shift = np.linalg.lstsq(
                    design, trace[mask], rcond=None
                )[0]
            else:
                scale = float(
                    np.dot(masked_template, trace[mask])
                    / np.dot(masked_template, masked_template)
                )
                shift = 0.0

            scales[index] = scale
            shifts[index] = shift
            surface[index] = scale * reference_baseline + shift

        return surface, scales, shifts

    def _fit_independent_baselines(
        self,
        fields: np.ndarray,
        data: np.ndarray,
        mask: np.ndarray,
        reference_index: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_times = data.shape[0]
        coefficients = np.empty((n_times, self.poly_order + 1), dtype=float)
        surface = np.empty_like(data)

        reference_coefficients = np.polyfit(
            fields[mask], data[reference_index, mask], self.poly_order
        )
        coefficients[reference_index] = reference_coefficients
        surface[reference_index] = np.polyval(reference_coefficients, fields)

        def fit_slice(index: int, initial: np.ndarray) -> np.ndarray:
            result = least_squares(
                lambda values: (
                    np.polyval(values, fields[mask]) - data[index, mask]
                ),
                x0=initial,
                method="trf",
            )
            if not result.success:
                raise RuntimeError(
                    f"Polynomial baseline fit failed at time index {index}: "
                    f"{result.message}"
                )
            coefficients[index] = result.x
            surface[index] = np.polyval(result.x, fields)
            return result.x

        previous = reference_coefficients
        for index in range(reference_index + 1, n_times):
            previous = fit_slice(index, previous)

        previous = reference_coefficients
        for index in range(reference_index - 1, -1, -1):
            previous = fit_slice(index, previous)

        return surface, coefficients

    def _as_time_field(
        self,
        spectrum: np.ndarray,
        n_times: int,
        n_fields: int,
    ) -> tuple[np.ndarray, Literal["time_field", "field_time"]]:
        data = np.asarray(spectrum)
        if data.ndim != 2:
            raise ValueError(f"spectrum must be 2D, got shape {data.shape}")

        time_field_shape = (n_times, n_fields)
        field_time_shape = (n_fields, n_times)
        if self.layout == "time_field":
            if data.shape != time_field_shape:
                raise ValueError(
                    f"time_field layout requires shape {time_field_shape}, got {data.shape}"
                )
            return data, "time_field"
        if self.layout == "field_time":
            if data.shape != field_time_shape:
                raise ValueError(
                    f"field_time layout requires shape {field_time_shape}, got {data.shape}"
                )
            return data.T, "field_time"

        matches_time_field = data.shape == time_field_shape
        matches_field_time = data.shape == field_time_shape
        if matches_time_field and matches_field_time:
            raise ValueError(
                "Cannot infer the layout of a square spectrum; set layout explicitly"
            )
        if matches_time_field:
            return data, "time_field"
        if matches_field_time:
            return data.T, "field_time"
        raise ValueError(
            f"spectrum shape {data.shape} matches neither {time_field_shape} "
            f"nor {field_time_shape}"
        )

    @staticmethod
    def _validate_axis(axis: np.ndarray, name: str) -> np.ndarray:
        values = np.asarray(axis, dtype=float)
        if values.ndim != 1 or values.size == 0:
            raise ValueError(f"{name} must be a non-empty 1D array")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains NaN or infinite values")
        return values

    @staticmethod
    def _normalized_bounds(
        bounds: Optional[tuple[float, float]], name: str
    ) -> tuple[float, float]:
        if bounds is None or len(bounds) != 2:
            raise ValueError(f"{name} must be a (lower, upper) pair")
        lower, upper = map(float, bounds)
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ValueError(f"{name} must contain finite values with lower < upper")
        return lower, upper

    @classmethod
    def _outside_bounds_mask(
        cls,
        axis: np.ndarray,
        bounds: Optional[tuple[float, float]],
        name: str,
    ) -> np.ndarray:
        lower, upper = cls._normalized_bounds(bounds, name)
        return (axis < lower) | (axis > upper)

    @classmethod
    def _inside_bounds_mask(
        cls,
        axis: np.ndarray,
        bounds: TimeBounds,
        name: str,
        required: bool,
    ) -> np.ndarray:
        if bounds is None:
            if required:
                raise ValueError(f"{name} is required for the enabled correction")
            return np.ones(axis.size, dtype=bool)
        if len(bounds) != 2:
            raise ValueError(f"{name} must be a (lower, upper) pair")

        lower_raw, upper_raw = bounds
        lower = None if lower_raw is None else float(lower_raw)
        upper = None if upper_raw is None else float(upper_raw)
        if lower is not None and not np.isfinite(lower):
            raise ValueError(f"The lower value of {name} must be finite or None")
        if upper is not None and not np.isfinite(upper):
            raise ValueError(f"The upper value of {name} must be finite or None")
        if lower is not None and upper is not None and lower >= upper:
            raise ValueError(f"{name} must have lower < upper")

        mask = np.ones(axis.size, dtype=bool)
        if lower is not None:
            mask &= axis >= lower
        if upper is not None:
            mask &= axis <= upper
        return mask


__all__ = ["BaselineCorrector2D"]
