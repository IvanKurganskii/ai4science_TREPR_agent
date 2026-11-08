"""Нормировка симулированного спектра под экспериментальный.

Использует встроенный функционал mars-epr (normalize_spectrum)
с добавлением линейной трансформации для совпадения:
- максимумов
- размаха (расстояния между максимумом и минимумом)
- минимизации среднеквадратичного отклонения

Формат входных данных:
    - numpy arrays: field (G), intensity (a.u.)
    - или пути к CSV-файлам (строка 1: поле, строка 2: интенсивность)

Выход:
    - Нормированный массив интенсивности
    - Словарь с диагностикой (scale, offset, rms, max_sim, max_exp, ...)
"""

import numpy as np
from typing import Union, Tuple, Dict, Optional

from mars.spectra_processing import normalize_spectrum


def _read_csv_spectrum(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Прочитать CSV-файл спектра (поле, интенсивность)."""
    data = np.loadtxt(filepath, delimiter=',', skiprows=0)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    field = data[0].astype(np.float64)
    intensity = data[1].astype(np.float64)
    return field, intensity


def _interpolate_to_grid(
    x_src: np.ndarray, y_src: np.ndarray, x_tgt: np.ndarray
) -> np.ndarray:
    """Интерполировать y_src на сетку x_tgt (линейная, с экстраполяцией краёв)."""
    from scipy.interpolate import interp1d
    f = interp1d(x_src, y_src, kind='linear', fill_value='extrapolate')
    return f(x_tgt)


def _grids_match(x1: np.ndarray, x2: np.ndarray) -> bool:
    """Проверить, совпадают ли две сетки."""
    if x1.shape != x2.shape:
        return False
    return np.allclose(x1, x2)


def normalize_sim_to_exp(
    sim_field: np.ndarray,
    sim_intensity: np.ndarray,
    exp_field: np.ndarray,
    exp_intensity: np.ndarray,
    mars_mode: str = "max",
    clip_to_exp_range: bool = True,
) -> Tuple[np.ndarray, Dict]:
    """Нормировать симулированный спектр под экспериментальный.

    Алгоритм:
    1. Интерполировать симуляцию на сетку эксперимента (если различаются).
    2. Применить базовую нормировку mars-epr (по умолчанию по максимуму).
    3. Линейная трансформация y_norm = a * y_base + b:
       - a = range_exp / range_base  (совпадение размаха)
       - b = max_exp - a * max_base  (совпадение максимума)
    4. Опционально: обрезать значения за пределами диапазона эксперимента.
    5. Вернуть нормированный спектр и диагностику.

    Args:
        sim_field: магнитное поле симуляции (G)
        sim_intensity: интенсивность симуляции (a.u.)
        exp_field: магнитное поле эксперимента (G)
        exp_intensity: интенсивность эксперимента (a.u.)
        mars_mode: режим базовой нормировки mars-epr ("max", "integral", "none")
        clip_to_exp_range: обрезать значения за пределами [min_exp, max_exp]

    Returns:
        normalized: нормированная интенсивность (совпадает с exp_field по длине)
        diagnostics: словарь с параметрами трансформации и метриками
    """
    sim_field = np.asarray(sim_field, dtype=np.float64)
    sim_intensity = np.asarray(sim_intensity, dtype=np.float64)
    exp_field = np.asarray(exp_field, dtype=np.float64)
    exp_intensity = np.asarray(exp_intensity, dtype=np.float64)

    # --- Шаг 1: интерполяция на общую сетку ---
    if not _grids_match(sim_field, exp_field):
        sim_intensity = _interpolate_to_grid(sim_field, sim_intensity, exp_field)

    # --- Шаг 2: базовая нормировка mars-epr ---
    base = normalize_spectrum(exp_field, sim_intensity, mode=mars_mode)
    base = np.asarray(base, dtype=np.float64)

    # --- Шаг 3: линейная трансформация ---
    max_sim = float(np.max(base))
    min_sim = float(np.min(base))
    range_sim = max_sim - min_sim

    max_exp = float(np.max(exp_intensity))
    min_exp = float(np.min(exp_intensity))
    range_exp = max_exp - min_exp

    if range_sim == 0:
        raise ValueError("Симулированный спектр константен — размах равен нулю")
    if range_exp == 0:
        raise ValueError("Экспериментальный спектр константен — размах равен нулю")

    a = range_exp / range_sim
    b = max_exp - a * max_sim

    normalized = a * base + b

    # --- Шаг 4: опциональная обрезка ---
    if clip_to_exp_range:
        normalized = np.clip(normalized, min_exp, max_exp)

    # --- Шаг 5: диагностика ---
    rms = float(np.sqrt(np.mean((normalized - exp_intensity) ** 2)))
    rms_rel = rms / (abs(max_exp) + abs(min_exp) + 1e-30)

    diagnostics = {
        "mars_mode": mars_mode,
        "scale_factor_a": a,
        "offset_b": b,
        "max_sim": max_sim,
        "min_sim": min_sim,
        "range_sim": range_sim,
        "max_exp": max_exp,
        "min_exp": min_exp,
        "range_exp": range_exp,
        "max_normalized": float(np.max(normalized)),
        "min_normalized": float(np.min(normalized)),
        "rms": rms,
        "rms_relative": rms_rel,
        "field_grid_matched": _grids_match(sim_field, exp_field),
    }

    return normalized, diagnostics


def normalize_sim_to_exp_from_files(
    sim_path: str,
    exp_path: str,
    mars_mode: str = "max",
    clip_to_exp_range: bool = True,
) -> Tuple[np.ndarray, Dict]:
    """Нормировка по путям к CSV-файлам.

    Args:
        sim_path: путь к CSV с симуляцией (строка 1: поле, строка 2: интенсивность)
        exp_path: путь к CSV с экспериментом
        mars_mode: режим базовой нормировки mars-epr
        clip_to_exp_range: обрезать значения за пределами диапазона эксперимента

    Returns:
        normalized: нормированная интенсивность
        diagnostics: словарь с параметрами трансформации
    """
    sim_field, sim_intensity = _read_csv_spectrum(sim_path)
    exp_field, exp_intensity = _read_csv_spectrum(exp_path)
    return normalize_sim_to_exp(
        sim_field, sim_intensity, exp_field, exp_intensity,
        mars_mode=mars_mode, clip_to_exp_range=clip_to_exp_range,
    )


def fit_linear_optimal(
    sim_field: np.ndarray,
    sim_intensity: np.ndarray,
    exp_field: np.ndarray,
    exp_intensity: np.ndarray,
) -> Tuple[np.ndarray, Dict]:
    """Оптимальная линейная подгонка (минимизация RMS).

    Прямая least-squares подгонка y_norm = a * y_sim + b.
    Не накладывает ограничений на максимум/размах — только минимизация RMS.

    Returns:
        normalized: нормированная интенсивность
        diagnostics: словарь с параметрами
    """
    sim_field = np.asarray(sim_field, dtype=np.float64)
    sim_intensity = np.asarray(sim_intensity, dtype=np.float64)
    exp_field = np.asarray(exp_field, dtype=np.float64)
    exp_intensity = np.asarray(exp_intensity, dtype=np.float64)

    # Интерполяция
    if not _grids_match(sim_field, exp_field):
        sim_intensity = _interpolate_to_grid(sim_field, sim_intensity, exp_field)

    # Линейная подгонка: y_norm = a * y_sim + b
    n = len(sim_intensity)
    A = np.vstack([sim_intensity, np.ones(n)]).T
    result, residuals, rank, s = np.linalg.lstsq(A, exp_intensity, rcond=None)
    a_opt, b_opt = result

    normalized = a_opt * sim_intensity + b_opt

    diagnostics = {
        "method": "least_squares",
        "scale_factor_a": float(a_opt),
        "offset_b": float(b_opt),
        "max_sim": float(np.max(sim_intensity)),
        "min_sim": float(np.min(sim_intensity)),
        "range_sim": float(np.max(sim_intensity) - np.min(sim_intensity)),
        "max_exp": float(np.max(exp_intensity)),
        "min_exp": float(np.min(exp_intensity)),
        "range_exp": float(np.max(exp_intensity) - np.min(exp_intensity)),
        "max_normalized": float(np.max(normalized)),
        "min_normalized": float(np.min(normalized)),
        "rms": float(np.sqrt(np.mean((normalized - exp_intensity) ** 2))),
        "field_grid_matched": _grids_match(sim_field, exp_field),
    }

    return normalized, diagnostics


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Использование:")
        print("  python spectra_normalization.py <sim.csv> <exp.csv> [mars_mode]")
        print("  mars_mode: max (по умолчанию), integral, none")
        sys.exit(1)

    sim_path = sys.argv[1]
    exp_path = sys.argv[2]
    mars_mode = sys.argv[3] if len(sys.argv) > 3 else "max"

    print(f"Симуляция: {sim_path}")
    print(f"Эксперимент: {exp_path}")
    print(f"Режим mars: {mars_mode}")
    print()

    normalized, diag = normalize_sim_to_exp_from_files(
        sim_path, exp_path, mars_mode=mars_mode
    )

    print("=== Диагностика ===")
    for k, v in diag.items():
        print(f"  {k}: {v}")

    # Сохранение результата
    out_path = sim_path.replace(".csv", "_normalized.csv")
    with open(out_path, "w") as f:
        f.write("# Normalized simulation (mars_mode={}, scale={:.6f}, offset={:.6f})\n".format(
            mars_mode, diag["scale_factor_a"], diag["offset_b"]))
        f.write("# Field (G)\n")
        # Перечитываем поле из exp
        exp_f, _ = _read_csv_spectrum(exp_path)
        f.write(",".join(f"{v:.7f}" for v in exp_f) + "\n")
        f.write(",".join(f"{v:.10e}" for v in normalized) + "\n")

    print(f"\nРезультат сохранён: {out_path}")
    print(f"RMS: {diag['rms']:.6e}")
    print(f"RMS относительный: {diag['rms_relative']:.6e}")
