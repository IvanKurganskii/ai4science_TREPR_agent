"""Статистические метрики для сравнения расчётных и экспериментальных ZFS-параметров.

Формулы — из data/statistic_formulas.md. y_calc — расчётные значения,
y_exp — экспериментальные (эталонные).
"""
from __future__ import annotations

import numpy as np


def mae(y_calc: np.ndarray, y_exp: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_calc - y_exp)))


def mape(y_calc: np.ndarray, y_exp: np.ndarray) -> float:
    """Mean Absolute Percentage Error, в процентах."""
    return float(100.0 * np.mean(np.abs((y_calc - y_exp) / y_exp)))


def rmse(y_calc: np.ndarray, y_exp: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_calc - y_exp) ** 2)))


def r2(y_calc: np.ndarray, y_exp: np.ndarray) -> float:
    """Коэффициент детерминации R^2."""
    ss_res = np.sum((y_exp - y_calc) ** 2)
    ss_tot = np.sum((y_exp - np.mean(y_exp)) ** 2)
    return float(1.0 - ss_res / ss_tot)