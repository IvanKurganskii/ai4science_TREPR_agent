"""Compute MAE, MAPE, RMSE, R^2 for |D|, |E|, |E/D| from DMRG table."""
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path("data/DMRG_table_semicolon.csv")


def _num(s: str) -> float:
    return float(s.replace("−", "-"))


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", skiprows=1, dtype=str)
    df.columns = ["no", "molecule", "D", "E", "ED_theory", "D_exp", "E_exp", "ED_exp", "refs"]
    for col in ["D", "E", "ED_theory", "D_exp", "E_exp", "ED_exp"]:
        df[col] = df[col].map(_num)
    return df


def mae(y_calc: np.ndarray, y_exp: np.ndarray) -> float:
    return float(np.mean(np.abs(y_calc - y_exp)))


def mape(y_calc: np.ndarray, y_exp: np.ndarray) -> float:
    return float(np.mean(np.abs((y_calc - y_exp) / y_exp)) * 100.0)


def rmse(y_calc: np.ndarray, y_exp: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_calc - y_exp) ** 2)))


def r2(y_calc: np.ndarray, y_exp: np.ndarray) -> float:
    ss_res = np.sum((y_exp - y_calc) ** 2)
    ss_tot = np.sum((y_exp - np.mean(y_exp)) ** 2)
    return float(1.0 - ss_res / ss_tot)


def metrics(y_calc: np.ndarray, y_exp: np.ndarray) -> dict[str, float]:
    return {
        "MAE": mae(y_calc, y_exp),
        "MAPE": mape(y_calc, y_exp),
        "RMSE": rmse(y_calc, y_exp),
        "R2": r2(y_calc, y_exp),
    }


def main() -> None:
    df = load(SRC)
    cases = {
        "|D|": (df["D"].abs().to_numpy(), df["D_exp"].to_numpy()),
        "|E|": (df["E"].abs().to_numpy(), df["E_exp"].to_numpy()),
        "|E/D|": (df["ED_theory"].to_numpy(), df["ED_exp"].to_numpy()),
    }
    for name, (calc, exp) in cases.items():
        m = metrics(calc, exp)
        print(f"{name}: MAE={m['MAE']:.4f} MAPE={m['MAPE']:.2f}% RMSE={m['RMSE']:.4f} R2={m['R2']:.4f}")


if __name__ == "__main__":
    main()