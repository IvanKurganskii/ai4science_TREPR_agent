"""Расчёт статистик по Table_1.csv и запись results/tables/statistics.csv."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.zfs_stats import mae, mape, rmse, r2

ROOT = Path(__file__).resolve().parents[1]
TABLE1 = ROOT / "results" / "tables" / "Table_1.csv"
OUT = ROOT / "results" / "tables" / "statistics.csv"


def load_table1(path: Path) -> dict[str, np.ndarray]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(next(csv.reader([line])))
    header = rows[0]
    idx = {name: i for i, name in enumerate(header)}
    data = rows[1:]
    d_theory = np.array([float(r[idx["D"]]) for r in data])
    e_theory = np.array([float(r[idx["E"]]) for r in data])
    d_exp = np.array([float(r[idx["absD"]]) for r in data])
    e_exp = np.array([float(r[idx["absE"]]) for r in data])
    eta_theory = np.array([float(r[idx["eta"]]) for r in data])
    eta_exp = np.array([float(r[idx["eta_exp"]]) for r in data])
    return {
        "D": (np.abs(d_theory), d_exp),
        "E": (np.abs(e_theory), e_exp),
        "eta": (eta_theory, eta_exp),
    }


def main() -> None:
    quantities = load_table1(TABLE1)
    with OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["quantity", "MAE", "MAPE", "RMSE", "R2"])
        for name, (calc, exp) in quantities.items():
            writer.writerow([
                name,
                f"{mae(calc, exp):.10f}",
                f"{mape(calc, exp):.10f}",
                f"{rmse(calc, exp):.10f}",
                f"{r2(calc, exp):.10f}",
            ])
    print(OUT.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()