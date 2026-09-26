"""Independent verification: recompute metrics straight from raw CSV."""
from pathlib import Path

import numpy as np

rows = [l.split() for l in Path("data/DMRG_table.csv").read_text(encoding="utf-8").splitlines() if l.strip()]
data = rows[2:]  # skip two header lines
N = len(data)
assert N == 16, f"expected 16 rows, got {N}"

D = np.array([float(r[2].replace("−", "-")) for r in data])
E = np.array([float(r[3].replace("−", "-")) for r in data])
ED = np.array([float(r[4]) for r in data])
D_exp = np.array([float(r[5]) for r in data])
E_exp = np.array([float(r[6]) for r in data])
ED_exp = np.array([float(r[7]) for r in data])


def mae(a, b): return np.mean(np.abs(a - b))
def mape(a, b): return np.mean(np.abs((a - b) / b)) * 100
def rmse(a, b): return np.sqrt(np.mean((a - b) ** 2))
def r2(a, b):
    return 1 - np.sum((b - a) ** 2) / np.sum((b - b.mean()) ** 2)


for name, calc, exp in [("|D|", np.abs(D), D_exp),
                        ("|E|", np.abs(E), E_exp),
                        ("|E/D|", ED, ED_exp)]:
    print(f"{name}: MAE={mae(calc, exp):.4f} MAPE={mape(calc, exp):.2f}% "
          f"RMSE={rmse(calc, exp):.4f} R2={r2(calc, exp):.4f}")