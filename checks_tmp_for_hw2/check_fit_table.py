"""Validate the current 1D/2D fit's row in the project's summary CSV table.

Standard-library only; no simulation, optimization or table modification.
Exit codes: 0 passed, 1 failed criteria, 2 invalid input/report destination.
"""

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sys


POPULATION_ATOL = 1e-6
BASE_COLUMNS = ("spectrum_id", "fit_id", "D_MHz", "E_MHz", "g", "px", "py", "pz", "lw_Gs")
LIFETIME_COLUMNS = ("t_x", "t_y", "t_z", "T_1")


def _number(text, name, errors):
    try:
        value = float(text)
    except (TypeError, ValueError, OverflowError):
        errors.append(f"{name}: required finite number, not empty/None/null")
        return None
    if not math.isfinite(value):
        errors.append(f"{name}: NaN and +/-Inf are forbidden")
        return None
    return value


def validate_parameters(row, dimension):
    """Return errors and parsed values; table D/E are MHz, lifetimes are us."""
    errors, values = [], {}
    basis = row.get("population_basis", "xyz")
    if basis not in ("xyz", "product", "eigen", "zfs", "multiplet", "zeeman"):
        errors.append("population_basis: choose xyz/product/eigen/zfs/multiplet/zeeman")
    else:
        values["population_basis"] = basis
    names = ("D_MHz", "E_MHz", "px", "py", "pz", "lw_Gs")
    if dimension == "2d":
        names += LIFETIME_COLUMNS
    for name in names:
        value = _number(row.get(name), name, errors)
        if value is not None:
            values[name] = value

    try:
        raw_g = row.get("g", "")
        if isinstance(raw_g, bool):
            raise ValueError
        g = json.loads(raw_g) if isinstance(raw_g, str) and raw_g.strip().startswith("[") else float(raw_g)
        numbers = g if isinstance(g, list) else [g]
        if isinstance(g, list) and len(g) not in (2, 3):
            raise ValueError
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in numbers):
            raise ValueError
        values["g"] = g
        if not all(1 <= v <= 4 for v in numbers):
            errors.append("g: each value must be in [1, 4]")
    except (TypeError, ValueError, OverflowError):
        errors.append("g: required finite number or JSON array of two/three finite numbers")

    D, E = values.get("D_MHz"), values.get("E_MHz")
    if D is not None and not 100 <= D <= 10000:
        errors.append("D_MHz: must be in [100, 10000]")
    if D is not None and E is not None and D > 0:
        ratio = E / D
        # Extreme invalid inputs must not introduce Infinity into the report.
        if not math.isfinite(ratio) or not ratio < 1/3:
            errors.append("E/D: must be strictly less than 1/3")
        if math.isfinite(ratio):
            values["E_over_D"] = ratio
    lw = values.get("lw_Gs")
    if lw is not None and not 0 < lw < 20:
        errors.append("lw_Gs: must be strictly between 0 and 20 G")
    pop = [values.get(k) for k in ("px", "py", "pz")]
    if all(v is not None for v in pop):
        if not all(0 <= v <= 1 for v in pop):
            errors.append("px/py/pz: each population must be in [0, 1]")
        else:
            values["population_sum"] = math.fsum(pop)
            if not math.isclose(values["population_sum"], 1, rel_tol=0, abs_tol=POPULATION_ATOL):
                errors.append("px+py+pz: must equal 1 within absolute tolerance 1e-6")
    if dimension == "2d":
        for name in LIFETIME_COLUMNS:
            value = values.get(name)
            if value is not None and not 0.1 <= value <= 100000:
                errors.append(f"{name}: must be in [0.1, 100000] us")
    return errors, values


def check_table(root, dimension, spectrum_id, fit_id):
    if dimension not in ("1d", "2d"):
        raise ValueError("dimension must be 1d or 2d")
    if not isinstance(spectrum_id, str) or not spectrum_id.strip() or not isinstance(fit_id, str) or not fit_id.strip():
        raise ValueError("Nonempty spectrum_id and fit_id are required")
    root = Path(root).resolve()
    table = root / "results" / "tables" / f"{dimension}-fit_results.csv"
    result = {"status": "failed", "dimension": dimension, "table": str(table),
              "spectrum_id": spectrum_id, "fit_id": fit_id, "errors": [],
              "population_atol": POPULATION_ATOL}
    try:
        raw = table.read_bytes()
    except OSError as exc:
        result.update(status="invalid", errors=[f"Cannot read fit table: {exc}"])
        return result
    result["table_sha256"] = hashlib.sha256(raw).hexdigest()
    try:
        rows = csv.reader(io.StringIO(raw.decode("utf-8-sig")), strict=True)
        header, selected = None, []
        for row in rows:
            if not row or all(not v.strip() for v in row) or row[0].lstrip().startswith("#"):
                continue
            if header is None:
                header = [v.strip() for v in row]
                required = BASE_COLUMNS + (LIFETIME_COLUMNS if dimension == "2d" else ())
                if len(set(header)) != len(header) or any(not k for k in header):
                    raise ValueError("Duplicate or empty table column names")
                missing = set(required) - set(header)
                if missing:
                    raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
                continue
            if len(row) != len(header):
                raise ValueError(f"CSV line {rows.line_num}: wrong number of columns")
            item = {k: v.strip() for k, v in zip(header, row)}
            if item["spectrum_id"] == spectrum_id and item["fit_id"] == fit_id:
                selected.append((rows.line_num, item))
        if header is None:
            raise ValueError("Fit table is empty")
    except (UnicodeError, csv.Error, ValueError) as exc:
        result.update(status="invalid", errors=[str(exc)])
        return result
    result["matching_rows"] = len(selected)
    if len(selected) != 1:
        result["errors"] = [f"Expected exactly one row for this spectrum_id/fit_id; found {len(selected)}"]
        return result
    line, row = selected[0]
    errors, values = validate_parameters(row, dimension)
    result.update(status="failed" if errors else "passed", errors=errors,
                  row_line=line, parameters=values)
    return result


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root", type=Path, required=True, help="Project root")
    cli.add_argument("--dimension", choices=("1d", "2d"), required=True)
    cli.add_argument("--spectrum-id", required=True, help="Exact ID of this slice or 2D measurement")
    cli.add_argument("--fit-id", required=True, help="Exact ID of the current saved fit")
    cli.add_argument("--report", type=Path, required=True, help="New JSON file inside the project; parent must exist")
    args = cli.parse_args(argv)
    try:
        result = check_table(args.root, args.dimension, args.spectrum_id, args.fit_id)
        root = args.root.resolve()
        report = (root / args.report).resolve()
        if not report.is_relative_to(root):
            raise ValueError("Report must be inside the project")
        payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        with report.open("x", encoding="utf-8") as handle:
            handle.write(payload)
    except (OSError, ValueError) as exc:
        result = {"status": "invalid", "errors": [str(exc)]}
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return {"passed": 0, "failed": 1, "invalid": 2}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
