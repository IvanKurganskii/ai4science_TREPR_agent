"""Resonance fields for the X/Y/Z principal axes of a high-J S=1 fit.

Solve adjacent eigenvalue gaps of H/h=(mu_B/h) B g_i S_i +
D(Sz**2-2/3)+E(Sx**2-Sy**2), rather than infer positions from powder extrema.
These are unbroadened orientation resonances, not guaranteed observed maxima.
Current g and D/E tensors share their principal frame, as in mars_simulation.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    from fit_draw_io import cli_result, fingerprint, finite_array, scalar
else:
    from .fit_draw_io import cli_result, fingerprint, finite_array, scalar

MU_B_OVER_H = 1.39962449171e10  # Hz/T; CODATA 2022, also used in fit-slice.md


def read_parameters(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = csv.DictReader(line for line in handle if not line.lstrip().startswith("#"))
            if not {"parameter", "value", "unit"} <= set(rows.fieldnames or ()):
                raise ValueError("Use the vertical parameter,value,unit export, not the summary fit table")
            value = {}
            for row in rows:
                if any(not isinstance(row.get(k), str) for k in ("parameter", "value", "unit")):
                    raise ValueError("Malformed parameter CSV row")
                key = row["parameter"].strip()
                if not key or key in value:
                    raise ValueError("Empty or duplicate parameter name")
                if key in ("D", "E", "freq") and row["unit"].strip() != "Hz":
                    raise ValueError(f"{key} must be in Hz")
                value[key] = row["value"]
    elif path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    else:
        raise ValueError("Parameters must be a current fitter CSV or JSON")
    if not isinstance(value, dict):
        raise ValueError("Parameter JSON must be an object")
    if "simulation" in value:
        if not isinstance(value["simulation"], dict):
            raise ValueError("simulation must be an object")
        value = value["simulation"].get("spin_model", value["simulation"])
    elif "fixed_spin_model" in value:
        value = value["fixed_spin_model"]
    elif "spin_model" in value:
        value = value["spin_model"]
    elif "parameters" in value:
        value = value["parameters"]
    if not isinstance(value, dict):
        raise ValueError("Saved parameters/spin_model must be an object")
    if "hamiltonian" in value:
        if value.get("model", "high_j") != "high_j":
            raise ValueError("Canonical S=1 calculation supports high_j only; specify fields explicitly for other models")
        if not isinstance(value["hamiltonian"], dict) or not isinstance(value.get("settings"), dict):
            raise ValueError("spin_model needs hamiltonian and settings objects")
        value = dict(value["hamiltonian"], freq=value["settings"]["freq"])
    if not {"D", "E", "freq"} <= value.keys() or not ("g" in value or "g_iso" in value):
        raise ValueError("Need full saved D, E, freq in Hz and g/g_iso")
    g = value.get("g", value.get("g_iso"))
    if isinstance(g, str):
        g = json.loads(g)
    return {"D": scalar(value["D"], "D_Hz"), "E": scalar(value["E"], "E_Hz"),
            "freq": scalar(value["freq"], "freq_Hz", 0.), "g": g}


def canonical_fields(D, E, freq, g):
    """Return six labelled fields in T, sorted by field; no population assignment.

    The zero-field span must be smaller than the microwave frequency. Outside
    this regime there can be more resonance branches; supply explicit fields.
    Degenerate X/Y (or all) positions are retained with separate axis labels.
    """
    from scipy.optimize import brentq
    D, E, freq = scalar(D, "D"), scalar(E, "E"), scalar(freq, "freq", 0.)
    if freq == 0 or abs(E) > abs(D)/3.+1e-12*max(1., abs(D)):
        raise ValueError("Need freq > 0 and |E| <= |D|/3")
    gs = finite_array(g, "g")
    if gs.ndim == 0:
        gs = np.repeat(gs, 3)
    if gs.shape != (3,) or np.any(gs <= 0):
        raise ValueError("g must be positive scalar or three aligned principal values")
    sx = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], complex)/np.sqrt(2.)
    sy = np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]], complex)/np.sqrt(2.)
    sz = np.diag([1., 0., -1.])
    h0 = D*(sz@sz-2.*np.eye(3)/3.)+E*(sx@sx-sy@sy)
    if not np.isfinite(h0).all() or np.ptp(np.linalg.eigvalsh(h0)) >= freq:
        raise ValueError("Zero-field splitting must be smaller than freq; specify explicit fields in this regime")
    positions = []
    for label, spin, gi in zip("XYZ", (sx, sy, sz), gs):
        upper = (freq+4.*(abs(D)+abs(E))+1.)/(MU_B_OVER_H*gi)
        resonances = []
        for lower in (0, 1):
            def detuning(field):
                energies = np.linalg.eigvalsh(h0+MU_B_OVER_H*gi*field*spin)
                return float(energies[lower+1]-energies[lower]-freq)
            field = float(brentq(detuning, 0., upper, xtol=1e-14, rtol=1e-12))
            resonances.append((field, lower))
        for side, (field, lower) in zip(("left", "right"), sorted(resonances)):
            positions.append({"label": label+"_"+side, "orientation": label,
                              "field_T": field, "levels": [lower, lower+1]})
    return sorted(positions, key=lambda p: p["field_T"])


def validate_positions(value, count=None):
    if isinstance(value, dict):
        value = value.get("positions")
    if not isinstance(value, list) or not value:
        raise ValueError("Positions must be a nonempty list of {label, field_T}")
    positions = []
    for i, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError("Each position needs a label and field_T")
        label = item.get("label", f"B{i+1}")
        if not isinstance(label, str) or not label.strip() or len(label) > 80:
            raise ValueError("Position labels must be nonempty, at most 80 characters")
        positions.append(dict(item, label=label, field_T=scalar(item["field_T"], "field_T", 0.)))
    if len({p["label"] for p in positions}) != len(positions):
        raise ValueError("Position labels must be unique (degenerate fields may coincide)")
    if count is not None and len(positions) != count:
        raise ValueError(f"Need exactly {count} field positions")
    return positions


def add_position_arguments(parser):
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--parameters", type=Path, help="Compute principal-axis fields from saved parameters")
    group.add_argument("--positions", type=Path, help="JSON containing positions with label and field_T")
    group.add_argument("--fields-T", type=float, nargs="+", help="Agent-selected fields in T, overriding theory")


def resolve_positions(args, count=None):
    if args.parameters:
        params = read_parameters(args.parameters)
        positions = canonical_fields(**params)
        info = {"method": "spin1_eigenvalue_resonances", "source": fingerprint(args.parameters), "parameters": params}
    elif args.positions:
        positions = json.loads(args.positions.read_text(encoding="utf-8-sig"))
        info = {"method": "supplied_positions", "source": fingerprint(args.positions)}
    else:
        positions = [{"label": f"B{i+1}", "field_T": field} for i, field in enumerate(args.fields_T)]
        info = {"method": "agent_selected_fields"}
    return validate_positions(positions, count), info


def _run(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    add_position_arguments(parser)
    parser.add_argument("--report", type=Path, help="Optional new JSON file")
    args = parser.parse_args(argv)
    positions, info = resolve_positions(args)
    result = {"status": "calculated", "positions": positions, **info,
              "note": "Unbroadened principal-axis fields; strain/linewidths do not move these nominal positions"}
    if args.report:
        with args.report.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False)+"\n")
    return result


def main(argv=None):
    return cli_result(_run, argv)


if __name__ == "__main__":
    raise SystemExit(main())
