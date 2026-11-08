"""Independent SVG presence/content checks; standard library only, no project imports.

Requires identifiable experimental and fitted data paths, not just axes/text.
Checks current-fit SHA256 and IDs embedded by drawing modules. This confirms
plot completeness, not numerical fit quality or physical assignments.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
MAX_SVG_BYTES = 30_000_000


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def curve_has_geometry(group, parents):
    if group is None:
        return False
    for path in group.iter():
        if path.tag.rsplit("}", 1)[-1] != "path":
            continue
        ancestor, hidden = path, False
        while ancestor is not None:
            style = ancestor.get("style", "").replace(" ", "").lower()
            if ("display:none" in style or "visibility:hidden" in style or
                    re.search(r"(?:^|;)(?:opacity|stroke-opacity|stroke-width):0(?:\.0*)?(?:;|$)", style) or
                    ancestor.get("opacity") == "0" or ancestor.get("stroke-opacity") == "0" or
                    ancestor.get("display") == "none" or ancestor.get("visibility") == "hidden"):
                hidden = True
                break
            ancestor = parents.get(ancestor)
        if hidden:
            continue
        if path.get("stroke") == "none" or "stroke:none" in path.get("style", "").replace(" ", "").lower():
            continue
        d = path.get("d", "")
        if not re.search(r"[lLcCqQhHvV]", d):
            continue
        numbers = [float(v) for v in re.findall(NUMBER, d)]
        if len(numbers) >= 4 and len(numbers) % 2 == 0 and all(math.isfinite(v) for v in numbers):
            points = list(zip(numbers[::2], numbers[1::2]))
            if len(set(points)) >= 2:
                return True
    return False


def check_figure(root, figure, fit, spectrum_id, fit_id, dimension):
    root = Path(root).resolve()
    figure, fit = (root/figure).resolve(), (root/fit).resolve()
    if dimension not in ("1d", "2d"):
        raise ValueError("dimension must be 1d or 2d")
    if not isinstance(spectrum_id, str) or not re.fullmatch(r"[\w.-]{1,150}", spectrum_id) or not str(fit_id).strip():
        raise ValueError("Nonempty fit_id and filename-compatible spectrum_id are required")
    if not figure.is_relative_to(root):
        raise ValueError("Figure must be inside the project")
    result = {"status": "failed", "dimension": dimension, "figure": str(figure),
              "spectrum_id": spectrum_id, "fit_id": fit_id, "errors": []}
    errors = result["errors"]
    if figure.name != f"fitdraw_{spectrum_id}.svg":
        errors.append("Figure name must be fitdraw_<spectrum_id>.svg")
    if not figure.is_file():
        errors.append("Figure is missing")
        return result
    size = figure.stat().st_size
    result["size_bytes"] = size
    if not 0 < size <= MAX_SVG_BYTES:
        errors.append("Figure is empty or exceeds the 30 MB inspection limit")
        return result
    raw = figure.read_bytes()
    if b"<!DOCTYPE" in raw.upper() and b"<!ENTITY" in raw.upper():
        raise ValueError("SVG entities are unsupported")
    tree = ET.fromstring(raw)
    if tree.tag != "{http://www.w3.org/2000/svg}svg":
        raise ValueError("File is not an SVG document")
    viewport = [float(v) for v in re.findall(NUMBER, tree.get("viewBox", ""))]
    if len(viewport) != 4 or not all(math.isfinite(v) for v in viewport) or min(viewport[2:]) <= 0:
        errors.append("SVG has no positive finite viewBox")
    payloads = []
    for element in tree.iter():
        if element.tag.rsplit("}", 1)[-1] == "description" and element.text:
            try:
                item = json.loads(element.text)
            except ValueError:
                continue
            if isinstance(item, dict) and item.get("schema") == "ai4science.fitdraw.v1":
                payloads.append(item)
    if len(payloads) != 1:
        errors.append("Missing or ambiguous fit drawing metadata")
        return result
    payload = payloads[0]
    fit_sha = digest(fit)
    result.update(fit_sha256=fit_sha, figure_sha256=hashlib.sha256(raw).hexdigest())
    if (payload.get("dimension"), payload.get("spectrum_id"), payload.get("fit_id")) != (dimension, spectrum_id, fit_id):
        errors.append("Figure belongs to another spectrum, dimension or fit_id")
    if not isinstance(payload.get("fit_source"), dict) or payload["fit_source"].get("sha256") != fit_sha:
        errors.append("Figure does not match the current saved fit SHA256; redraw it")
    if payload.get("normalization") != "none_saved_scale":
        errors.append("Unknown plotting scale")
    expected = (["fitdraw_1d_slice_0"] if dimension == "1d" else
                [f"fitdraw_2d_{kind}_{i}" for kind in ("kinetic", "slice") for i in range(6)])
    nodes, duplicates = {}, set()
    parents = {child: parent for parent in tree.iter() for child in parent}
    for node in tree.iter():
        identifier = node.get("id")
        if identifier in nodes:
            duplicates.add(identifier)
        if identifier:
            nodes[identifier] = node
    if duplicates:
        errors.append("SVG has duplicate element IDs")
    panels = payload.get("panels", [])
    if not isinstance(panels, list) or len(panels) != len(expected) or any(not isinstance(p, dict) for p in panels):
        errors.append(f"Expected {len(expected)} plot panels")
        return result
    panel_map = {p.get("id"): p for p in panels}
    if set(panel_map) != set(expected):
        errors.append("Expected one 1D overlay or exactly six kinetics and six spectral overlays")
    families = {"kinetic": [], "slice": []}
    for identifier in expected:
        panel = panel_map.get(identifier, {})
        for series in ("experiment", "fit"):
            summary = panel.get(series, {})
            if (not isinstance(summary, dict) or not finite(summary.get("n_points")) or summary["n_points"] < 2 or
                    not all(finite(summary.get(k)) for k in ("min", "max", "abs_max")) or
                    summary.get("abs_max", -1) < 0):
                errors.append(f"{identifier}/{series}: no valid data summary")
            if not curve_has_geometry(nodes.get(identifier+"_"+series), parents):
                errors.append(f"{identifier}/{series}: missing or empty visible data curve")
        family = "kinetic" if "_kinetic_" in identifier else "slice"
        families[family].append(panel)
    for family, group in families.items():
        for series in ("experiment", "fit"):
            if group and not any(isinstance(p.get(series), dict) and finite(p[series].get("abs_max")) and
                                 p[series]["abs_max"] > 0 for p in group):
                errors.append(f"{family}: all {series} curves are zero")
    result["n_panels"] = len(panels)
    result["status"] = "failed" if errors else "passed"
    return result


def main(dimension, argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--spectrum-id", required=True)
    parser.add_argument("--fit-id", required=True)
    parser.add_argument("--report", type=Path, required=True, help="New JSON inside project; parent must exist")
    args = parser.parse_args(argv)
    try:
        result = check_figure(args.root, args.figure, args.fit, args.spectrum_id, args.fit_id, dimension)
        report = (args.root.resolve()/args.report).resolve()
        if not report.is_relative_to(args.root.resolve()):
            raise ValueError("Report must be inside the project")
        with report.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=True, indent=2, allow_nan=False)+"\n")
    except (OSError, ValueError, TypeError, KeyError, ET.ParseError, OverflowError) as exc:
        result = {"status": "invalid", "errors": [str(exc)]}
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return {"passed": 0, "failed": 1, "invalid": 2}[result["status"]]
