"""Validate JSON inputs before importing the numerical backend."""

import math
from collections.abc import Mapping


DEFAULT_SETTINGS = {
    "freq": 9.8e9, "temperature": 80.0, "mesh": 25,
    "frame": [0.0, 0.0, 0.0], "device": "cpu", "dtype": "float64", "harmonic": 0,
}


def object_fields(value, allowed, required=(), name="parameters"):
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    unknown, missing = set(value) - set(allowed), set(required) - set(value)
    if unknown:
        raise ValueError(f"Unknown {name} fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"Missing {name} fields: {sorted(missing)}")
    return dict(value)


def number(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite; use null for a disabled lifetime")
    if positive and result <= 0 or nonnegative and result < 0:
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return result


def integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def settings(value=None):
    value = object_fields({} if value is None else value, DEFAULT_SETTINGS, name="settings")
    result = {**DEFAULT_SETTINGS, **value}
    result["freq"] = number(result["freq"], "freq (Hz)", positive=True)
    result["temperature"] = number(result["temperature"], "temperature (K)", positive=True)
    result["mesh"] = integer(result["mesh"], "mesh", 2)
    if result["dtype"] not in ("float32", "float64"):
        raise ValueError("dtype must be float32 or float64")
    if result["device"] not in ("cpu", "cuda", "auto"):
        raise ValueError("device must be cpu, cuda or auto")
    if integer(result["harmonic"], "harmonic") not in (0, 1):
        raise ValueError("harmonic must be 0 (absorption) or 1 (first derivative)")
    frame = result["frame"]
    if not isinstance(frame, (list, tuple)) or len(frame) != 3:
        raise ValueError("frame must contain three ZYZ Euler angles in radians")
    result["frame"] = [number(v, "frame") for v in frame]
    return result


def hamiltonian(model, value):
    required = ("g", "D", "E", "gauss", "lorentz") if model == "high_j" else (
        "g0", "delta_g", "D", "E", "J", "gauss", "lorentz")
    result = object_fields(value, required, required, "hamiltonian")
    for key in required:
        if key == "g" and isinstance(result[key], (list, tuple)):
            if len(result[key]) != 3:
                raise ValueError("g must be a scalar or three principal values")
            result[key] = [number(v, "g", positive=True) for v in result[key]]
        else:
            result[key] = number(result[key], key, positive=key in ("g", "g0"),
                                 nonnegative=key in ("gauss", "lorentz"))
    if abs(result["E"]) > abs(result["D"]) / 3.0 + 1e-12 * max(1, abs(result["D"])):
        raise ValueError("D and E in Hz must satisfy |E| <= |D|/3")
    if result["gauss"] == result["lorentz"] == 0:
        raise ValueError("At least one linewidth must be positive")
    if model == "moderate_j" and result["g0"] <= abs(result["delta_g"]) / 2:
        raise ValueError("Both g0 +/- delta_g/2 must be positive")
    return result


def population_names(model):
    return ("p_x", "p_y", "p_z") if model == "high_j" else ("p_s", "p_x", "p_y", "p_z")


def lifetime_names(model):
    return ("tau_intra", "tau_out_x", "tau_out_y", "tau_out_z") if model == "high_j" else (
        "tau_outer", "tau_inner", "tau_out_s", "tau_out_x", "tau_out_y", "tau_out_z")


def populations(model, value):
    names = population_names(model)
    result = object_fields(value, names, names, "populations")
    for key in names:
        result[key] = number(result[key], key, nonnegative=True)
        if result[key] > 1:
            raise ValueError(f"{key} must be in [0, 1]")
    return result


def context(model, value):
    names = population_names(model) + lifetime_names(model)
    result = object_fields(value, names, names, "context")
    result.update(populations(model, {k: result[k] for k in population_names(model)}))
    for key in lifetime_names(model):
        if result[key] is not None:
            result[key] = number(result[key], key + " (us)", positive=True)
            if not math.isfinite(1e6 / result[key]):
                raise ValueError(f"{key} is too small to represent its rate")
    return result


def axis(value, name, *, max_points=1_000_000):
    """Accept explicit values or an inclusive uniform grid, never both."""
    value = object_fields(value, ("values", "start", "stop", "points"), name=name)
    if "values" in value:
        if set(value) != {"values"} or not isinstance(value["values"], (list, tuple)):
            raise ValueError(f"{name}: values cannot be combined with grid fields")
        raw = value["values"]
        if not 1 <= len(raw) <= max_points:
            raise ValueError(f"{name} must contain 1..{max_points} values")
        values = [number(v, name, nonnegative=True) for v in raw]
    else:
        object_fields(value, ("start", "stop", "points"), ("start", "stop", "points"), name)
        start = number(value["start"], name + ".start", nonnegative=True)
        stop = number(value["stop"], name + ".stop", nonnegative=True)
        count = integer(value["points"], name + ".points", 1)
        if count > max_points or stop < start or (count > 1 and stop == start):
            raise ValueError(f"{name}: invalid grid range or point count")
        if count == 1 and start != stop:
            raise ValueError(f"{name}: one-point grid requires start == stop")
        values = [start] if count == 1 else [start + (stop-start)*i/(count-1) for i in range(count)]
    if any(b <= a for a, b in zip(values, values[1:])):
        raise ValueError(f"{name} must be strictly increasing")
    if name == "field_axis" and len(values) < 2:
        raise ValueError("field_axis needs at least two points, in tesla")
    return values
