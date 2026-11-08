"""Shared fit configuration for all triplet fitters; see README.md.

JSON: {"parameters": {name: {"value": ..., "bounds": [lo, hi], "vary": bool}},
       "settings": {"harmonic": 0, "mesh": 20}}.
Frequency parameters use Hz, linewidths T, temperature K. D defaults to
500--3000 MHz. Fixed values remain in every full model dictionary.
Population constraints are enforced on physical, unit-sum populations.
"""
import copy
import json
import math
from pathlib import Path


POP_NAMES = ("pop_x", "pop_y", "pop_z")
POPULATION_BASES = ("xyz", "product", "eigen", "zfs", "multiplet", "zeeman")
ALIASES = {"g": "g_iso", "px": "pop_x", "py": "pop_y", "pz": "pop_z",
           "p_x": "pop_x", "p_y": "pop_y", "p_z": "pop_z"}
DEFAULT_PARAMETERS = {
    "g_iso": {"value": 2.0023, "bounds": [2.0, 2.12], "vary": True},
    "D": {"value": 1.5e9, "bounds": [500e6, 3000e6], "vary": True},
    "E": {"value": 20e6, "bounds": [5e6, 40e6], "vary": True},
    "ham_strain": {"value": 0.0, "bounds": [0.0, 1e8], "vary": False},
    "gauss": {"value": 1e-4, "bounds": [1e-5, 5e-4], "vary": True},
    "lorentz": {"value": 0.0, "bounds": [0.0, 5e-4], "vary": False},
    "pop_x": {"value": 0.2, "bounds": [0.0, 1.0], "vary": True},
    "pop_y": {"value": 0.5, "bounds": [0.0, 1.0], "vary": True},
    "pop_z": {"value": 0.3, "bounds": [0.0, 1.0], "vary": True},
    "freq": {"value": 9.8e9, "bounds": None, "vary": False},
    "temperature": {"value": 10.0, "bounds": None, "vary": False},
    "basis": {"value": "xyz", "choices": ["xyz", "product"], "vary": False},
}
NUMERIC_NAMES = tuple(k for k in DEFAULT_PARAMETERS if k != "basis")
PARAMETER_UNITS = {"D": "Hz", "E": "Hz", "ham_strain": "Hz", "freq": "Hz",
                   "gauss": "T", "lorentz": "T", "temperature": "K"}


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return value


def _domain(name, value):
    if name in ("g_iso", "freq", "temperature") and value <= 0:
        raise ValueError(f"{name} must be positive")
    if name in ("ham_strain", "gauss", "lorentz") and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    if name in POP_NAMES and not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1]")


def canonical_name(name):
    name = ALIASES.get(name, name)
    if name not in DEFAULT_PARAMETERS:
        raise ValueError(f"Unknown fit parameter {name!r}; choose from {tuple(DEFAULT_PARAMETERS)}")
    return name


def read_config(path):
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict) or set(value) - {"schema_version", "parameters", "settings"}:
        raise ValueError("Fit config must contain parameters/settings and optional schema_version")
    if value.get("schema_version", 1) != 1:
        raise ValueError("Unsupported fit config schema_version")
    return value


class FitConfig:
    """Validated physical search space, with a derived final free population.

    Setting vary=False fixes the value; its unused search bounds do not reject
    that value. Only free populations are projected to a feasible initial
    unit-sum vector. No simulated or fitted spectrum is changed by this class.
    """

    def __init__(self, parameters=None, settings=None):
        self.parameters = copy.deepcopy(DEFAULT_PARAMETERS)
        if parameters is not None:
            if not isinstance(parameters, dict):
                raise ValueError("parameters must be an object")
            seen = set()
            for supplied, spec in parameters.items():
                name = canonical_name(supplied)
                if name in seen or not isinstance(spec, dict):
                    raise ValueError(f"Duplicate parameter alias or invalid specification for {name}")
                seen.add(name)
                allowed = {"value", "vary", "choices" if name == "basis" else "bounds"}
                if set(spec) - allowed:
                    raise ValueError(f"Unknown specification fields for {name}: {sorted(set(spec)-allowed)}")
                self.parameters[name].update(copy.deepcopy(spec))
                if name == "basis" and "value" in spec and "choices" not in spec:
                    chosen = spec["value"]
                    if chosen in POPULATION_BASES and chosen not in self.parameters[name]["choices"]:
                        self.parameters[name]["choices"].append(chosen)
        self.settings = {"harmonic": 0, "mesh": 20}
        if settings is not None:
            if not isinstance(settings, dict) or set(settings) - set(self.settings):
                raise ValueError("settings accepts only harmonic and mesh")
            self.settings.update(settings)
        if type(self.settings["harmonic"]) is not int or self.settings["harmonic"] not in (0, 1):
            raise ValueError("harmonic must be 0 or 1")
        if type(self.settings["mesh"]) is not int or self.settings["mesh"] < 5:
            raise ValueError("mesh must be an integer >= 5")
        for name, spec in self.parameters.items():
            if type(spec["vary"]) is not bool:
                raise ValueError(f"{name}.vary must be true or false")
            if name == "basis":
                choices = spec["choices"]
                if (not isinstance(choices, list) or not choices or len(set(choices)) != len(choices)
                        or any(c not in POPULATION_BASES for c in choices) or spec["value"] not in choices):
                    raise ValueError(f"basis value/choices must select from {POPULATION_BASES}")
                continue
            spec["value"] = _number(spec["value"], name+".value")
            _domain(name, spec["value"])
            bounds = spec["bounds"]
            if bounds is not None:
                if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                    raise ValueError(f"{name}.bounds must be [low, high] or null")
                lo, hi = [_number(v, name+".bounds") for v in bounds]
                if lo >= hi or not math.isfinite(hi-lo):
                    raise ValueError(f"{name}: low must be less than high; use vary=false to fix a value")
                _domain(name, lo)
                _domain(name, hi)
                spec["bounds"] = [lo, hi]
            if spec["vary"]:
                if bounds is None:
                    raise ValueError(f"Supply bounds before varying {name}")
                # A new search interval can exclude the inherited/default starting value.
                spec["value"] = min(max(spec["value"], lo), hi)
        self.population_variables = tuple(k for k in POP_NAMES if self.parameters[k]["vary"])
        self.population_pivot = self.population_variables[-1] if self.population_variables else None
        self.free_names = tuple(k for k in NUMERIC_NAMES
                                if self.parameters[k]["vary"] and k != self.population_pivot)
        self._initial_populations()
        self.check_point(self.values())

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) - {"schema_version", "parameters", "settings"}:
            raise ValueError("Invalid fit configuration keys")
        if value.get("schema_version", 1) != 1:
            raise ValueError("Unsupported fit config schema_version")
        return cls(value.get("parameters"), value.get("settings"))

    def _initial_populations(self):
        varying = self.population_variables
        fixed_sum = math.fsum(self.parameters[k]["value"] for k in POP_NAMES if k not in varying)
        remaining = 1.0 - fixed_sum
        if not varying:
            if not math.isclose(fixed_sum, 1, rel_tol=0, abs_tol=1e-10):
                raise ValueError("Fixed populations must sum to 1")
            return
        lo = [self.parameters[k]["bounds"][0] for k in varying]
        hi = [self.parameters[k]["bounds"][1] for k in varying]
        if remaining < sum(lo)-1e-12 or remaining > sum(hi)+1e-12:
            raise ValueError("Population bounds and fixed values have no unit-sum solution")
        original = [self.parameters[k]["value"] for k in varying]
        if math.isclose(math.fsum(original), remaining, rel_tol=0, abs_tol=1e-12):
            pivot = self.population_pivot
            self.parameters[pivot]["value"] = min(1., max(0., 1.0-math.fsum(
                self.parameters[k]["value"] for k in POP_NAMES if k != pivot)))
            return
        left = min(a-v for a, v in zip(lo, original))-1
        right = max(b-v for b, v in zip(hi, original))+1
        for _ in range(80):
            shift = (left+right)/2
            trial = [min(max(v+shift, a), b) for v, a, b in zip(original, lo, hi)]
            if sum(trial) < remaining:
                left = shift
            else:
                right = shift
        projected = [min(max(v+(left+right)/2, a), b) for v, a, b in zip(original, lo, hi)]
        for name, value in zip(varying, projected):
            self.parameters[name]["value"] = value
        pivot = self.population_pivot
        self.parameters[pivot]["value"] = 1.0-math.fsum(self.parameters[k]["value"] for k in POP_NAMES if k != pivot)

    def values(self):
        return {k: spec["value"] for k, spec in self.parameters.items()}

    @property
    def basis_choices(self):
        spec = self.parameters["basis"]
        return ([spec["value"]] + [c for c in spec["choices"] if c != spec["value"]]
                if spec["vary"] else [spec["value"]])

    @property
    def n_free_model(self):
        return len(self.free_names) + int(len(self.basis_choices) > 1)

    def check_point(self, params):
        for name in NUMERIC_NAMES:
            value = _number(params[name], name)
            _domain(name, value)
            spec = self.parameters[name]
            if spec["vary"]:
                lo, hi = spec["bounds"]
                tol = 1e-12*max(1, abs(lo), abs(hi))
                if not lo-tol <= value <= hi+tol:
                    raise ValueError(f"{name} is outside the selected bounds")
            elif value != spec["value"]:
                raise ValueError(f"Fixed parameter {name} was changed")
        if params["basis"] not in self.basis_choices:
            raise ValueError("basis is outside the selected choices")
        if not math.isclose(math.fsum(params[k] for k in POP_NAMES), 1, rel_tol=0, abs_tol=1e-10):
            raise ValueError("Populations must sum to 1")
        if abs(params["E"]) > abs(params["D"])/3 + 1e-12*max(1, abs(params["D"])):
            raise ValueError("The ZFS convention requires |E| <= |D|/3")
        if not any(params[k] > 0 for k in ("gauss", "lorentz", "ham_strain")):
            raise ValueError("At least one broadening parameter must be positive")
        return params

    def start_vector(self):
        return [(self.parameters[k]["value"]-self.parameters[k]["bounds"][0]) /
                (self.parameters[k]["bounds"][1]-self.parameters[k]["bounds"][0]) for k in self.free_names]

    def decode(self, vector, basis=None, *, check=True):
        if len(vector) != len(self.free_names):
            raise ValueError("Optimizer vector does not match free parameter names")
        params = self.values()
        for name, coordinate in zip(self.free_names, vector):
            lo, hi = self.parameters[name]["bounds"]
            params[name] = lo + float(coordinate)*(hi-lo)
        if self.population_pivot:
            params[self.population_pivot] = 1.0-math.fsum(params[k] for k in POP_NAMES if k != self.population_pivot)
        if basis is not None:
            params["basis"] = basis
        return self.check_point(params) if check else params

    def suggest(self, trial):
        params = self.values()
        for name in self.free_names:
            if name not in POP_NAMES:
                lo, hi = self.parameters[name]["bounds"]
                params[name] = trial.suggest_float(name, lo, hi)
        remaining = 1.0-math.fsum(params[k] for k in POP_NAMES if k not in self.population_variables)
        for index, name in enumerate(self.population_variables):
            rest = self.population_variables[index+1:]
            lo, hi = self.parameters[name]["bounds"]
            lo = max(lo, remaining-sum(self.parameters[k]["bounds"][1] for k in rest))
            hi = min(hi, remaining-sum(self.parameters[k]["bounds"][0] for k in rest))
            if hi < lo-1e-12:
                raise ValueError("Infeasible population sample")
            value = (remaining if not rest else (lo if hi-lo <= 1e-14 else trial.suggest_float(name, lo, hi)))
            params[name] = min(max(value, lo), hi)
            remaining -= params[name]
        if len(self.basis_choices) > 1:
            params["basis"] = trial.suggest_categorical("basis", self.parameters["basis"]["choices"])
        return self.check_point(params)

    def margins(self, vector, basis=None):
        params = self.decode(vector, basis, check=False)
        margins = list(vector) + [1-float(v) for v in vector]
        for name in self.population_variables:
            lo, hi = self.parameters[name]["bounds"]
            margins.extend((params[name]-lo, hi-params[name]))
        margins.append((abs(params["D"])/3-abs(params["E"])) / max(1, abs(self.parameters["D"]["value"])))
        return margins

    def to_dict(self, values=None):
        parameters = copy.deepcopy(self.parameters)
        if values is not None:
            self.check_point(values)
            for name in parameters:
                parameters[name]["value"] = values[name]
        return {"schema_version": 1, "parameters": parameters, "settings": dict(self.settings)}


def add_fit_arguments(parser):
    parser.add_argument("--fit-config", type=Path, help="JSON with parameters/value/bounds/vary and numerical settings")
    parser.add_argument("--bounds", action="append", nargs=3, default=[], metavar=("PARAM", "LOW", "HIGH"), help="Override numeric bounds, in SI units; repeatable")
    parser.add_argument("--initial", action="append", nargs=2, default=[], metavar=("PARAM", "VALUE"), help="Override initial/fixed value; repeatable")
    parser.add_argument("--fix", action="append", nargs=2, default=[], metavar=("PARAM", "VALUE"), help="Fix a parameter to this value; repeatable")
    parser.add_argument("--vary", action="append", default=[], metavar="PARAM", help="Enable variation of a parameter; repeatable")
    parser.add_argument("--print-fit-config", action="store_true", help="Print resolved configuration without reading data or fitting")


def config_from_args(args, *, inherited=None, initial=None, defaults=None):
    definitions = copy.deepcopy(DEFAULT_PARAMETERS)
    settings = {"harmonic": 0, "mesh": 20}
    def overlay(value):
        if not isinstance(value, dict) or set(value)-{"schema_version", "parameters", "settings"}:
            raise ValueError("Invalid fit config keys")
        if value.get("schema_version", 1) != 1:
            raise ValueError("Unsupported fit config schema_version")
        entries = value.get("parameters", {})
        if not isinstance(entries, dict):
            raise ValueError("parameters must be an object")
        seen = set()
        for key, spec in entries.items():
            name = canonical_name(key)
            if name in seen:
                raise ValueError(f"Duplicate parameter alias for {name}")
            seen.add(name)
            if not isinstance(spec, dict):
                raise ValueError(f"Invalid specification for {name}")
            definitions[name].update(spec)
            if name == "basis" and "value" in spec and "choices" not in spec:
                chosen = spec["value"]
                if chosen in POPULATION_BASES and chosen not in definitions[name]["choices"]:
                    definitions[name]["choices"].append(chosen)
        extra = value.get("settings", {})
        if not isinstance(extra, dict) or set(extra)-set(settings):
            raise ValueError("settings accepts only harmonic and mesh")
        settings.update(extra)
    for name, value in (defaults or {}).items():
        definitions[canonical_name(name)]["value"] = value
    if inherited:
        overlay(inherited)
    for name, value in (initial or {}).items():
        if name in definitions or name in ALIASES:
            definitions[canonical_name(name)]["value"] = value
            if name == "basis" and value in POPULATION_BASES and value not in definitions[name]["choices"]:
                definitions[name]["choices"].append(value)
        elif name in settings:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not float(value).is_integer():
                raise ValueError(f"Inherited {name} must be an integer")
            settings[name] = int(value)
    if args.fit_config:
        overlay(read_config(args.fit_config))
    for name in NUMERIC_NAMES + ("basis",):
        value = getattr(args, name, None)
        if value is not None:
            definitions[name]["value"] = value
            if name == "basis" and value in POPULATION_BASES and value not in definitions[name]["choices"]:
                definitions[name]["choices"].append(value)
    for name in settings:
        if getattr(args, name, None) is not None:
            settings[name] = getattr(args, name)
    def cli_value(name, raw):
        if name == "basis" and raw in POPULATION_BASES and raw not in definitions[name]["choices"]:
            definitions[name]["choices"].append(raw)
        return raw if name == "basis" else float(raw)
    for supplied, raw in args.initial:
        name = canonical_name(supplied)
        definitions[name]["value"] = cli_value(name, raw)
    for supplied, low, high in args.bounds:
        name = canonical_name(supplied)
        if name == "basis":
            raise ValueError("basis is categorical; configure its choices instead of numeric bounds")
        definitions[name]["bounds"] = [float(low), float(high)]
    fixed = {canonical_name(item[0]) for item in args.fix}
    varying = {canonical_name(name) for name in args.vary}
    if fixed & varying:
        raise ValueError(f"Cannot both --fix and --vary: {sorted(fixed & varying)}")
    for supplied, raw in args.fix:
        name = canonical_name(supplied)
        definitions[name].update(value=cli_value(name, raw), vary=False)
    for name in varying:
        definitions[name]["vary"] = True
    return FitConfig(definitions, settings)
