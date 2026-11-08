"""Search contract for high-J 2D kinetics; Hamiltonian/linewidths are immutable.

All times are in microseconds. Populations refer to the chosen preparation basis,
with unit sum. See README_2D.md for JSON/CLI, aliases and the rise formula.
"""

import copy
import json
import math
from pathlib import Path

from .fit_parameters import add_fit_arguments
from ..mars_simulation.validation import POPULATION_BASES, population_basis as validate_population_basis


POP_NAMES = ("p_x", "p_y", "p_z")
TIME_NAMES = ("tau_intra", "tau_out_x", "tau_out_y", "tau_out_z", "rise_time_us")
ALIASES = {"pop_x": "p_x", "pop_y": "p_y", "pop_z": "p_z",
           "px": "p_x", "py": "p_y", "pz": "p_z",
           "t_x": "tau_out_x", "t_y": "tau_out_y", "t_z": "tau_out_z",
           "T1": "tau_intra", "T_1": "tau_intra"}
DEFAULT_PARAMETERS = {
    **{name: {"value": value, "bounds": [0., 1.], "vary": True, "scale": "linear"}
       for name, value in zip(POP_NAMES, (.2, .5, .3))},
    **{name: {"value": 10., "bounds": [.1, 100000.], "vary": True, "scale": "log"}
       for name in TIME_NAMES[:-1]},
    "rise_time_us": {"value": .2, "bounds": [.01, 1000.], "vary": True, "scale": "log"},
}


def number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def canonical_name(name):
    name = ALIASES.get(name, name)
    if name not in DEFAULT_PARAMETERS:
        raise ValueError(f"Unknown context parameter {name!r}; only {tuple(DEFAULT_PARAMETERS)} may vary. "
                         "Hamiltonian, linewidths and experimental settings stay fixed from 1D.")
    return name


def _domain(name, value):
    if name in POP_NAMES:
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
    elif value <= 0 or not math.isfinite(1e6/value):
        raise ValueError(f"{name} must be positive and have a finite inverse rate")


def _entries(value):
    if (not isinstance(value, dict) or set(value)-{"schema_version", "parameters", "population_basis"}
            or type(value.get("schema_version", 1)) is not int or value.get("schema_version", 1) != 1):
        raise ValueError("2D context config accepts schema_version=1, parameters and population_basis")
    if "population_basis" in value:
        validate_population_basis(value["population_basis"])
    entries = value.get("parameters", {})
    if not isinstance(entries, dict):
        raise ValueError("parameters must be an object")
    result = {}
    for supplied, spec in entries.items():
        name = canonical_name(supplied)
        if name in result or not isinstance(spec, dict) or set(spec)-{"value", "bounds", "vary", "scale"}:
            raise ValueError(f"Duplicate alias or invalid specification for {name}")
        result[name] = copy.deepcopy(spec)
    return result


class ContextFitConfig:
    """Bounded kinetic search space, with one dependent free population.

    With N enabled populations only N-1 are independent; fixed probabilities
    never move. Null disables a time term and is allowed only with vary=false.
    """

    def __init__(self, parameters=None, population_basis="xyz"):
        self.population_basis = validate_population_basis(population_basis)
        self.parameters = copy.deepcopy(DEFAULT_PARAMETERS)
        for name, spec in _entries({"parameters": {} if parameters is None else parameters}).items():
            self.parameters[name].update(spec)
        for name, spec in self.parameters.items():
            if type(spec["vary"]) is not bool or spec["scale"] not in ("linear", "log"):
                raise ValueError(f"{name}: vary must be boolean; scale must be linear or log")
            if name in POP_NAMES and spec["scale"] != "linear":
                raise ValueError("Populations require linear scale")
            if spec["value"] is None:
                if name in POP_NAMES or spec["vary"]:
                    raise ValueError(f"{name}: null is allowed only for a fixed time")
            else:
                spec["value"] = number(spec["value"], name+".value")
                _domain(name, spec["value"])
            bounds = spec["bounds"]
            if bounds is not None:
                if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                    raise ValueError(f"{name}.bounds must be [low, high] or null")
                lo, hi = [number(v, name+".bounds") for v in bounds]
                _domain(name, lo)
                _domain(name, hi)
                if lo >= hi or not math.isfinite(hi-lo):
                    raise ValueError(f"{name}: low must be less than high; use vary=false to fix")
                spec["bounds"] = [lo, hi]
            if spec["vary"]:
                if bounds is None:
                    raise ValueError(f"Supply bounds before varying {name}")
                spec["value"] = min(max(spec["value"], lo), hi)
        self.population_variables = tuple(k for k in POP_NAMES if self.parameters[k]["vary"])
        self.population_pivot = self.population_variables[-1] if self.population_variables else None
        self.free_names = tuple(k for k in self.parameters
                                if self.parameters[k]["vary"] and k != self.population_pivot)
        self._initial_populations()
        self.check_point(self.values())

    @classmethod
    def from_dict(cls, value):
        return cls(_entries(value), value.get("population_basis", "xyz"))

    def _initial_populations(self):
        varying = self.population_variables
        remaining = 1.-math.fsum(self.parameters[k]["value"] for k in POP_NAMES if k not in varying)
        if not varying:
            if not math.isclose(remaining, 0, abs_tol=1e-10):
                raise ValueError("Fixed populations must sum to 1")
            return
        lo = [self.parameters[k]["bounds"][0] for k in varying]
        hi = [self.parameters[k]["bounds"][1] for k in varying]
        if remaining < math.fsum(lo)-1e-12 or remaining > math.fsum(hi)+1e-12:
            raise ValueError("Population bounds and fixed values have no unit-sum solution")
        original = [self.parameters[k]["value"] for k in varying]
        if not math.isclose(math.fsum(original), remaining, abs_tol=1e-12, rel_tol=0):
            left, right = min(a-v for a, v in zip(lo, original))-1, max(b-v for b, v in zip(hi, original))+1
            for _ in range(80):
                shift = (left+right)/2
                if math.fsum(min(max(v+shift, a), b) for v, a, b in zip(original, lo, hi)) < remaining:
                    left = shift
                else:
                    right = shift
            for name, a, b, value in zip(varying, lo, hi, original):
                self.parameters[name]["value"] = min(max(value+(left+right)/2, a), b)
        pivot = self.population_pivot
        self.parameters[pivot]["value"] = min(1., max(0., 1.-math.fsum(
            self.parameters[k]["value"] for k in POP_NAMES if k != pivot)))

    @property
    def n_free_model(self):
        return len(self.free_names)

    def values(self):
        return {k: spec["value"] for k, spec in self.parameters.items()}

    def check_point(self, params):
        if not isinstance(params, dict) or set(params) != set(self.parameters):
            raise ValueError("Point must contain all and only the context fit parameters")
        for name, spec in self.parameters.items():
            value = params[name]
            if value is None:
                if spec["vary"] or spec["value"] is not None:
                    raise ValueError(f"{name}: unexpected null")
                continue
            value = number(value, name)
            _domain(name, value)
            if spec["vary"]:
                lo, hi = spec["bounds"]
                tol = 1e-12*max(abs(lo), abs(hi), 1)
                if not lo-tol <= value <= hi+tol:
                    raise ValueError(f"{name} is outside selected bounds")
            elif value != spec["value"]:
                raise ValueError(f"Fixed parameter {name} was changed")
        if not math.isclose(math.fsum(params[k] for k in POP_NAMES), 1., abs_tol=1e-10, rel_tol=0):
            raise ValueError("Populations must sum to 1")
        return params

    def _coordinate(self, name, value):
        spec = self.parameters[name]
        lo, hi = spec["bounds"]
        if spec["scale"] == "log":
            return (math.log(value)-math.log(lo))/(math.log(hi)-math.log(lo))
        return (value-lo)/(hi-lo)

    def start_vector(self):
        return [self._coordinate(k, self.parameters[k]["value"]) for k in self.free_names]

    def decode(self, vector):
        if len(vector) != len(self.free_names):
            raise ValueError("Optimizer vector does not match free context parameters")
        params = self.values()
        for name, raw in zip(self.free_names, vector):
            x = number(float(raw), name+" coordinate")
            if not -1e-12 <= x <= 1+1e-12:
                raise ValueError("Optimizer coordinate is outside [0,1]")
            x = min(1., max(0., x))
            spec = self.parameters[name]
            lo, hi = spec["bounds"]
            params[name] = (lo if x == 0 else hi if x == 1 else
                            math.exp(math.log(lo)+x*(math.log(hi)-math.log(lo)))
                            if spec["scale"] == "log" else lo+x*(hi-lo))
        if self.population_pivot:
            params[self.population_pivot] = 1.-math.fsum(params[k] for k in POP_NAMES if k != self.population_pivot)
        return self.check_point(params)

    def suggest(self, trial):
        params = self.values()
        for name in self.free_names:
            if name not in POP_NAMES:
                spec = self.parameters[name]
                params[name] = trial.suggest_float(name, *spec["bounds"], log=spec["scale"] == "log")
        remaining = 1.-math.fsum(params[k] for k in POP_NAMES if k not in self.population_variables)
        for index, name in enumerate(self.population_variables):
            rest = self.population_variables[index+1:]
            lo, hi = self.parameters[name]["bounds"]
            lo = max(lo, remaining-math.fsum(self.parameters[k]["bounds"][1] for k in rest))
            hi = min(hi, remaining-math.fsum(self.parameters[k]["bounds"][0] for k in rest))
            if hi < lo-1e-12:
                raise ValueError("Infeasible population sample")
            value = remaining if not rest else lo if hi-lo < 1e-14 else trial.suggest_float(name, lo, hi)
            params[name] = min(max(value, lo), hi)
            remaining -= params[name]
        return self.check_point(params)

    def to_dict(self, values=None):
        result = copy.deepcopy(self.parameters)
        if values is not None:
            self.check_point(values)
            for name in result:
                result[name]["value"] = values[name]
        return {"schema_version": 1, "parameters": result, "population_basis": self.population_basis}


def config_from_args(args, *, inherited=None, initial=None):
    definitions = copy.deepcopy(DEFAULT_PARAMETERS)
    basis = "xyz"
    if inherited is not None:
        for name, spec in _entries(inherited).items():
            definitions[name].update(spec)
        basis = inherited.get("population_basis", basis)
    for supplied, value in (initial or {}).items():
        if supplied == "population_basis":
            basis = validate_population_basis(value)
        else:
            definitions[canonical_name(supplied)]["value"] = value
    if args.fit_config:
        value = json.loads(Path(args.fit_config).read_text(encoding="utf-8-sig"))
        for name, spec in _entries(value).items():
            definitions[name].update(spec)
        basis = value.get("population_basis", basis)
    def cli_value(raw):
        return None if raw.lower() in ("null", "none") else float(raw)
    for supplied, raw in args.initial:
        definitions[canonical_name(supplied)]["value"] = cli_value(raw)
    for supplied, low, high in args.bounds:
        definitions[canonical_name(supplied)]["bounds"] = [float(low), float(high)]
    fixed = {canonical_name(item[0]) for item in args.fix}
    varying = {canonical_name(name) for name in args.vary}
    if fixed & varying:
        raise ValueError(f"Cannot both --fix and --vary: {sorted(fixed & varying)}")
    for supplied, raw in args.fix:
        definitions[canonical_name(supplied)].update(value=cli_value(raw), vary=False)
    for name in varying:
        definitions[name]["vary"] = True
    if getattr(args, "basis", None) is not None:
        basis = validate_population_basis(args.basis)
    return ContextFitConfig(definitions, basis)
