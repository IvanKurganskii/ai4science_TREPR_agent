"""Bounded independent refinement through MaRS/ Nevergrad Cobyla.

The parent process enforces the wall-clock limit, including backend imports.
Only the worker imports torch/MaRS. Existing spectra and parameters are read-only.
"""
from __future__ import annotations

import contextlib
import copy
from collections import OrderedDict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

import fit_noise as noise


class BudgetReached(Exception):
    pass


class InvalidCandidate(noise.CheckError):
    """A bad trial can be penalized; a bad initial spectrum still fails the check."""


FIT_DEPENDENCIES = {"mars": "mars-epr", "torch": "torch", "tqdm": "tqdm", "nevergrad": "nevergrad",
                    "optuna": "optuna", "optuna_dashboard": "optuna-dashboard", "emcee": "emcee",
                    "numdifftools": "numdifftools", "optuna_integration": "optuna-integration",
                    "hdbscan": "hdbscan"}


def missing_backend():
    """Cheap preflight: never spend a minute importing torch just to find an absent extra."""
    return [package for module, package in FIT_DEPENDENCIES.items()
            if module not in sys.modules and importlib.util.find_spec(module) is None]


def finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise noise.CheckError(f"{label} must be a finite number")
    return float(value)


def node(state, path):
    parts = path.split(".")
    current = state
    for part in parts[:-1]:
        current = current[int(part)] if isinstance(current, list) else current[part]
    key = int(parts[-1]) if isinstance(current, list) else parts[-1]
    return current, key


def get_value(state, path):
    current, key = node(state, path)
    return finite(current[key], path)


def set_value(state, path, value):
    current, key = node(state, path)
    current[key] = finite(value, path)


def allowed_path(path):
    return isinstance(path, str) and len(path) <= 100 and (
        path.startswith(("simulation.hamiltonian.", "simulation.populations.", "simulation.context."))
        or path in ("simulation.rise_time_us", "calibration.scale", "calibration.offset"))


class Problem:
    """JSON-only simulation snapshot and independent, dimensionless variables."""
    def __init__(self, value, dimension):
        if not isinstance(value, dict) or set(value)-{"model", "dimension", "objective", "simulation", "calibration", "variables", "relations", "constraints"}:
            raise noise.CheckError("Unknown or invalid problem fields")
        if value.get("objective") != "noise_k":
            raise noise.CheckError("Declare objective='noise_k'; this check supports the same K/MSE problem, not another loss")
        self.model = value["model"]
        self.dimension = value["dimension"]
        if self.model not in ("high_j", "moderate_j") or self.dimension not in (
                ("1d",) if dimension == "1d" else ("2d", "2d_full")):
            raise noise.CheckError("Problem model/dimension does not match this checker")
        self.state = {"simulation": copy.deepcopy(value["simulation"]),
                      "calibration": {"scale": 1., "offset": 0., **value.get("calibration", {})}}
        if set(self.state["calibration"]) != {"scale", "offset"}:
            raise noise.CheckError("Calibration contains unknown fields")
        # Axes come from the saved prediction; the problem cannot silently replace them.
        allowed = {"hamiltonian", "settings", "spin_model", "populations", "context",
                   "population_mode", "rise_time_us", "normalization", "device", "population_basis"}
        if not isinstance(self.state["simulation"], dict) or set(self.state["simulation"])-allowed:
            raise noise.CheckError("Invalid simulation fields; axes must come from the saved fit")
        self.variables = value["variables"]
        if not isinstance(self.variables, list) or not 1 <= len(self.variables) <= 32:
            raise noise.CheckError("Declare 1..32 independent variables with original bounds")
        self.names, self.initial = [], {}
        for index, variable in enumerate(self.variables):
            if not isinstance(variable, dict) or set(variable) != {"path", "bounds"}:
                raise noise.CheckError("Each variable needs exactly path and bounds")
            path = variable["path"]
            if not allowed_path(path) or path in self.names:
                raise noise.CheckError("Invalid or duplicate variable path")
            bounds = variable["bounds"]
            if not isinstance(bounds, list) or len(bounds) != 2:
                raise noise.CheckError("Variable bounds must contain two numbers")
            lo, hi = [finite(v, path) for v in bounds]
            start = get_value(self.state, path)
            if not lo <= start <= hi or lo >= hi:
                raise noise.CheckError(f"Initial {path} must be inside finite lower < upper bounds")
            self.names.append(path)
            self.initial[f"v{index}"] = (start-lo)/(hi-lo)
        relations = value.get("relations", [])
        if not isinstance(relations, list) or len(relations) > 32:
            raise noise.CheckError("Invalid linear relations")
        by_target = {}
        for relation in relations:
            if not isinstance(relation, dict) or set(relation) != {"target", "constant", "terms"}:
                raise noise.CheckError("Relation needs target, constant and terms")
            target = relation["target"]
            if not allowed_path(target) or target in self.names or target in by_target:
                raise noise.CheckError("Relation target must be unique and cannot be independently varied")
            get_value(self.state, target)
            finite(relation["constant"], target)
            if not isinstance(relation["terms"], dict):
                raise noise.CheckError("Relation terms must map paths to coefficients")
            for path, coefficient in relation["terms"].items():
                if not allowed_path(path):
                    raise noise.CheckError("Invalid relation source")
                get_value(self.state, path)
                finite(coefficient, path)
            by_target[target] = relation
        self.relations, visiting, visited = [], set(), set()

        def visit(target):
            if target in visiting:
                raise noise.CheckError("Cyclic linear relations")
            if target in visited:
                return
            visiting.add(target)
            for source in by_target[target]["terms"]:
                if source in by_target:
                    visit(source)
            visiting.remove(target)
            visited.add(target)
            self.relations.append(by_target[target])

        for target in by_target:
            visit(target)
        initial_state = self.make_state(self.initial)
        for relation in self.relations:
            path = relation["target"]
            if not np.isclose(get_value(initial_state, path), get_value(self.state, path), rtol=1e-10, atol=1e-12):
                raise noise.CheckError("Initial parameters do not satisfy their declared relations")

        self.constraints = value.get("constraints", [])
        if not isinstance(self.constraints, list) or len(self.constraints) > 32:
            raise noise.CheckError("Invalid additional parameter constraints")
        seen = set()
        for constraint in self.constraints:
            if not isinstance(constraint, dict) or set(constraint) != {"path", "bounds"}:
                raise noise.CheckError("Each constraint needs exactly path and bounds")
            path, bounds = constraint["path"], constraint["bounds"]
            if not isinstance(path, str) or not allowed_path(path) or path in seen:
                raise noise.CheckError("Invalid or duplicate constrained path")
            seen.add(path)
            if not isinstance(bounds, list) or len(bounds) != 2:
                raise noise.CheckError("Constraint bounds must contain two numbers")
            lo, hi = [finite(v, path) for v in bounds]
            if lo > hi:
                raise noise.CheckError("Constraint bounds must satisfy low <= high")
            get_value(initial_state, path)
        self.check_constraints(initial_state)

    def check_constraints(self, state):
        for constraint in self.constraints:
            value = finite(get_value(state, constraint["path"]), constraint["path"])
            lo, hi = [finite(v, constraint["path"]) for v in constraint["bounds"]]
            tol = 1e-12*max(1., abs(lo), abs(hi))
            if not lo-tol <= value <= hi+tol:
                raise ValueError(f"{constraint['path']} violates its original constrained bounds")

    def make_state(self, parameters):
        state = copy.deepcopy(self.state)
        for index, variable in enumerate(self.variables):
            u = finite(parameters[f"v{index}"], variable["path"])
            if not -1e-12 <= u <= 1+1e-12:
                raise noise.CheckError("Optimizer proposed a point outside the original bounds")
            lo, hi = variable["bounds"]
            set_value(state, variable["path"], lo+np.clip(u, 0., 1.)*(hi-lo))
        for relation in self.relations:
            value = relation["constant"] + sum(coefficient*get_value(state, path)
                                               for path, coefficient in relation["terms"].items())
            set_value(state, relation["target"], value)
        return state

    def changed_values(self, state):
        paths = self.names + [relation["target"] for relation in self.relations]
        return {path: get_value(state, path) for path in paths}


def comparison(initial, best, atol, rtol):
    gain = max(0., initial-best)
    tolerance = atol+rtol*initial
    return {"improvement": gain, "tolerance": tolerance,
            "relative_improvement": gain/initial if initial > 0 else 0.,
            "stable": gain <= tolerance}


def fit_arrays(args, dimension):
    path = noise.input_path(args.fit, args.root)
    prediction = noise.read_prediction(path, args, dimension)
    if dimension == "1d":
        data = noise.read_slice(noise.input_path(args.slice, args.root), args.csv_field_unit)
        return data.values, prediction.values, prediction.fields, None, None
    data = noise.read_map(noise.input_path(args.data, args.root), args)
    field_indices = noise.match_indices(data.fields, prediction.fields, "data/fit field")
    times = prediction.times + args.fit_time_offset_ns*1e-9
    time_indices = noise.match_indices(data.times, times, "data/fit time")
    mask = times > args.noise_window_ns[1]*1e-9
    if args.fit_window_ns is not None:
        lo, hi = np.asarray(args.fit_window_ns)*1e-9
        mask &= (times >= lo) & (times <= hi)
    return data.values[np.ix_(time_indices[mask], field_indices)], prediction.values[mask], prediction.fields, prediction.times, mask


class TrialSimulator:
    def __init__(self, problem, engine, measured, fields, times, args, baseline, comparison_mask=None):
        self.problem, self.engine = problem, engine
        self.measured, self.fields, self.times, self.args = measured, fields, times, args
        self.comparison_mask = comparison_mask
        self.noise_variance = baseline["metrics"]["noise_sigma"]**2
        self.df = baseline["metrics"]["residual_df_nominal"]
        self.calls, self.model_calls, self.invalid_calls = 0, 0, 0
        self.feasible_points = set()
        self.current_invalid = False
        self.best_loss = float("inf")
        self.best_state = None
        self.cache = OrderedDict()
        self.physical_key, self.physical_value = None, None

    def validate_state(self, state):
        self.problem.check_constraints(state)
        from check_support import validation as v
        simulation = state["simulation"]
        if self.problem.dimension == "2d":
            self.engine._validate_snapshot(self.problem.model, simulation["spin_model"], simulation.get("device"))
        else:
            self.engine.make_spin_model(self.problem.model, simulation["hamiltonian"], simulation.get("settings"))
        if self.problem.dimension == "1d":
            v.populations(self.problem.model, simulation["populations"])
        else:
            v.context(self.problem.model, simulation["context"])
        if simulation.get("rise_time_us") is not None:
            v.number(simulation["rise_time_us"], "rise_time_us", positive=True)
        if finite(state["calibration"]["scale"], "calibration.scale") == 0:
            raise noise.CheckError("Zero calibration scale")
        finite(state["calibration"]["offset"], "calibration.offset")

    def evaluate(self, parameters):
        state = self.problem.make_state(parameters)
        self.validate_state(state)
        simulation = state["simulation"]
        physical_key = json.dumps(simulation, sort_keys=True, allow_nan=False)
        if physical_key != self.physical_key:
            result = self.engine.simulate(self.problem.model, self.problem.dimension,
                fields_T=self.fields, **({} if self.times is None else {"times_s": self.times}), **simulation)
            self.model_calls += 1
            raw = noise.array(result["spectrum"], "regenerated spectrum", self.measured.ndim)
            expected_shape = (len(self.fields),) if self.times is None else (len(self.times), len(self.fields))
            if raw.shape != expected_shape:
                raise RuntimeError("MaRS regenerated incompatible dimensions")
            if self.comparison_mask is not None:
                # Keep kinetics and maxabs normalization on the original full grid;
                # apply the comparison window only after the forward calculation.
                raw = raw[self.comparison_mask]
            if raw.shape != self.measured.shape:
                raise RuntimeError("MaRS regenerated incompatible dimensions")
            if not np.any(raw != 0):
                raise InvalidCandidate("MaRS regenerated a zero spectrum in the comparison window")
            self.physical_value, self.physical_key = raw.copy(), physical_key
        raw = self.physical_value*state["calibration"]["scale"] + state["calibration"]["offset"]
        aligned = raw
        if self.args.alignment != "none":
            design = raw.reshape(-1, 1) if self.args.alignment == "scale" else np.column_stack((raw.ravel(), np.ones(raw.size)))
            coefficients, _, rank, _ = np.linalg.lstsq(design, self.measured.ravel(), rcond=None)
            if rank != design.shape[1]:
                raise RuntimeError("Rank-deficient amplitude alignment")
            aligned = raw*coefficients[0] + (coefficients[1] if len(coefficients) > 1 else 0.)
        loss = float(np.sum((self.measured-aligned)**2)/self.df/self.noise_variance)
        if not np.isfinite(loss):
            raise RuntimeError("Nonfinite refinement metric")
        if loss < self.best_loss:
            self.best_loss, self.best_state = loss, copy.deepcopy(state)
        return raw, aligned, loss

    def __call__(self, *arguments):
        import torch
        parameters = arguments[-1]
        if self.calls >= self.args.budget:
            raise BudgetReached()
        if self.calls == 0 and any(abs(parameters[name]-value) > 1e-10 for name, value in self.problem.initial.items()):
            raise RuntimeError("MaRS Cobyla did not start from the agent parameters")
        self.calls += 1
        key = tuple(float(parameters[f"v{i}"]) for i in range(len(self.problem.variables)))
        if key in self.cache:
            aligned, invalid = self.cache[key]
            self.cache.move_to_end(key)
            self.current_invalid = invalid
            if not invalid:
                self.feasible_points.add(key)
            else:
                self.invalid_calls += 1
            return torch.as_tensor(aligned, dtype=torch.float64)
        try:
            state = self.problem.make_state(parameters)
            self.validate_state(state)
        except (ValueError, KeyError):
            self.invalid_calls += 1
            self.current_invalid = True
            aligned = np.zeros_like(self.measured)
        else:
            try:
                _, aligned, _ = self.evaluate(parameters)
            except InvalidCandidate:
                self.invalid_calls += 1
                self.current_invalid = True
                aligned = np.zeros_like(self.measured)
            else:
                self.current_invalid = False
                self.feasible_points.add(key)
        self.cache[key] = (aligned.copy(), self.current_invalid)
        if len(self.cache) > 4:
            self.cache.popitem(last=False)
        return torch.as_tensor(aligned, dtype=torch.float64)


def worker(args, dimension):
    # Keep all backend diagnostics out of the tool's JSON channel.
    missing = missing_backend()
    if missing:
        raise RuntimeError("MaRS fitting dependencies are missing: " + ", ".join(missing) +
                           ". Use the MaRS environment with its optimization dependencies installed.")
    import torch
    from importlib.metadata import version
    from mars.optimization import ParameterSpace, ParamSpec, SpectrumFitter, Spectrum2DFitter
    from mars.optimization.objectives import BaseObjectiveFunction
    from check_support.engine import SimulationEngine

    args.root = Path(args.root).resolve()
    parameter_path = noise.input_path(args.parameters, args.root)
    if parameter_path.stat().st_size > 100000:
        raise noise.CheckError("Parameter snapshot exceeds 100000 bytes")
    problem = Problem(json.loads(parameter_path.read_text(encoding="utf-8")), dimension)
    if args.n_parameters < len(problem.variables):
        raise noise.CheckError("n-parameters cannot be smaller than the declared independent variable count")
    if args.alignment != "none" and "calibration.scale" in problem.names:
        raise noise.CheckError("Do not vary a scale that is already fitted by --alignment")
    if args.alignment == "affine" and "calibration.offset" in problem.names:
        raise noise.CheckError("Do not vary an offset that is already fitted by --alignment")
    baseline, _ = noise.check(args, dimension)
    measured, saved, fields, times, comparison_mask = fit_arrays(args, dimension)
    simulator = TrialSimulator(problem, SimulationEngine(), measured, fields, times, args, baseline, comparison_mask)
    regenerated, _, initial_loss = simulator.evaluate(problem.initial)
    peak = max(float(np.max(np.abs(saved))), np.finfo(float).tiny)
    reproducibility = float(np.max(np.abs(regenerated-saved))/peak)
    if reproducibility > args.spectrum_rtol:
        raise noise.CheckError("Reported parameters do not reproduce the saved spectrum within --spectrum-rtol")
    space = ParameterSpace([ParamSpec(name=f"v{i}", bounds=(0., 1.), default=problem.initial[f"v{i}"])
                            for i in range(len(problem.variables))])

    class NoiseObjective(BaseObjectiveFunction):
        def __call__(self, predicted, target):
            if simulator.current_invalid:
                return torch.tensor(max(1., initial_loss)*1e6, dtype=torch.float64)
            return torch.sum((predicted-target)**2)/simulator.df/simulator.noise_variance

    kwargs = dict(y_exp=measured, param_space=space, spectra_simulator=simulator,
                  objective=NoiseObjective(), norm_mode="none", dtype=torch.float64, device=torch.device("cpu"))
    fitter = (SpectrumFitter(x_exp=fields, **kwargs) if dimension == "1d" else
              Spectrum2DFitter(x1_exp=fields, x2_exp=times[comparison_mask], **kwargs))
    try:
        fitter.fit(backend="ng", optimizer="Cobyla", budget=args.budget,
                   show_progress=False, return_best_spectrum=False, seed=None)
    except BudgetReached:
        # MaRS additionally reevaluates its recommendation after spending the budget.
        # The independently tracked feasible best already includes the initial point.
        if simulator.calls != args.budget:
            raise
    if len(simulator.feasible_points) < 2:
        raise RuntimeError("The check did not explore enough feasible points")
    control = comparison(initial_loss, simulator.best_loss, args.improvement_atol, args.improvement_rtol)
    noise_passed = args.max_k is not None and initial_loss <= args.max_k
    status = "passed" if noise_passed and control["stable"] else (
        "diagnostic" if args.max_k is None and control["stable"] else "failed")
    result = {"status": status, "dimension": dimension, "objective": "noise_k", "backend": "mars.optimization / nevergrad Cobyla",
              "mars_version": version("mars-epr"), "budget": args.budget, "objective_calls": simulator.calls,
              "time_limit_seconds": args.time_limit_seconds,
              "model_simulations": simulator.model_calls, "invalid_candidates": simulator.invalid_calls,
              "distinct_feasible_points": len(simulator.feasible_points),
              "saved_k": baseline["metrics"]["k_variance_ratio"], "initial_k": initial_loss,
              "best_k": simulator.best_loss, "max_k": args.max_k, "noise_passed": noise_passed,
              "spectrum_relative_error": reproducibility, "comparison": control,
              "initial_parameters": problem.changed_values(problem.state),
              "best_parameters": problem.changed_values(simulator.best_state),
              "inputs_sha256": baseline["sha256"],
              "warnings": ["This is a bounded local check, not a proof of a global minimum",
                           "Shared free parameters require a joint check; this checker handles one dataset"]}
    with parameter_path.open("rb") as handle:
        result["inputs_sha256"][parameter_path.relative_to(args.root).as_posix()] = hashlib.file_digest(handle, "sha256").hexdigest()
    return result


def run_child(command, seconds):
    started = time.monotonic()
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                             start_new_session=(os.name == "posix"))
    try:
        stdout, stderr = child.communicate(timeout=seconds)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            child.kill()
        child.communicate()
        return {"status": "inconclusive", "reason": "time_limit", "time_limit_seconds": seconds,
                "elapsed_seconds": time.monotonic()-started}
    if child.returncode:
        try:
            report = json.loads(stdout)
        except (ValueError, TypeError):
            raise RuntimeError("Refinement worker failed: " + stderr[-500:])
    else:
        report = json.loads(stdout)
    if not isinstance(report, dict) or "status" not in report:
        raise RuntimeError("Invalid response from refinement worker")
    report["elapsed_seconds"] = time.monotonic()-started
    return report


def parser(dimension):
    cli = noise.parser(dimension)
    cli.description = __doc__
    cli.add_argument("--parameters", required=True, help="JSON simulation snapshot, variable paths/bounds and linear relations")
    cli.add_argument("--budget", type=int, default=500, help="Maximum optimizer objective calls")
    cli.add_argument("--time-limit-seconds", type=float, default=300., help="Hard limit on the entire worker, including imports")
    cli.add_argument("--improvement-atol", type=float, required=True, help="Predeclared allowed absolute improvement in K")
    cli.add_argument("--improvement-rtol", type=float, required=True, help="Predeclared allowed improvement relative to initial K")
    cli.add_argument("--spectrum-rtol", type=float, default=1e-5, help="Reproduction tolerance relative to the saved spectrum peak")
    return cli


def validate_args(args):
    if args.budget < 2 or args.budget > 100000:
        raise noise.CheckError("Budget must be between 2 and 100000")
    for name in ("time_limit_seconds", "spectrum_rtol"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise noise.CheckError(f"{name} must be finite and positive")
    if args.n_parameters < 0:
        raise noise.CheckError("n-parameters must be nonnegative")
    if args.max_k is not None and (not np.isfinite(args.max_k) or args.max_k <= 0):
        raise noise.CheckError("max-k must be finite and positive")
    for name in ("improvement_atol", "improvement_rtol"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise noise.CheckError(f"{name} must be finite and nonnegative")
    if args.improvement_atol == args.improvement_rtol == 0:
        raise noise.CheckError("Declare a nonzero tolerance instead of requiring exact equality")
    for name in ("slice_scale", "slice_offset", "fit_time_offset_ns"):
        if hasattr(args, name) and not np.isfinite(getattr(args, name)):
            raise noise.CheckError(f"{name} must be finite")
    if getattr(args, "slice_scale", 1) == 0:
        raise noise.CheckError("slice-scale must be nonzero")


def main(dimension, argv=None, *, in_worker=False):
    argv = sys.argv[1:] if argv is None else argv
    args = parser(dimension).parse_args(argv)
    try:
        validate_args(args)
        if in_worker:
            with open(os.devnull, "w") as quiet, contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                result = worker(args, dimension)
        else:
            result = run_child([sys.executable, "-B", str(Path(__file__).resolve()), "--worker", dimension, *argv],
                               args.time_limit_seconds)
            if args.report:
                root = Path(args.root).resolve()
                path = Path(args.report)
                path = (path if path.is_absolute() else root/path).resolve()
                if not path.is_relative_to(root):
                    raise noise.CheckError("Report must stay inside the project")
                with path.open("x", encoding="utf-8") as handle:
                    handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2)+"\n")
                result["report"] = path.relative_to(root).as_posix()
        code = 0 if result["status"] == "passed" else 1 if result["status"] == "failed" else 2
        if not in_worker:
            result = {key: value for key, value in result.items()
                      if key not in ("initial_parameters", "best_parameters", "inputs_sha256")}
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return code
    except Exception as error:
        # An unavailable backend, invalid parameters or a broken check never pass.
        print(json.dumps({"status": "invalid", "error": str(error)[:700]}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker" and sys.argv[2] in ("1d", "2d"):
        raise SystemExit(main(sys.argv[2], sys.argv[3:], in_worker=True))
    raise SystemExit("Use check_refit_1d.py or check_refit_2d.py")
