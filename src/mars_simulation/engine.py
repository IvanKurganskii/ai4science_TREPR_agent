"""Numerical implementations, independent of notebook globals and agent SDKs."""

import copy
import importlib
import json
import os
import sys
import threading
from collections import OrderedDict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from . import validation as v


def _backend():
    # Windows conda DLL discovery must precede numpy/torch imports.
    if sys.platform == "win32":
        dll = Path(sys.prefix) / "Library" / "bin"
        if dll.is_dir() and str(dll) not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = str(dll) + os.pathsep + os.environ.get("PATH", "")
    try:
        np = importlib.import_module("numpy")
        torch = importlib.import_module("torch")
        mars = importlib.import_module("mars")
        return np, torch, mars.spin_model, mars.population, mars.spectra_manager
    except ImportError as exc:
        raise RuntimeError("Install mars-epr in the Python environment running these tools") from exc


def _weights(np, model, values, dimension, mode):
    if mode not in ("notebook", "raw", "minmax", "max"):
        raise ValueError("population_mode must be notebook, raw, minmax or max")
    if mode == "notebook":
        mode = "minmax" if model == "high_j" and dimension == "1d" else (
            "raw" if model == "high_j" else "max")
    weights = np.array([values[k] for k in v.population_names(model)], dtype=float)
    if mode == "minmax":
        span = float(weights.max() - weights.min())
        weights = (weights-weights.min())/span if span > np.finfo(float).eps else np.full(len(weights), 1/len(weights))
    elif mode == "max":
        maximum = float(weights.max())
        weights = weights/maximum if maximum > np.finfo(float).eps else np.full(len(weights), 1/len(weights))
    return weights.tolist(), mode


def _rate(lifetime):
    return 0.0 if lifetime is None else 1e6 / lifetime


class SimulationEngine:
    """Small bounded cache; calls on one engine are serialized.

    2D takes a complete JSON spin_model snapshot returned by 1D/full; it has no
    dependency on earlier calls, widgets, saved pickles or process-local IDs.
    """

    def __init__(self, cache_size=4, max_points=1_000_000, threads=2):
        self.cache_size = v.integer(cache_size, "cache_size", 1)
        self.max_points = v.integer(max_points, "max_points", 2)
        self.threads = v.integer(threads, "threads", 1)
        self._cache = OrderedDict()
        self._lock = threading.RLock()

    def clear_cache(self):
        with self._lock:
            self._cache.clear()

    def make_spin_model(self, model, hamiltonian, settings=None):
        if model not in ("high_j", "moderate_j"):
            raise ValueError("model must be high_j or moderate_j")
        return {"model": model, "hamiltonian": v.hamiltonian(model, hamiltonian),
                "settings": v.settings(settings)}

    def _validate_snapshot(self, model, snapshot, device=None):
        snapshot = v.object_fields(snapshot, ("model", "hamiltonian", "settings"),
                                   ("model", "hamiltonian", "settings"), "spin_model")
        if snapshot["model"] != model:
            raise ValueError(f"This tool requires a {model} spin_model")
        snapshot = self.make_spin_model(model, snapshot["hamiltonian"], snapshot["settings"])
        if device is not None:
            snapshot["settings"]["device"] = device
            snapshot["settings"] = v.settings(snapshot["settings"])
        return snapshot

    def _manager(self, backend, snapshot, fields, dimension):
        np, torch, spin_model, _, spectra_manager = backend
        config, h = snapshot["settings"], snapshot["hamiltonian"]
        requested_device = config["device"]
        device = "cuda" if requested_device == "auto" and torch.cuda.is_available() else (
            "cpu" if requested_device == "auto" else requested_device)
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable in this environment; use device='cpu' or 'auto'")
        dtype = getattr(torch, config["dtype"])
        effective = copy.deepcopy(snapshot)
        effective["settings"]["device"] = device
        key = (json.dumps(effective, sort_keys=True), dimension, tuple(fields))
        if key in self._cache:
            self._cache.move_to_end(key)
            return (*self._cache[key], effective)
        # Evict before allocating another mesh/eigensystem.
        if len(self._cache) >= self.cache_size:
            self._cache.popitem(last=False)
        args = {"device": device, "dtype": dtype, "frame": config["frame"]}
        de = spin_model.DEInteraction((h["D"], h["E"]), **args)
        if snapshot["model"] == "high_j":
            system = spin_model.SpinSystem(
                electrons=[1.0], g_tensors=[spin_model.Interaction(h["g"], **args)],
                electron_electron=[(0, 0, de)], device=device, dtype=dtype)
        else:
            gs = [h["g0"] + h["delta_g"]/2, h["g0"] - h["delta_g"]/2]
            exchange = spin_model.Interaction(h["J"], **args)
            system = spin_model.SpinSystem(
                electrons=[0.5, 0.5], g_tensors=[spin_model.Interaction(g, **args) for g in gs],
                electron_electron=[(0, 1, de+exchange)], device=device, dtype=dtype)
        sample = spin_model.MultiOrientedSample(
            base_spin_system=system, gauss=h["gauss"], lorentz=h["lorentz"],
            mesh=(config["mesh"], config["mesh"]), device=device, dtype=dtype)
        cls = spectra_manager.StationarySpectra if dimension == "1d" else spectra_manager.CoupledTimeSpectra
        manager = cls(freq=config["freq"], sample=sample, harmonic=config["harmonic"],
                      temperature=config["temperature"], context=None,
                      output_eigenvector=True, recompute_spin_parameters=False,
                      device=device, dtype=dtype)
        item = (sample, manager, device, dtype)
        self._cache[key] = item
        return (*item, effective)

    def _context(self, backend, model, sample, weights, kinetic, device, dtype):
        _, torch, _, population, _ = backend
        args = {"sample": sample, "device": device, "dtype": dtype}
        losses = None if kinetic is None else [_rate(kinetic["tau_out_"+k]) for k in (
            ("x", "y", "z") if model == "high_j" else ("s", "x", "y", "z"))]
        xyz = population.Context(basis="xyz", init_populations=weights, decay_rates=losses, **args)
        if kinetic is None:
            return xyz
        if model == "high_j":
            matrix = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], device=device, dtype=dtype)
            matrix *= _rate(kinetic["tau_intra"])
            basis = "multiplet"
        else:
            outer = torch.tensor([[0, 1, 1, 0], [1, 0, 0, 1],
                                  [1, 0, 0, 1], [0, 1, 1, 0]], device=device, dtype=dtype)
            inner = torch.tensor([[0, 0, 0, 0], [0, 0, 1, 0],
                                  [0, 1, 0, 0], [0, 0, 0, 0]], device=device, dtype=dtype)
            matrix = outer*_rate(kinetic["tau_outer"]) + inner*_rate(kinetic["tau_inner"])
            basis = "eigen"
        return xyz + population.Context(basis=basis, thermal_rates=matrix, **args)

    def simulate(self, model, dimension, *, hamiltonian=None, settings=None, spin_model=None,
                 fields_T, times_s=None, populations=None, context=None,
                 population_mode="notebook", rise_time_us=None, normalization="none", device=None):
        """Return CPU numpy arrays and the effective, portable spin_model snapshot.

        fields_T/times_s are explicit SI-unit axes. Lifetimes in context and
        rise_time_us use microseconds; null disables an individual rate.
        """
        if dimension not in ("1d", "2d", "2d_full"):
            raise ValueError("dimension must be 1d, 2d or 2d_full")
        if dimension == "2d":
            if spin_model is None or hamiltonian is not None or settings is not None:
                raise ValueError("2d requires spin_model and cannot override its Hamiltonian/settings")
            snapshot = self._validate_snapshot(model, spin_model, device)
        else:
            if spin_model is not None or device is not None:
                raise ValueError("Use hamiltonian/settings for 1d and 2d_full")
            snapshot = self.make_spin_model(model, hamiltonian, settings)
        field_values = fields_T.tolist() if hasattr(fields_T, "tolist") else fields_T
        fields = v.axis({"values": field_values}, "field_axis", max_points=self.max_points)
        times = None
        kinetic = None
        if dimension == "1d":
            if times_s is not None or context is not None or rise_time_us is not None:
                raise ValueError("1d accepts populations only; kinetics/rise need a 2d tool")
            values = v.populations(model, populations)
        else:
            if populations is not None:
                raise ValueError("2d populations belong inside context")
            time_values = times_s.tolist() if hasattr(times_s, "tolist") else times_s
            times = v.axis({"values": time_values}, "time_axis", max_points=self.max_points)
            kinetic = v.context(model, context)
            values = {k: kinetic[k] for k in v.population_names(model)}
            if rise_time_us is not None:
                rise_time_us = v.number(rise_time_us, "rise_time_us", positive=True)
        if len(fields) * (len(times) if times else 1) > self.max_points:
            raise ValueError(f"Simulation exceeds max_points={self.max_points}")
        if normalization not in ("none", "maxabs"):
            raise ValueError("normalization must be none or maxabs")
        if population_mode not in ("notebook", "raw", "minmax", "max"):
            raise ValueError("Invalid population_mode")
        with self._lock:
            backend = _backend()
            np, torch = backend[:2]
            torch.set_num_threads(self.threads)
            weights, weight_mode = _weights(np, model, values, dimension, population_mode)
            kind = "1d" if dimension == "1d" else "2d"
            sample, manager, actual_device, dtype, effective = self._manager(backend, snapshot, fields, kind)
            current = self._context(backend, model, sample, weights, kinetic, actual_device, dtype)
            manager.update_context(current)
            field_tensor = torch.as_tensor(fields, device=actual_device, dtype=dtype)
            with torch.no_grad():
                if times is None:
                    raw = manager(sample, field_tensor).detach().cpu().numpy()
                    expected = (len(fields),)
                else:
                    prepend_zero = rise_time_us is not None and times[0] != 0
                    evaluated = ([0.0] + times) if prepend_zero else times
                    time_tensor = torch.as_tensor(evaluated, device=actual_device, dtype=dtype)
                    raw = manager(sample, field_tensor, time_tensor).detach().cpu().numpy()
                    expected = (len(evaluated), len(fields))
                    if raw.shape != expected and raw.shape == expected[::-1]:
                        raw = raw.T
                    if raw.shape != expected:
                        raise RuntimeError(f"Unexpected MaRS map shape {raw.shape}, expected {expected}")
                    if rise_time_us is not None:
                        initial = raw[0].copy()
                        raw = raw[1:] if prepend_zero else raw
                        raw = raw - np.exp(-np.asarray(times)[:, None]/(rise_time_us*1e-6))*initial[None, :]
                    expected = (len(times), len(fields))
            raw = np.asarray(raw, dtype=float)
            if raw.shape != expected or not np.isfinite(raw).all():
                raise RuntimeError("MaRS returned an unexpected shape or nonfinite spectrum")
            peak = float(np.max(np.abs(raw)))
            divisor = peak if normalization == "maxabs" and peak > 0 else 1.0
            try:
                mars_version = version("mars-epr")
            except PackageNotFoundError:
                mars_version = "unavailable"
            return {
                "fields_T": np.asarray(fields), "times_s": None if times is None else np.asarray(times),
                "raw_spectrum": raw.copy(), "spectrum": raw.copy()/divisor,
                "spin_model": effective,
                "metadata": {"dimension": dimension, "shape": list(raw.shape),
                    "axis_order": ["field"] if times is None else ["time", "field"],
                    "mars_epr_version": mars_version, "numpy_version": np.__version__,
                    "torch_version": str(torch.__version__), "python_version": sys.version.split()[0],
                    "population_mode": weight_mode, "input_populations": values,
                    "effective_populations": weights, "context": kinetic,
                    "rise_time_us": rise_time_us, "normalization": normalization,
                    "normalization_divisor": divisor, "threads": self.threads,
                    "relaxation_basis": None if times is None else (
                        "multiplet" if model == "high_j" else "eigen"),
                    "population_basis": "molecular_xyz" if model == "high_j" else "singlet_molecular_xyz",
                    "backend": "MaRS StationarySpectra" if times is None else "MaRS CoupledTimeSpectra"},
            }
