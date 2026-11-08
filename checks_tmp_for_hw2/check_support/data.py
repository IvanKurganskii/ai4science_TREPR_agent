"""Canonical time/field datasets and portable CSV/NPZ/NumPy-PKL input."""

import copy
import hashlib
import importlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

from ._runtime import numpy


TIME_SCALES = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9, "ps": 1e-12}
FIELD_SCALES = {"T": 1.0, "mT": 1e-3, "G": 1e-4}


def fields(value, allowed, required=(), name="options"):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    missing, unknown = set(required)-set(value), set(value)-set(allowed)
    if missing or unknown:
        raise ValueError(f"{name}: missing {sorted(missing)}; unknown {sorted(unknown)}")
    return dict(value)


def plain(value):
    """Convert metadata to JSON without retaining pickle-specific objects."""
    np = numpy()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("Metadata contains nonfinite values")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"Unsupported metadata type {type(value).__name__}")


@dataclass
class EPRDataset:
    """All working arrays use SI units and shape (time, field)."""
    fields_T: object
    times_s: object
    signal: object
    name: str
    metadata: dict

    @classmethod
    def from_arrays(cls, fields_T, times_s, signal, *, name="spectrum", layout="time_field", metadata=None):
        np = numpy()
        if layout not in ("time_field", "field_time", "auto"):
            raise ValueError("layout must be time_field, field_time or auto")
        raw = np.asarray(signal)
        if np.iscomplexobj(raw):
            raise ValueError("Complex signal is unsupported; select a phased real channel first")
        b, t, y = np.asarray(fields_T, dtype=float), np.asarray(times_s, dtype=float), np.asarray(raw, dtype=float)
        if b.ndim != 1 or t.ndim != 1 or min(b.size, t.size) < 2:
            raise ValueError("Field and time axes must be 1D with at least two points")
        if not (np.isfinite(b).all() and np.isfinite(t).all() and np.isfinite(y).all()):
            raise ValueError("Axes and signal must be finite; no NaN interpolation is performed")
        tf, ft = (t.size, b.size), (b.size, t.size)
        if layout == "auto":
            if y.shape == tf == ft:
                raise ValueError("Square matrix is ambiguous; specify layout")
            if y.shape == tf:
                layout = "time_field"
            elif y.shape == ft:
                layout = "field_time"
            else:
                raise ValueError("Signal shape matches neither axis layout")
        if y.shape != (tf if layout == "time_field" else ft):
            raise ValueError("Signal shape does not match the declared layout")
        if layout == "field_time":
            y = y.T
        order_b, order_t = np.argsort(b), np.argsort(t)
        b, t = b[order_b], t[order_t]
        if np.any(np.diff(b) <= 0) or np.any(np.diff(t) <= 0):
            raise ValueError("Axes must have unique coordinates")
        if np.any(b < 0):
            raise ValueError("Field values must be nonnegative")
        info = copy.deepcopy(metadata or {})
        info.setdefault("history", [])
        if not isinstance(info["history"], list):
            raise ValueError("metadata.history must be a list")
        info["axis_order"] = ["time", "field"]
        info["units"] = {"field": "T", "time": "s", "signal": "a.u."}
        return cls(b.copy(), t.copy(), np.ascontiguousarray(y[np.ix_(order_t, order_b)]), str(name), info)

    def copy(self, signal=None):
        return EPRDataset(self.fields_T.copy(), self.times_s.copy(),
                          self.signal.copy() if signal is None else signal.copy(),
                          self.name, copy.deepcopy(self.metadata))


class NumpyUnpickler(pickle.Unpickler):
    """Accept the notebook's plain dictionaries/NumPy arrays, not arbitrary classes."""
    def find_class(self, module, name):
        np = numpy()
        allowed = {("numpy", "ndarray"): np.ndarray, ("numpy", "dtype"): np.dtype}
        core = "numpy._core" if hasattr(np, "_core") else "numpy.core"
        multiarray = importlib.import_module(core+".multiarray")
        numeric = importlib.import_module(core+".numeric")
        for prefix in ("numpy.core", "numpy._core"):
            allowed.update({(prefix+".multiarray", "_reconstruct"): multiarray._reconstruct,
                            (prefix+".multiarray", "scalar"): multiarray.scalar,
                            (prefix+".numeric", "_frombuffer"): numeric._frombuffer})
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"Unsupported pickle class {module}.{name}; use CSV/NPZ or a plain NumPy dataset")
        return allowed[(module, name)]


class DatasetLoader:
    """Resolve relative inputs against one explicit host working directory."""
    def __init__(self, working_dir=None, max_points=10_000_000):
        self.working_dir = Path(working_dir or Path.cwd()).resolve()
        self.max_points = max_points

    def _payload(self, source):
        source = fields(source, ("path", "dataset", "time_unit", "field_unit", "layout", "signal_key"), ("path",), "source")
        if not isinstance(source["path"], str) or not source["path"].strip():
            raise ValueError("source.path must be a file path")
        path = Path(source["path"])
        path = (path if path.is_absolute() else self.working_dir/path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            np = numpy()
            matrix = np.genfromtxt(path, delimiter=",", dtype=float)
            if matrix.ndim != 2 or min(matrix.shape) < 3:
                raise ValueError("CSV must have field coordinates in the first row and time coordinates in the first column")
            payload = {"datasets": {path.stem: {"field_G": matrix[0, 1:], "time": matrix[1:, 0], "signal": matrix[1:, 1:]}}}
        elif suffix == ".npz":
            np = numpy()
            with np.load(path, allow_pickle=False) as archive:
                item = {key: archive[key].copy() for key in archive.files}
            info = json.loads(str(item.pop("metadata_json"))) if "metadata_json" in item else {}
            name = info.get("dataset", path.stem)
            payload = {"datasets": {name: item}, "metadata": info}
        elif suffix in (".pkl", ".pickle"):
            with path.open("rb") as handle:
                payload = NumpyUnpickler(handle).load()
            if isinstance(payload, dict) and {"field_G", "time", "signal"} <= payload.keys():
                payload = {"datasets": {path.stem: payload}}
        else:
            raise ValueError("Input format must be CSV, NPZ or plain NumPy PKL")
        if not isinstance(payload, dict) or not isinstance(payload.get("datasets"), dict) or not payload["datasets"]:
            raise ValueError("Input contains no datasets mapping")
        return source, path, payload

    def load(self, source):
        np = numpy()
        options, path, payload = self._payload(source)
        datasets = payload["datasets"]
        selected = options.get("dataset")
        if selected is None:
            if len(datasets) != 1:
                raise ValueError("Multiple datasets: inspect the file and supply source.dataset")
            selected = next(iter(datasets))
        if selected not in datasets:
            raise ValueError(f"Dataset {selected!r} not found; available: {sorted(datasets)}")
        item = datasets[selected]
        if not isinstance(item, dict):
            raise ValueError("Dataset entry must be a dictionary")
        field_key = next((k for k in ("fields_T", "field_G", "fields") if k in item), None)
        time_key = next((k for k in ("times_s", "time", "times") if k in item), None)
        signal_key = options.get("signal_key", "signal")
        if field_key is None or time_key is None or signal_key not in item:
            raise ValueError("Dataset must contain field/time axes and signal; use source.signal_key for another NPZ signal")
        native_b = "T" if field_key == "fields_T" else "G" if field_key == "field_G" else None
        native_t = "s" if time_key == "times_s" else item.get("time_unit")
        field_unit, time_unit = options.get("field_unit", native_b), options.get("time_unit", native_t)
        if field_unit not in FIELD_SCALES or time_unit not in TIME_SCALES:
            raise ValueError("Specify source.time_unit (s/ms/us/ns/ps) and field_unit (T/mT/G) for axes without explicit units")
        if native_b is not None and field_unit != native_b or native_t is not None and time_unit != native_t:
            raise ValueError("Unit override contradicts the explicit axis key; fields_T/times_s are SI and field_G is gauss")
        y = np.asarray(item[signal_key])
        if y.size > self.max_points:
            raise ValueError(f"Dataset exceeds max_points={self.max_points}")
        metadata = copy.deepcopy(payload.get("metadata", {}))
        if not isinstance(metadata, dict):
            raise ValueError("Input metadata must be an object")
        # Carry acquisition records forward for notebook-compatible PKL export.
        if "parameters" in payload:
            metadata["parameters"] = plain(payload["parameters"])
        for key in ("baseline_correction", "denoising"):
            if key in item:
                metadata[key] = plain(item[key])
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if "source" in metadata:
            lineage = metadata.setdefault("source_history", [])
            if not isinstance(lineage, list):
                raise ValueError("metadata.source_history must be a list")
            lineage.append(copy.deepcopy(metadata["source"]))
        metadata["source"] = {"path": str(path), "dataset": selected, "sha256": digest,
                              "field_unit": field_unit, "time_unit": time_unit,
                              "signal_key": signal_key}
        return EPRDataset.from_arrays(np.asarray(item[field_key], float)*FIELD_SCALES[field_unit],
            np.asarray(item[time_key], float)*TIME_SCALES[time_unit], y, name=selected,
            layout=options.get("layout", "time_field"), metadata=metadata)

    def inspect(self, source):
        options, path, payload = self._payload(source)
        if options.get("dataset") is not None:
            names = [options["dataset"]]
        else:
            names = list(payload["datasets"])
        records = []
        for name in names:
            if name not in payload["datasets"]:
                raise ValueError(f"Dataset {name!r} not found")
            item = payload["datasets"][name]
            if not isinstance(item, dict):
                raise ValueError("Dataset entry must be an object")
            np = numpy()
            signal_key = options.get("signal_key", "signal")
            y = np.asarray(item[signal_key]) if signal_key in item else None
            record = {"name": name, "keys": sorted(item), "shape": list(y.shape) if y is not None else None}
            for axis_key in ("fields_T", "field_G", "fields", "times_s", "time", "times"):
                if axis_key in item:
                    a = np.asarray(item[axis_key], dtype=float)
                    finite = a[np.isfinite(a)]
                    record[axis_key] = {"points": int(a.size), "range": [float(finite.min()), float(finite.max())] if finite.size else None}
            if y is not None:
                record["finite"] = bool(np.isfinite(y).all())
            records.append(record)
        return {"path": str(path), "datasets": records, "parameter_names": sorted(payload.get("parameters", {}))}
