"""Six JSON-callable tools with compact reports and reproducible artifacts."""

import copy
import json
import pickle
import re
import sys
import uuid
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from ._runtime import numpy
from .data import DatasetLoader, TIME_SCALES, fields, plain
from .engine import CleaningEngine, integer
from .schemas import TOOL_SCHEMAS, schemas


class MarsCleaningTools:
    def __init__(self, output_dir=None, *, working_dir=None, engine=None, max_points=10_000_000):
        self.working_dir = Path(working_dir or Path.cwd()).resolve()
        destination = Path(output_dir) if output_dir is not None else self.working_dir/"results"/"mars_simulations"
        self.output_dir = (destination if destination.is_absolute() else self.working_dir/destination).resolve()
        self.loader = DatasetLoader(self.working_dir, integer(max_points, "max_points", 4))
        self.engine = engine if engine is not None else CleaningEngine()

    @property
    def schemas(self):
        return schemas()

    def dispatch(self, name, arguments):
        try:
            if name not in {s["function"]["name"] for s in TOOL_SCHEMAS}:
                raise ValueError(f"Unknown cleaning tool {name!r}")
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise ValueError("Arguments must be an object")
            schema = next(s["function"]["parameters"] for s in TOOL_SCHEMAS if s["function"]["name"] == name)
            fields(arguments, schema["properties"], schema["required"], "tool arguments")
            return getattr(self, name)(**arguments)
        except Exception as exc:
            return {"ok": False, "tool": name, "error": {"type": type(exc).__name__, "message": str(exc)}}

    def inspect_epr_2d(self, source, singular_values=0):
        integer(singular_values, "singular_values")
        if singular_values > 100:
            raise ValueError("singular_values must be <=100")
        result = {"ok": True, **self.loader.inspect(source)}
        if singular_values:
            dataset = self.loader.load(source)
            result["analysis"] = self.engine.singular_values(dataset, singular_values)
        return result

    def import_epr_2d(self, source, output=None):
        options = self._output(output)
        dataset = self.loader.load(source)
        return self._save(dataset, dataset, {}, options)

    def correct_epr_2d_baseline(self, source, baseline, output=None):
        options = self._output(output)
        original = self.loader.load(source)
        cleaned, diagnostic = self.engine.correct_baseline(original, baseline)
        return self._save(cleaned, original, diagnostic, options)

    def denoise_epr_2d(self, source, denoising, output=None):
        options = self._output(output)
        original = self.loader.load(source)
        cleaned = self.engine.denoise(original, denoising)
        return self._save(cleaned, original, {}, options)

    def clean_epr_2d(self, source, baseline=None, denoising=None, output=None):
        options = self._output(output)
        original = self.loader.load(source)
        cleaned, diagnostic = self.engine.clean(original, baseline=baseline, denoising=denoising)
        return self._save(cleaned, original, diagnostic, options)

    def export_epr_2d(self, source, output=None):
        options = self._output(output, default_formats=["npz", "pkl"])
        dataset = self.loader.load(source)
        return self._save(dataset, dataset, {}, options)

    def _output(self, output, default_formats=None):
        value = fields({} if output is None else output, ("name", "formats", "time_unit", "save_plot", "include_data"), name="output")
        result = {"name": "epr_2d", "formats": default_formats or ["npz"], "time_unit": "ns",
                  "save_plot": True, "include_data": False, **value}
        if not isinstance(result["name"], str) or not re.fullmatch(r"[\w-]{1,80}", result["name"]):
            raise ValueError("output.name must contain 1..80 letters, digits, _ or -")
        if (not isinstance(result["formats"], list) or not result["formats"]
                or not all(isinstance(f, str) and f in ("npz", "csv", "pkl") for f in result["formats"])
                or len(set(result["formats"])) != len(result["formats"])):
            raise ValueError("output.formats must list unique npz/csv/pkl formats")
        if result["time_unit"] not in TIME_SCALES:
            raise ValueError("Unknown output.time_unit")
        for key in ("save_plot", "include_data"):
            if not isinstance(result[key], bool):
                raise ValueError(f"output.{key} must be boolean")
        return result

    def _save(self, dataset, original, diagnostics, options):
        np = numpy()
        if options["include_data"] and dataset.signal.size > 10_000:
            raise ValueError("include_data is limited to 10000 points; use NPZ for larger maps")
        metadata = copy.deepcopy(dataset.metadata)
        metadata.update({"dataset": dataset.name, "shape": list(dataset.signal.shape),
                         "created_utc": datetime.now(timezone.utc).isoformat(),
                         "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                                      "scipy": version("scipy"), "mars_epr": version("mars-epr"),
                                      "PyWavelets": version("PyWavelets")},
                         "threads": self.engine.threads})
        metadata = plain(metadata)
        serialized = json.dumps(metadata, ensure_ascii=False, allow_nan=False)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stem = self.output_dir/(options["name"]+"_"+uuid.uuid4().hex)
        artifacts = {}
        metadata_path = stem.with_suffix(".json")
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        artifacts["metadata"] = str(metadata_path)
        if "npz" in options["formats"]:
            path = stem.with_suffix(".npz")
            np.savez_compressed(path, fields_T=dataset.fields_T, times_s=dataset.times_s,
                signal=dataset.signal, input_signal=original.signal, removed_signal=original.signal-dataset.signal,
                metadata_json=np.asarray(serialized), **diagnostics)
            artifacts["npz"] = str(path)
        if "csv" in options["formats"]:
            matrix = np.empty((len(dataset.times_s)+1, len(dataset.fields_T)+1))
            matrix[0, 0] = 0
            matrix[0, 1:] = dataset.fields_T*1e4
            matrix[1:, 0] = dataset.times_s/TIME_SCALES[options["time_unit"]]
            matrix[1:, 1:] = dataset.signal
            path = stem.with_suffix(".csv")
            np.savetxt(path, matrix, delimiter=",", fmt="%.17g")
            artifacts["csv"] = str(path)
        if "pkl" in options["formats"]:
            item = {"field_G": dataset.fields_T*1e4,
                    "time": dataset.times_s/TIME_SCALES[options["time_unit"]],
                    "time_unit": options["time_unit"], "signal": dataset.signal.copy()}
            for key in ("baseline_correction", "denoising"):
                if key in metadata:
                    item[key] = copy.deepcopy(metadata[key])
            storage_metadata = {**metadata, "units": {"field": "G", "time": options["time_unit"], "signal": "a.u."}}
            payload = {"datasets": {dataset.name: item}, "parameters": metadata.get("parameters", {}), "metadata": storage_metadata}
            path = stem.with_suffix(".pkl")
            with path.open("wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            artifacts["pkl"] = str(path)
        warnings = [w for step in metadata["history"] for w in step.get("warnings", [])]
        if options["save_plot"]:
            try:
                path = stem.with_suffix(".png")
                self._plot(original, dataset, path)
                artifacts["plot"] = str(path)
            except Exception as exc:
                warnings.append(f"Data saved but plot failed: {type(exc).__name__}: {exc}")
        removed = original.signal-dataset.signal
        result = {"ok": True, "dataset": dataset.name, "shape": list(dataset.signal.shape),
                  "axis_order": ["time", "field"], "artifacts": artifacts, "metadata": metadata,
                  "summary": {"field_range_T": [float(dataset.fields_T[0]), float(dataset.fields_T[-1])],
                              "time_range_s": [float(dataset.times_s[0]), float(dataset.times_s[-1])],
                              "input_rms": float(np.sqrt(np.mean(original.signal**2))),
                              "output_rms": float(np.sqrt(np.mean(dataset.signal**2))),
                              "removed_rms": float(np.sqrt(np.mean(removed**2))),
                              "output_maxabs": float(np.max(np.abs(dataset.signal)))}, "warnings": warnings}
        if options["include_data"]:
            result["data"] = {"fields_T": dataset.fields_T.tolist(), "times_s": dataset.times_s.tolist(), "signal": dataset.signal.tolist()}
        return result

    @staticmethod
    def _plot(original, cleaned, path):
        np = numpy()
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        fig = Figure(figsize=(12, 7), constrained_layout=True)
        FigureCanvasAgg(fig)
        axes = fig.subplots(2, 3)
        y0, y1 = original.signal, cleaned.signal
        peak = max(float(np.max(np.abs(y0))), float(np.max(np.abs(y1)))) or 1.0
        ts, bs = max(1, len(cleaned.times_s)//700), max(1, len(cleaned.fields_T)//700)
        for ax, signal, label in zip(axes[0], (y0, y1, y0-y1), ("Input", "Output", "Removed")):
            image = ax.pcolormesh(cleaned.fields_T[::bs]*1e3, cleaned.times_s[::ts]*1e6,
                signal[::ts, ::bs], shading="auto", cmap="RdBu_r", vmin=-peak, vmax=peak)
            ax.set(xlabel="Field (mT)", ylabel="Time (µs)", title=label)
            fig.colorbar(image, ax=ax, label="Signal (a.u.)")
        ti = int(np.argmax(np.max(np.abs(y1), axis=1)))
        bi = int(np.argmax(np.max(np.abs(y1), axis=0)))
        for signal, label in ((y0, "Input"), (y1, "Output")):
            axes[1, 0].plot(cleaned.fields_T*1e3, signal[ti], label=label)
            axes[1, 1].plot(cleaned.times_s*1e6, signal[:, bi], label=label)
        axes[1, 0].set(xlabel="Field (mT)", ylabel="Signal", title=f"Slice at {cleaned.times_s[ti]*1e6:.3g} µs")
        axes[1, 1].set(xlabel="Time (µs)", ylabel="Signal", title=f"Trace at {cleaned.fields_T[bi]*1e3:.3g} mT")
        axes[1, 0].legend()
        axes[1, 1].legend()
        axes[1, 2].hist((y0-y1).ravel(), bins=60)
        axes[1, 2].set(xlabel="Removed signal", ylabel="Count", title="Removed distribution")
        fig.suptitle(cleaned.name)
        fig.savefig(path, dpi=130)
