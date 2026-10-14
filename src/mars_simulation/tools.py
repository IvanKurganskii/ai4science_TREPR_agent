"""Six callable tools, JSON dispatcher, and compact simulation artifacts."""

import copy
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import validation as v
from .engine import SimulationEngine
from .schemas import TOOL_SCHEMAS


class MarsSimulationTools:
    """Instantiate once per agent session; no API keys or agent SDK required."""

    def __init__(self, output_dir=None, *, engine=None):
        self.output_dir = Path(output_dir or Path.cwd() / "results" / "mars_simulations").resolve()
        self.engine = engine if engine is not None else SimulationEngine()

    @property
    def schemas(self):
        return copy.deepcopy(TOOL_SCHEMAS)

    def dispatch(self, name, arguments):
        """Execute one tool call; return a JSON-serializable result or error.

        arguments may be a decoded dict or the JSON string from function.arguments.
        Only the six named simulation tools are exposed by this dispatcher.
        """
        try:
            known = {item["function"]["name"] for item in TOOL_SCHEMAS}
            if name not in known:
                raise ValueError(f"Unknown tool {name!r}; available: {sorted(known)}")
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments must be a JSON object")
            return getattr(self, name)(**arguments)
        except Exception as exc:
            return {"ok": False, "tool": name,
                    "error": {"type": type(exc).__name__, "message": str(exc)}}

    def simulate_high_j_1d(self, **arguments):
        return self._run("high_j", "1d", arguments)

    def simulate_high_j_2d(self, **arguments):
        return self._run("high_j", "2d", arguments)

    def simulate_high_j_2d_full(self, **arguments):
        return self._run("high_j", "2d_full", arguments)

    def simulate_moderate_j_1d(self, **arguments):
        return self._run("moderate_j", "1d", arguments)

    def simulate_moderate_j_2d(self, **arguments):
        return self._run("moderate_j", "2d", arguments)

    def simulate_moderate_j_2d_full(self, **arguments):
        return self._run("moderate_j", "2d_full", arguments)

    def _run(self, model, dimension, arguments):
        schema = next(item["function"]["parameters"] for item in TOOL_SCHEMAS
                      if item["function"]["name"] == f"simulate_{model}_{dimension}")
        arguments = v.object_fields(arguments, schema["properties"], schema["required"], "tool arguments")
        output = v.object_fields(arguments.get("output", {}), ("name", "save_plot", "include_data"), name="output")
        name = output.get("name", f"{model}_{dimension}")
        if not isinstance(name, str) or not re.fullmatch(r"[\w-]{1,80}", name):
            raise ValueError("output.name must contain 1..80 letters, digits, _ or -")
        for key in ("save_plot", "include_data"):
            if key in output and not isinstance(output[key], bool):
                raise ValueError(f"output.{key} must be boolean")
        fields = v.axis(arguments["field_axis"], "field_axis", max_points=self.engine.max_points)
        times = None if dimension == "1d" else v.axis(
            arguments["time_axis"], "time_axis", max_points=self.engine.max_points)
        if output.get("include_data", False) and len(fields)*(len(times) if times else 1) > 10_000:
            raise ValueError("include_data is limited to 10000 samples; use the NPZ artifact for larger maps")
        numerical_args = {key: value for key, value in arguments.items()
                          if key not in ("field_axis", "time_axis", "output")}
        result = self.engine.simulate(model, dimension, fields_T=fields, times_s=times, **numerical_args)
        return self._save(result, name, output)

    def _save(self, result, name, output):
        # All artifact paths are generated under the host-configured output_dir.
        import numpy as np
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stem = self.output_dir / (name + "_" + uuid.uuid4().hex)
        npz_path, json_path = stem.with_suffix(".npz"), stem.with_suffix(".json")
        metadata = copy.deepcopy(result["metadata"])
        metadata["created_utc"] = datetime.now(timezone.utc).isoformat()
        metadata["spin_model"] = result["spin_model"]
        arrays = {key: result[key] for key in ("fields_T", "raw_spectrum", "spectrum")}
        if result["times_s"] is not None:
            arrays["times_s"] = result["times_s"]
        arrays["metadata_json"] = np.asarray(json.dumps(metadata, ensure_ascii=False, allow_nan=False))
        np.savez_compressed(npz_path, **arrays)
        json_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        artifacts = {"npz": str(npz_path), "metadata": str(json_path)}
        warnings = []
        if output.get("save_plot", True):
            try:
                plot_path = stem.with_suffix(".png")
                self._plot(result, plot_path)
                artifacts["plot"] = str(plot_path)
            except Exception as exc:
                warnings.append(f"Spectrum saved, but plot failed: {type(exc).__name__}: {exc}")
        signal = result["spectrum"]
        response = {"ok": True, "model": result["spin_model"]["model"],
                    "dimension": metadata["dimension"], "shape": list(signal.shape),
                    "axis_order": metadata["axis_order"], "spin_model": result["spin_model"],
                    "metadata": metadata, "artifacts": artifacts,
                    "summary": {"minimum": float(signal.min()), "maximum": float(signal.max()),
                                "maxabs": float(np.max(np.abs(signal))),
                                "field_range_T": [float(result["fields_T"][0]), float(result["fields_T"][-1])]},
                    "warnings": warnings}
        if result["times_s"] is not None:
            response["summary"]["time_range_s"] = [float(result["times_s"][0]), float(result["times_s"][-1])]
        if output.get("include_data", False):
            response["data"] = {key: result[key].tolist() for key in (
                ("fields_T", "spectrum") if result["times_s"] is None else ("fields_T", "times_s", "spectrum"))}
        return response

    @staticmethod
    def _plot(result, path):
        # Agg canvas avoids GUI windows and does not change a notebook's backend.
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        import numpy as np
        fig = Figure(figsize=(7, 4.5), constrained_layout=True)
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        fields, times, signal = result["fields_T"]*1e3, result["times_s"], result["spectrum"]
        if times is None or len(times) == 1:
            ax.plot(fields, signal if times is None else signal[0])
            ax.set_ylabel("Signal (a.u.)")
            ax.axhline(0, color="grey", linewidth=0.5)
        else:
            peak = float(np.max(np.abs(signal))) or 1.0
            image = ax.pcolormesh(fields, times*1e6, signal, shading="auto", cmap="RdBu_r", vmin=-peak, vmax=peak)
            fig.colorbar(image, ax=ax, label="Signal (a.u.)")
            ax.set_ylabel("Time (µs)")
        ax.set_xlabel("Magnetic field (mT)")
        ax.set_title(result["spin_model"]["model"] + " / " + result["metadata"]["dimension"])
        fig.savefig(path, dpi=150)
