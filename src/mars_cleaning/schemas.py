"""Framework-independent function schemas for the six cleaning tools."""

import copy


def obj(properties, required=()):
    return {"type": "object", "properties": properties, "required": list(required), "additionalProperties": False}


def num(description, **kwargs):
    return {"type": "number", "description": description, **kwargs}


SOURCE = obj({
    "path": {"type": "string", "description": "CSV, NPZ or plain NumPy PKL. Relative paths resolve from the host working directory."},
    "dataset": {"type": "string", "description": "Required for a multi-dataset PKL; inspect first."},
    "time_unit": {"type": "string", "enum": ["s", "ms", "us", "ns", "ps"],
                  "description": "Explicit unit for raw time axes without declared units. Notebook exports use ns."},
    "field_unit": {"type": "string", "enum": ["T", "mT", "G"],
                   "description": "fields_T implies T; field_G implies G. Other fields require a unit."},
    "layout": {"type": "string", "enum": ["time_field", "field_time", "auto"], "default": "time_field",
               "description": "Notebook/canonical layout is time first. Auto rejects ambiguous square matrices."},
    "signal_key": {"type": "string", "default": "signal", "description": "NPZ intensity key; simulation artifacts use spectrum or raw_spectrum."},
}, ("path",))

BASELINE = obj({
    "field_bounds_T": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number"},
                       "description": "Inclusive useful-signal interval in T; polynomial fits use points OUTSIDE it."},
    "time_bounds_s": {"type": "array", "minItems": 2, "maxItems": 2,
                      "items": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                      "description": "Inclusive useful-signal interval in s; background averages points OUTSIDE it. null endpoint means unbounded."},
    "poly_order": {"type": "integer", "minimum": 0, "default": 1},
    "subtract_average_background": {"type": "boolean", "default": True},
    "correct_baseline": {"type": "boolean", "default": True},
    "baseline_mode": {"type": "string", "enum": ["scaled_reference", "independent"], "default": "scaled_reference"},
    "fit_intensity_shift": {"type": "boolean", "default": True},
})

DENOISING = obj({
    "method": {"type": "string", "enum": ["svd", "gaussian", "wavelet", "median", "combined"]},
    "svd_rank": {"type": "integer", "minimum": 1, "default": 10, "description": "SVD/combined only; choose using singular values and residuals."},
    "sigma_t": num("Gaussian/combined time width in SAMPLES; notebook default 2.", minimum=0),
    "sigma_b": num("Gaussian/combined field width in SAMPLES; notebook default 1.", minimum=0),
    "wavelet": {"type": "string", "default": "db4", "description": "Wavelet only, applied along time at each field."},
    "wavelet_level": {"type": "integer", "minimum": 1, "default": 4},
    "wavelet_threshold": {"anyOf": [{"type": "number", "minimum": 0}, {"type": "null"}],
                          "description": "Wavelet only; null/omission estimates the MAD-based universal threshold."},
    "median_kernel": {"type": "integer", "minimum": 1, "default": 3, "description": "Median only; must be odd."},
}, ("method",))

OUTPUT = obj({
    "name": {"type": "string", "description": "Optional filename prefix (letters, digits, _ or -); files have unique suffixes."},
    "save_plot": {"type": "boolean", "default": True},
    "include_data": {"type": "boolean", "default": False, "description": "Inline SI axes/signal for <=10000 points; prefer NPZ for larger maps."},
    "formats": {"type": "array", "minItems": 1, "uniqueItems": True,
                "items": {"type": "string", "enum": ["npz", "csv", "pkl"]},
                "description": "Default [npz]. JSON metadata always saved. Use pkl for fitting-notebook compatibility."},
    "time_unit": {"type": "string", "enum": ["s", "ms", "us", "ns", "ps"], "default": "ns",
                  "description": "CSV/PKL export time unit; field export is G. NPZ always uses SI."},
})


def schema(name, description, properties, required):
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": obj(properties, required)}}


TOOL_SCHEMAS = [
    schema("inspect_epr_2d", "List datasets/axes without modifying a file. Optional singular-value diagnostics require selecting one dataset.",
           {"source": SOURCE, "singular_values": {"type": "integer", "minimum": 0, "maximum": 100, "default": 0}}, ["source"]),
    schema("import_epr_2d", "Import one dataset into canonical SI NPZ with (time,field) layout; no cleaning is performed.",
           {"source": SOURCE, "output": OUTPUT}, ["source"]),
    schema("correct_epr_2d_baseline", "Notebook background/polynomial correction: useful-signal bounds define regions excluded from estimation. Retains the full measured grid.",
           {"source": SOURCE, "baseline": BASELINE, "output": OUTPUT}, ["source", "baseline"]),
    schema("denoise_epr_2d", "Apply the notebook's MaRS SVD/Gaussian/wavelet/median/combined filtering. Does not correct background or normalize amplitude.",
           {"source": SOURCE, "denoising": DENOISING, "output": OUTPUT}, ["source", "denoising"]),
    schema("clean_epr_2d", "Run baseline/background correction then denoising; supply at least one stage. Saves history and diagnostics without overwriting input.",
           {"source": SOURCE, "baseline": BASELINE, "denoising": DENOISING, "output": OUTPUT}, ["source"]),
    schema("export_epr_2d", "Export one dataset as NPZ, axis-header CSV and/or notebook-compatible PKL; no additional cleaning. Default NPZ+PKL.",
           {"source": SOURCE, "output": OUTPUT}, ["source"]),
]


def schemas():
    return copy.deepcopy(TOOL_SCHEMAS)
