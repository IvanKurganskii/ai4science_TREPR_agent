"""JSON function schemas usable without a specific agent/provider SDK."""

import copy


def _object(properties, required=()):
    return {"type": "object", "properties": properties, "required": list(required),
            "additionalProperties": False}


def _number(description, **constraints):
    return {"type": "number", "description": description, **constraints}


def _axis(unit, minimum):
    return {"description": f"Strictly increasing axis in {unit}; endpoints are included.",
            "oneOf": [
                _object({"start": _number(f"First point in {unit}", minimum=0),
                         "stop": _number(f"Last point in {unit}", minimum=0),
                         "points": {"type": "integer", "minimum": minimum}},
                        ("start", "stop", "points")),
                _object({"values": {"type": "array", "minItems": minimum,
                                    "items": _number(f"Point in {unit}", minimum=0)}}, ("values",))]}


SETTINGS_SCHEMA = _object({
    "freq": _number("Microwave frequency in Hz; default 9.8e9", exclusiveMinimum=0),
    "temperature": _number("Temperature in kelvin; default 80", exclusiveMinimum=0),
    "mesh": {"type": "integer", "minimum": 2, "default": 25,
             "description": "One resolution for both powder meshes: (mesh, mesh)."},
    "frame": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "number"},
              "description": "Common molecular tensor frame, ZYZ Euler angles in radians; default [0,0,0]."},
    "device": {"type": "string", "enum": ["cpu", "cuda", "auto"], "default": "cpu"},
    "dtype": {"type": "string", "enum": ["float64", "float32"], "default": "float64"},
    "harmonic": {"type": "integer", "enum": [0, 1], "default": 0,
                 "description": "0 absorption; 1 first field derivative."},
})


def hamiltonian_schema(model):
    properties = {
        "D": _number("D in Hz, tensor diag(-D/3+E,-D/3-E,2D/3)."),
        "E": _number("E in Hz; |E| <= |D|/3."),
        "gauss": _number("Gaussian FWHM in tesla.", minimum=0),
        "lorentz": _number("Lorentzian FWHM in tesla; at least one linewidth must be positive.", minimum=0),
        "ham_strain": {"description": "Optional residual Gaussian FWHM in Hz: scalar or three principal widths; default 0.",
                       "anyOf": [{"type": "number", "minimum": 0},
                                 {"type": "array", "minItems": 3, "maxItems": 3,
                                  "items": {"type": "number", "minimum": 0}}]},
    }
    if model == "high_j":
        properties["g"] = {"description": "Dimensionless g factor or three principal values for effective S=1.",
                           "anyOf": [{"type": "number", "exclusiveMinimum": 0},
                                     {"type": "array", "minItems": 3, "maxItems": 3,
                                      "items": {"type": "number", "exclusiveMinimum": 0}}]}
    else:
        properties.update({
            "g0": _number("Mean isotropic g factor; g1,2 = g0 +/- delta_g/2.", exclusiveMinimum=0),
            "delta_g": _number("Signed isotropic g1-g2 difference, dimensionless."),
            "J": _number("Exchange in Hz, convention +J*S1 dot S2; negative J is allowed."),
        })
    return _object(properties, [key for key in properties if key != "ham_strain"])


def populations_schema(model):
    names = ("p_x", "p_y", "p_z") if model == "high_j" else ("p_s", "p_x", "p_y", "p_z")
    return _object({name: _number("Relative molecular XYZ (or singlet) weight, not a unit-sum probability.",
                                 minimum=0, maximum=1) for name in names}, names)


def context_schema(model):
    properties = populations_schema(model)["properties"]
    lifetimes = {"tau_intra": "T1 parameter: neighboring energy-ordered eigenstate links"} if model == "high_j" else {
        "tau_outer": "Energy-ordered eigenstate links 1-2, 1-3, 2-4, 3-4",
        "tau_inner": "Energy-ordered eigenstate link 2-3"}
    losses = ("x", "y", "z") if model == "high_j" else ("s", "x", "y", "z")
    lifetimes.update({"tau_out_"+name: "Loss from molecular sublevel "+name for name in losses})
    for name, description in lifetimes.items():
        properties[name] = {"description": description + "; lifetime in microseconds, rate=1e6/tau s^-1. null disables this rate.",
                            "anyOf": [{"type": "number", "exclusiveMinimum": 0}, {"type": "null"}]}
    return _object(properties, properties)


OUTPUT_SCHEMA = _object({
    "name": {"type": "string", "description": "Optional artifact name prefix: letters, digits, _ or -."},
    "save_plot": {"type": "boolean", "default": True},
    "include_data": {"type": "boolean", "default": False,
                     "description": "Inline axes and spectrum only for <=10000 samples; artifacts always contain all samples."},
})


def tool_schema(model, dimension):
    label = "High-J effective S=1 triplet" if model == "high_j" else "Moderate-J two S=1/2 radical pair"
    properties = {
        "field_axis": _axis("tesla", 2),
        "population_basis": {"type": "string", "enum": ["xyz", "product", "eigen", "zfs", "multiplet", "zeeman"],
                             "default": "xyz", "description": "Basis of the ordered initial populations only; T1 always eigen, sublevel loss always xyz."},
        "population_mode": {"type": "string", "enum": ["notebook", "raw", "minmax", "max"],
                            "default": "notebook", "description": "notebook: high-J 1D minmax, high-J 2D raw, moderate-J max."},
        "normalization": {"type": "string", "enum": ["none", "maxabs"], "default": "none",
                          "description": "Optional whole-spectrum amplitude normalization preserving E/A signs."},
        "output": copy.deepcopy(OUTPUT_SCHEMA),
    }
    required = ["field_axis"]
    if dimension == "2d":
        properties["spin_model"] = _object({"model": {"type": "string", "enum": [model]},
                                           "hamiltonian": hamiltonian_schema(model),
                                           "settings": copy.deepcopy(SETTINGS_SCHEMA)},
                                          ("model", "hamiltonian", "settings"))
        properties["device"] = {"type": "string", "enum": ["cpu", "cuda", "auto"],
                                "description": "Optional execution-device override for a saved spin_model."}
        required.append("spin_model")
    else:
        properties["hamiltonian"] = hamiltonian_schema(model)
        properties["settings"] = copy.deepcopy(SETTINGS_SCHEMA)
        required.append("hamiltonian")
    if dimension == "1d":
        properties["populations"] = populations_schema(model)
        required.append("populations")
        description = label + ": stationary field spectrum without kinetic relaxation; returns reusable spin_model."
    else:
        properties["time_axis"] = _axis("seconds since preparation (t=0)", 1)
        properties["context"] = context_schema(model)
        properties["rise_time_us"] = _number(
            "Optional positive rise time in us: S(B,t)-S(B,0)*exp(-t/T); omit for raw MaRS kinetics.", exclusiveMinimum=0)
        required.extend(("time_axis", "context"))
        description = label + (": time-field map varying context with a fixed spin_model from 1D/full." if dimension == "2d" else
                               ": full time-field map with explicit Hamiltonian and relaxation context; both may change between calls.")
    return {"type": "function", "function": {"name": f"simulate_{model}_{dimension}",
                                               "description": description,
                                               "parameters": _object(properties, required)}}


TOOL_SCHEMAS = [tool_schema(model, dimension) for model in ("high_j", "moderate_j")
                for dimension in ("1d", "2d", "2d_full")]
