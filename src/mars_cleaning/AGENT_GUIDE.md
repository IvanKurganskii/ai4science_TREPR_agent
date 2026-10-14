# Instructions for the spectrum-cleaning agent

Register `MarsCleaningTools().schemas` as function tools in your agent host.
Execute returned function calls with `tools.dispatch(name, arguments)`; do not
ask the language model to implement its own numerical filters. The schemas and
dispatcher do not depend on a DeepSeek/Qwen SDK.

Recommended sequence:

1. Inspect the input. For a multi-dataset PKL select a dataset by its actual name.
   Declare the measured time unit if it is absent; never infer it from magnitude.
2. Import to canonical NPZ if a reusable unmodified dataset is useful. Read its
   diagnostic plot or inspect its axes/slices in the host.
3. Choose the useful field/time intervals from the experiment. Background points
   lie outside the time interval; polynomial sidebands lie outside the field
   interval. A pre-trigger-only background is `[trigger_time_s, null]`.
   Do not use a late-time interval as background unless its signal is negligible.
4. Run `correct_epr_2d_baseline`, inspect the removed map and sidebands, and keep
   the corrected NPZ path from `result.artifacts.npz`. Start with a low polynomial
   order. Disable the background step if no signal-free times were measured.
5. Inspect singular values of that corrected NPZ if considering SVD, then run
   `denoise_epr_2d` with an explicit method. Assess whether removed components
   contain spectral or kinetic features. The notebook defaults are available,
   but do not determine the appropriate rank/width for a new experiment.
6. Export a satisfactory cleaned NPZ to PKL for the fitting notebook, or CSV for
   exchange. Keep the metadata and diagnostics with the cleaned map.

`clean_epr_2d` combines steps 4 and 5 after their settings are selected. It always
applies correction before filtering. Bounds are in T/s and **never crop** the
map. Gaussian widths are in time/field samples, median kernels are odd sample
counts and wavelet filtering acts along time at each field. Pass only the chosen
method's parameters. Preserve signal polarity and amplitude; the tools do not
automatically phase or normalize the data.

Artifacts default to `<working directory>/results/mars_simulations`; run the host
from the project root or set `working_dir` explicitly. Use returned artifact
paths for subsequent calls instead of predicting the unique filenames. A
successful result has `ok: true`; an unsuccessful dispatched call has `ok: false`
and an `error` object. Do not proceed with an artifact path from a failed call.
Arrays can remain on disk; do not request inline data for large maps.

Example registration alongside the existing simulation tools:

```python
from pathlib import Path
from src.mars_cleaning import MarsCleaningTools
from src.mars_simulation import MarsSimulationTools

cleaning = MarsCleaningTools(working_dir=project_root)
simulation = MarsSimulationTools(output_dir=Path(project_root)/"results"/"mars_simulations")
tool_schemas = cleaning.schemas + simulation.schemas
owners = {
    schema["function"]["name"]: owner
    for owner in (cleaning, simulation)
    for schema in owner.schemas
}

def execute_tool(name, arguments):
    if name not in owners:
        return {"ok": False, "error": {"message": f"Unknown tool: {name}"}}
    return owners[name].dispatch(name, arguments)
```

For direct in-memory work, use `CleaningEngine` and
`src.mars_cleaning.data.EPRDataset.from_arrays(fields_T, times_s, signal)`.
`engine.clean(dataset, baseline=..., denoising=...)` returns the new dataset and
diagnostics without writing files. Its canonical signal layout is `(time, field)`.
