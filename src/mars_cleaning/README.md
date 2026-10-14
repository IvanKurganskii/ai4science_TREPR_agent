# MaRS agent tools for cleaning 2D EPR spectra

Python callables and JSON function schemas for DeepSeek/Qwen or another agent
host. The numerical workflow follows `Fit Dir/rho-ndi_data_clean.ipynb`:
background subtraction, polynomial field-baseline correction, then optional
MaRS denoising. The notebook and its helper modules are not runtime dependencies.
`baseline.py` is a bundled copy of the notebook's `BaselineCorrector2D`.

## Setup and use

Use the existing MaRS environment (Python 3.11). Additional requirements are in
`requirements.txt`; Nevergrad and CUDA are not needed for cleaning. The tools
were checked with MaRS-epr 0.0.5.post5. For an environment shared with older
Nevergrad code, NumPy 1.26.4 remains an appropriate compatibility choice;
cleaning was also tested in the current NumPy 2.4.6 environment.

Run from the project root so that `src` can be imported:

```python
from src.mars_cleaning import MarsCleaningTools

tools = MarsCleaningTools()
schemas = tools.schemas
result = tools.dispatch("inspect_epr_2d", {
    "source": {"path": "data/raw_spectrum.csv", "time_unit": "ns"}
})
```

Default outputs go to **`<working directory>/results/mars_simulations`**. To anchor
paths explicitly, use `MarsCleaningTools(working_dir=project_root)`.
`output_dir` overrides the destination; a relative override is resolved from
`working_dir`. Copying the package into another project does not require editing
any paths. Tests are optional and are not imported by the tools.

| Tool | Purpose |
| --- | --- |
| `inspect_epr_2d` | List datasets, shapes and axis ranges; optionally compute singular values and cumulative squared-singular-value fractions. |
| `import_epr_2d` | Convert a selected dataset into canonical SI NPZ without cleaning. |
| `correct_epr_2d_baseline` | Subtract the average background and/or fit field baselines. |
| `denoise_epr_2d` | Apply SVD, Gaussian, wavelet, median or combined filtering. |
| `clean_epr_2d` | Apply correction followed by denoising, or just one selected stage. |
| `export_epr_2d` | Write NPZ, CSV and/or notebook-compatible PKL without further processing. |

Direct methods have the same names and arguments. `dispatch` accepts a dictionary
or a JSON argument string and returns structured errors (`ok: false`). Direct
methods raise exceptions. See `AGENT_GUIDE.md`, `tool_schemas.json` and `examples/`.

## Data conventions

All working maps have shape **`(number of times, number of fields)`**. Numerical
API arrays, bounds and NPZ axes use **tesla and seconds**. Negative time points
are allowed. Axes are sorted together with the signal; duplicate coordinates,
nonfinite samples and complex channels are rejected. No interpolation,
normalization, sign change or cropping is performed.

Accepted files:

- Notebook CSV: first row is field in G, first column is time, interior is the
  signal. Set `source.time_unit` explicitly; a CSV header does not declare it.
- Notebook PKL: `datasets[name]` contains `field_G`, `time`, `signal`. Select
  `source.dataset` for a multi-dataset file and declare the original time unit.
  Acquisition `parameters` are carried into metadata and PKL exports. The loader
  accepts plain dictionaries/NumPy arrays, not pickled custom Python objects.
- NPZ: `fields_T`, `times_s`, `signal` and optional JSON `metadata_json`.
  Alternate axis keys `fields` and `times` need explicit units. For a simulation
  artifact, set `source.signal_key` to its intensity key, e.g. `spectrum`.

The default layout is `time_field`. Use `field_time` for a transposed input;
`auto` rejects square maps because their axis order cannot be inferred. MaRS
filtering expects `(field, time)` internally; the adapter transposes both ways.

## Correction and filtering

`field_bounds_T` and `time_bounds_s` describe the **useful-signal intervals**;
points **outside** those inclusive intervals supply the baseline/background.
They do not select an output crop. For background correction using pre-trigger
points only, specify `time_bounds_s: [0, null]`. `[null, upper]` uses only points
after `upper`. There must be usable background points and sufficient field
sideband points for the selected polynomial order.

Both corrections default to enabled. Polynomial order defaults to 1. The
`scaled_reference` mode fits the strongest time slice, then fits its baseline
scale and an intensity shift for each slice (`fit_intensity_shift: true`).
`independent` fits each slice with adjacent-slice initial guesses. Internally
the fits retain the notebook's G/ns coordinates for matching numerical
conditioning. Disable either step with `subtract_average_background: false`
or `correct_baseline: false`; the disabled step's bounds can be omitted.

Filtering defaults match the notebook widgets: SVD rank 10; Gaussian widths
`sigma_t: 2`, `sigma_b: 1`; wavelet `db4`, level 4 and automatic MAD-based
threshold; median kernel 3. Choose `method` explicitly. Supply only parameters
relevant to that method. Widths and kernel sizes are in **samples**, not SI units.
MaRS `combined` applies SVD followed by Gaussian filtering at **half** the
requested widths. SVD rank is capped by the smaller map dimension. On a
nonuniform grid the filters still use sample indices; this is recorded in the
report. The default engine limits BLAS to two threads during numerical calls;
override with `CleaningEngine(threads=...)` if needed.

Use field/time slices and the removed map to assess filtering. Singular-value
fractions are diagnostics, not a guarantee that rejected components are noise.
Signal-free regions and filtering settings should be selected for the actual
dataset rather than copied from the illustrative examples.

## Saved results

Names have unique suffixes and input files are never overwritten. JSON metadata
is always saved. NPZ is the default format and includes SI axes, cleaned signal,
input signal, removed signal and correction diagnostics. Metadata records source
SHA-256, source lineage, correction/filter options, backend-source SHA-256 and
package versions. An optional PNG shows input/output/removed maps and selected
slices. Large arrays stay in files; `include_data: true` is limited to 10,000
points.

CSV and PKL exports use field in G and the selected time unit (default ns).
PKL preserves the notebook's `datasets`/`parameters` structure and adds an
explicit `time_unit`. `export_epr_2d` defaults to both NPZ and PKL. Load the PKL
into the existing fitting notebook as a cleaned dataset; load NPZ in Python with
`numpy.load(path, allow_pickle=False)`.

## CLI and checks

```powershell
python -m src.mars_cleaning --schemas
python -m src.mars_cleaning --request src/mars_cleaning/examples/inspect.json
python -m src.mars_cleaning --serve
python -B -m unittest src.mars_cleaning.tests.test_cleaning -v
```

The line-based server reads `{"id": ..., "name": ..., "arguments": {...}}` and
returns `{"id": ..., "result": {...}}` per line. Backend output goes to stderr.
The examples reference placeholder data; replace their paths and bounds first.
The tests use temporary synthetic datasets, check signal recovery, agreement
with MaRS filters, axis handling, tool calls and NPZ/CSV/PKL round trips.
