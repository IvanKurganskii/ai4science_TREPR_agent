"""Check counts, text headers, and nonzero signal in results/spectra/slices.

Usage: python checks/spectra_sliced.py results/spectra
Processed counts must match data/raw_spectra by text-file extension. Slice
counts must be at least those counts, since several slices per spectrum are
allowed. Each slice needs a text header before numeric data. Only aggregate
counts are checked; filenames and processing/integration metadata are not.
Each slice must have finite numeric values and at least one nonzero intensity.
The default layout is two numeric rows: field axis, then intensities. For
coordinate/intensity columns, use --layout columns. Axes are never signal.
Exit status: 0 = passed, 1 = failed.
"""

import argparse
from pathlib import Path

if __package__:
    from .spectra_cleaned import PROJECT_DIR, spectrum_files, counts, check_counts, check_spectrum
else:
    from spectra_cleaned import PROJECT_DIR, spectrum_files, counts, check_counts, check_spectrum


def validate(results_dir, raw_dir, *, layout="rows"):
    raw = spectrum_files(raw_dir)
    processed_dir = results_dir / "processed"
    slices_dir = results_dir / "slices"
    processed = spectrum_files(processed_dir)
    slices = spectrum_files(slices_dir)
    errors = []
    if not raw:
        errors.append(f"{raw_dir}: no source spectra found")
    for directory in (processed_dir, slices_dir):
        if not directory.is_dir():
            errors.append(f"{directory}: directory does not exist")
    errors.extend(check_counts(counts(raw), counts(processed), "processed"))
    errors.extend(check_counts(counts(raw), counts(slices), "slices", at_least=True))
    errors.extend(error for path in slices if (error := check_spectrum(path, layout=layout)))
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_DIR / "data" / "raw_spectra")
    parser.add_argument("--layout", choices=("rows", "columns"), default="rows",
                        help="slice layout: axis/intensity rows (default), or coordinate/intensity columns")
    args = parser.parse_args()
    errors = validate(args.results_dir, args.raw_dir, layout=args.layout)
    for error in errors:
        print(f"FAIL: {error}")
    print("FAILED" if errors else "OK: slice counts, headers, and finite nonzero signal")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
