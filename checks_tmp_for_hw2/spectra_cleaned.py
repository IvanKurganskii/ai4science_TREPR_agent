"""Check counts, text headers, and nonzero signal in results/spectra/processed.

Usage: python checks/spectra_cleaned.py results/spectra
Text spectra (.csv, .tsv, .txt, .dat, .asc) are counted recursively by extension.
JSON metadata and other auxiliary files are ignored. A nonempty text header
must precede the first numeric row; blank lines alone do not count as a header.
By default, the first numeric row and column are coordinate axes; only the
remaining matrix is signal. Use --layout columns for field/intensity pairs.
All numeric values must be finite, and at least one intensity must be nonzero.
No amplitude threshold is imposed. This does not establish processing quality
or physical signal onset.
Exit status: 0 = passed, 1 = failed.
"""

import argparse
from collections import Counter
import math
from pathlib import Path
import re


PROJECT_DIR = Path(__file__).resolve().parents[1]
SPECTRUM_TYPES = {".csv", ".tsv", ".txt", ".dat", ".asc"}


def spectrum_files(directory):
    return sorted(path for path in directory.rglob("*")
                  if path.is_file() and path.suffix.lower() in SPECTRUM_TYPES)


def counts(files):
    return Counter(path.suffix.lower() for path in files)


def numeric_row(line):
    tokens = [token for token in re.split(r"[,;\s]+", line.strip()) if token]
    if not tokens:
        return False
    try:
        for token in tokens:
            float(token)
    except ValueError:
        return False
    return True


def check_header(path):
    try:
        with path.open(encoding="utf-8-sig") as stream:
            has_header = False
            for line in stream:
                text = line.strip()
                if not text:
                    continue
                if numeric_row(text):
                    return None if has_header else f"{path}: numeric data starts without a text header"
                if any(character.isalpha() for character in text):
                    has_header = True
        return f"{path}: no numeric spectrum data after the header"
    except (OSError, UnicodeError) as exc:
        return f"{path}: cannot read UTF-8 text: {exc}"


def check_signal(path, *, layout):
    """Reject zero/nonfinite signal without counting coordinate axes as signal.

    grid: first row and first column are axes; signal is the interior matrix.
    rows: first row is the axis, second row contains intensities (slice format).
    columns: exactly two columns, coordinate followed by intensity.
    Explicit layouts avoid guessing the meaning of an ambiguous 2-by-2 array.
    """
    if layout not in {"grid", "rows", "columns"}:
        raise ValueError(f"unknown spectrum layout: {layout}")
    row_count = 0
    width = None
    has_nonzero_signal = False
    try:
        with path.open(encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, start=1):
                text = line.strip()
                if not text or text.startswith("#"):
                    continue
                tokens = [token for token in re.split(r"[,;\s]+", text) if token]
                try:
                    values = [float(token) for token in tokens]
                except ValueError:
                    if row_count:
                        return f"{path}: line {line_number}: nonnumeric text inside spectrum data"
                    continue
                if width is None:
                    width = len(values)
                    if width < 2:
                        return f"{path}: expected coordinate axes and intensity data, found fewer than two columns"
                elif len(values) != width:
                    return f"{path}: line {line_number}: inconsistent number of numeric columns"
                if not all(math.isfinite(value) for value in values):
                    return f"{path}: line {line_number}: nonfinite numeric value (NaN or infinity)"
                row_count += 1
                if layout == "grid":
                    if row_count == 1:
                        continue
                    signal = values[1:]
                elif layout == "rows":
                    if row_count == 1:
                        continue
                    if row_count > 2:
                        return f"{path}: rows layout requires exactly two numeric rows (axis and intensity); use --layout columns for coordinate/intensity pairs"
                    signal = values
                else:
                    if width != 2:
                        return f"{path}: columns layout requires exactly two numeric columns (coordinate and intensity)"
                    signal = values[1:]
                has_nonzero_signal |= any(value != 0.0 for value in signal)
        if not row_count:
            return f"{path}: no numeric spectrum data"
        if layout in {"grid", "rows"} and row_count < 2:
            return f"{path}: coordinate axis exists but intensity data is missing"
        if not has_nonzero_signal:
            return f"{path}: all signal intensities are zero (coordinate axes excluded)"
        return None
    except (OSError, UnicodeError) as exc:
        return f"{path}: cannot read UTF-8 text: {exc}"


def check_spectrum(path, *, layout):
    return check_header(path) or check_signal(path, layout=layout)


def check_counts(expected, actual, label, *, at_least=False):
    errors = []
    for extension in sorted(expected.keys() | actual.keys()):
        need, found = expected[extension], actual[extension]
        print(f"{label} {extension}: reference={need}, found={found}")
        if (found < need if at_least else found != need):
            relation = "at least" if at_least else "exactly"
            errors.append(f"{label} {extension}: expected {relation} {need} files, found {found}")
    return errors


def validate(results_dir, raw_dir, *, layout="grid"):
    raw = spectrum_files(raw_dir)
    processed_dir = results_dir / "processed"
    processed = spectrum_files(processed_dir)
    errors = []
    if not raw:
        errors.append(f"{raw_dir}: no source spectra found")
    if not processed_dir.is_dir():
        errors.append(f"{processed_dir}: directory does not exist")
    errors.extend(check_counts(counts(raw), counts(processed), "processed"))
    errors.extend(error for path in processed if (error := check_spectrum(path, layout=layout)))
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_DIR / "data" / "raw_spectra")
    parser.add_argument("--layout", choices=("grid", "columns"), default="grid",
                        help="numeric data layout: grid with axes in the first row/column (default), or coordinate/intensity columns")
    args = parser.parse_args()
    errors = validate(args.results_dir, args.raw_dir, layout=args.layout)
    for error in errors:
        print(f"FAIL: {error}")
    print("FAILED" if errors else "OK: processed spectra counts, headers, and finite nonzero signal")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
