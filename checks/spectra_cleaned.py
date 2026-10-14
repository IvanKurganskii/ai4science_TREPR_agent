"""Check counts and text headers in results/spectra/processed.

Usage: python checks/spectra_cleaned.py results/spectra
Text spectra (.csv, .tsv, .txt, .dat, .asc) are counted recursively by extension.
JSON metadata and other auxiliary files are ignored. A nonempty text header
must precede the first numeric row; blank lines alone do not count as a header.
This checks file structure, not processing quality or physical signal onset.
Exit status: 0 = passed, 1 = failed.
"""

import argparse
from collections import Counter
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


def check_counts(expected, actual, label, *, at_least=False):
    errors = []
    for extension in sorted(expected.keys() | actual.keys()):
        need, found = expected[extension], actual[extension]
        print(f"{label} {extension}: reference={need}, found={found}")
        if (found < need if at_least else found != need):
            relation = "at least" if at_least else "exactly"
            errors.append(f"{label} {extension}: expected {relation} {need} files, found {found}")
    return errors


def validate(results_dir, raw_dir):
    raw = spectrum_files(raw_dir)
    processed_dir = results_dir / "processed"
    processed = spectrum_files(processed_dir)
    errors = []
    if not raw:
        errors.append(f"{raw_dir}: no source spectra found")
    if not processed_dir.is_dir():
        errors.append(f"{processed_dir}: directory does not exist")
    errors.extend(check_counts(counts(raw), counts(processed), "processed"))
    errors.extend(error for path in processed if (error := check_header(path)))
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_DIR / "data" / "raw_spectra")
    args = parser.parse_args()
    errors = validate(args.results_dir, args.raw_dir)
    for error in errors:
        print(f"FAIL: {error}")
    print("FAILED" if errors else "OK: processed spectra counts and headers")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
