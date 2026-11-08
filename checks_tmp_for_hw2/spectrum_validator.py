"""Validation utilities for 2D EPR spectra (time × field).

Checks that a spectrum file contains valid data:
- No NaN values
- No Inf values
- Not all values are zero simultaneously
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional


def validate_spectrum_array(signal: np.ndarray) -> Tuple[bool, List[str]]:
    """Validate a single 2D spectrum array.

    Args:
        signal: 2D numpy array of shape (time, field).

    Returns:
        (is_valid, list_of_error_messages)
    """
    errors: List[str] = []

    if signal.ndim != 2:
        errors.append(f"Expected 2D array, got {signal.ndim}D")
        return False, errors

    if signal.size == 0:
        errors.append("Empty spectrum array")
        return False, errors

    if np.any(np.isnan(signal)):
        nan_count = int(np.sum(np.isnan(signal)))
        errors.append(f"Contains {nan_count} NaN value(s)")

    if np.any(np.isinf(signal)):
        inf_count = int(np.sum(np.isinf(signal)))
        errors.append(f"Contains {inf_count} Inf value(s)")

    if np.all(signal == 0):
        errors.append("All values are zero")

    is_valid = len(errors) == 0
    return is_valid, errors


def validate_spectrum_file(filepath: Path) -> Tuple[bool, List[str]]:
    """Validate a spectrum CSV file.

    Expects CSV with:
    - First row: time_value, field1, field2, ...
    - Subsequent rows: time_value, intensity1, intensity2, ...

    Args:
        filepath: Path to the CSV file.

    Returns:
        (is_valid, list_of_error_messages)
    """
    errors: List[str] = []

    if not filepath.exists():
        errors.append(f"File not found: {filepath}")
        return False, errors

    try:
        data = np.loadtxt(filepath, delimiter=',')
    except Exception as e:
        errors.append(f"Failed to load CSV: {e}")
        return False, errors

    # First row is header: time, field1, field2, ...
    # Data starts from row 1, col 1 (skip time column and header row)
    if data.ndim == 1:
        errors.append("CSV contains only one row or column")
        return False, errors

    signal = data[1:, 1:]

    if signal.size == 0:
        errors.append("No signal data found (header-only file)")
        return False, errors

    is_valid, array_errors = validate_spectrum_array(signal)
    errors.extend(array_errors)

    return is_valid, errors


def validate_spectra_directory(directory: Path) -> Tuple[int, int, List[Tuple[Path, bool, List[str]]]]:
    """Validate all CSV files in a directory.

    Args:
        directory: Path to directory containing spectrum CSV files.

    Returns:
        (total_count, valid_count, list_of (filepath, is_valid, errors))
    """
    results: List[Tuple[Path, bool, List[str]]] = []
    valid_count = 0

    if not directory.exists():
        return 0, 0, results

    csv_files = sorted(directory.glob("*.csv"))

    for csv_file in csv_files:
        is_valid, errors = validate_spectrum_file(csv_file)
        results.append((csv_file, is_valid, errors))
        if is_valid:
            valid_count += 1

    return len(csv_files), valid_count, results


if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: python spectrum_validator.py <path> [--directory]")
        sys.exit(1)

    target = Path(sys.argv[1])
    is_directory = "--directory" in sys.argv

    if is_directory:
        total, valid, results = validate_spectra_directory(target)
        print(f"Directory: {target}")
        print(f"Total files: {total}, Valid: {valid}")
        for filepath, is_valid, errors in results:
            status = "OK" if is_valid else "FAIL"
            print(f"  [{status}] {filepath.name}")
            for err in errors:
                print(f"    - {err}")
        sys.exit(0 if valid == total else 1)
    else:
        is_valid, errors = validate_spectrum_file(target)
        status = "OK" if is_valid else "FAIL"
        print(f"[{status}] {target.name}")
        for err in errors:
            print(f"  - {err}")
        sys.exit(0 if is_valid else 1)
