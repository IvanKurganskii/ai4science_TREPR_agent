#!/usr/bin/env python3
"""
Recreate slice files from processed 2D spectra.
For each processed spectrum:
  - Find the time index with max absolute intensity (max slice)
  - Find the late time where signal stabilizes (late slice)
  - Write slice CSV files with proper headers and data

NOTE: Processed files have first data row = field values (not intensity).
      Actual intensity data starts from row 1.
"""

import os
import csv
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path("/mnt/c/Users/kurga/OneDrive/Документы/YSDA/AI4Science/my_project/results/spectra/processed")
SLICES_DIR = Path("/mnt/c/Users/kurga/OneDrive/Документы/YSDA/AI4Science/my_project/results/spectra/slices")
SLICES_DIR.mkdir(parents=True, exist_ok=True)


def read_processed_spectrum(filepath):
    """Read a processed 2D spectrum file.
    
    Processed file format:
      - Comment lines starting with '#'
      - First data row: time=0, field_value_1, field_value_2, ...
      - Subsequent rows: time_T, intensity_T_1, intensity_T_2, ...
    
    Returns:
        comments: list of comment lines
        field_values: array of field values (from first data row, cols 1:)
        time_values: array of time values (from data rows 1:, col 0)
        data: 2D array of intensities (time x field), shape (n_times, n_fields)
    """
    comments = []
    rows = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                comments.append(line)
            else:
                rows.append([float(x) for x in line.split(',')])
    
    rows = np.array(rows)
    
    # First data row contains field values, not intensity
    field_values = rows[0, 1:]
    
    # Actual intensity data starts from row 1
    time_values = rows[1:, 0]
    data = rows[1:, 1:]  # intensity values only
    
    return comments, field_values, time_values, data


def find_max_intensity_index(data):
    """Find the time index with maximum absolute intensity across all fields."""
    abs_data = np.abs(data)
    max_per_time = np.max(abs_data, axis=1)
    return np.argmax(max_per_time)


def find_late_time_index(time_values, data, min_fraction=0.8):
    """Find the time index where the spectrum has stabilized (late time).
    
    Look at the last fraction of time points and find where the shape
    stops changing significantly.
    """
    n_times = len(time_values)
    start_idx = int(n_times * min_fraction)
    
    # Use the last time point as reference
    last_data = data[-1]
    
    # Find the first time point after start_idx where the data is close to the last
    for i in range(start_idx, n_times):
        current_data = data[i]
        # Use relative difference where signal is strong
        mask = np.abs(last_data) > 1e-6
        if np.sum(mask) > 0:
            rel_diff = np.mean(np.abs((current_data[mask] - last_data[mask]) / last_data[mask]))
            if rel_diff < 0.05:  # 5% threshold
                return i
    
    # If no stable point found, return the last few points
    return n_times - 1


def write_slice_file(output_path, comments, field_values, time, intensities, 
                     slice_type, integration_window=None):
    """Write a slice CSV file with proper headers."""
    with open(output_path, 'w', newline='') as f:
        # Write header comments
        for comment in comments:
            f.write(comment + '\n')
        
        # Write slice-specific header
        f.write(f'# Slice: {output_path.name}\n')
        f.write(f'# Integration time range: [{time:.1f}, {time:.1f}] ns\n')
        f.write(f'# Method: integration around {"max intensity" if slice_type == "max" else "late time"}\n')
        f.write(f'# Processing: {comments[0].replace("# Processed spectrum: ", "")}\n')
        
        if integration_window is not None:
            f.write(f'# Integration window: {integration_window}\n')
        
        # Write field values row
        f.write(','.join(f'{fv:.7f}' for fv in field_values) + '\n')
        
        # Write intensity values row
        f.write(','.join(f'{iv:.16e}' for iv in intensities) + '\n')


def main():
    processed_files = sorted(PROCESSED_DIR.glob("*.csv"))
    
    for proc_file in processed_files:
        print(f"\nProcessing: {proc_file.name}")
        
        # Read the processed spectrum
        comments, field_values, time_values, data = read_processed_spectrum(proc_file)
        
        print(f"  Time range: {time_values[0]:.1f} - {time_values[-1]:.1f} ns")
        print(f"  Number of time points: {len(time_values)}")
        print(f"  Number of field points: {len(field_values)}")
        print(f"  Field range: {field_values[0]:.1f} - {field_values[-1]:.1f}")
        
        # Find max intensity index
        max_idx = find_max_intensity_index(data)
        max_time = time_values[max_idx]
        max_intensities = data[max_idx]
        print(f"  Max intensity at t = {max_time:.1f} ns (index {max_idx})")
        print(f"  Max intensity range: {max_intensities.min():.6e} / {max_intensities.max():.6e}")
        
        # Find late time index
        late_idx = find_late_time_index(time_values, data)
        late_time = time_values[late_idx]
        late_intensities = data[late_idx]
        print(f"  Late time at t = {late_time:.1f} ns (index {late_idx})")
        print(f"  Late intensity range: {late_intensities.min():.6e} / {late_intensities.max():.6e}")
        
        # Extract and write max slice
        max_slice_name = proc_file.stem.replace("_processed", "_slice_max") + ".csv"
        max_slice_path = SLICES_DIR / max_slice_name
        write_slice_file(max_slice_path, comments, field_values, max_time, max_intensities, "max")
        print(f"  Written: {max_slice_name}")
        
        # Extract and write late slice
        late_slice_name = proc_file.stem.replace("_processed", "_slice_late") + ".csv"
        late_slice_path = SLICES_DIR / late_slice_name
        write_slice_file(late_slice_path, comments, field_values, late_time, late_intensities, "late")
        print(f"  Written: {late_slice_name}")
        
        # Verify the slices are non-zero
        max_nonzero = np.sum(np.abs(max_intensities) > 1e-10)
        late_nonzero = np.sum(np.abs(late_intensities) > 1e-10)
        print(f"  Max slice non-zero points: {max_nonzero}/{len(max_intensities)}")
        print(f"  Late slice non-zero points: {late_nonzero}/{len(late_intensities)}")


if __name__ == "__main__":
    main()