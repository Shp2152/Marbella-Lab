#!/usr/bin/env python3
"""
XPS Data Plotter
Plots fitted XPS data with raw spectrum and individual fitted peaks.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import sys
from typing import List


def adjust_color_brightness(color, brightness_factor):
    """
    Adjust the brightness of a color.

    Args:
        color: RGB color tuple (values 0-1)
        brightness_factor: Factor to adjust brightness (>1 for lighter, <1 for darker)

    Returns:
        Adjusted RGB color tuple
    """
    return tuple(min(1.0, c * brightness_factor) for c in color[:3])


def find_peak_columns(columns):
    """
    Find all peak columns matching the pattern [n/1].

    Args:
        columns: List of column names from the CSV

    Returns:
        List of peak column names
    """
    peak_columns = []
    for col in columns:
        if col.startswith('[') and col.endswith('/1]'):
            peak_columns.append(col)
    return peak_columns


def plot_xps_spectrum(csv_file, output_folder, base_color=(0.2, 0.4, 0.6)):
    """
    Plot XPS spectrum with raw data and fitted peaks.

    Args:
        csv_file: Path to the CSV file
        output_folder: Path to output folder
        base_color: RGB color tuple for the spectrum (dark shade)

    Returns:
        None
    """
    # Read the CSV file - handle different formats
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Skipping {csv_file}: {e}")
        return

    # Check required columns
    if 'Energy' not in df.columns or 'Spectrum' not in df.columns:
        print(f"Skipping {csv_file}: Required columns (Energy, Spectrum) not found")
        return

    # Find peak columns
    peak_columns = find_peak_columns(df.columns)

    # Skip files without fitted peaks
    if not peak_columns:
        print(f"Skipping {csv_file}: No fitted peaks found")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot spectrum (raw data) in dark color
    ax.plot(df['Energy'], df['Spectrum'],
            color=base_color, linewidth=2)

    # Calculate lighter color for fitted peaks (1.5x brighter)
    peak_color = adjust_color_brightness(base_color, 1.5)

    # Plot each fitted peak in lighter shade
    for peak_col in peak_columns:
        ax.plot(df['Energy'], df[peak_col],
                color=peak_color, linewidth=1.5, alpha=0.8)

    # Set labels and title
    ax.set_xlabel('Energy (eV)', fontsize=16)
    ax.set_ylabel('Intensity (c/s)', fontsize=16)

    # Get filename without extension for title
    filename = Path(csv_file).stem
    ax.set_title(f'XPS: {filename}', fontsize=18)

    # Set tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Invert x-axis (typical for XPS)
    ax.invert_xaxis()

    # Tight layout
    plt.tight_layout()

    # Save figure
    output_file = output_folder / f"{filename}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_file}")


def process_csv_files(csv_files):
    """
    Process a list of CSV files and create plots.

    Args:
        csv_files: List of paths to CSV files
    """
    for csv_file in csv_files:
        csv_path = Path(csv_file)

        # Get the sample folder name (parent directory)
        sample_folder = csv_path.parent.name

        # Create output folder
        output_folder = Path('data/processed') / sample_folder
        output_folder.mkdir(parents=True, exist_ok=True)

        # Plot the spectrum
        plot_xps_spectrum(csv_file, output_folder)


def interactive_file_selection() -> List[Path]:
    """
    Interactive file selection interface.
    Allows user to navigate through folders and select CSV files.

    Returns:
        List of selected CSV file paths
    """
    raw_data_path = Path('data/raw')

    if not raw_data_path.exists():
        print("Error: data/raw directory not found")
        return []

    # Get all sample folders
    sample_folders = sorted([f for f in raw_data_path.iterdir() if f.is_dir()])

    if not sample_folders:
        print("No sample folders found in data/raw/")
        return []

    # Display available folders
    print("\n=== XPS Data Plotter ===")
    print("\nAvailable sample folders:")
    for idx, folder in enumerate(sample_folders, 1):
        print(f"  {idx}. {folder.name}")

    # Select folder
    while True:
        folder_choice = input("\nSelect a folder number (or 'q' to quit): ").strip()
        if folder_choice.lower() == 'q':
            return []
        try:
            folder_idx = int(folder_choice) - 1
            if 0 <= folder_idx < len(sample_folders):
                selected_folder = sample_folders[folder_idx]
                break
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # Get all CSV files with fitted peaks in the selected folder
    csv_files = []
    all_csv_files = list(selected_folder.glob('*.csv'))

    for csv_file in all_csv_files:
        # Check if file has fitted peaks
        try:
            df = pd.read_csv(csv_file)
            if 'Energy' in df.columns and 'Spectrum' in df.columns:
                peak_columns = find_peak_columns(df.columns)
                if peak_columns:
                    csv_files.append(csv_file)
        except:
            continue

    # Sort CSV files alphabetically
    csv_files = sorted(csv_files)

    if not csv_files:
        print(f"\nNo valid XPS fitted data files found in {selected_folder.name}/")
        return []

    # Display available CSV files
    print(f"\n=== Files in {selected_folder.name} ===")
    for idx, csv_file in enumerate(csv_files, 1):
        print(f"  {idx}. {csv_file.name}")

    # Select files
    print("\nSelect files to plot:")
    print("  - Enter numbers separated by commas (e.g., 1,3,5)")
    print("  - Enter 'all' to select all files")
    print("  - Enter 'more' to add files from another folder")
    print("  - Enter 'q' to quit")

    selected_files = []

    while True:
        file_choice = input("\nYour selection: ").strip()
        if file_choice.lower() == 'q':
            return selected_files if selected_files else []
        elif file_choice.lower() == 'all':
            selected_files.extend(csv_files)
            print(f"Added {len(csv_files)} file(s). Total: {len(selected_files)} file(s)")
            return selected_files
        elif file_choice.lower() == 'more':
            # Add files from current folder first if any were selected
            if selected_files:
                print(f"Current selection: {len(selected_files)} file(s)")
            # Recursively call to select from another folder
            more_files = interactive_file_selection()
            if more_files:
                selected_files.extend(more_files)
                print(f"\nTotal files selected: {len(selected_files)}")
            return selected_files
        else:
            try:
                indices = [int(x.strip()) - 1 for x in file_choice.split(',')]
                temp_selected = []
                for idx in indices:
                    if 0 <= idx < len(csv_files):
                        temp_selected.append(csv_files[idx])
                    else:
                        print(f"Warning: Index {idx + 1} is out of range, skipping.")

                if temp_selected:
                    selected_files.extend(temp_selected)
                    print(f"Added {len(temp_selected)} file(s). Total: {len(selected_files)} file(s)")

                    # Ask if user wants to add more files from another folder
                    add_more = input("\nAdd files from another folder? (y/n): ").strip().lower()
                    if add_more == 'y':
                        more_files = interactive_file_selection()
                        if more_files:
                            selected_files.extend(more_files)
                            print(f"\nTotal files selected: {len(selected_files)}")
                    return selected_files
                else:
                    print("No valid files selected. Please try again.")
            except ValueError:
                print("Invalid input. Please enter numbers separated by commas.")


def plot_multiple_spectra(csv_files: List[Path], output_folder: Path, output_filename: str = None, offset_factor: float = 1.2, normalize: bool = False):
    """
    Plot multiple XPS spectra on the same figure with y-axis offset.

    Args:
        csv_files: List of CSV file paths
        output_folder: Path to output folder
        output_filename: Name for the output file (without extension). If None, defaults to "multi_spectrum_comparison"
        offset_factor: Factor to offset y-axis (multiplier for max intensity)
        normalize: If True, normalize each spectrum so their maximum intensities are equal
    """
    if not csv_files:
        return

    # Define color palette for different spectra
    color_palette = [
        (0.2, 0.4, 0.6),  # Blue
        (0.6, 0.2, 0.2),  # Red
        (0.2, 0.6, 0.2),  # Green
        (0.6, 0.4, 0.2),  # Orange
        (0.4, 0.2, 0.6),  # Purple
        (0.2, 0.6, 0.6),  # Cyan
        (0.6, 0.6, 0.2),  # Yellow-green
        (0.6, 0.2, 0.4),  # Magenta
    ]

    # Create figure with portrait orientation (taller than wide)
    fig, ax = plt.subplots(figsize=(8, 12))

    current_offset = 0
    max_intensity_list = []

    # First pass: determine max intensities for proper offsetting
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if 'Spectrum' in df.columns:
                max_intensity_list.append(df['Spectrum'].max())
        except:
            continue

    if not max_intensity_list:
        print("Error: Could not read any spectra")
        return

    # Calculate offset based on maximum intensity
    if normalize:
        # When normalized, all spectra will have max intensity of 1.0
        base_offset = 1.0 * offset_factor
    else:
        # Use original max intensity for offset
        base_offset = max(max_intensity_list) * offset_factor

    # Second pass: plot all spectra
    for idx, csv_file in enumerate(csv_files):
        try:
            df = pd.read_csv(csv_file)

            if 'Energy' not in df.columns or 'Spectrum' not in df.columns:
                print(f"Skipping {csv_file.name}: Missing required columns")
                continue

            peak_columns = find_peak_columns(df.columns)

            # Get color for this spectrum
            base_color = color_palette[idx % len(color_palette)]
            peak_color = adjust_color_brightness(base_color, 1.5)

            # Get filename for label
            filename = csv_file.stem

            # Normalize if requested
            if normalize:
                # Find min and max across Spectrum and all peak columns
                all_values = df['Spectrum'].tolist()
                for peak_col in peak_columns:
                    all_values.extend(df[peak_col].tolist())

                min_intensity = min(all_values)
                max_intensity = max(all_values)

                # Calculate normalization: (value - min) / (max - min)
                intensity_range = max_intensity - min_intensity
                if intensity_range > 0:
                    baseline = min_intensity
                    norm_factor = 1.0 / intensity_range
                else:
                    baseline = 0
                    norm_factor = 1.0
            else:
                baseline = 0
                norm_factor = 1.0

            # Plot spectrum with offset (subtract baseline, then normalize)
            ax.plot(df['Energy'], (df['Spectrum'] - baseline) * norm_factor + current_offset,
                    color=base_color, linewidth=2)

            # Plot fitted peaks with offset (subtract baseline, then normalize)
            for peak_col in peak_columns:
                ax.plot(df['Energy'], (df[peak_col] - baseline) * norm_factor + current_offset,
                        color=peak_color, linewidth=1.5, alpha=0.8)

            # Increment offset for next spectrum
            current_offset += base_offset

        except Exception as e:
            print(f"Error plotting {csv_file.name}: {e}")
            continue

    # Set labels and title
    ax.set_xlabel('Energy (eV)', fontsize=16)
    ylabel = 'Intensity (normalized) - offset' if normalize else 'Intensity (c/s) - offset'
    ax.set_ylabel(ylabel, fontsize=16)
    title = 'XPS Spectra Comparison (Normalized)' if normalize else 'XPS Spectra Comparison'
    ax.set_title(title, fontsize=18)

    # Set tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Invert x-axis
    ax.invert_xaxis()

    # Tight layout
    plt.tight_layout()

    # Save figure with custom filename
    if output_filename is None:
        output_filename = "multi_spectrum_comparison"

    # Ensure filename doesn't have extension
    if output_filename.endswith('.png'):
        output_filename = output_filename[:-4]

    output_file = output_folder / f"{output_filename}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nSaved multi-spectrum plot: {output_file}")


def main():
    """
    Main function to run the script.
    """
    # Interactive mode
    print("=== XPS Plotter - Interactive Mode ===")
    print("\nMode selection:")
    print("  1. Plot individual spectra (one file per plot)")
    print("  2. Plot multiple spectra on same figure (with offset)")
    print("  3. Process all files in data/raw/ (batch mode)")

    mode = input("\nSelect mode (1-3): ").strip()

    if mode == '1':
        # Individual plots
        selected_files = interactive_file_selection()
        if selected_files:
            print(f"\nProcessing {len(selected_files)} file(s)...")
            process_csv_files(selected_files)
            print("\nDone!")

    elif mode == '2':
        # Multiple spectra on same plot
        selected_files = interactive_file_selection()
        if selected_files:
            print(f"\nPlotting {len(selected_files)} spectra on same figure...")

            # Prompt for normalization
            normalize_choice = input("\nNormalize peak heights? (y/n): ").strip().lower()
            normalize = (normalize_choice == 'y')

            # Prompt for output filename
            print("\nEnter a name for the output file (press Enter for default 'multi_spectrum_comparison'):")
            custom_filename = input("Filename: ").strip()
            if not custom_filename:
                custom_filename = None  # Use default

            # Check if files are from multiple folders
            unique_folders = set(f.parent.name for f in selected_files)
            if len(unique_folders) > 1:
                # Files from multiple folders - use a combined folder name
                output_folder_name = "multi_folder_comparison"
                output_folder = Path('data/processed') / output_folder_name
            else:
                # All files from same folder
                sample_folder = selected_files[0].parent.name
                output_folder = Path('data/processed') / sample_folder

            output_folder.mkdir(parents=True, exist_ok=True)
            plot_multiple_spectra(selected_files, output_folder, output_filename=custom_filename, normalize=normalize)
            print("\nDone!")

    elif mode == '3':
        # Batch mode - process all files
        raw_data_path = Path('data/raw')
        csv_files = list(raw_data_path.glob('**/*.csv'))

        if not csv_files:
            print("No CSV files found in data/raw/")
            return

        print(f"\nFound {len(csv_files)} CSV files")
        print("Processing...")
        process_csv_files(csv_files)
        print("\nDone!")

    else:
        print("Invalid mode selection.")


if __name__ == "__main__":
    main()
