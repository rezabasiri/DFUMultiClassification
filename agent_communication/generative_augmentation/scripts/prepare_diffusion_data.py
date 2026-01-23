#!/usr/bin/env python3
"""
Prepare DFU images for diffusion model training

Organizes images from raw data into phase-specific folders using the original CSV,
NOT the best_matching.csv (which only includes a subset of images).

This script:
1. Reads the original DataMaster_Processed_V12_WithMissing.csv
2. Parses all images from Depth_RGB folder
3. Matches images to CSV records to get phase labels
4. Copies/symlinks images to organized folders: data/diffusion/{modality}/{phase}/

Usage:
    python prepare_diffusion_data.py --output_dir data/diffusion --use_symlinks
"""

import os
import re
import argparse
import shutil
from pathlib import Path
from typing import Dict, Optional, List
import pandas as pd
from collections import defaultdict


def parse_filename(fname: str) -> Optional[Dict]:
    """
    Parse DFU image filename to extract metadata.

    Format: {random}_P{patient:3d}{appt:2d}{dfu:1d}{B/A}{D/T}{R/M/L}{Z/W}.png

    Components:
    - random: Random identifier
    - patient: 3-digit patient number
    - appt: 2-digit appointment number
    - dfu: 1-digit DFU number
    - B/A: Before/After debridement
    - D/T: Depth/Thermal camera
    - R/M/L: Right/Middle/Left angle
    - Z/W: Zoomed/Wide view
    """
    match = re.match(r'(\d+)_P(\d{3})(\d{2})(\d)([BA])([DT])([RML])([ZW])\.png', fname)
    if match:
        return {
            'random_id': match.group(1),
            'patient': int(match.group(2)),
            'appt': int(match.group(3)),
            'dfu': int(match.group(4)),
            'debride': match.group(5),  # B=Before, A=After
            'camera': match.group(6),   # D=Depth, T=Thermal
            'angle': match.group(7),    # R/M/L
            'view': match.group(8)      # Z/W
        }
    return None


def get_phase_for_image(img_info: Dict, df: pd.DataFrame) -> Optional[str]:
    """
    Look up the healing phase for an image based on patient/appt/dfu.
    """
    matches = df[(df['Patient#'] == img_info['patient']) &
                 (df['Appt#'] == img_info['appt']) &
                 (df['DFU#'] == img_info['dfu'])]
    if len(matches) > 0:
        phase = matches.iloc[0]['Healing Phase Abs']
        if pd.notna(phase) and phase in ['I', 'P', 'R']:
            return phase
    return None


def prepare_diffusion_data(
    data_root: str,
    output_dir: str,
    modalities: List[str] = ['Depth_RGB'],
    use_symlinks: bool = True
) -> Dict[str, Dict[str, int]]:
    """
    Organize images into phase-specific folders for diffusion training.

    Args:
        data_root: Path to raw data directory
        output_dir: Path to output directory for organized data
        modalities: List of modalities to process
        use_symlinks: If True, create symlinks; if False, copy files

    Returns:
        Dictionary with counts per modality and phase
    """
    data_root = Path(data_root)
    output_dir = Path(output_dir)

    # Load CSV
    csv_path = data_root / 'DataMaster_Processed_V12_WithMissing.csv'
    print(f"Loading CSV from {csv_path}")
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    print(f"Loaded {len(df)} records")

    # Track counts
    counts = defaultdict(lambda: defaultdict(int))

    for modality in modalities:
        # Map modality names
        if modality.lower() == 'rgb' or modality == 'Depth_RGB':
            src_folder = 'Depth_RGB'
            out_modality = 'rgb'
        elif modality.lower() == 'depth_map' or modality == 'Depth_Map_IMG':
            src_folder = 'Depth_Map_IMG'
            out_modality = 'depth_map'
        elif modality.lower() == 'thermal_rgb' or modality == 'Thermal_RGB':
            src_folder = 'Thermal_RGB'
            out_modality = 'thermal_rgb'
        elif modality.lower() == 'thermal_map' or modality == 'Thermal_Map_IMG':
            src_folder = 'Thermal_Map_IMG'
            out_modality = 'thermal_map'
        else:
            src_folder = modality
            out_modality = modality.lower()

        src_dir = data_root / src_folder
        if not src_dir.exists():
            print(f"Warning: Source directory not found: {src_dir}")
            continue

        print(f"\nProcessing {src_folder} -> {out_modality}")

        # Get all images
        images = list(src_dir.glob('*.png'))
        print(f"Found {len(images)} images")

        # Create output directories for each phase
        for phase in ['I', 'P', 'R']:
            phase_dir = output_dir / out_modality / phase
            phase_dir.mkdir(parents=True, exist_ok=True)

        # Process each image
        skipped = 0
        for img_path in images:
            info = parse_filename(img_path.name)
            if info is None:
                skipped += 1
                continue

            phase = get_phase_for_image(info, df)
            if phase is None:
                skipped += 1
                continue

            # Create output path
            out_path = output_dir / out_modality / phase / img_path.name

            # Skip if already exists
            if out_path.exists():
                counts[out_modality][phase] += 1
                continue

            # Create symlink or copy
            if use_symlinks:
                # Remove existing symlink if broken
                if out_path.is_symlink():
                    out_path.unlink()
                out_path.symlink_to(img_path.resolve())
            else:
                shutil.copy2(img_path, out_path)

            counts[out_modality][phase] += 1

        print(f"Skipped {skipped} images (could not parse or match to CSV)")

    return counts


def main():
    parser = argparse.ArgumentParser(description='Prepare DFU images for diffusion training')
    parser.add_argument('--data_root', type=str,
                        default='/workspace/DFUMultiClassification/data/raw',
                        help='Path to raw data directory')
    parser.add_argument('--output_dir', type=str,
                        default='/workspace/DFUMultiClassification/data/diffusion',
                        help='Path to output directory')
    parser.add_argument('--modalities', type=str, nargs='+',
                        default=['Depth_RGB'],
                        help='Modalities to process')
    parser.add_argument('--use_symlinks', action='store_true', default=True,
                        help='Use symlinks instead of copying files')
    parser.add_argument('--copy', action='store_true',
                        help='Copy files instead of symlinking')

    args = parser.parse_args()

    use_symlinks = not args.copy

    print("=" * 60)
    print("DFU Diffusion Data Preparation")
    print("=" * 60)
    print(f"Data root: {args.data_root}")
    print(f"Output dir: {args.output_dir}")
    print(f"Modalities: {args.modalities}")
    print(f"Mode: {'symlinks' if use_symlinks else 'copy'}")
    print("=" * 60)

    counts = prepare_diffusion_data(
        data_root=args.data_root,
        output_dir=args.output_dir,
        modalities=args.modalities,
        use_symlinks=use_symlinks
    )

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    total = 0
    for modality, phase_counts in counts.items():
        print(f"\n{modality}:")
        for phase in ['I', 'P', 'R']:
            count = phase_counts[phase]
            print(f"  {phase}: {count} images")
            total += count

    print(f"\nTotal images organized: {total}")
    print("=" * 60)


if __name__ == '__main__':
    main()
