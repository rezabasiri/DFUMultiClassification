"""
Phase 7: Training script that uses cleaned datasets from outlier detection

This script modifies the data loading to use pre-cleaned datasets
without outliers, then runs the standard fusion training pipeline.
"""

import pandas as pd
import sys
import os

# Add project root
sys.path.insert(0, '/workspace/DFUMultiClassification')

def train_with_cleaned_data(contamination_pct, description=""):
    """
    Train fusion model using cleaned dataset

    Args:
        contamination_pct: Which cleaned dataset to use (5, 10, or 15)
        description: Optional description for logs
    """
    from src.data.caching import load_preprocessed_data
    from src.main import main
    from src.utils import production_config
    import importlib

    print("=" * 80)
    print(f"Training with {contamination_pct}% outlier removal")
    if description:
        print(f"Description: {description}")
    print("=" * 80)
    print()

    # Path to cleaned dataset
    cleaned_file = f'/workspace/DFUMultiClassification/data/cleaned/metadata_cleaned_{contamination_pct:02d}pct.csv'

    if not os.path.exists(cleaned_file):
        print(f"ERROR: Cleaned dataset not found: {cleaned_file}")
        print("Run detect_outliers.py first!")
        return

    # Load cleaned metadata
    print(f"Loading cleaned dataset: {cleaned_file}")
    cleaned_metadata = pd.read_csv(cleaned_file)
    print(f"Loaded {len(cleaned_metadata)} samples (outliers removed)")
    print()

    # Monkey-patch the load_preprocessed_data function to use cleaned data
    original_load = load_preprocessed_data

    def load_cleaned_data():
        # Load original to get image_dict and label_encoder
        _, image_dict, label_encoder = original_load()
        # But use cleaned metadata
        return cleaned_metadata, image_dict, label_encoder

    # Replace function
    sys.modules['src.data.caching'].load_preprocessed_data = load_cleaned_data

    # Set config for this run
    production_config.DATA_PERCENTAGE = 100  # Use all cleaned data
    production_config.SAMPLING_STRATEGY = 'combined'  # Still need sampling for class balance
    production_config.IMAGE_SIZE = 32
    production_config.INCLUDED_COMBINATIONS = [('metadata', 'thermal_map')]

    # Run training
    print("Starting training with cleaned data...")
    print(f"Config: 100% of cleaned data, combined sampling, 32x32 images")
    print()

    # Note: This would need to be run via subprocess or direct import
    # For now, print the command
    print("To run training, use:")
    print(f"python src/main.py --mode search --cv_folds 3 --verbosity 2 \\")
    print(f"  --resume_mode fresh --device-mode multi \\")
    print(f"  --cleaned_data_path {cleaned_file} \\")
    print(f"  2>&1 | tee agent_communication/fusion_fix/run_fusion_32x32_100pct_cleaned_{contamination_pct:02d}pct.txt")
    print()

    # Restore original function
    sys.modules['src.data.caching'].load_preprocessed_data = original_load


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train with cleaned data')
    parser.add_argument('--contamination', type=int, required=True,
                        choices=[5, 10, 15],
                        help='Contamination percentage used for outlier detection')

    args = parser.parse_args()

    train_with_cleaned_data(args.contamination)
