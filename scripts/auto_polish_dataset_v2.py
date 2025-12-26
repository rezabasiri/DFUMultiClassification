"""
Two-Phase Intelligent Dataset Polishing via Bayesian Threshold Optimization

This script uses a smart two-phase approach to find optimal misclassification
filtering thresholds:

PHASE 1: Misclassification Detection (Run Once)
- Tests each modality individually (default: metadata, depth_rgb, depth_map, thermal_map)
- Runs training N times per modality (e.g., 10) with different random seeds
- Accumulates misclassification counts across all modality runs (max count = N * num_modalities)
- Creates comprehensive misclassification profile
- Time: ~30-60 minutes for N=10 runs per modality (e.g., 40 runs total for 4 modalities)

PHASE 2: Bayesian Threshold Optimization
- Uses Bayesian optimization to find optimal thresholds
- Search space: percentage-based (P: 30-70%, I: 50-90%, R: 80-100% of N)
- For each candidate threshold combination:
  * Filters dataset based on thresholds
  * Trains metadata with cv_folds=3, n_runs=1
  * Evaluates combined score: 0.4×macro_f1 + 0.4×min_f1 + 0.2×kappa
- Finds best thresholds in ~20 evaluations
- Time: ~1-2 hours for 20 evaluations with cv_folds=3

Key Advantages:
- Much fewer total runs than iterative approach (30 vs 50+)
- Systematically explores threshold space instead of arbitrary reduction
- Balances speed (Phase 1: n_runs=10) with precision (Phase 2: Bayesian)
- Safety constraint: rejects thresholds that filter >50% of data

Usage:
    # Run both phases with metadata-only evaluation in Phase 2
    python scripts/auto_polish_dataset_v2.py --phase2-modalities metadata

    # Run both phases with all modalities combined in Phase 2
    python scripts/auto_polish_dataset_v2.py --phase2-modalities metadata+depth_rgb+depth_map+thermal_map

    # Run Phase 1 with 100 runs per modality (400 total for 4 modalities)
    python scripts/auto_polish_dataset_v2.py --phase2-modalities metadata --phase1-only --phase1-n-runs 100

    # Run Phase 1 with custom modalities (e.g., only metadata and depth_rgb)
    python scripts/auto_polish_dataset_v2.py --phase2-modalities metadata --phase1-only --phase1-modalities metadata depth_rgb

    # Just Phase 2 (if Phase 1 already completed) with combined modalities
    python scripts/auto_polish_dataset_v2.py --phase2-modalities metadata+depth_rgb --phase2-only

    # Custom optimization budget
    python scripts/auto_polish_dataset_v2.py --phase2-modalities metadata --n-evaluations 30

GPU Configuration:
    # Use all available GPUs (multi-GPU mode with MirroredStrategy)
    python scripts/auto_polish_dataset_v2.py --phase2-modalities metadata --device-mode multi

    # Use specific GPUs (custom mode)
    python scripts/auto_polish_dataset_v2.py --phase2-modalities metadata --device-mode custom --custom-gpus 0 1

    # Single GPU (auto-select best, default)
    python scripts/auto_polish_dataset_v2.py --phase2-modalities metadata --device-mode single

    # CPU only (no GPUs)
    python scripts/auto_polish_dataset_v2.py --phase2-modalities metadata --device-mode cpu
"""

import argparse
import os
import sys
import json
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import get_project_paths, cleanup_for_resume_mode
from src.utils.gpu_config import setup_device_strategy


class BayesianDatasetPolisher:
    """Two-phase dataset polishing with Bayesian threshold optimization."""

    def __init__(self,
                 modalities,
                 min_f1_per_class=0.42,
                 min_macro_f1=0.48,
                 min_kappa=0.35,
                 phase1_n_runs=10,
                 phase1_cv_folds=1,
                 phase1_modalities=None,
                 phase1_data_percentage=100,
                 phase2_cv_folds=3,
                 phase2_n_evaluations=20,
                 phase2_data_percentage=100,
                 base_random_seed=42,
                 min_dataset_fraction=0.5,
                 min_f1_threshold=0.25,
                 min_samples_per_class=30,
                 max_class_imbalance_ratio=5.0,
                 min_retention_per_class=0.90,
                 device_mode='single',
                 custom_gpus=None,
                 min_gpu_memory=8.0,
                 include_display_gpus=False):
        """
        Initialize the Bayesian dataset polisher.

        Args:
            modalities: List of modalities to train in Phase 2 (e.g., ['metadata'], ['depth_rgb', 'depth_map'])
            min_f1_per_class: Minimum F1 score for each class
            min_macro_f1: Minimum macro F1 score
            min_kappa: Minimum Cohen's Kappa
            phase1_n_runs: Number of runs per modality in Phase 1 for misclass detection (default: 10)
            phase1_cv_folds: CV folds in Phase 1 (default: 1 for speed)
            phase1_modalities: List of modalities to test individually in Phase 1 (default: ['metadata', 'depth_rgb', 'depth_map', 'thermal_map'])
            phase1_data_percentage: Percentage of data to use in Phase 1 (default: 100)
            phase2_cv_folds: CV folds in Phase 2 for evaluation (default: 3)
            phase2_n_evaluations: Number of Bayesian optimization iterations (default: 20)
            phase2_data_percentage: Percentage of data to use in Phase 2 (default: 100)
            base_random_seed: Base random seed
            min_dataset_fraction: Minimum fraction of dataset to keep (default: 0.5)
            min_f1_threshold: Hard constraint - reject if any class F1 < this (default: 0.25)
            min_samples_per_class: Hard constraint - reject if any class < this many samples (default: 30)
            max_class_imbalance_ratio: Hard constraint - reject if largest/smallest class ratio > this (default: 5.0)
            min_retention_per_class: Minimum fraction of samples to retain per class (default: 0.90)
                                     Optimization skipped if this cannot be achieved.
            device_mode: GPU mode - 'cpu', 'single', 'multi', or 'custom' (default: 'single')
            custom_gpus: List of GPU IDs for 'custom' mode (e.g., [0, 1])
            min_gpu_memory: Minimum GPU memory in GB (default: 8.0)
            include_display_gpus: Allow training on display GPUs (default: False)
        """
        if 'metadata' not in modalities:
            raise ValueError("Modalities must include 'metadata' for polishing")

        self.modalities = modalities
        self.min_f1_per_class = min_f1_per_class
        self.min_macro_f1 = min_macro_f1
        self.min_kappa = min_kappa
        self.phase1_n_runs = phase1_n_runs
        self.phase1_cv_folds = phase1_cv_folds
        self.phase1_modalities = phase1_modalities if phase1_modalities is not None else ['metadata', 'depth_rgb', 'depth_map', 'thermal_map']
        self.phase1_data_percentage = phase1_data_percentage
        self.phase2_cv_folds = phase2_cv_folds
        self.phase2_n_evaluations = phase2_n_evaluations
        self.phase2_data_percentage = phase2_data_percentage
        self.base_random_seed = base_random_seed
        self.min_dataset_fraction = min_dataset_fraction
        self.min_f1_threshold = min_f1_threshold
        self.min_samples_per_class = min_samples_per_class
        self.max_class_imbalance_ratio = max_class_imbalance_ratio
        self.min_retention_per_class = min_retention_per_class

        # GPU configuration
        self.device_mode = device_mode
        self.custom_gpus = custom_gpus
        self.min_gpu_memory = min_gpu_memory
        self.include_display_gpus = include_display_gpus

        # Get project paths
        self.directory, self.result_dir, self.root = get_project_paths()

        # Tracking
        self.original_dataset_size = None
        self.optimization_history = []
        self.best_thresholds = None
        self.best_score = -np.inf

    def _calculate_phase2_batch_size(self):
        """
        Calculate appropriate batch size for Phase 2 based on number of image modalities.

        Phase 1 tests ONE modality at a time with GLOBAL_BATCH_SIZE.
        Phase 2 tests MULTIPLE modalities simultaneously, requiring proportional reduction.

        Returns:
            int: Adjusted batch size for Phase 2
        """
        from src.utils.production_config import GLOBAL_BATCH_SIZE

        # Count image modalities (exclude metadata which is small)
        image_modalities = [m for m in self.modalities if m != 'metadata']
        num_image_modalities = len(image_modalities)

        if num_image_modalities == 0:
            # Metadata only - use full batch size
            return GLOBAL_BATCH_SIZE

        # Divide batch size by number of image modalities to maintain similar memory usage
        adjusted_batch_size = max(16, GLOBAL_BATCH_SIZE // num_image_modalities)

        print(f"\n{'='*80}")
        print("BATCH SIZE ADJUSTMENT FOR PHASE 2")
        print(f"{'='*80}")
        print(f"Phase 1 batch size (1 modality at a time): {GLOBAL_BATCH_SIZE}")
        print(f"Phase 2 modalities: {self.modalities}")
        print(f"  Image modalities: {image_modalities} (count: {num_image_modalities})")
        print(f"  Adjusted batch size: {GLOBAL_BATCH_SIZE} / {num_image_modalities} = {adjusted_batch_size}")
        print(f"  Memory reduction: ~{num_image_modalities}x less per batch")
        print(f"{'='*80}\n")

        return adjusted_batch_size

    def _find_misclass_file(self):
        """
        Find the misclassification file, checking multiple locations.
        Returns the path if found, None otherwise.
        """
        from src.utils.config import get_output_paths
        output_paths = get_output_paths(self.result_dir)

        # Possible locations for the misclass file
        possible_paths = [
            os.path.join(output_paths['misclassifications'], 'frequent_misclassifications_saved.csv'),
            os.path.join(self.result_dir, 'misclassifications_saved', 'frequent_misclassifications_saved.csv'),
            os.path.join(self.result_dir, 'misclassifications', 'frequent_misclassifications_saved.csv'),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        return None

    def calculate_min_thresholds_for_retention(self, target_retention=0.90):
        """
        Calculate the minimum threshold for each class to retain at least target_retention
        fraction of samples.

        For example, with target_retention=0.90, finds the threshold that keeps 90% of
        each class's samples.

        Args:
            target_retention: Fraction of samples to retain per class (0.0-1.0)

        Returns:
            dict: Minimum thresholds like {'I': 52, 'P': 55, 'R': 48} or None if impossible
            dict: Retention info with stats for each class
        """
        misclass_file = self._find_misclass_file()
        if misclass_file is None:
            print("⚠️  No misclassification file found, cannot calculate auto thresholds")
            return None, None

        # Load misclassification data
        df = pd.read_csv(misclass_file)

        # Get original counts from best_matching.csv
        best_matching_file = os.path.join(self.result_dir, 'best_matching.csv')
        if not os.path.exists(best_matching_file):
            print("⚠️  best_matching.csv not found, cannot calculate auto thresholds")
            return None, None

        best_matching = pd.read_csv(best_matching_file)
        best_matching['Sample_ID'] = (
            'P' + best_matching['Patient#'].astype(str).str.zfill(3) +
            'A' + best_matching['Appt#'].astype(str).str.zfill(2) +
            'D' + best_matching['DFU#'].astype(str)
        )
        original_counts = best_matching.groupby('Healing Phase Abs')['Sample_ID'].nunique().to_dict()

        # Get max misclass count per sample (handling duplicates)
        max_misclass = df.groupby(['Sample_ID', 'True_Label'])['Misclass_Count'].max().reset_index()

        min_thresholds = {}
        retention_info = {}

        for phase in ['I', 'P', 'R']:
            phase_data = max_misclass[max_misclass['True_Label'] == phase]
            original_count = original_counts.get(phase, 0)

            if original_count == 0 or len(phase_data) == 0:
                min_thresholds[phase] = 1  # Default to 1 if no data
                retention_info[phase] = {
                    'original': 0,
                    'min_threshold': 1,
                    'retained': 0,
                    'retention_pct': 0.0
                }
                continue

            counts = phase_data['Misclass_Count'].values
            min_to_keep = int(np.ceil(original_count * target_retention))

            # Find the minimum threshold that keeps at least min_to_keep samples
            # Threshold T keeps samples where count < T, so we need: original - (count >= T) >= min_to_keep
            # Equivalently: (count >= T) <= original - min_to_keep = max_to_exclude
            max_to_exclude = original_count - min_to_keep

            # Sort counts descending - the first max_to_exclude samples can be excluded
            sorted_counts = np.sort(counts)[::-1]

            if max_to_exclude <= 0:
                # Cannot exclude any samples - need threshold higher than max count
                min_threshold = int(counts.max()) + 1
            elif max_to_exclude >= len(sorted_counts):
                # Can exclude all samples - threshold of 1 would work but let's be reasonable
                min_threshold = int(counts.min())
            else:
                # The threshold should be: sorted_counts[max_to_exclude - 1] + 1
                # This excludes exactly max_to_exclude samples (those with highest counts)
                # But we want AT LEAST target_retention, so threshold >= that value
                min_threshold = int(sorted_counts[max_to_exclude - 1]) + 1

            # Calculate actual retention at this threshold
            retained = np.sum(counts < min_threshold)
            retention_pct = retained / original_count * 100 if original_count > 0 else 0

            min_thresholds[phase] = min_threshold
            retention_info[phase] = {
                'original': original_count,
                'min_threshold': min_threshold,
                'retained': retained,
                'retention_pct': retention_pct,
                'count_range': (int(counts.min()), int(counts.max())),
                'count_median': float(np.median(counts))
            }

        return min_thresholds, retention_info

    def get_class_counts(self, thresholds):
        """
        Get class counts after applying filtering thresholds.

        FIXED: Now correctly calculates by:
        1. Getting original class distribution from best_matching.csv
        2. Subtracting samples that would be excluded (Misclass_Count >= threshold)
        3. Properly handling duplicate Sample_ID entries by using max count per sample

        Args:
            thresholds: Dict like {'I': 5, 'P': 3, 'R': 8}

        Returns:
            dict: Class counts like {'I': 120, 'P': 180, 'R': 50}
        """
        # Find misclass file (checks multiple locations)
        misclass_file = self._find_misclass_file()

        # Step 1: Get original class distribution from best_matching.csv
        best_matching_file = os.path.join(self.result_dir, 'best_matching.csv')
        if not os.path.exists(best_matching_file):
            print(f"⚠️  best_matching.csv not found, returning zeros")
            return {'I': 0, 'P': 0, 'R': 0}

        best_matching = pd.read_csv(best_matching_file)
        best_matching['Sample_ID'] = (
            'P' + best_matching['Patient#'].astype(str).str.zfill(3) +
            'A' + best_matching['Appt#'].astype(str).str.zfill(2) +
            'D' + best_matching['DFU#'].astype(str)
        )

        # Original unique samples per class
        original_counts = best_matching.groupby('Healing Phase Abs')['Sample_ID'].nunique().to_dict()

        if misclass_file is None:
            # No filtering - return original counts
            return {
                'I': original_counts.get('I', 0),
                'P': original_counts.get('P', 0),
                'R': original_counts.get('R', 0)
            }

        # Step 2: Load misclassification data and get max count per Sample_ID
        df = pd.read_csv(misclass_file)

        # Group by Sample_ID and True_Label to get max misclass count
        # (same sample can have multiple entries for different predicted labels)
        max_misclass = df.groupby(['Sample_ID', 'True_Label'])['Misclass_Count'].max().reset_index()

        # Step 3: Count excluded samples per class
        excluded_per_class = {}
        for phase in ['I', 'P', 'R']:
            threshold = thresholds.get(phase, float('inf'))
            excluded = max_misclass[
                (max_misclass['True_Label'] == phase) &
                (max_misclass['Misclass_Count'] >= threshold)
            ]['Sample_ID'].nunique()
            excluded_per_class[phase] = excluded

        # Step 4: Calculate remaining = original - excluded
        remaining_counts = {
            'I': max(0, original_counts.get('I', 0) - excluded_per_class.get('I', 0)),
            'P': max(0, original_counts.get('P', 0) - excluded_per_class.get('P', 0)),
            'R': max(0, original_counts.get('R', 0) - excluded_per_class.get('R', 0))
        }

        return remaining_counts

    def calculate_constraint_penalties(self, thresholds, metrics, filtered_size=None):
        """
        Calculate smooth constraint penalties (integrated into optimization score).

        Uses exponential penalties that increase smoothly as constraints are violated.
        This gives the Bayesian optimizer gradient information to learn from.

        Args:
            thresholds: Dict like {'I': 5, 'P': 3, 'R': 8}
            metrics: Dict with performance metrics
            filtered_size: Number of samples after filtering (optional, for dataset size penalty)

        Returns:
            dict: Penalties for each constraint (0 = satisfied, >0 = violated)
        """
        penalties = {}

        # Penalty 0: Dataset size (exponential penalty below threshold)
        if filtered_size is not None and self.original_dataset_size is not None:
            min_size = int(self.original_dataset_size * self.min_dataset_fraction)
            if filtered_size < min_size:
                # Heavy exponential penalty for dataset too small
                violation = (min_size - filtered_size) / min_size  # 0-1 range
                penalties['dataset_size'] = 5 * (np.exp(3 * violation) - 1)  # Heavy penalty
            else:
                penalties['dataset_size'] = 0.0
        else:
            penalties['dataset_size'] = 0.0

        # Penalty 1: Minimum F1 per class (log-barrier + exponential penalty)
        # Uses log-barrier that approaches infinity as any F1 approaches zero
        # This provides smooth gradients for Bayesian optimization
        min_f1 = min(metrics['f1_per_class'].values())

        # Log-barrier penalty: smooth increase as F1 decreases toward zero
        # Formula: -k * log(f1 + eps) where k controls steepness
        # Designed to be ~0 for F1 > 0.5, then increase smoothly toward infinity at F1=0
        epsilon = 0.01  # Small constant to avoid log(0)
        # Scale factor: at F1=0.5, penalty ≈ 0; at F1=0.1, penalty ≈ 1; at F1=0, penalty → ∞
        k = 0.3  # Controls steepness
        log_barrier = k * max(0, -np.log((min_f1 + epsilon) / (0.5 + epsilon)))

        # Exponential component for threshold violation (kicks in below min_f1_threshold)
        if min_f1 < self.min_f1_threshold:
            violation = self.min_f1_threshold - min_f1
            exp_penalty = np.exp(5 * violation) - 1
        else:
            exp_penalty = 0.0

        # Combined: log-barrier provides smooth gradient toward zero, exp adds extra push below threshold
        penalties['min_f1'] = log_barrier + exp_penalty

        # Penalty 2 & 3: Class counts and imbalance
        class_counts = self.get_class_counts(thresholds)
        counts = np.array(list(class_counts.values()))

        # Empty class - extreme penalty
        if np.any(counts == 0):
            penalties['empty_class'] = 10.0  # Very high but still distinguishable
        else:
            penalties['empty_class'] = 0.0

            # Minimum samples per class (smooth penalty below threshold)
            min_samples = min(counts)
            if min_samples < self.min_samples_per_class:
                violation = (self.min_samples_per_class - min_samples) / self.min_samples_per_class
                penalties['min_samples'] = 2 * violation  # Linear penalty, scaled
            else:
                penalties['min_samples'] = 0.0

            # Maximum class imbalance ratio (smooth penalty above threshold)
            ratio = max(counts) / min(counts)
            if ratio > self.max_class_imbalance_ratio:
                violation = (ratio - self.max_class_imbalance_ratio) / self.max_class_imbalance_ratio
                penalties['imbalance'] = 2 * violation  # Linear penalty, scaled
            else:
                penalties['imbalance'] = 0.0

        return penalties

    def check_hard_constraints(self, thresholds, metrics):
        """
        Check EXTREME violations only (catastrophic failures).

        Most constraints are now soft penalties integrated into the score.
        This only catches truly unusable solutions (e.g., empty dataset).

        Args:
            thresholds: Dict like {'I': 5, 'P': 3, 'R': 8}
            metrics: Dict with performance metrics

        Returns:
            tuple: (is_valid, rejection_reason)
        """
        # Only reject if completely catastrophic (empty dataset)
        class_counts = self.get_class_counts(thresholds)
        counts = np.array(list(class_counts.values()))

        if np.sum(counts) == 0:
            return False, "No samples remaining after filtering"

        return True, None

    def get_class_balance_score(self, thresholds):
        """
        Calculate class balance score (0-1) for filtered dataset.

        FIXED: Now uses the corrected get_class_counts() method.

        Higher score = better balance between classes.
        Uses coefficient of variation (CV) of class sizes.

        Returns:
            float: Balance score (1.0 = perfect balance, 0.0 = extreme imbalance)
        """
        # Use the corrected get_class_counts method
        class_counts = self.get_class_counts(thresholds)
        counts = np.array([class_counts.get(c, 0) for c in ['I', 'P', 'R']])

        # If any class is empty, return 0
        if np.any(counts == 0):
            return 0.0

        # If total is 0, return 0
        if np.sum(counts) == 0:
            return 0.0

        # Calculate coefficient of variation (lower = better balance)
        mean_count = np.mean(counts)
        std_count = np.std(counts)
        cv = std_count / mean_count if mean_count > 0 else 0

        # Convert to score (0-1), where 1 = perfect balance
        # CV of 0 = perfect balance (score 1.0)
        # CV of 1 = high imbalance (score ~0.4)
        # CV > 2 = extreme imbalance (score ~0)
        balance_score = 1.0 / (1.0 + cv)

        return balance_score

    def calculate_combined_score(self, metrics, thresholds, filtered_size=None):
        """
        Calculate enhanced combined optimization score with soft constraint penalties.

        Formula: base_score - total_penalty

        Base Score: 0.3×macro_f1 + 0.5×min_per_class_f1 + 0.1×kappa + 0.1×balance_score

        Penalties (smooth, integrated for Bayesian optimization):
        - dataset_size_penalty: Heavy exponential when filtered dataset < 50% of original
        - min_f1_penalty: Exponential penalty when any class F1 < 0.25
        - min_samples_penalty: Linear penalty when any class < 30 samples
        - imbalance_penalty: Linear penalty when class ratio > 5.0
        - empty_class_penalty: Large penalty when any class has 0 samples

        Args:
            metrics: Dict with 'macro_f1', 'f1_per_class', 'kappa'
            thresholds: Dict with threshold values
            filtered_size: Number of samples after filtering (for dataset size penalty)

        Returns:
            tuple: (final_score, penalties_dict)
        """
        # Base performance score
        macro_f1 = metrics['macro_f1']
        min_f1 = min(metrics['f1_per_class'].values())
        kappa = metrics['kappa']
        balance_score = self.get_class_balance_score(thresholds)

        base_score = 0.3 * macro_f1 + 0.5 * min_f1 + 0.1 * kappa + 0.1 * balance_score

        # Calculate constraint penalties (smooth, differentiable)
        penalties = self.calculate_constraint_penalties(thresholds, metrics, filtered_size)
        total_penalty = sum(penalties.values())

        # Final score = base performance - penalties
        final_score = base_score - total_penalty

        return final_score, penalties

    def get_original_dataset_size(self):
        """Get original dataset size before any filtering."""
        # Try multiple sources to get dataset size

        # Option 1: Read from balanced CSV if it exists
        data_path = os.path.join(self.directory, 'balanced_combined_healing_phases.csv')
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            return len(df)

        # Option 2: Get from misclassification CSV (count unique Sample_IDs)
        from src.utils.config import get_output_paths
        output_paths = get_output_paths(self.result_dir)
        misclass_file = os.path.join(output_paths['misclassifications'], 'frequent_misclassifications_total.csv')
        if os.path.exists(misclass_file):
            df = pd.read_csv(misclass_file)
            unique_samples = df['Sample_ID'].nunique()
            return unique_samples

        # Option 3: Read from raw data
        raw_data_path = os.path.join(self.directory, 'data', 'raw', 'DataMaster_Processed_V12_WithMissing.csv')
        if os.path.exists(raw_data_path):
            df = pd.read_csv(raw_data_path)
            return len(df)

        return None

    def get_filtered_dataset_size(self, thresholds):
        """
        Estimate how many samples would remain after filtering.

        FIXED: Uses _find_misclass_file() helper and handles duplicate Sample_IDs correctly.

        Args:
            thresholds: Dict like {'I': 5, 'P': 3, 'R': 8}

        Returns:
            Number of samples that would remain
        """
        # Find misclass file (checks multiple locations)
        misclass_file = self._find_misclass_file()

        if misclass_file is None:
            print(f"⚠️  Misclassification file not found in any location")
            return self.original_dataset_size

        df = pd.read_csv(misclass_file)

        # If original_dataset_size not set, get it from best_matching.csv
        if self.original_dataset_size is None:
            best_matching_file = os.path.join(self.result_dir, 'best_matching.csv')
            if os.path.exists(best_matching_file):
                best_matching = pd.read_csv(best_matching_file)
                best_matching['Sample_ID'] = (
                    'P' + best_matching['Patient#'].astype(str).str.zfill(3) +
                    'A' + best_matching['Appt#'].astype(str).str.zfill(2) +
                    'D' + best_matching['DFU#'].astype(str)
                )
                self.original_dataset_size = best_matching['Sample_ID'].nunique()
            else:
                self.original_dataset_size = df['Sample_ID'].nunique()
            print(f"ℹ️  Inferred dataset size: {self.original_dataset_size} samples")

        # Group by Sample_ID and True_Label to get max misclass count
        # (same sample can have multiple entries for different predicted labels)
        max_misclass = df.groupby(['Sample_ID', 'True_Label'])['Misclass_Count'].max().reset_index()

        # Count samples that would be excluded
        excluded_samples = set()
        for phase, threshold in thresholds.items():
            high_misclass = max_misclass[
                (max_misclass['True_Label'] == phase) &
                (max_misclass['Misclass_Count'] >= threshold)
            ]['Sample_ID'].unique().tolist()
            excluded_samples.update(high_misclass)

        excluded_count = len(excluded_samples)
        remaining_count = self.original_dataset_size - excluded_count

        return remaining_count

    def phase1_detect_misclassifications(self):
        """
        Phase 1: Run training multiple times to accumulate misclassifications.

        Returns:
            bool: Success status
        """
        print("\n" + "="*70)
        print("PHASE 1: MISCLASSIFICATION DETECTION")
        print("="*70)

        # Get original dataset size
        self.original_dataset_size = self.get_original_dataset_size()
        if self.original_dataset_size:
            min_samples = int(self.original_dataset_size * self.min_dataset_fraction)
            print(f"Dataset: {self.original_dataset_size} samples (min after filtering: {min_samples})")

        total_runs = self.phase1_n_runs * len(self.phase1_modalities)
        print(f"Testing {len(self.phase1_modalities)} modalities individually: {self.phase1_modalities}")
        print(f"Running {self.phase1_n_runs} runs per modality (total {total_runs} runs)")
        print(f"Misclassification counts will be out of {total_runs}")
        print(f"CV folds={self.phase1_cv_folds}, data={self.phase1_data_percentage}%, verbosity=silent\n")

        # Clean up everything for fresh start
        cleanup_for_resume_mode('fresh')

        # Clear misclassifications directory for fresh start
        from src.utils.config import get_output_paths
        import shutil
        output_paths = get_output_paths(self.result_dir)
        misclass_dir = output_paths['misclassifications']
        if os.path.exists(misclass_dir):
            shutil.rmtree(misclass_dir)
        os.makedirs(misclass_dir, exist_ok=True)

        # Temporarily override INCLUDED_COMBINATIONS
        config_path = project_root / 'src' / 'utils' / 'production_config.py'
        backup_path = project_root / 'src' / 'utils' / 'production_config.py.backup'

        with open(config_path, 'r') as f:
            original_config = f.read()
        with open(backup_path, 'w') as f:
            f.write(original_config)

        try:
            import re
            # Loop through each modality individually
            total_runs = self.phase1_n_runs * len(self.phase1_modalities)
            run_counter = 0

            for modality_name in self.phase1_modalities:
                print(f"\n{'='*70}")
                print(f"Testing modality: {modality_name} ({self.phase1_n_runs} runs)")
                print(f"{'='*70}")

                # Update config for this specific modality
                modified_config = re.sub(
                    r'INCLUDED_COMBINATIONS\s*=\s*\[[\s\S]*?\n\]',
                    f"INCLUDED_COMBINATIONS = [\n    ('{modality_name}',),  # Temporary: Phase 1 detection\n]",
                    original_config
                )
                with open(config_path, 'w') as f:
                    f.write(modified_config)

                # Run multiple times with different seeds for this modality
                for run_idx in tqdm(range(1, self.phase1_n_runs + 1),
                                   desc=f"Phase 1 Progress ({modality_name})",
                                   unit="run",
                                   total=self.phase1_n_runs,
                                   position=0,
                                   leave=True):
                    run_counter += 1
                    # Clean up predictions/models/patient splits from previous run
                    # We MUST delete patient splits to force regeneration with new random seed
                    if run_idx > 1:
                        import glob
                        from src.utils.config import get_output_paths
                        output_paths = get_output_paths(self.result_dir)

                        # Delete everything except misclassification CSV (which accumulates)
                        patterns = [
                            os.path.join(output_paths['checkpoints'], '*predictions*.npy'),
                            os.path.join(output_paths['checkpoints'], '*pred*.npy'),
                            os.path.join(output_paths['checkpoints'], '*label*.npy'),
                            os.path.join(output_paths['checkpoints'], 'patient_split_*.npz'),  # DELETE to force new splits!
                            os.path.join(output_paths['models'], '*.h5'),
                        ]
                        for pattern in patterns:
                            for file_path in glob.glob(pattern):
                                try:
                                    os.remove(file_path)
                                except Exception:
                                    pass

                    # Set random seed via environment variable (use run_counter for unique seeds)
                    os.environ['CROSS_VAL_RANDOM_SEED'] = str(self.base_random_seed + run_counter)

                    # ALWAYS use fresh mode for Phase 1 runs - ensures clean training each time
                    cmd = [
                        sys.executable, 'src/main.py',
                        '--mode', 'search',
                        '--cv_folds', str(self.phase1_cv_folds),
                        '--data_percentage', str(self.phase1_data_percentage),
                        '--verbosity', '0',
                        '--resume_mode', 'fresh',  # Force fresh training for each run
                        '--device-mode', self.device_mode,
                        '--min-gpu-memory', str(self.min_gpu_memory)
                    ]

                    # Add custom GPU IDs if specified
                    if self.device_mode == 'custom' and self.custom_gpus:
                        cmd.extend(['--custom-gpus'] + [str(gpu) for gpu in self.custom_gpus])

                    # Add display GPU flag if enabled
                    if self.include_display_gpus:
                        cmd.append('--include-display-gpus')

                    # Use os.system to avoid TensorFlow context conflicts (same fix as Phase 2)
                    # Save current directory
                    original_cwd = os.getcwd()
                    try:
                        os.chdir(project_root)
                        cmd_str = ' '.join(str(arg) for arg in cmd)
                        return_code = os.system(f"{cmd_str} >/dev/null 2>&1")
                    finally:
                        os.chdir(original_cwd)

                    if return_code != 0:
                        print(f"\n❌ Training failed on {modality_name} run {run_idx}")
                        return False

            # Clear environment variable
            if 'CROSS_VAL_RANDOM_SEED' in os.environ:
                del os.environ['CROSS_VAL_RANDOM_SEED']

            print(f"\n{'='*70}")
            print(f"✅ PHASE 1 COMPLETE")
            print(f"{'='*70}")

            # Save Phase 1 misclassification data (preserve it before Phase 2 starts)
            import shutil
            import glob
            output_paths = get_output_paths(self.result_dir)
            misclass_file = os.path.join(output_paths['misclassifications'], 'frequent_misclassifications_total.csv')
            misclass_saved = os.path.join(output_paths['misclassifications'], 'frequent_misclassifications_saved.csv')

            if os.path.exists(misclass_file):
                shutil.copy2(misclass_file, misclass_saved)
                print(f"\n💾 Saved Phase 1 misclassification data:")
                print(f"   - frequent_misclassifications_saved.csv (total)")

                # Also save per-modality files for later analysis
                misclass_files = glob.glob(os.path.join(output_paths['misclassifications'], 'frequent_misclassifications_*.csv'))
                saved_modality_files = []
                for f in misclass_files:
                    if f.endswith('_saved.csv') or f.endswith('_total.csv'):
                        continue  # Skip the saved and total files
                    # Rename modality file: frequent_misclassifications_metadata.csv -> frequent_misclassifications_metadata_saved.csv
                    base_name = os.path.basename(f)
                    saved_name = base_name.replace('.csv', '_saved.csv')
                    saved_path = os.path.join(output_paths['misclassifications'], saved_name)
                    shutil.copy2(f, saved_path)
                    saved_modality_files.append(saved_name)

                for name in saved_modality_files:
                    print(f"   - {name}")

                # Delete all non-saved misclassification tracking files to prevent Phase 2 from updating them
                # Phase 2 will create new files, but we'll ignore them and use only the saved files
                for f in misclass_files:
                    if not f.endswith('_saved.csv'):  # Keep only the saved files
                        try:
                            os.remove(f)
                        except:
                            pass
                print(f"🔒 Locked Phase 1 data - Phase 2 training won't contaminate counts")

            # Analyze misclassifications
            self.show_misclass_summary()

            return True

        finally:
            # Restore config
            with open(backup_path, 'r') as f:
                original_config = f.read()
            with open(config_path, 'w') as f:
                f.write(original_config)
            backup_path.unlink(missing_ok=True)

    def show_misclass_summary(self):
        """Show summary of accumulated misclassifications."""
        from src.utils.config import get_output_paths
        output_paths = get_output_paths(self.result_dir)

        # Try saved file first (Phase 1 locked data), fallback to total file
        misclass_saved = os.path.join(output_paths['misclassifications'], 'frequent_misclassifications_saved.csv')
        misclass_total = os.path.join(output_paths['misclassifications'], 'frequent_misclassifications_total.csv')

        if os.path.exists(misclass_saved):
            misclass_file = misclass_saved
        elif os.path.exists(misclass_total):
            misclass_file = misclass_total
        else:
            print("\n⚠️  No misclassification file found")
            return

        df = pd.read_csv(misclass_file)

        print(f"\nMisclassification Summary:")
        print(f"  Total unique misclassified samples: {len(df)}")
        print(f"  Max count observed: {df['Misclass_Count'].max()}")

        print(f"\n  By class:")
        for phase in ['I', 'P', 'R']:
            phase_df = df[df['True_Label'] == phase]
            if len(phase_df) > 0:
                print(f"    {phase}: {len(phase_df)} samples, " +
                      f"counts {phase_df['Misclass_Count'].min()}-{phase_df['Misclass_Count'].max()}, " +
                      f"median={phase_df['Misclass_Count'].median():.1f}")

    def phase2_optimize_thresholds(self):
        """
        Phase 2: Use Bayesian optimization to find optimal thresholds.

        Returns:
            tuple: (success, best_thresholds)
        """
        print("\n" + "="*70)
        print("PHASE 2: BAYESIAN THRESHOLD OPTIMIZATION")
        print("="*70)

        # Ensure original dataset size is set (in case Phase 1 was skipped)
        if self.original_dataset_size is None:
            self.original_dataset_size = self.get_original_dataset_size()
            if self.original_dataset_size:
                print(f"\nOriginal dataset size: {self.original_dataset_size} samples")
                min_samples = int(self.original_dataset_size * self.min_dataset_fraction)
                print(f"Minimum dataset size after filtering: {min_samples} samples ({self.min_dataset_fraction*100:.0f}%)")

        # Verify Phase 1 saved misclassification file exists
        from src.utils.config import get_output_paths
        output_paths = get_output_paths(self.result_dir)
        misclass_file = os.path.join(output_paths['misclassifications'], 'frequent_misclassifications_saved.csv')
        if not os.path.exists(misclass_file):
            print(f"\n❌ ERROR: Phase 1 misclassification file not found: {misclass_file}")
            print("   Phase 1 must complete successfully before running Phase 2.")
            print("   Run without --phase2-only to execute both phases.\n")
            return False, None

        # Check if scikit-optimize is available
        try:
            from skopt import gp_minimize
            from skopt.space import Integer
            from skopt.utils import use_named_args
        except ImportError:
            print("\n❌ scikit-optimize not found. Install with: pip install scikit-optimize")
            print("   Falling back to grid search...")
            return self.phase2_grid_search()

        # Calculate automatic threshold bounds based on data retention target
        target_retention = self.min_retention_per_class
        min_thresholds, retention_info = self.calculate_min_thresholds_for_retention(target_retention)

        if min_thresholds is None:
            print(f"\n❌ ERROR: Cannot calculate automatic threshold bounds")
            print("   Misclassification data not available.")
            return False, None

        # Display retention analysis
        print(f"\n{'='*70}")
        print(f"AUTOMATIC THRESHOLD CALCULATION (Target: {target_retention*100:.0f}% retention per class)")
        print(f"{'='*70}")

        can_optimize = True
        for phase in ['I', 'P', 'R']:
            info = retention_info[phase]
            print(f"\n  Class {phase}:")
            print(f"    Original samples: {info['original']}")
            print(f"    Misclass count range: {info['count_range'][0]}-{info['count_range'][1]}, median={info['count_median']:.1f}")
            print(f"    Min threshold for {target_retention*100:.0f}% retention: {info['min_threshold']}")
            print(f"    Actual retention at this threshold: {info['retained']}/{info['original']} ({info['retention_pct']:.1f}%)")

            # Check if threshold exceeds max observed count (meaning we can't retain enough)
            if info['min_threshold'] > info['count_range'][1]:
                print(f"    ⚠️  WARNING: Threshold {info['min_threshold']} exceeds max count {info['count_range'][1]}")
                print(f"       Cannot achieve {target_retention*100:.0f}% retention - all samples have high misclass counts")
                can_optimize = False

        # Check if optimization should be skipped
        if not can_optimize:
            print(f"\n{'='*70}")
            print(f"⚠️  SKIPPING PHASE 2 OPTIMIZATION")
            print(f"{'='*70}")
            print(f"\nReason: Cannot achieve {target_retention*100:.0f}% data retention.")
            print("All samples have been frequently misclassified, suggesting:")
            print("  1. The classification task may be inherently difficult")
            print("  2. The misclassification tracking has accumulated too many counts")
            print("  3. Consider running Phase 1 with fewer runs to reduce count accumulation")
            print("\nRecommendation: Use NO filtering (keep all samples) or manually set thresholds.")

            # Return "success" but with no thresholds (no filtering)
            return True, None

        # Define search space with automatic minimum bounds
        # Minimum = threshold to keep 90% retention
        # Maximum = max observed misclass count + 1 (no filtering at upper bound)
        max_count_overall = max(info['count_range'][1] for info in retention_info.values())
        upper_bound = max_count_overall + 1

        search_space = [
            Integer(min_thresholds['P'], upper_bound, name='threshold_P'),
            Integer(min_thresholds['I'], upper_bound, name='threshold_I'),
            Integer(min_thresholds['R'], upper_bound, name='threshold_R')
        ]

        print(f"\n{'='*70}")
        print(f"SEARCH SPACE (auto-calculated to preserve ≥{target_retention*100:.0f}% per class)")
        print(f"{'='*70}")
        print(f"  P: {search_space[0].low}-{search_space[0].high} (min threshold to keep {target_retention*100:.0f}%)")
        print(f"  I: {search_space[1].low}-{search_space[1].high} (min threshold to keep {target_retention*100:.0f}%)")
        print(f"  R: {search_space[2].low}-{search_space[2].high} (min threshold to keep {target_retention*100:.0f}%)")

        print(f"\nOptimization Settings:")
        print(f"  Evaluations: {self.phase2_n_evaluations}")
        print(f"  CV folds per evaluation: {self.phase2_cv_folds}")
        print(f"  Score: 0.3×macro_f1 + 0.5×min_f1 + 0.1×kappa + 0.1×balance - penalties")
        print(f"\nSoft Constraints (smooth penalties guide optimizer):")
        print(f"  Dataset size: Target ≥{self.min_dataset_fraction*100:.0f}% of original (heavy exp penalty)")
        print(f"  Min F1 per class: Target ≥{self.min_f1_threshold} (exponential penalty)")
        print(f"  Min samples per class: Target ≥{self.min_samples_per_class} (linear penalty)")
        print(f"  Max class imbalance: Target ≤{self.max_class_imbalance_ratio}x (linear penalty)")
        print(f"\nAll constraints are soft - optimizer learns from violations!\n")

        # Objective function
        @use_named_args(search_space)
        def objective(**thresholds):
            """Objective function for Bayesian optimization (to minimize, so we negate)."""
            eval_num = len(self.optimization_history) + 1

            print(f"\n{'─'*70}")
            print(f"EVALUATION {eval_num}/{self.phase2_n_evaluations}")
            print(f"{'─'*70}")
            print(f"Trying thresholds: P={thresholds['threshold_P']}, " +
                  f"I={thresholds['threshold_I']}, R={thresholds['threshold_R']}")

            # Check dataset size constraint
            threshold_dict = {
                'P': thresholds['threshold_P'],
                'I': thresholds['threshold_I'],
                'R': thresholds['threshold_R']
            }

            filtered_size = self.get_filtered_dataset_size(threshold_dict)
            min_size = int(self.original_dataset_size * self.min_dataset_fraction)

            print(f"Dataset after filtering: {filtered_size}/{self.original_dataset_size} samples " +
                  f"({filtered_size/self.original_dataset_size*100:.1f}%)")

            # No hard rejection - let penalty guide the optimizer
            # Train and evaluate (even if below min_size, penalty will handle it)
            metrics = self.train_with_thresholds(threshold_dict)

            if metrics is None:
                print(f"❌ Training failed")
                penalty_score = -10.0
                self.optimization_history.append({
                    'evaluation': eval_num,
                    'thresholds': threshold_dict,
                    'score': penalty_score,
                    'filtered_size': filtered_size,
                    'rejected': True,
                    'failed': True
                })
                return -penalty_score

            # Check hard constraints
            is_valid, rejection_reason = self.check_hard_constraints(threshold_dict, metrics)
            if not is_valid:
                print(f"❌ Rejected: {rejection_reason}")
                penalty_score = -10.0
                self.optimization_history.append({
                    'evaluation': eval_num,
                    'thresholds': threshold_dict,
                    'score': penalty_score,
                    'filtered_size': filtered_size,
                    'rejected': True,
                    'rejection_reason': rejection_reason
                })
                return -penalty_score

            # Calculate score with penalties (including dataset size penalty)
            balance_score = self.get_class_balance_score(threshold_dict)
            score, penalties = self.calculate_combined_score(metrics, threshold_dict, filtered_size)

            print(f"Results:")
            print(f"  Macro F1: {metrics['macro_f1']:.4f}")
            print(f"  Min F1: {min(metrics['f1_per_class'].values()):.4f}")
            print(f"  Kappa: {metrics['kappa']:.4f}")
            print(f"  Balance: {balance_score:.4f}")
            if sum(penalties.values()) > 0:
                print(f"  Penalties: {sum(penalties.values()):.3f} " +
                      f"({', '.join(f'{k}:{v:.2f}' for k, v in penalties.items() if v > 0)})")
            print(f"  Combined Score: {score:.4f}")

            # Track best
            if score > self.best_score:
                self.best_score = score
                self.best_thresholds = threshold_dict
                print(f"  ✨ New best score!")

            self.optimization_history.append({
                'evaluation': eval_num,
                'thresholds': threshold_dict,
                'metrics': metrics,
                'score': score,
                'filtered_size': filtered_size,
                'rejected': False
            })

            # Return negative for minimization
            return -score

        # Run Bayesian optimization
        print(f"\n🔍 Starting Bayesian optimization...\n")

        result = gp_minimize(
            objective,
            search_space,
            n_calls=self.phase2_n_evaluations,
            random_state=self.base_random_seed,
            verbose=False
        )

        print(f"\n{'='*70}")
        print(f"✅ PHASE 2 COMPLETE")
        print(f"{'='*70}")
        print(f"\nBest thresholds found: {self.best_thresholds}")
        print(f"Best score: {self.best_score:.4f}")

        return True, self.best_thresholds

    def phase2_grid_search(self):
        """Fallback grid search if Bayesian optimization not available."""
        print("\n🔍 Using grid search (fallback)...")
        print("⚠️  This will be slower than Bayesian optimization")

        # Define coarse grid (symmetric ranges 10-100% for all classes)
        p_range = range(max(1, int(0.10 * self.phase1_n_runs)), int(1.00 * self.phase1_n_runs) + 1, 2)
        i_range = range(max(1, int(0.10 * self.phase1_n_runs)), int(1.00 * self.phase1_n_runs) + 1, 2)
        r_range = range(max(1, int(0.10 * self.phase1_n_runs)), int(1.00 * self.phase1_n_runs) + 1, 2)

        from itertools import product
        grid = list(product(p_range, i_range, r_range))

        # Limit to n_evaluations
        if len(grid) > self.phase2_n_evaluations:
            import random
            random.seed(self.base_random_seed)
            grid = random.sample(grid, self.phase2_n_evaluations)

        print(f"Testing {len(grid)} threshold combinations...\n")

        for eval_num, (p, i, r) in enumerate(grid, 1):
            threshold_dict = {'P': p, 'I': i, 'R': r}

            print(f"\nEvaluation {eval_num}/{len(grid)}")
            print(f"Thresholds: {threshold_dict}")

            # Get filtered dataset size
            filtered_size = self.get_filtered_dataset_size(threshold_dict)
            min_size = int(self.original_dataset_size * self.min_dataset_fraction)
            print(f"Dataset after filtering: {filtered_size}/{self.original_dataset_size} samples " +
                  f"({filtered_size/self.original_dataset_size*100:.1f}%)")

            # No hard rejection - let penalty guide optimization
            # Train and evaluate
            metrics = self.train_with_thresholds(threshold_dict)
            if metrics is None:
                continue

            # Check hard constraints (only catastrophic failures)
            is_valid, rejection_reason = self.check_hard_constraints(threshold_dict, metrics)
            if not is_valid:
                print(f"❌ Rejected: {rejection_reason}")
                continue

            score, penalties = self.calculate_combined_score(metrics, threshold_dict, filtered_size)
            balance_score = self.get_class_balance_score(threshold_dict)
            penalty_str = f", Penalties: {sum(penalties.values()):.2f}" if sum(penalties.values()) > 0 else ""
            print(f"Score: {score:.4f} (Balance: {balance_score:.3f}, Min F1: {min(metrics['f1_per_class'].values()):.3f}{penalty_str})")

            if score > self.best_score:
                self.best_score = score
                self.best_thresholds = threshold_dict
                print(f"✨ New best!")

            self.optimization_history.append({
                'evaluation': eval_num,
                'thresholds': threshold_dict,
                'metrics': metrics,
                'score': score,
                'filtered_size': filtered_size
            })

        return True, self.best_thresholds

    def train_with_thresholds(self, thresholds):
        """
        Train metadata with given thresholds and return metrics.

        Args:
            thresholds: Dict like {'I': 5, 'P': 3, 'R': 8}

        Returns:
            dict: Metrics or None if training failed
        """
        # Clean up from previous evaluation
        cleanup_for_resume_mode('from_data')

        # Temporarily override config
        config_path = project_root / 'src' / 'utils' / 'production_config.py'
        backup_path = project_root / 'src' / 'utils' / 'production_config.py.backup'

        with open(config_path, 'r') as f:
            original_config = f.read()
        with open(backup_path, 'w') as f:
            f.write(original_config)

        try:
            import re

            # Calculate adjusted batch size for Phase 2
            adjusted_batch_size = self._calculate_phase2_batch_size()

            # Build the modality tuple string, e.g., "('metadata', 'depth_rgb')"
            modality_tuple = "(" + ", ".join(f"'{m}'" for m in self.modalities) + ",)"

            # Modify INCLUDED_COMBINATIONS
            modified_config = re.sub(
                r'INCLUDED_COMBINATIONS\s*=\s*\[[\s\S]*?\n\]',
                f"INCLUDED_COMBINATIONS = [\n    {modality_tuple},  # Temporary: Phase 2 evaluation\n]",
                original_config
            )

            # Modify GLOBAL_BATCH_SIZE for Phase 2
            modified_config = re.sub(
                r'GLOBAL_BATCH_SIZE\s*=\s*\d+',
                f"GLOBAL_BATCH_SIZE = {adjusted_batch_size}",
                modified_config
            )

            with open(config_path, 'w') as f:
                f.write(modified_config)

            # Run training with thresholds - use fresh mode for clean evaluation
            cmd = [
                sys.executable, 'src/main.py',
                '--mode', 'search',
                '--cv_folds', str(self.phase2_cv_folds),
                '--data_percentage', str(self.phase2_data_percentage),
                '--verbosity', '0',  # Minimal output during optimization
                '--resume_mode', 'fresh',  # Force fresh training for each evaluation
                '--threshold_I', str(thresholds['I']),
                '--threshold_P', str(thresholds['P']),
                '--threshold_R', str(thresholds['R']),
                '--device-mode', self.device_mode,
                '--min-gpu-memory', str(self.min_gpu_memory)
            ]

            # Add custom GPU IDs if specified
            if self.device_mode == 'custom' and self.custom_gpus:
                cmd.extend(['--custom-gpus'] + [str(gpu) for gpu in self.custom_gpus])

            # Add display GPU flag if enabled
            if self.include_display_gpus:
                cmd.append('--include-display-gpus')

            print(f"⏳ Training with cv_folds={self.phase2_cv_folds} (fresh mode)...")

            # Save current directory
            original_cwd = os.getcwd()
            try:
                # Change to project root for execution
                os.chdir(project_root)

                # Use os.system instead of subprocess.run to avoid TensorFlow context conflicts
                # Build command string with proper quoting
                cmd_str = ' '.join(str(arg) for arg in cmd)
                # Redirect output to temp file for debugging
                temp_output = project_root / 'phase2_training_output.tmp'
                return_code = os.system(f"{cmd_str} >{temp_output} 2>&1")

            finally:
                # Restore original directory
                os.chdir(original_cwd)

            if return_code != 0:
                print(f"\n{'='*80}")
                print("❌ TRAINING FAILED")
                print(f"{'='*80}")
                print(f"Return code: {return_code}")
                print(f"\nCommand that failed:")
                print(' '.join(cmd))

                # Show last part of output for debugging
                if temp_output.exists():
                    with open(temp_output, 'r') as f:
                        output = f.read()
                        print(f"\n{'─'*80}")
                        print("OUTPUT (last 3000 chars):")
                        print(f"{'─'*80}")
                        print(output[-3000:])

                print(f"{'='*80}\n")
                return None

            # Extract metrics from CSV
            return self.extract_metrics_from_files()

        finally:
            # Restore config
            with open(backup_path, 'r') as f:
                original_config = f.read()
            with open(config_path, 'w') as f:
                f.write(original_config)
            backup_path.unlink(missing_ok=True)

    def extract_metrics_from_files(self):
        """Extract performance metrics from CSV files."""
        metrics = {
            'f1_per_class': {'I': 0.0, 'P': 0.0, 'R': 0.0},
            'macro_f1': 0.0,
            'kappa': 0.0,
            'accuracy': 0.0
        }

        csv_file = os.path.join(self.result_dir, 'csv', 'modality_results_averaged.csv')
        if not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0:
            return None

        try:
            df = pd.read_csv(csv_file)
            metadata_rows = df[df['Modalities'].str.contains('metadata', case=False, na=False)]
            if len(metadata_rows) == 0:
                return None

            row = metadata_rows.iloc[-1]
            metrics['macro_f1'] = float(row.get('Macro Avg F1-score (Mean)', 0.0))
            metrics['kappa'] = float(row.get("Cohen's Kappa (Mean)", 0.0))
            metrics['accuracy'] = float(row.get('Accuracy (Mean)', 0.0))

            for i, cls in enumerate(['I', 'P', 'R']):
                col_name = f'Class {i} F1-score (Mean)'
                if col_name in row:
                    metrics['f1_per_class'][cls] = float(row[col_name])

            return metrics

        except Exception as e:
            print(f"⚠️  Could not extract metrics: {e}")
            return None

    def save_results(self):
        """Save optimization results to JSON."""
        results_file = os.path.join(self.result_dir, 'bayesian_optimization_results.json')

        # Convert numpy types to native Python types for JSON serialization
        def convert_to_native(obj):
            """Recursively convert numpy types to native Python types."""
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        results = {
            'timestamp': datetime.now().isoformat(),
            'phase1_n_runs': self.phase1_n_runs,
            'phase2_n_evaluations': self.phase2_n_evaluations,
            'best_thresholds': convert_to_native(self.best_thresholds),
            'best_score': float(self.best_score),
            'optimization_history': convert_to_native(self.optimization_history),
            'original_dataset_size': int(self.original_dataset_size) if self.original_dataset_size else None
        }

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n📊 Results saved to: {results_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Two-phase Bayesian dataset polishing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run both phases with metadata-only evaluation in Phase 2
  python scripts/auto_polish_dataset_v2.py --phase2-modalities metadata

  # Run both phases with all modalities combined in Phase 2
  python scripts/auto_polish_dataset_v2.py --phase2-modalities metadata+depth_rgb+depth_map+thermal_map

  # Just Phase 1 with 100 runs per modality (400 total)
  python scripts/auto_polish_dataset_v2.py --phase2-modalities metadata --phase1-only --phase1-n-runs 100

  # Phase 1 with custom modalities (only metadata and depth_rgb)
  python scripts/auto_polish_dataset_v2.py --phase2-modalities metadata --phase1-only --phase1-modalities metadata depth_rgb

  # Just Phase 2 with combined modalities (if Phase 1 already done)
  python scripts/auto_polish_dataset_v2.py --phase2-modalities metadata+depth_rgb --phase2-only

  # More thorough optimization
  python scripts/auto_polish_dataset_v2.py --phase2-modalities metadata --n-evaluations 30

GPU Configuration:
  # Multi-GPU mode (use all available GPUs >=8GB)
  python scripts/auto_polish_dataset_v2.py --phase2-modalities metadata --device-mode multi

  # Custom GPU selection
  python scripts/auto_polish_dataset_v2.py --phase2-modalities metadata --device-mode custom --custom-gpus 0 1

  # Single GPU (default, auto-select best)
  python scripts/auto_polish_dataset_v2.py --phase2-modalities metadata --device-mode single

  # CPU only
  python scripts/auto_polish_dataset_v2.py --phase2-modalities metadata --device-mode cpu
        """
    )

    parser.add_argument('--phase2-modalities', type=str, required=True,
                        help='Modalities for Phase 2 evaluation. Use + to combine modalities. '
                             'Examples: "metadata", "metadata+depth_rgb", "metadata+depth_rgb+depth_map+thermal_map"')

    parser.add_argument('--phase1-only', action='store_true',
                        help='Run only Phase 1 (misclassification detection)')

    parser.add_argument('--phase2-only', action='store_true',
                        help='Run only Phase 2 (threshold optimization)')

    parser.add_argument('--phase1-n-runs', type=int, default=10,
                        help='Number of runs per modality in Phase 1 (default: 10)')

    parser.add_argument('--phase1-modalities', nargs='+', default=['metadata', 'depth_rgb', 'depth_map', 'thermal_map'],
                        help='Modalities to test individually in Phase 1 (default: metadata depth_rgb depth_map thermal_map)')

    parser.add_argument('--phase1-data-percentage', type=int, default=100,
                        help='Percentage of data to use in Phase 1 (default: 100)')

    parser.add_argument('--n-evaluations', type=int, default=20,
                        help='Number of Bayesian optimization evaluations (default: 20)')

    parser.add_argument('--phase2-data-percentage', type=int, default=100,
                        help='Percentage of data to use in Phase 2 (default: 100)')

    parser.add_argument('--min-dataset-fraction', type=float, default=0.5,
                        help='Minimum fraction of dataset to keep (default: 0.5)')

    parser.add_argument('--min-retention-per-class', type=float, default=0.90,
                        help='Minimum retention per class (default: 0.90). '
                             'Optimization is skipped if this cannot be achieved.')

    # GPU configuration flags
    parser.add_argument('--device-mode', type=str, choices=['cpu', 'single', 'multi', 'custom'],
                        default='single',
                        help='GPU mode: cpu (no GPUs), single (auto-select best), multi (all available), custom (specify IDs)')
    parser.add_argument('--custom-gpus', type=int, nargs='+', default=None,
                        help='GPU IDs for custom mode (e.g., --custom-gpus 0 1)')
    parser.add_argument('--min-gpu-memory', type=float, default=8.0,
                        help='Minimum GPU memory in GB (default: 8.0)')
    parser.add_argument('--include-display-gpus', action='store_true',
                        help='Allow training on display GPUs (default: exclude them)')

    args = parser.parse_args()

    # Parse phase2-modalities (e.g., "metadata+depth_rgb" -> ['metadata', 'depth_rgb'])
    phase2_modalities = [m.strip() for m in args.phase2_modalities.split('+')]

    valid_modalities = {'metadata', 'depth_rgb', 'depth_map', 'thermal_map'}
    invalid = set(phase2_modalities) - valid_modalities
    if invalid:
        print(f"❌ Error: Invalid modalities: {invalid}")
        print(f"   Valid options: {valid_modalities}")
        sys.exit(1)

    if not phase2_modalities:
        print("❌ Error: --phase2-modalities cannot be empty")
        sys.exit(1)

    polisher = BayesianDatasetPolisher(
        modalities=phase2_modalities,
        phase1_n_runs=args.phase1_n_runs,
        phase1_modalities=args.phase1_modalities,
        phase1_data_percentage=args.phase1_data_percentage,
        phase2_n_evaluations=args.n_evaluations,
        phase2_data_percentage=args.phase2_data_percentage,
        min_dataset_fraction=args.min_dataset_fraction,
        min_retention_per_class=args.min_retention_per_class,
        device_mode=args.device_mode,
        custom_gpus=args.custom_gpus,
        min_gpu_memory=args.min_gpu_memory,
        include_display_gpus=args.include_display_gpus
    )

    # Run phases
    if not args.phase2_only:
        print("\n🚀 Starting Phase 1: Misclassification Detection\n")
        success = polisher.phase1_detect_misclassifications()
        if not success:
            print("\n❌ Phase 1 failed")
            sys.exit(1)

    if not args.phase1_only:
        print("\n🚀 Starting Phase 2: Bayesian Threshold Optimization\n")
        success, best_thresholds = polisher.phase2_optimize_thresholds()
        if not success:
            print("\n❌ Phase 2 failed")
            sys.exit(1)

        # Handle case where optimization was skipped (thresholds=None)
        if best_thresholds is None:
            print("\n" + "="*70)
            print("⚠️  OPTIMIZATION SKIPPED")
            print("="*70)
            print(f"\nCannot achieve {args.min_retention_per_class*100:.0f}% retention per class.")
            print("Recommendation: Use NO filtering for training:")
            print(f"   python src/main.py --mode search --cv_folds 5")
            print("\nOr reduce --min-retention-per-class to allow more aggressive filtering.")
        else:
            polisher.save_results()

            print("\n" + "="*70)
            print("🎉 OPTIMIZATION COMPLETE")
            print("="*70)
            print(f"\nOptimal thresholds: {best_thresholds}")
            print(f"Optimization score: {polisher.best_score:.4f}")
            print(f"\n💡 Use these thresholds for final training:")
            print(f"   python src/main.py --mode search --cv_folds 5 \\")
            print(f"       --threshold_I {best_thresholds['I']} \\")
            print(f"       --threshold_P {best_thresholds['P']} \\")
            print(f"       --threshold_R {best_thresholds['R']}")

    sys.exit(0)


if __name__ == '__main__':
    main()
