"""
Confidence-Based Filtering for Dataset Cleaning

This script identifies "bad" samples by analyzing model prediction confidence.
Low-confidence samples are often annotation errors, ambiguous cases, or outliers
that hurt model performance.

Key Features:
- Faster than iterative misclassification tracking (requires only 1 training run)
- Multiple confidence metrics: max_prob, margin, entropy
- Per-class or global filtering modes
- Safety limits to protect minority classes
- Can run standalone or integrate with main training pipeline

How it works:
1. Train model normally (or use existing trained model)
2. Collect predictions and confidence scores for all samples
3. Identify samples with lowest confidence (bottom X%)
4. Save low-confidence sample IDs for filtering
5. Optionally retrain with filtered dataset

Usage:
    # Run with default settings (15% filtering, per-class mode)
    python scripts/confidence_based_filtering.py

    # Custom filtering percentage
    python scripts/confidence_based_filtering.py --percentile 20

    # Use margin-based confidence instead of max_prob
    python scripts/confidence_based_filtering.py --metric margin

    # Global filtering (not per-class)
    python scripts/confidence_based_filtering.py --mode global

    # Automatically retrain after identifying bad samples
    python scripts/confidence_based_filtering.py --retrain

Configuration:
    See src/utils/production_config.py for:
    - USE_CONFIDENCE_FILTERING: Enable/disable during training
    - CONFIDENCE_FILTER_PERCENTILE: Filtering threshold
    - CONFIDENCE_FILTER_MODE: 'per_class' or 'global'
    - CONFIDENCE_METRIC: 'max_prob', 'margin', or 'entropy'
"""

import argparse
import os
import sys
import json
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import get_project_paths, get_output_paths, CLASS_LABELS


class ConfidenceBasedFilter:
    """Identifies bad samples using prediction confidence analysis."""

    def __init__(self,
                 percentile=15,
                 mode='per_class',
                 metric='max_prob',
                 min_samples_per_class=50,
                 max_class_removal_pct=30,
                 cv_folds=3,
                 data_percentage=100.0,
                 verbosity=2):
        """
        Initialize the confidence-based filter.

        Args:
            percentile: Bottom X% of samples to flag as low-confidence (default: 15)
            mode: 'per_class' = filter bottom X% per class, 'global' = filter bottom X% overall
            metric: Confidence metric to use:
                    'max_prob' = max softmax probability (simple, fast)
                    'margin' = top1 - top2 probability (uncertainty measure)
                    'entropy' = prediction entropy (information-theoretic)
            min_samples_per_class: Minimum samples to keep per class (safety limit)
            max_class_removal_pct: Maximum percentage to remove from ANY class (protects minorities)
            cv_folds: Number of CV folds for training
            data_percentage: Percentage of data to use
            verbosity: Output verbosity level
        """
        self.percentile = percentile
        self.mode = mode
        self.metric = metric
        self.min_samples_per_class = min_samples_per_class
        self.max_class_removal_pct = max_class_removal_pct
        self.cv_folds = cv_folds
        self.data_percentage = data_percentage
        self.verbosity = verbosity

        # Get project paths
        self.directory, self.result_dir, self.root = get_project_paths()
        self.output_paths = get_output_paths(self.result_dir)

        # Results storage
        self.sample_confidences = {}  # sample_id -> confidence_score
        self.sample_predictions = {}  # sample_id -> predicted_class
        self.sample_true_labels = {}  # sample_id -> true_class
        self.low_confidence_samples = []  # List of sample IDs to filter

    def calculate_confidence(self, probabilities):
        """
        Calculate confidence score based on selected metric.

        Args:
            probabilities: Array of shape (N, num_classes) with softmax probabilities

        Returns:
            Array of shape (N,) with confidence scores
        """
        if self.metric == 'max_prob':
            # Maximum probability (simple confidence)
            return np.max(probabilities, axis=1)

        elif self.metric == 'margin':
            # Margin between top-2 probabilities (uncertainty measure)
            sorted_probs = np.sort(probabilities, axis=1)
            return sorted_probs[:, -1] - sorted_probs[:, -2]

        elif self.metric == 'entropy':
            # Prediction entropy (higher = more uncertain, so we return negative)
            # Clip to avoid log(0)
            probs_clipped = np.clip(probabilities, 1e-10, 1.0)
            entropy = -np.sum(probs_clipped * np.log(probs_clipped), axis=1)
            # Normalize to [0, 1] where 1 = max confidence (min entropy)
            max_entropy = np.log(probabilities.shape[1])  # log(num_classes)
            return 1.0 - (entropy / max_entropy)

        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def run_training(self):
        """
        Run training to collect predictions and confidence scores.

        Returns:
            True if training succeeded, False otherwise
        """
        print("\n" + "="*70)
        print("CONFIDENCE-BASED FILTERING: TRAINING PHASE")
        print("="*70)
        print(f"\nConfiguration:")
        print(f"  Percentile: {self.percentile}% (remove bottom {self.percentile}%)")
        print(f"  Mode: {self.mode}")
        print(f"  Metric: {self.metric}")
        print(f"  CV Folds: {self.cv_folds}")
        print(f"  Data Percentage: {self.data_percentage}%")

        # Prepare command
        cmd = [
            'python', 'src/main.py',
            '--mode', 'search',
            '--cv_folds', str(self.cv_folds),
            '--data_percentage', str(self.data_percentage),
            '--verbosity', str(self.verbosity),
            '--resume_mode', 'fresh',
        ]

        print(f"\nRunning: {' '.join(cmd)}")
        print("\n" + "-"*70)

        # Run training
        result = subprocess.run(cmd, cwd=project_root)

        if result.returncode != 0:
            print(f"\n❌ Training failed with return code {result.returncode}")
            return False

        print("\n✅ Training completed successfully")
        return True

    def collect_predictions(self):
        """
        Collect predictions from saved numpy files.

        Returns:
            True if predictions were found and loaded, False otherwise
        """
        print("\n" + "="*70)
        print("COLLECTING PREDICTIONS")
        print("="*70)

        checkpoint_dir = self.output_paths['checkpoints']

        # Look for prediction files (format: {modality}_predictions_fold{N}.npy)
        pred_files = list(Path(checkpoint_dir).glob('*_predictions_*.npy'))
        label_files = list(Path(checkpoint_dir).glob('*_labels_*.npy'))
        sample_id_files = list(Path(checkpoint_dir).glob('*_sample_ids_*.npy'))

        if not pred_files:
            print(f"⚠️  No prediction files found in {checkpoint_dir}")
            print("   Looking for alternative sources...")

            # Try to load from misclassification tracking if available
            misclass_file = os.path.join(self.result_dir, 'frequent_misclassifications_total.csv')
            if os.path.exists(misclass_file):
                return self._load_from_misclass_file(misclass_file)

            print("❌ No prediction data found. Run training first.")
            return False

        print(f"Found {len(pred_files)} prediction files")

        # Aggregate predictions across folds
        all_sample_ids = []
        all_predictions = []
        all_labels = []
        all_probabilities = []

        for pred_file in pred_files:
            try:
                # Load predictions
                predictions = np.load(pred_file)

                # Try to find matching labels and sample_ids
                base_name = pred_file.stem.replace('_predictions', '')
                label_file = pred_file.parent / f"{base_name}_labels{pred_file.suffix}"
                sample_id_file = pred_file.parent / f"{base_name}_sample_ids{pred_file.suffix}"

                if label_file.exists():
                    labels = np.load(label_file)
                else:
                    labels = None

                if sample_id_file.exists():
                    sample_ids = np.load(sample_id_file)
                else:
                    # Generate placeholder IDs
                    sample_ids = np.arange(len(predictions))

                all_predictions.append(predictions)
                all_sample_ids.append(sample_ids)
                if labels is not None:
                    all_labels.append(labels)

                print(f"  Loaded {len(predictions)} predictions from {pred_file.name}")

            except Exception as e:
                print(f"  ⚠️  Error loading {pred_file.name}: {e}")
                continue

        if not all_predictions:
            print("❌ Failed to load any predictions")
            return False

        # Combine all predictions
        all_predictions = np.vstack(all_predictions)
        all_sample_ids = np.concatenate(all_sample_ids)

        if all_labels:
            all_labels = np.concatenate(all_labels)
        else:
            all_labels = None

        # Calculate confidence scores
        confidences = self.calculate_confidence(all_predictions)
        predicted_classes = np.argmax(all_predictions, axis=1)

        # Store results
        for i, sample_id in enumerate(all_sample_ids):
            sid = str(sample_id) if isinstance(sample_id, (int, np.integer)) else sample_id
            self.sample_confidences[sid] = float(confidences[i])
            self.sample_predictions[sid] = int(predicted_classes[i])
            if all_labels is not None:
                self.sample_true_labels[sid] = int(all_labels[i]) if isinstance(all_labels[i], (int, np.integer)) else int(np.argmax(all_labels[i]))

        print(f"\n✅ Collected confidence scores for {len(self.sample_confidences)} samples")
        return True

    def _load_from_misclass_file(self, misclass_file):
        """
        Load sample information from misclassification tracking file.

        This is a fallback when prediction numpy files are not available.
        """
        print(f"Loading from misclassification file: {misclass_file}")

        try:
            df = pd.read_csv(misclass_file)

            if 'Sample_ID' not in df.columns or 'True_Label' not in df.columns:
                print("❌ Misclassification file missing required columns")
                return False

            # Use misclassification count as inverse confidence
            # Higher misclass count = lower confidence
            if 'Misclass_Count' in df.columns:
                max_count = df['Misclass_Count'].max()
                for _, row in df.iterrows():
                    sid = str(row['Sample_ID'])
                    # Convert misclass count to confidence (inverse relationship)
                    confidence = 1.0 - (row['Misclass_Count'] / (max_count + 1))
                    self.sample_confidences[sid] = confidence
                    self.sample_true_labels[sid] = CLASS_LABELS.index(row['True_Label'])

                print(f"✅ Loaded {len(self.sample_confidences)} samples from misclassification file")
                return True
            else:
                print("❌ No Misclass_Count column found")
                return False

        except Exception as e:
            print(f"❌ Error loading misclassification file: {e}")
            return False

    def identify_low_confidence_samples(self):
        """
        Identify samples with low confidence scores.

        Returns:
            List of sample IDs to filter
        """
        print("\n" + "="*70)
        print("IDENTIFYING LOW-CONFIDENCE SAMPLES")
        print("="*70)

        if not self.sample_confidences:
            print("❌ No confidence scores available")
            return []

        # Convert to DataFrame for easier analysis
        df = pd.DataFrame({
            'sample_id': list(self.sample_confidences.keys()),
            'confidence': list(self.sample_confidences.values()),
            'true_label': [self.sample_true_labels.get(sid, -1) for sid in self.sample_confidences.keys()]
        })

        print(f"\nTotal samples: {len(df)}")
        print(f"Confidence metric: {self.metric}")
        print(f"Filtering mode: {self.mode}")
        print(f"Target percentile: {self.percentile}%")

        # Safety: Maximum percentage that can be removed from ANY class (protect minorities)
        MAX_CLASS_REMOVAL_PCT = self.max_class_removal_pct

        # Calculate threshold(s)
        if self.mode == 'global':
            # Global threshold - bottom X% overall
            threshold = np.percentile(df['confidence'], self.percentile)
            low_conf_mask = df['confidence'] <= threshold

            print(f"\nGlobal confidence threshold: {threshold:.4f}")

            # SAFETY CHECK: Ensure no class loses too many samples
            print(f"\nApplying class balance protection (max {MAX_CLASS_REMOVAL_PCT}% removal per class):")
            for label_idx, label_name in enumerate(CLASS_LABELS):
                class_mask = df['true_label'] == label_idx
                class_df = df[class_mask]

                if len(class_df) == 0:
                    continue

                # Count how many would be flagged from this class
                class_flagged_mask = low_conf_mask & class_mask
                num_flagged = class_flagged_mask.sum()

                # Calculate limits
                max_removable_pct = int(len(class_df) * MAX_CLASS_REMOVAL_PCT / 100)
                min_to_keep = max(self.min_samples_per_class, len(class_df) - max_removable_pct)
                max_to_remove = len(class_df) - min_to_keep

                if num_flagged > max_to_remove:
                    # Too many flagged - keep only the lowest confidence ones up to limit
                    class_indices = df[class_mask].index
                    class_confidences = df.loc[class_indices, 'confidence']

                    # Sort by confidence, keep only bottom max_to_remove
                    sorted_class_indices = class_confidences.sort_values().index
                    indices_to_unflag = sorted_class_indices[max_to_remove:]

                    # Unflag the ones above limit
                    low_conf_mask.loc[indices_to_unflag] = False

                    print(f"  ⚠️  {label_name}: Limited removal {num_flagged} → {max_to_remove} "
                          f"(protecting minority class, keeping {min_to_keep}/{len(class_df)})")
                else:
                    print(f"  {label_name}: {num_flagged}/{len(class_df)} flagged "
                          f"({100*num_flagged/len(class_df):.1f}%) - OK")

        else:  # per_class
            # Per-class threshold - bottom X% per class
            low_conf_mask = pd.Series([False] * len(df), index=df.index)

            print(f"\nPer-class confidence thresholds:")
            for label_idx, label_name in enumerate(CLASS_LABELS):
                class_mask = df['true_label'] == label_idx
                class_df = df[class_mask]

                if len(class_df) == 0:
                    continue

                threshold = np.percentile(class_df['confidence'], self.percentile)

                # Apply safety limits:
                # 1. Never go below min_samples_per_class
                # 2. Never remove more than MAX_CLASS_REMOVAL_PCT from any class
                max_removable_pct = int(len(class_df) * MAX_CLASS_REMOVAL_PCT / 100)
                max_removable_percentile = int(len(class_df) * self.percentile / 100)

                num_to_keep = max(
                    self.min_samples_per_class,
                    len(class_df) - max_removable_pct,
                    len(class_df) - max_removable_percentile
                )
                num_to_remove = max(0, len(class_df) - num_to_keep)

                if num_to_remove > 0:
                    # Sort by confidence and mark bottom ones
                    sorted_indices = class_df['confidence'].nsmallest(num_to_remove).index
                    low_conf_mask.loc[sorted_indices] = True

                actual_flagged = low_conf_mask[class_mask].sum()
                pct_flagged = 100 * actual_flagged / len(class_df) if len(class_df) > 0 else 0

                # Warning if we had to limit removal
                if num_to_remove < max_removable_percentile:
                    print(f"  {label_name}: threshold={threshold:.4f}, "
                          f"samples={len(class_df)}, "
                          f"flagged={actual_flagged} ({pct_flagged:.1f}%) "
                          f"⚠️ LIMITED (min_samples={self.min_samples_per_class})")
                else:
                    print(f"  {label_name}: threshold={threshold:.4f}, "
                          f"samples={len(class_df)}, "
                          f"flagged={actual_flagged} ({pct_flagged:.1f}%)")

        # Get low-confidence sample IDs
        self.low_confidence_samples = df[low_conf_mask]['sample_id'].tolist()

        # Statistics
        total_flagged = len(self.low_confidence_samples)
        pct_flagged = 100 * total_flagged / len(df)

        print(f"\n" + "-"*70)
        print(f"LOW-CONFIDENCE SAMPLES IDENTIFIED")
        print(f"-"*70)
        print(f"Total flagged: {total_flagged} ({pct_flagged:.1f}%)")

        # Per-class breakdown
        print(f"\nPer-class breakdown:")
        for label_idx, label_name in enumerate(CLASS_LABELS):
            class_flagged = sum(1 for sid in self.low_confidence_samples
                               if self.sample_true_labels.get(sid, -1) == label_idx)
            class_total = sum(1 for sid, lbl in self.sample_true_labels.items()
                             if lbl == label_idx)
            if class_total > 0:
                pct = 100 * class_flagged / class_total
                print(f"  {label_name}: {class_flagged}/{class_total} flagged ({pct:.1f}%)")

        return self.low_confidence_samples

    def save_results(self):
        """
        Save filtering results to files.

        Returns:
            Path to the results file
        """
        print("\n" + "="*70)
        print("SAVING RESULTS")
        print("="*70)

        # Create results dictionary
        results = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'percentile': self.percentile,
                'mode': self.mode,
                'metric': self.metric,
                'min_samples_per_class': self.min_samples_per_class,
                'max_class_removal_pct': self.max_class_removal_pct,
                'cv_folds': self.cv_folds,
                'data_percentage': self.data_percentage
            },
            'statistics': {
                'total_samples': len(self.sample_confidences),
                'low_confidence_samples': len(self.low_confidence_samples),
                'filtered_percentage': 100 * len(self.low_confidence_samples) / len(self.sample_confidences) if self.sample_confidences else 0
            },
            'low_confidence_sample_ids': self.low_confidence_samples,
            'confidence_summary': {
                'mean': np.mean(list(self.sample_confidences.values())),
                'std': np.std(list(self.sample_confidences.values())),
                'min': np.min(list(self.sample_confidences.values())),
                'max': np.max(list(self.sample_confidences.values())),
                'median': np.median(list(self.sample_confidences.values()))
            }
        }

        # Save JSON results
        results_file = os.path.join(self.result_dir, 'confidence_filter_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✅ Saved results to: {results_file}")

        # Save low-confidence samples CSV (for easy viewing/editing)
        if self.low_confidence_samples:
            csv_data = []
            for sid in self.low_confidence_samples:
                csv_data.append({
                    'Sample_ID': sid,
                    'Confidence': self.sample_confidences.get(sid, 0),
                    'True_Label': CLASS_LABELS[self.sample_true_labels[sid]] if sid in self.sample_true_labels else 'Unknown',
                    'Predicted_Label': CLASS_LABELS[self.sample_predictions[sid]] if sid in self.sample_predictions else 'Unknown'
                })

            csv_file = os.path.join(self.result_dir, 'confidence_low_samples.csv')
            pd.DataFrame(csv_data).to_csv(csv_file, index=False)
            print(f"✅ Saved low-confidence samples to: {csv_file}")

        # Save all confidence scores (for analysis)
        all_scores_data = []
        for sid, conf in self.sample_confidences.items():
            all_scores_data.append({
                'Sample_ID': sid,
                'Confidence': conf,
                'True_Label': CLASS_LABELS[self.sample_true_labels[sid]] if sid in self.sample_true_labels else 'Unknown',
                'Is_Low_Confidence': sid in self.low_confidence_samples
            })

        all_scores_file = os.path.join(self.result_dir, 'confidence_all_samples.csv')
        pd.DataFrame(all_scores_data).to_csv(all_scores_file, index=False)
        print(f"✅ Saved all confidence scores to: {all_scores_file}")

        return results_file

    def run_filtered_training(self):
        """
        Run training again with low-confidence samples filtered out.

        Returns:
            True if training succeeded, False otherwise
        """
        print("\n" + "="*70)
        print("FILTERED TRAINING: EXCLUDING LOW-CONFIDENCE SAMPLES")
        print("="*70)

        if not self.low_confidence_samples:
            print("⚠️  No low-confidence samples identified, skipping filtered training")
            return True

        # For now, we'll use the exclusion list approach
        # In production, this would integrate with the data loading pipeline
        exclusion_file = os.path.join(self.result_dir, 'confidence_exclusion_list.txt')
        with open(exclusion_file, 'w') as f:
            for sid in self.low_confidence_samples:
                f.write(f"{sid}\n")

        print(f"Created exclusion list: {exclusion_file}")
        print(f"Samples to exclude: {len(self.low_confidence_samples)}")

        # Run training with exclusion
        # Note: This requires integration with main.py to read the exclusion list
        # For now, we'll set an environment variable that main.py can check
        os.environ['CONFIDENCE_EXCLUSION_FILE'] = exclusion_file

        cmd = [
            'python', 'src/main.py',
            '--mode', 'search',
            '--cv_folds', str(self.cv_folds),
            '--data_percentage', str(self.data_percentage),
            '--verbosity', str(self.verbosity),
            '--resume_mode', 'fresh',
        ]

        print(f"\nRunning filtered training: {' '.join(cmd)}")
        print("\n" + "-"*70)

        result = subprocess.run(cmd, cwd=project_root, env=os.environ.copy())

        # Clean up
        if 'CONFIDENCE_EXCLUSION_FILE' in os.environ:
            del os.environ['CONFIDENCE_EXCLUSION_FILE']

        if result.returncode != 0:
            print(f"\n❌ Filtered training failed with return code {result.returncode}")
            return False

        print("\n✅ Filtered training completed successfully")
        return True

    def run(self, skip_training=False, retrain=False):
        """
        Run the complete confidence-based filtering pipeline.

        Args:
            skip_training: If True, skip initial training and try to load existing predictions
            retrain: If True, run training again with filtered dataset

        Returns:
            Tuple of (success, low_confidence_sample_ids)
        """
        print("\n" + "="*70)
        print("CONFIDENCE-BASED FILTERING PIPELINE")
        print("="*70)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Step 1: Train model (or skip if requested)
        if not skip_training:
            if not self.run_training():
                return False, []
        else:
            print("\n⏭️  Skipping training, will use existing predictions")

        # Step 2: Collect predictions
        if not self.collect_predictions():
            return False, []

        # Step 3: Identify low-confidence samples
        low_conf_samples = self.identify_low_confidence_samples()

        # Step 4: Save results
        self.save_results()

        # Step 5: Optionally retrain with filtered data
        if retrain and low_conf_samples:
            print("\n" + "="*70)
            print("RETRAINING WITH FILTERED DATASET")
            print("="*70)
            if not self.run_filtered_training():
                return False, low_conf_samples

        print("\n" + "="*70)
        print("✅ CONFIDENCE-BASED FILTERING COMPLETE")
        print("="*70)
        print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nSummary:")
        print(f"  Total samples analyzed: {len(self.sample_confidences)}")
        print(f"  Low-confidence samples: {len(low_conf_samples)}")
        print(f"  Results saved to: {self.result_dir}")

        if low_conf_samples and not retrain:
            print(f"\n💡 To retrain with filtered dataset, run with --retrain flag")

        return True, low_conf_samples


# =============================================================================
# Helper Functions for main.py Integration
# =============================================================================


def get_exclusion_list_path():
    """Get the path to the confidence exclusion list file."""
    _, result_dir, _ = get_project_paths()
    return os.path.join(result_dir, 'confidence_exclusion_list.txt')


def get_results_file_path():
    """Get the path to the confidence filter results JSON file."""
    _, result_dir, _ = get_project_paths()
    return os.path.join(result_dir, 'confidence_filter_results.json')


def exclusion_list_exists():
    """Check if a confidence exclusion list already exists."""
    return os.path.exists(get_exclusion_list_path())


def load_exclusion_set():
    """
    Load the set of sample IDs to exclude.

    Returns:
        set: Set of sample IDs (as strings) to exclude, empty set if file doesn't exist
    """
    exclusion_file = get_exclusion_list_path()
    if not os.path.exists(exclusion_file):
        return set()

    try:
        with open(exclusion_file, 'r') as f:
            return set(line.strip() for line in f if line.strip())
    except Exception as e:
        print(f"Warning: Failed to load exclusion list: {e}")
        return set()


def apply_confidence_filter_to_dataframe(df, sample_id_column='depth_rgb'):
    """
    Apply confidence-based filtering to a DataFrame.

    Args:
        df: pandas DataFrame to filter
        sample_id_column: Column to use as sample identifier (default: 'depth_rgb')

    Returns:
        tuple: (filtered_df, num_excluded)
    """
    from collections import Counter

    excluded_ids = load_exclusion_set()

    if not excluded_ids:
        return df, 0

    original_count = len(df)

    # Create mask for samples to keep
    sample_ids = df[sample_id_column].astype(str)
    keep_mask = ~sample_ids.isin(excluded_ids)

    filtered_df = df[keep_mask].copy()
    num_excluded = original_count - len(filtered_df)

    if num_excluded > 0:
        print(f"  Confidence filtering: excluded {num_excluded}/{original_count} samples "
              f"({100*num_excluded/original_count:.1f}%)")

        # Log per-class breakdown if available
        if 'Healing Phase Abs' in df.columns:
            orig_dist = Counter(df['Healing Phase Abs'])
            filt_dist = Counter(filtered_df['Healing Phase Abs'])
            print(f"  Per-class breakdown:")
            for cls in ['I', 'P', 'R']:
                orig = orig_dist.get(cls, 0)
                filt = filt_dist.get(cls, 0)
                removed = orig - filt
                if orig > 0:
                    print(f"    {cls}: {filt}/{orig} (removed {removed}, {100*removed/orig:.1f}%)")

    return filtered_df, num_excluded


def run_confidence_filtering_pipeline(
    percentile=None,
    mode=None,
    metric=None,
    min_samples_per_class=None,
    max_class_removal_pct=None,
    cv_folds=1,
    data_percentage=100.0,
    force_recompute=False,
    verbosity=2
):
    """
    Run the confidence filtering pipeline from main.py.

    This function is called by main.py when USE_CONFIDENCE_FILTERING=True
    and no exclusion list exists (or force_recompute=True).

    Args:
        percentile: Filter percentile (uses config default if None)
        mode: Filter mode (uses config default if None)
        metric: Confidence metric (uses config default if None)
        min_samples_per_class: Min samples per class (uses config default if None)
        max_class_removal_pct: Max removal % per class (uses config default if None)
        cv_folds: Number of CV folds for preliminary training (default: 1 for speed)
        data_percentage: Data percentage to use
        force_recompute: If True, recompute even if exclusion list exists
        verbosity: Output verbosity level

    Returns:
        tuple: (success, excluded_sample_ids)
    """
    # Import config values for defaults
    from src.utils.production_config import (
        CONFIDENCE_FILTER_PERCENTILE,
        CONFIDENCE_FILTER_MODE,
        CONFIDENCE_METRIC,
        CONFIDENCE_FILTER_MIN_SAMPLES,
        CONFIDENCE_FILTER_MAX_CLASS_REMOVAL_PCT
    )

    # Use config defaults if not specified
    if percentile is None:
        percentile = CONFIDENCE_FILTER_PERCENTILE
    if mode is None:
        mode = CONFIDENCE_FILTER_MODE
    if metric is None:
        metric = CONFIDENCE_METRIC
    if min_samples_per_class is None:
        min_samples_per_class = CONFIDENCE_FILTER_MIN_SAMPLES
    if max_class_removal_pct is None:
        max_class_removal_pct = CONFIDENCE_FILTER_MAX_CLASS_REMOVAL_PCT

    # Check if we can skip
    if not force_recompute and exclusion_list_exists():
        print(f"\n✓ Using existing confidence exclusion list: {get_exclusion_list_path()}")
        excluded_ids = load_exclusion_set()
        print(f"  {len(excluded_ids)} samples will be excluded")
        return True, list(excluded_ids)

    print("\n" + "="*70)
    print("CONFIDENCE-BASED FILTERING (Preliminary Training Phase)")
    print("="*70)
    print(f"Running preliminary training to identify low-confidence samples...")
    print(f"  Percentile: {percentile}%")
    print(f"  Mode: {mode}")
    print(f"  Metric: {metric}")
    print(f"  CV Folds: {cv_folds}")

    # Create and run filter
    filter_obj = ConfidenceBasedFilter(
        percentile=percentile,
        mode=mode,
        metric=metric,
        min_samples_per_class=min_samples_per_class,
        max_class_removal_pct=max_class_removal_pct,
        cv_folds=cv_folds,
        data_percentage=data_percentage,
        verbosity=verbosity
    )

    success, low_conf_samples = filter_obj.run(
        skip_training=False,
        retrain=False  # Don't retrain here, main.py will handle that
    )

    return success, low_conf_samples


def get_confidence_filter_stats():
    """
    Get statistics from the most recent confidence filtering run.

    Returns:
        dict or None: Statistics dictionary if results exist
    """
    results_file = get_results_file_path()

    if not os.path.exists(results_file):
        return None

    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        return {
            'timestamp': results.get('timestamp'),
            'config': results.get('config', {}),
            'statistics': results.get('statistics', {}),
            'num_excluded': len(results.get('low_confidence_sample_ids', []))
        }
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Identify bad samples using prediction confidence analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default settings (15% filtering, per-class mode, max_prob metric)
    python scripts/confidence_based_filtering.py

    # More aggressive filtering (remove bottom 20%)
    python scripts/confidence_based_filtering.py --percentile 20

    # Use margin-based confidence (top1 - top2 probability)
    python scripts/confidence_based_filtering.py --metric margin

    # Use entropy-based confidence (information-theoretic)
    python scripts/confidence_based_filtering.py --metric entropy

    # Global filtering instead of per-class
    python scripts/confidence_based_filtering.py --mode global

    # Skip training, just analyze existing predictions
    python scripts/confidence_based_filtering.py --skip-training

    # Run full pipeline including retraining with filtered data
    python scripts/confidence_based_filtering.py --retrain

    # Quick test with less data
    python scripts/confidence_based_filtering.py --data-percentage 20 --cv-folds 1

Confidence Metrics:
    max_prob: Maximum softmax probability (simple, fast)
              High value = model is confident in its prediction

    margin:   Difference between top-2 probabilities
              High value = model clearly prefers one class over others

    entropy:  Prediction entropy (information-theoretic)
              High value = low entropy = confident prediction

Filtering Modes:
    per_class: Remove bottom X% from EACH class (protects minority classes)
    global:    Remove bottom X% overall (may remove more from difficult classes)
"""
    )

    parser.add_argument('--percentile', type=float, default=15,
                       help='Bottom X%% of samples to flag as low-confidence (default: 15)')

    parser.add_argument('--mode', choices=['per_class', 'global'], default='per_class',
                       help='Filtering mode (default: per_class)')

    parser.add_argument('--metric', choices=['max_prob', 'margin', 'entropy'], default='max_prob',
                       help='Confidence metric to use (default: max_prob)')

    parser.add_argument('--min-samples', type=int, default=50,
                       help='Minimum samples to keep per class (default: 50)')

    parser.add_argument('--max-class-removal', type=int, default=30,
                       help='Maximum %% to remove from ANY class, protects minorities (default: 30)')

    parser.add_argument('--cv-folds', type=int, default=3,
                       help='Number of CV folds for training (default: 3)')

    parser.add_argument('--data-percentage', type=float, default=100.0,
                       help='Percentage of data to use (default: 100.0)')

    parser.add_argument('--verbosity', type=int, default=2,
                       help='Output verbosity level (default: 2)')

    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training, use existing predictions')

    parser.add_argument('--retrain', action='store_true',
                       help='Retrain with filtered dataset after identifying bad samples')

    args = parser.parse_args()

    # Create filter
    filter = ConfidenceBasedFilter(
        percentile=args.percentile,
        mode=args.mode,
        metric=args.metric,
        min_samples_per_class=args.min_samples,
        max_class_removal_pct=args.max_class_removal,
        cv_folds=args.cv_folds,
        data_percentage=args.data_percentage,
        verbosity=args.verbosity
    )

    # Run pipeline
    success, low_conf_samples = filter.run(
        skip_training=args.skip_training,
        retrain=args.retrain
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
