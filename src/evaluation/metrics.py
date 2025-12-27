import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.utils.config import CLASS_LABELS
from src.utils.verbosity import vprint

def track_misclassifications(y_true, y_pred, sample_ids, selected_modalities, result_dir):
    """
    Track uniquely misclassified examples and update the CSV file.

    FIXED: Now counts at SAMPLE LEVEL (how many folds was sample misclassified in),
    not image level. Each sample counted once per fold regardless of:
    - How many images it has
    - What the predicted label was

    Args:
        y_true: True labels (numpy array)
        y_pred: Predicted labels (numpy array)
        sample_ids: Array of sample identifiers (shape: [N, 3] for [Patient, Appt, DFU])
        result_dir: Directory to save the CSV file (caller passes misclassifications subdirectory)
    """
    modality_str = '_'.join(sorted(selected_modalities))
    misclass_file = os.path.join(result_dir, f'frequent_misclassifications_{modality_str}.csv')
    misclass_file_total = os.path.join(result_dir, f'frequent_misclassifications_total.csv')

    # Create DataFrame of current misclassifications
    misclassified_mask = (y_true != y_pred)
    if misclassified_mask.any():
        current_misclass = pd.DataFrame({
            'Patient': sample_ids[misclassified_mask, 0].astype(int),
            'Appointment': sample_ids[misclassified_mask, 1].astype(int),
            'DFU': sample_ids[misclassified_mask, 2].astype(int),
            'True_Label': [CLASS_LABELS[int(y)] for y in y_true[misclassified_mask]]
        })

        # Add unique identifier column
        current_misclass['Sample_ID'] = (
            'P' + current_misclass['Patient'].astype(str).str.zfill(3) +
            'A' + current_misclass['Appointment'].astype(str).str.zfill(2) +
            'D' + current_misclass['DFU'].astype(str)
        )

        # FIXED: Drop duplicates at SAMPLE LEVEL only (ignore predicted label)
        # This ensures each sample counts once per fold, regardless of:
        # - How many images it has
        # - Whether different images got different wrong predictions
        current_misclass = current_misclass.drop_duplicates(
            subset=['Sample_ID', 'True_Label']
        )
        # Save Modality-wise misclassifications
        if os.path.exists(misclass_file):
            # Load existing misclassifications
            existing_misclass = pd.read_csv(misclass_file)

            # Update counts for existing misclassifications
            for _, row in current_misclass.iterrows():
                mask = (
                    (existing_misclass['Sample_ID'] == row['Sample_ID']) &
                    (existing_misclass['True_Label'] == row['True_Label'])
                )

                if mask.any():
                    existing_misclass.loc[mask, 'Misclass_Count'] += 1
                else:
                    new_row = row.to_dict()
                    new_row['Misclass_Count'] = 1
                    existing_misclass = pd.concat([
                        existing_misclass,
                        pd.DataFrame([new_row])
                    ], ignore_index=True)
        else:
            # Create new DataFrame with counts
            existing_misclass = current_misclass.copy()
            existing_misclass['Misclass_Count'] = 1

        # Sort by misclassification count and save
        existing_misclass = existing_misclass.sort_values(
            ['Misclass_Count', 'Sample_ID'],
            ascending=[False, True]
        )
        existing_misclass.to_csv(misclass_file, index=False)
        
        # Save total misclassifications
        if os.path.exists(misclass_file_total):
            # Load existing misclassifications
            existing_misclass = pd.read_csv(misclass_file_total)

            # Update counts for existing misclassifications
            for _, row in current_misclass.iterrows():
                mask = (
                    (existing_misclass['Sample_ID'] == row['Sample_ID']) &
                    (existing_misclass['True_Label'] == row['True_Label'])
                )

                if mask.any():
                    existing_misclass.loc[mask, 'Misclass_Count'] += 1
                else:
                    new_row = row.to_dict()
                    new_row['Misclass_Count'] = 1
                    existing_misclass = pd.concat([
                        existing_misclass,
                        pd.DataFrame([new_row])
                    ], ignore_index=True)
        else:
            # Create new DataFrame with counts
            existing_misclass = current_misclass.copy()
            existing_misclass['Misclass_Count'] = 1

        # Sort by misclassification count and save
        existing_misclass = existing_misclass.sort_values(
            ['Misclass_Count', 'Sample_ID'],
            ascending=[False, True]
        )
        existing_misclass.to_csv(misclass_file_total, index=False)

def analyze_misclassifications(result_dir):
    """
    Analyze and print summary statistics of misclassifications.
    
    Args:
        result_dir: Directory containing the misclassifications CSV
    """
    misclass_file = os.path.join(result_dir, 'frequent_misclassifications.csv')
    if not os.path.exists(misclass_file):
        vprint("No misclassification, level=1 file found.")
        return
    
    df = pd.read_csv(misclass_file)
    
    # Print overall statistics
    print("\nMisclassification Analysis:")
    print(f"Total unique misclassified samples: {len(df)}")
    print(f"\nTop 10 most frequently misclassified samples:")
    print(df.head(10)[['Sample_ID', 'True_Label', 'Predicted_Label', 'Misclass_Count']])
    
    # Analyze transition patterns
    transitions = df.groupby(['True_Label', 'Predicted_Label'])['Misclass_Count'].sum()
    print("\nMisclassification Patterns:")
    print(transitions)
    
    # Plot confusion matrix of misclassifications
    plt.figure(figsize=(10, 8))
    confusion = pd.pivot_table(
        df,
        values='Misclass_Count',
        index='True_Label',
        columns='Predicted_Label',
        aggfunc='sum',
        fill_value=0
    )
    sns.heatmap(confusion, annot=True, fmt='.0f', cmap='Blues')
    plt.title('Misclassification Pattern Heatmap')
    plt.savefig(os.path.join(result_dir, 'misclassification_patterns.png'))
    plt.close()

def filter_frequent_misclassifications(data, result_dir, thresholds={'I': 12, 'P': 9, 'R': 12}):
    """
    Filter out samples that are frequently misclassified based on healing phase-specific thresholds.

    FIXED: Now properly handles duplicate Sample_ID entries by using max Misclass_Count per sample.
    Also adds fail-fast assertions and detailed logging.

    Args:
        data: Original DataFrame
        result_dir: Base results directory (will check multiple subdirectories for misclassification CSV)
        thresholds: Dictionary with misclassification count thresholds for each class

    Returns:
        Filtered DataFrame
    """
    # ASSERTION 1: Input data must be valid
    assert data is not None and len(data) > 0, "❌ Input data is empty or None!"
    original_size = len(data)

    # Check multiple possible locations for misclassification file
    # (auto_polish_dataset_v2.py may save to different subdirectories)
    possible_paths = [
        os.path.join(result_dir, 'misclassifications_saved', 'frequent_misclassifications_saved.csv'),
        os.path.join(result_dir, 'misclassifications', 'frequent_misclassifications_saved.csv'),
        os.path.join(result_dir, 'frequent_misclassifications_saved.csv'),  # Legacy direct path
    ]

    misclass_file = None
    for path in possible_paths:
        if os.path.exists(path):
            misclass_file = path
            vprint(f"Found misclassification file: {os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}", level=2)
            break

    if misclass_file is None:
        vprint("No misclassification file found. Using original dataset.", level=1)
        return data

    # Load misclassification data
    misclass_df = pd.read_csv(misclass_file)

    # ASSERTION 2: Misclass file must have required columns
    assert 'Sample_ID' in misclass_df.columns, "❌ Missing Sample_ID column in misclass file!"
    assert 'True_Label' in misclass_df.columns, "❌ Missing True_Label column in misclass file!"
    assert 'Misclass_Count' in misclass_df.columns, "❌ Missing Misclass_Count column in misclass file!"

    # Handle duplicate Sample_IDs: same sample can be misclassified as different labels
    # Use max Misclass_Count per Sample_ID for exclusion decision
    max_misclass = misclass_df.groupby(['Sample_ID', 'True_Label'])['Misclass_Count'].max().reset_index()

    # Create set of samples to exclude for each class
    samples_to_exclude = set()
    exclusion_details = {}
    for phase, threshold in thresholds.items():
        high_misclass = max_misclass[
            (max_misclass['True_Label'] == phase) &
            (max_misclass['Misclass_Count'] >= threshold)
        ]['Sample_ID'].unique().tolist()
        samples_to_exclude.update(high_misclass)
        exclusion_details[phase] = len(high_misclass)

    # Create sample ID column in original data
    data = data.copy()  # Avoid modifying original
    data['Sample_ID'] = (
        'P' + data['Patient#'].astype(str).str.zfill(3) +
        'A' + data['Appt#'].astype(str).str.zfill(2) +
        'D' + data['DFU#'].astype(str)
    )

    # ASSERTION 3: Sample IDs must be generated
    assert 'Sample_ID' in data.columns, "❌ Failed to create Sample_ID column!"
    assert data['Sample_ID'].notna().all(), "❌ Some Sample_IDs are NaN!"

    # Check overlap between data and misclass file
    data_ids = set(data['Sample_ID'].unique())
    misclass_ids = set(misclass_df['Sample_ID'].unique())
    overlap = data_ids.intersection(misclass_ids)

    if len(overlap) == 0:
        print(f"⚠️  WARNING: No overlap between data and misclass IDs!")
        print(f"   Data sample IDs: {list(data_ids)[:3]}...")
        print(f"   Misclass sample IDs: {list(misclass_ids)[:3]}...")

    # Filter out samples
    filtered_data = data[~data['Sample_ID'].isin(samples_to_exclude)].copy()

    # Get unique sample counts
    original_unique = data['Sample_ID'].nunique()
    filtered_unique = filtered_data['Sample_ID'].nunique()
    removed_unique = original_unique - filtered_unique

    # Remove temporary Sample_ID column
    filtered_data = filtered_data.drop('Sample_ID', axis=1)

    # Detailed logging
    print(f"\n{'='*60}")
    print(f"FILTERING SUMMARY")
    print(f"{'='*60}")
    print(f"Thresholds: I={thresholds.get('I', 'N/A')}, P={thresholds.get('P', 'N/A')}, R={thresholds.get('R', 'N/A')}")
    print(f"\nExcluded samples per class:")
    for phase in ['I', 'P', 'R']:
        print(f"  Class {phase}: {exclusion_details.get(phase, 0)} samples")
    print(f"\nTotal unique samples to exclude: {len(samples_to_exclude)}")
    print(f"\nDataset size (rows): {original_size} -> {len(filtered_data)} ({len(filtered_data)/original_size*100:.1f}%)")
    print(f"Unique samples: {original_unique} -> {filtered_unique} (removed {removed_unique})")

    # Check class distribution after filtering
    if 'Healing Phase Abs' in filtered_data.columns:
        print(f"\nClass distribution after filtering:")
        for phase in ['I', 'P', 'R']:
            count = len(filtered_data[filtered_data['Healing Phase Abs'] == phase])
            print(f"  Class {phase}: {count} rows")
    print(f"{'='*60}\n")

    # ASSERTION 4: Filtered size must be valid
    assert len(filtered_data) >= 0, f"❌ Negative filtered size!"

    # ASSERTION 5: Warn if too much was filtered
    if len(filtered_data) == 0:
        print(f"⚠️  WARNING: ALL SAMPLES FILTERED OUT!")
        print(f"   Check if thresholds are too aggressive: {thresholds}")
    elif len(filtered_data) < original_size * 0.2:
        print(f"⚠️  WARNING: Over 80% of data filtered out!")
        print(f"   Consider adjusting thresholds: {thresholds}")

    return filtered_data