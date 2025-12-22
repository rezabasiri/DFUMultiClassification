import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.utils.config import CLASS_LABELS, get_output_paths
from src.utils.verbosity import vprint

def track_misclassifications(y_true, y_pred, sample_ids, selected_modalities, result_dir):
    """
    Track uniquely misclassified examples and update the CSV file.

    Args:
        y_true: True labels (numpy array)
        y_pred: Predicted labels (numpy array)
        sample_ids: Array of sample identifiers
        result_dir: Directory to save the CSV file
    """
    # Use organized output paths - save to misclassifications subdirectory
    output_paths = get_output_paths(result_dir)
    misclass_dir = output_paths['misclassifications']

    modality_str = '_'.join(sorted(selected_modalities))
    misclass_file = os.path.join(misclass_dir, f'frequent_misclassifications_{modality_str}.csv')
    misclass_file_total = os.path.join(misclass_dir, f'frequent_misclassifications_total.csv')
    
    # Create DataFrame of current misclassifications
    misclassified_mask = (y_true != y_pred)
    if misclassified_mask.any():
        current_misclass = pd.DataFrame({
            'Patient': sample_ids[misclassified_mask, 0].astype(int),
            'Appointment': sample_ids[misclassified_mask, 1].astype(int),
            'DFU': sample_ids[misclassified_mask, 2].astype(int),
            'True_Label': [CLASS_LABELS[int(y)] for y in y_true[misclassified_mask]],
            'Predicted_Label': [CLASS_LABELS[int(y)] for y in y_pred[misclassified_mask]]
        })
        
        # Add unique identifier column
        current_misclass['Sample_ID'] = (
            'P' + current_misclass['Patient'].astype(str).str.zfill(3) + 
            'A' + current_misclass['Appointment'].astype(str).str.zfill(2) + 
            'D' + current_misclass['DFU'].astype(str)
        )
        
        # Drop duplicates based on Sample_ID, True_Label, and Predicted_Label
        current_misclass = current_misclass.drop_duplicates(
            subset=['Sample_ID', 'True_Label', 'Predicted_Label']
        )
        # Save Modality-wise misclassifications
        if os.path.exists(misclass_file):
            # Load existing misclassifications
            existing_misclass = pd.read_csv(misclass_file)
            
            # Update counts for existing misclassifications
            for _, row in current_misclass.iterrows():
                mask = (
                    (existing_misclass['Sample_ID'] == row['Sample_ID']) &
                    (existing_misclass['True_Label'] == row['True_Label']) &
                    (existing_misclass['Predicted_Label'] == row['Predicted_Label'])
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
                    (existing_misclass['True_Label'] == row['True_Label']) &
                    (existing_misclass['Predicted_Label'] == row['Predicted_Label'])
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

    Args:
        data: Original DataFrame
        result_dir: Directory containing the misclassifications CSV
        thresholds: Dictionary with misclassification count thresholds for each class

    Returns:
        Filtered DataFrame
    """
    # Use organized output paths - read from misclassifications subdirectory
    output_paths = get_output_paths(result_dir)
    misclass_dir = output_paths['misclassifications']
    misclass_file = os.path.join(misclass_dir, 'frequent_misclassifications_saved.csv')
    if not os.path.exists(misclass_file):
        vprint("No misclassification, level=1 file found. Using original dataset.")
        return data
    
    # Load misclassification data
    misclass_df = pd.read_csv(misclass_file)
    
    # Create set of samples to exclude for each class
    samples_to_exclude = set()
    for phase, threshold in thresholds.items():
        high_misclass = misclass_df[
            (misclass_df['True_Label'] == phase) & 
            (misclass_df['Misclass_Count'] >= threshold)
        ]['Sample_ID'].tolist()
        samples_to_exclude.update(high_misclass)
    
    # Convert samples to exclude to a list and print info
    samples_to_exclude = list(samples_to_exclude)
    print(f"\nExcluding {len(samples_to_exclude)} frequently misclassified samples:")
    for phase in ['I', 'P', 'R']:
        count = len([s for s in samples_to_exclude if 
                    s in misclass_df[misclass_df['True_Label'] == phase]['Sample_ID'].values])
        print(f"Class {phase}: {count} samples")
    
    # Create sample ID column in original data
    data['Sample_ID'] = (
        'P' + data['Patient#'].astype(str).str.zfill(3) + 
        'A' + data['Appt#'].astype(str).str.zfill(2) + 
        'D' + data['DFU#'].astype(str)
    )
    
    # Filter out samples
    filtered_data = data[~data['Sample_ID'].isin(samples_to_exclude)].copy()
    
    # Remove temporary Sample_ID column
    filtered_data = filtered_data.drop('Sample_ID', axis=1)
    
    print(f"\nOriginal dataset size: {len(data)}")
    print(f"Filtered dataset size: {len(filtered_data)}")
    
    return filtered_data