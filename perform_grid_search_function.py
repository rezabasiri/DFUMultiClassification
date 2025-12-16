# Standard libraries
import os
import csv
import gc
import argparse
import numpy as np

# TensorFlow
import tensorflow as tf


def perform_grid_search(data_percentage=100, train_patient_percentage=0.8, n_runs=3):
    """
    Perform grid search over loss function parameters.
    """
    # Define parameter grids
    param_grid = {
        'ordinal_weight': [0.1, 0.5, 1.0, 2.0, 6.5],
        'gamma': [0.1, 2.0, 3.0, 4.0],
        'alpha': [
            [1, 1, 1],  # Equal weights
            [0.598, 0.315, 1.597],  # Your current weights
            [1, 0.5, 2],  # Alternative weights
            [1.5, 0.5, 1.5]
        ]
    }
    
    # Create results CSV
    results_file = os.path.join(result_dir, 'loss_parameter_search_results.csv')
    fieldnames = ['ordinal_weight', 'gamma', 'alpha', 
                  'accuracy', 'f1_macro', 'f1_weighted',
                  'f1_I', 'f1_P', 'f1_R', 'kappa']
    
    # Create CSV file with headers if it doesn't exist
    if not os.path.exists(results_file):
        with open(results_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    
    # Track best parameters
    best_score = 0
    best_params = None
    
    # Iterate over all parameter combinations
    for ordinal_weight in param_grid['ordinal_weight']:
        for gamma in param_grid['gamma']:
            for alpha in param_grid['alpha']:
                print(f"\nTesting parameters:")
                print(f"Ordinal Weight: {ordinal_weight}")
                print(f"Gamma: {gamma}")
                print(f"Alpha: {alpha}")
                
                # Define custom loss function with current parameters
                def get_current_loss():
                    return get_focal_ordinal_loss(
                        num_classes=3,
                        ordinal_weight=ordinal_weight,
                        gamma=gamma,
                        alpha=alpha
                    )
                
                # Store original loss function
                original_loss = globals().get('get_focal_ordinal_loss')
                
                try:
                    # Replace loss function temporarily
                    globals()['get_focal_ordinal_loss'] = get_current_loss
                    
                    # Run your existing main function
                    metrics = main_search(
                        data_percentage=data_percentage,
                        train_patient_percentage=train_patient_percentage,
                        n_runs=n_runs
                    )
                    
                    # Prepare results
                    result = {
                        'ordinal_weight': ordinal_weight,
                        'gamma': gamma,
                        'alpha': str(alpha),  # Convert list to string for CSV
                        'accuracy': metrics.get('accuracy', 0),
                        'f1_macro': metrics.get('f1_macro', 0),
                        'f1_weighted': metrics.get('f1_weighted', 0),
                        'f1_I': metrics.get('f1_classes', [0, 0, 0])[0],
                        'f1_P': metrics.get('f1_classes', [0, 0, 0])[1],
                        'f1_R': metrics.get('f1_classes', [0, 0, 0])[2],
                        'kappa': metrics.get('kappa', 0)
                    }
                    
                    # Append results to CSV
                    with open(results_file, 'a', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writerow(result)
                    
                    # Update best parameters based on f1_weighted
                    current_score = result['f1_weighted']
                    if current_score > best_score:
                        best_score = current_score
                        best_params = {
                            'ordinal_weight': ordinal_weight,
                            'gamma': gamma,
                            'alpha': alpha
                        }
                        
                    print(f"\nCurrent Results:")
                    print(f"F1 Weighted: {result['f1_weighted']:.4f}")
                    print(f"Kappa: {result['kappa']:.4f}")
                    
                finally:
                    # Restore original loss function
                    globals()['get_focal_ordinal_loss'] = original_loss
                    
                # Clear memory
                tf.keras.backend.clear_session()
                gc.collect()
    
    print("\nGrid Search Complete!")
    print("\nBest Parameters:")
    print(f"Ordinal Weight: {best_params['ordinal_weight']}")
    print(f"Gamma: {best_params['gamma']}")
    print(f"Alpha: {best_params['alpha']}")
    print(f"Best F1 Weighted: {best_score:.4f}")
    
    return best_params, results_file