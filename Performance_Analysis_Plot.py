import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
import os


def create_performance_plots(result_dir, n_runs=5):
    """
    Create performance progression and per-class analysis plots averaging across runs
    """
    # Configuration mapping
    config_mapping = {
        'a32': {'modalities': ['metadata'], 'num_modalities': 1},
        'b32': {'modalities': ['depth_rgb'], 'num_modalities': 1},
        'c32': {'modalities': ['depth_map'], 'num_modalities': 1},
        'd32': {'modalities': ['thermal_map'], 'num_modalities': 1},
        'e32': {'modalities': ['metadata', 'depth_rgb'], 'num_modalities': 2},
        'h32': {'modalities': ['depth_rgb', 'depth_map'], 'num_modalities': 2},
        'i32': {'modalities': ['depth_rgb', 'thermal_map'], 'num_modalities': 2},
        'j32': {'modalities': ['depth_map', 'thermal_map'], 'num_modalities': 2},
        'k32': {'modalities': ['metadata', 'depth_rgb', 'depth_map'], 'num_modalities': 3},
        'z32': {'modalities': ['metadata', 'depth_rgb', 'thermal_map'], 'num_modalities': 3},
        'm32': {'modalities': ['metadata', 'depth_map', 'thermal_map'], 'num_modalities': 3},
        'n32': {'modalities': ['depth_rgb', 'depth_map', 'thermal_map'], 'num_modalities': 3},
        'p32': {'modalities': ['metadata', 'depth_rgb', 'depth_map', 'thermal_map'], 'num_modalities': 4},
    }

    # Initialize data structures to store results across runs
    all_runs_results = {config_key: {
        'f1_weighted': [],
        'f1_I': [],
        'f1_P': [],
        'f1_R': []
    } for config_key in config_mapping.keys()}

    # Collect results from all runs
    for run_number in range(1, n_runs + 1):
        for config_key, config in config_mapping.items():
            try:
                pred_file = os.path.join(result_dir, f'pred_run{run_number}_{config_key}_valid.npy')
                label_file = os.path.join(result_dir, f'true_label_run{run_number}_{config_key}_valid.npy')
                
                if os.path.exists(pred_file) and os.path.exists(label_file):
                    y_pred = np.load(pred_file)
                    y_true = np.load(label_file)
                    
                    # Calculate metrics
                    y_pred_classes = np.argmax(y_pred, axis=1)
                    f1_per_class = f1_score(y_true, y_pred_classes, average=None)
                    f1_weighted = f1_score(y_true, y_pred_classes, average='weighted')
                    
                    # Store results
                    all_runs_results[config_key]['f1_weighted'].append(f1_weighted)
                    all_runs_results[config_key]['f1_I'].append(f1_per_class[0])
                    all_runs_results[config_key]['f1_P'].append(f1_per_class[1])
                    all_runs_results[config_key]['f1_R'].append(f1_per_class[2])
                    
            except Exception as e:
                print(f"Error processing {config_key} run {run_number}: {str(e)}")

    # Calculate averages and standard deviations
    averaged_results = []
    for config_key, config in config_mapping.items():
        if all_runs_results[config_key]['f1_weighted']:  # If we have results for this config
            averaged_results.append({
                'config': config_key,
                'modalities': '+'.join(config['modalities']),
                'num_modalities': config['num_modalities'],
                'f1_weighted': np.mean(all_runs_results[config_key]['f1_weighted']),
                'f1_weighted_std': np.std(all_runs_results[config_key]['f1_weighted']),
                'f1_I': np.mean(all_runs_results[config_key]['f1_I']),
                'f1_I_std': np.std(all_runs_results[config_key]['f1_I']),
                'f1_P': np.mean(all_runs_results[config_key]['f1_P']),
                'f1_P_std': np.std(all_runs_results[config_key]['f1_P']),
                'f1_R': np.mean(all_runs_results[config_key]['f1_R']),
                'f1_R_std': np.std(all_runs_results[config_key]['f1_R'])
            })

    # 1. Performance Progression Plot
    plt.figure(figsize=(12, 6))
    
    # Group by number of modalities
    modality_groups = {}
    for result in averaged_results:
        num_mod = result['num_modalities']
        if num_mod not in modality_groups:
            modality_groups[num_mod] = {
                'means': [],
                'stds': []
            }
        modality_groups[num_mod]['means'].append(result['f1_weighted'])
        modality_groups[num_mod]['stds'].append(result['f1_weighted_std'])

    # Calculate means and std for each group
    x = sorted(modality_groups.keys())
    means = [np.mean(modality_groups[k]['means']) for k in x]
    stds = [np.sqrt(np.mean(np.square(modality_groups[k]['stds']))) for k in x]  # Combined std

    plt.errorbar(x, means, yerr=stds, fmt='o-', capsize=5, capthick=2, 
                linewidth=2, markersize=8, label='Mean Weighted F1-Score')

    # Add individual points
    for result in averaged_results:
        plt.scatter(result['num_modalities'], result['f1_weighted'], 
                   alpha=0.4, color='gray')

    plt.xlabel('Number of Modalities')
    plt.ylabel('Weighted F1-Score')
    plt.title(f'Performance Progression with Increasing Modalities\n(Averaged across {n_runs} runs)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save progression plot
    plt.savefig(os.path.join(result_dir, 'performance_progression_averaged2.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Per-Class Performance Analysis
    plt.figure(figsize=(15, 6))
    
    # Sort results by number of modalities and then by weighted F1
    averaged_results.sort(key=lambda x: (x['num_modalities'], x['f1_weighted']))
    
    # Prepare data for plotting
    labels = [r['modalities'] for r in averaged_results]
    f1_I = [r['f1_I'] for r in averaged_results]
    f1_I_std = [r['f1_I_std'] for r in averaged_results]
    f1_P = [r['f1_P'] for r in averaged_results]
    f1_P_std = [r['f1_P_std'] for r in averaged_results]
    f1_R = [r['f1_R'] for r in averaged_results]
    f1_R_std = [r['f1_R_std'] for r in averaged_results]

    x = np.arange(len(labels))
    width = 0.25

    # Create grouped bar plot with error bars
    plt.bar(x - width, f1_I, width, yerr=f1_I_std, label='Inflammatory', 
            color='#ff9999', capsize=3)
    plt.bar(x, f1_P, width, yerr=f1_P_std, label='Proliferative', 
            color='#99ff99', capsize=3)
    plt.bar(x + width, f1_R, width, yerr=f1_R_std, label='Remodeling', 
            color='#9999ff', capsize=3)

    plt.xlabel('Modality Combinations')
    plt.ylabel('F1-Score')
    plt.title(f'Per-Class Performance Across Modality Combinations\n(Averaged across {n_runs} runs)')
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # # Add value labels on top of bars
    # def add_value_labels(bars, stds):
    #     for bar, std in zip(bars, stds):
    #         height = bar.get_height()
    #         plt.text(bar.get_x() + bar.get_width()/2., height,
    #                 f'{height:.2f}±{std:.2f}',
    #                 ha='center', va='bottom', rotation=0,
    #                 fontsize=8)

    # # Add value labels for all bar groups
    # add_value_labels(plt.gca().containers[0], f1_I_std)
    # add_value_labels(plt.gca().containers[1], f1_P_std)
    # add_value_labels(plt.gca().containers[2], f1_R_std)

    plt.tight_layout()
    
    # Save per-class analysis plot
    plt.savefig(os.path.join(result_dir, 'per_class_performance_averaged2.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

    return averaged_results

# Usage example
# result_dir = r"C:\Users\90rez\OneDrive - University of Toronto\PhDUofT\ZivotData\Codes\MultimodalClassification\Phase_Specefic_Calssification_With_Generative_Augmentation\results_dir\checkpoints"
result_dir = r"C:\Users\90rez\OneDrive - University of Toronto\PhDUofT\GoogleDrive\PhD_Academic\Papers\Paper3_JPaper_MultiModalClassification\Code\checkpoints"
results = create_performance_plots(result_dir, n_runs=5)