import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import os

def get_display_name(modality_name):
    """Convert internal modality names to display names"""
    name_mapping = {
        'metadata': 'Metadata',
        'depth_rgb': 'RGB',
        'depth_map': 'Depth',
        'thermal_map': 'Thermal'
    }
    
    if '+' in modality_name:
        # Handle combined modalities
        parts = modality_name.split('+')
        return '+'.join(name_mapping.get(part.strip(), part.strip()) for part in parts)
    else:
        # Handle single modality
        return name_mapping.get(modality_name, modality_name)
    
def calculate_roc_curves(y_true, y_pred_proba, class_labels=['I', 'P', 'R']):
    """
    Calculate ROC curves for multi-class classification.
    
    Args:
        y_true: True labels (1D array of class indices)
        y_pred_proba: Predicted probabilities (2D array of shape (n_samples, n_classes))
        class_labels: List of class names
        
    Returns:
        Dictionary containing FPR, TPR, and AUC for each class and micro-average
    """
    # Binarize the labels for one-vs-rest ROC
    n_classes = len(class_labels)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # Calculate ROC curve and ROC area for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    # Calculate per-class ROC curves
    for i in range(n_classes):
        fpr[class_labels[i]], tpr[class_labels[i]], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc[class_labels[i]] = auc(fpr[class_labels[i]], tpr[class_labels[i]])
    
    # Calculate micro-average ROC curve
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Calculate macro-average ROC curve
    all_fpr = np.unique(np.concatenate([fpr[class_labels[i]] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[class_labels[i]], tpr[class_labels[i]])
    
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Create interpolated curves with fixed number of points
    n_points = 100
    interpolated_curves = []
    fpr_points = np.linspace(0, 1, n_points)
    
    for i in range(n_points):
        point = {
            'fpr': fpr_points[i],
            'tpr_micro': np.interp(fpr_points[i], fpr['micro'], tpr['micro']),
            'tpr_macro': np.interp(fpr_points[i], fpr['macro'], tpr['macro'])
        }
        for j, class_label in enumerate(class_labels):
            point[f'tpr_{class_label}'] = np.interp(fpr_points[i], fpr[class_label], tpr[class_label])
        interpolated_curves.append(point)
    
    return {
        'curves': interpolated_curves,
        'auc_scores': roc_auc,
        'raw_data': {
            'fpr': fpr,
            'tpr': tpr
        }
    }

def plot_class_specific_roc(y_true, y_pred_proba, modality_name, save_dir):
    """
    Generate class-specific ROC curves for a single modality or combination
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        modality_name: Name of the modality or combination
        save_dir: Directory to save the plots
    """
    plt.figure(figsize=(10, 10))
    
    # Calculate ROC curves
    roc_data = calculate_roc_curves(y_true, y_pred_proba)
    
    # Plot curves
    classes = ['I', 'P', 'R']
    colors = ['blue', 'green', 'red']
    
    # Plot random baseline
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    
    # Plot class-specific curves
    for cls, color in zip(classes, colors):
        fpr = roc_data['raw_data']['fpr'][cls]
        tpr = roc_data['raw_data']['tpr'][cls]
        auc_score = roc_data['auc_scores'][cls]
        plt.plot(fpr, tpr, color=color, 
                label=f'{cls} (AUC = {auc_score:.3f})',
                lw=2)

    # Add micro-average
    fpr_micro = roc_data['raw_data']['fpr']['micro']
    tpr_micro = roc_data['raw_data']['tpr']['micro']
    auc_micro = roc_data['auc_scores']['micro']
    plt.plot(fpr_micro, tpr_micro, color='darkorange',
            label=f'Combined (AUC = {auc_micro:.3f})',
            lw=2)

    # # Add macro-average
    # fpr_macro = roc_data['raw_data']['fpr']['macro']
    # tpr_macro = roc_data['raw_data']['tpr']['macro']
    # auc_macro = roc_data['auc_scores']['macro']
    # plt.plot(fpr_macro, tpr_macro, color='purple',
    #         label=f'Macro-average (AUC = {auc_macro:.3f})',
    #         lw=2)

    # Customize plot
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - {modality_name}')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # Save plot
    save_path = os.path.join(save_dir, f'roc_{modality_name.lower().replace(" ", "_")}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return roc_data

def get_config_key_for_modalities(modalities, config_mapping):
    """
    Get the config key (e.g., 'a32') for a given set of modalities
    """
    modalities_set = set(modalities)
    for key, config in config_mapping.items():
        if set(config['modalities']) == modalities_set:
            return key
    return None
def interpolate_roc_curve(fpr, tpr, n_points=1000):
    """Interpolate ROC curve to fixed number of points"""
    mean_fpr = np.linspace(0, 1, n_points)
    mean_tpr = np.interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    mean_tpr[-1] = 1.0
    return mean_fpr, mean_tpr

def calculate_roc_curves_multi_run(result_dir, config_key, n_runs=5):
    """Calculate ROC curves averaged across multiple runs"""
    all_runs_curves = {
        'I': {'tpr': [], 'fpr': [], 'auc': []},
        'P': {'tpr': [], 'fpr': [], 'auc': []},
        'R': {'tpr': [], 'fpr': [], 'auc': []},
        'micro': {'tpr': [], 'fpr': [], 'auc': []}
    }
    
    for run in range(1, n_runs + 1):
        try:
            pred_file = os.path.join(result_dir, f'pred_run{run}_{config_key}_valid.npy')
            label_file = os.path.join(result_dir, f'true_label_run{run}_{config_key}_valid.npy')
            
            if os.path.exists(pred_file) and os.path.exists(label_file):
                y_pred = np.load(pred_file)
                y_true = np.load(label_file)
                
                # Calculate per-run ROC curves
                roc_data = calculate_roc_curves(y_true, y_pred)
                
                # Store interpolated curves
                for class_label in ['I', 'P', 'R', 'micro']:
                    fpr = roc_data['raw_data']['fpr'][class_label]
                    tpr = roc_data['raw_data']['tpr'][class_label]
                    interp_fpr, interp_tpr = interpolate_roc_curve(fpr, tpr)
                    
                    all_runs_curves[class_label]['fpr'].append(interp_fpr)
                    all_runs_curves[class_label]['tpr'].append(interp_tpr)
                    all_runs_curves[class_label]['auc'].append(roc_data['auc_scores'][class_label])
                    
        except Exception as e:
            print(f"Error processing run {run} for {config_key}: {str(e)}")
    
    # Calculate mean and std for curves and AUC scores
    averaged_curves = {}
    for class_label in ['I', 'P', 'R', 'micro']:
        if all_runs_curves[class_label]['tpr']:
            # Calculate mean and std of TPR values
            tpr_values = np.array(all_runs_curves[class_label]['tpr'])
            mean_tpr = np.mean(tpr_values, axis=0)
            std_tpr = np.std(tpr_values, axis=0)
            
            # Calculate mean and std of AUC scores
            auc_scores = np.array(all_runs_curves[class_label]['auc'])
            mean_auc = np.mean(auc_scores)
            std_auc = np.std(auc_scores)
            
            averaged_curves[class_label] = {
                'fpr': all_runs_curves[class_label]['fpr'][0],  # Use first run's FPR values
                'mean_tpr': mean_tpr,
                'std_tpr': std_tpr,
                'mean_auc': mean_auc,
                'std_auc': std_auc
            }
    
    return averaged_curves

def plot_class_specific_roc_multi_run(result_dir, config_key, modality_name, save_dir, n_runs=5):
    """Generate class-specific ROC curves averaged across runs"""
    # Set font sizes
    SMALL_SIZE = 14
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 18
    LEGEND_SIZE = 16

    plt.figure(figsize=(10, 10))
    
    averaged_curves = calculate_roc_curves_multi_run(result_dir, config_key, n_runs)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
    
    colors = {
        'I': '#FF9999',
        'P': '#99CC99',
        'R': '#9999FF',
        'micro': '#FF7F0E'
    }
    
    for class_label in ['I', 'P', 'R', 'micro']:
        if class_label in averaged_curves:
            curve_data = averaged_curves[class_label]
            display_name = 'Combined' if class_label == 'micro' else class_label
            
            plt.plot(curve_data['fpr'], curve_data['mean_tpr'], 
                    color=colors[class_label],
                    label=f'{display_name} (AUC = {curve_data["mean_auc"]:.3f} ± {curve_data["std_auc"]:.3f})',
                    lw=2)
            
            plt.fill_between(curve_data['fpr'],
                           curve_data['mean_tpr'] - curve_data['std_tpr'],
                           curve_data['mean_tpr'] + curve_data['std_tpr'],
                           color=colors[class_label], alpha=0.2)
    
    plt.xlabel('False Positive Rate', fontsize=MEDIUM_SIZE)
    plt.ylabel('True Positive Rate', fontsize=MEDIUM_SIZE)
    plt.title(f'ROC Curves - {modality_name}\n(Averaged across {n_runs} runs)', 
             fontsize=BIGGER_SIZE, pad=20)
    plt.legend(loc='lower right', fontsize=LEGEND_SIZE)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=SMALL_SIZE)
    plt.yticks(fontsize=SMALL_SIZE)
    
    save_path = os.path.join(save_dir, f'roc_{modality_name.lower().replace(" ", "_")}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return averaged_curves
def analyze_modality_combinations_multi_run(result_dir, save_dir='roc_analysis', n_runs=5):
    """Analyze ROC curves for different modality combinations across multiple runs"""
    # Set font sizes
    SMALL_SIZE = 14  # For tick labels
    MEDIUM_SIZE = 16  # For axis labels
    BIGGER_SIZE = 18  # For titles
    LEGEND_SIZE = 16  # For legend text

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=LEGEND_SIZE)   # legend fontsize

    config_mapping = {
        'a32': {'modalities': ['metadata']},
        'b32': {'modalities': ['depth_rgb']},
        'c32': {'modalities': ['depth_map']},
        'd32': {'modalities': ['thermal_map']},
        'j32': {'modalities': ['depth_map', 'thermal_map']},
        'k32': {'modalities': ['metadata', 'depth_rgb', 'depth_map']},
        'p32': {'modalities': ['metadata', 'depth_rgb', 'depth_map', 'thermal_map']}
    }

    os.makedirs(os.path.join(result_dir, save_dir), exist_ok=True)
    
    plt.figure(figsize=(15, 10))
    plt.plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
    
    colors = sns.color_palette('husl', n_colors=len(config_mapping))
    modality_aucs = {}
    
    # Process each configuration
    for i, (config_key, config) in enumerate(config_mapping.items()):
        try:
            raw_name = '+'.join(config['modalities'])
            display_name = get_display_name(raw_name)
            averaged_curves = calculate_roc_curves_multi_run(result_dir, config_key, n_runs)
            
            if 'micro' in averaged_curves:
                curve_data = averaged_curves['micro']
                
                plt.plot(curve_data['fpr'], curve_data['mean_tpr'],
                        color=colors[i],
                        label=f'{display_name} (AUC = {curve_data["mean_auc"]:.3f} ± {curve_data["std_auc"]:.3f})',
                        lw=2)
                
                modality_aucs[display_name] = {
                    'mean': curve_data["mean_auc"],
                    'std': curve_data["std_auc"]
                }
                
                plot_class_specific_roc_multi_run(result_dir, config_key, display_name, 
                                                os.path.join(result_dir, save_dir), n_runs)
                
        except Exception as e:
            print(f"Error processing {config_key}: {str(e)}")
    
    plt.xlabel('False Positive Rate', fontsize=MEDIUM_SIZE)
    plt.ylabel('True Positive Rate', fontsize=MEDIUM_SIZE)
    plt.title(f'ROC Curves - Modality Comparison\n(Averaged across {n_runs} runs)', 
             fontsize=BIGGER_SIZE, pad=20)
    plt.legend(loc='lower right', fontsize=LEGEND_SIZE)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=SMALL_SIZE)
    plt.yticks(fontsize=SMALL_SIZE)
    
    save_path = os.path.join(result_dir, save_dir, 'modality_comparison_roc.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create AUC comparison bar plot
    plt.figure(figsize=(12, 6))
    modalities = list(modality_aucs.keys())
    means = [modality_aucs[m]['mean'] for m in modalities]
    stds = [modality_aucs[m]['std'] for m in modalities]
    
    bars = plt.bar(range(len(modalities)), means, yerr=stds, capsize=5)
    plt.xticks(range(len(modalities)), modalities, rotation=45, ha='right', fontsize=SMALL_SIZE)
    plt.yticks(fontsize=SMALL_SIZE)
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + stds[i],
                f'{means[i]:.3f}\n±{stds[i]:.3f}',
                ha='center', va='bottom', fontsize=SMALL_SIZE)
    
    plt.xlabel('Modality Configuration', fontsize=MEDIUM_SIZE)
    plt.ylabel('AUC Score', fontsize=MEDIUM_SIZE)
    plt.title(f'AUC Score Comparison Across Modalities\n(Averaged across {n_runs} runs)', 
             fontsize=BIGGER_SIZE, pad=20)
    plt.tight_layout()
    
    save_path = os.path.join(result_dir, save_dir, 'auc_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return modality_aucs
def analyze_modality_combinations(result_dir, save_dir='roc_analysis', run_number=1):
    """
    Analyze ROC curves for different modality combinations
    """
    # Define config mapping
    config_mapping = {
        'a32': {'modalities': ['metadata']},
        'b32': {'modalities': ['depth_rgb']},
        'c32': {'modalities': ['depth_map']},
        'd32': {'modalities': ['thermal_map']},
        'j32': {'modalities': ['depth_map', 'thermal_map']},
        'k32': {'modalities': ['metadata', 'depth_rgb', 'depth_map']},
        'p32': {'modalities': ['metadata', 'depth_rgb', 'depth_map', 'thermal_map']}
    }

    os.makedirs(os.path.join(result_dir, save_dir), exist_ok=True)
    
    plt.figure(figsize=(15, 10))
    plt.plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
    
    colors = sns.color_palette('husl', n_colors=len(config_mapping))
    modality_aucs = {}
    legend_handles = []

    # Process each configuration
    for i, (config_key, config) in enumerate(config_mapping.items()):
        try:
            # Construct file paths using config key
            pred_file = os.path.join(result_dir, f'pred_run{run_number}_{config_key}_valid.npy')
            label_file = os.path.join(result_dir, f'true_label_run{run_number}_{config_key}_valid.npy')
            
            if os.path.exists(pred_file) and os.path.exists(label_file):
                y_pred = np.load(pred_file)
                y_true = np.load(label_file)
                
                # Convert modality list to display name
                display_name = '+'.join(config['modalities'])
                
                roc_data = calculate_roc_curves(y_true, y_pred)
                fpr = roc_data['raw_data']['fpr']['micro']
                tpr = roc_data['raw_data']['tpr']['micro']
                auc_score = roc_data['auc_scores']['micro']
                
                line = plt.plot(fpr, tpr, color=colors[i], 
                              label=f'{display_name} (AUC = {auc_score:.3f})',
                              lw=2)
                legend_handles.append(line[0])
                modality_aucs[display_name] = auc_score
                
                # Also generate individual ROC plot for this configuration
                plot_class_specific_roc(y_true, y_pred, display_name, 
                                      os.path.join(result_dir, save_dir))
                
        except Exception as e:
            print(f"Error processing {config_key}: {str(e)}")
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Modality Comparison')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(result_dir, save_dir, 'modality_comparison_roc.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create AUC comparison bar plot
    plt.figure(figsize=(12, 6))
    modalities = list(modality_aucs.keys())
    aucs = list(modality_aucs.values())
    
    bars = plt.bar(range(len(modalities)), aucs)
    plt.xticks(range(len(modalities)), modalities, rotation=45, ha='right')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.xlabel('Modality Configuration')
    plt.ylabel('AUC Score')
    plt.title('AUC Score Comparison Across Modalities')
    plt.tight_layout()
    
    save_path = os.path.join(result_dir, save_dir, 'auc_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return modality_aucs

def calculate_confidence_intervals(y_true, y_pred_proba, n_bootstraps=1000):
    """
    Calculate confidence intervals for ROC curves using bootstrapping
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        n_bootstraps: Number of bootstrap samples
    """
    n_samples = len(y_true)
    roc_auc_list = []
    
    for i in range(n_bootstraps):
        # Bootstrap sample indices
        indices = np.random.randint(0, n_samples, n_samples)
        
        # Calculate ROC curves for this bootstrap sample
        if len(y_pred_proba.shape) > 1:  # Multi-class
            roc_data = calculate_roc_curves(y_true[indices], y_pred_proba[indices])
            roc_auc_list.append(roc_data['auc_scores']['micro'])
        else:  # Binary
            fpr, tpr, _ = roc_curve(y_true[indices], y_pred_proba[indices])
            roc_auc_list.append(auc(fpr, tpr))
    
    # Calculate confidence intervals
    confidence_intervals = np.percentile(roc_auc_list, [2.5, 97.5])
    
    return {
        'mean_auc': np.mean(roc_auc_list),
        'ci_lower': confidence_intervals[0],
        'ci_upper': confidence_intervals[1]
    }

def plot_roc_with_confidence(y_true, y_pred_proba, title, save_path):
    """
    Plot ROC curves with confidence intervals
    """
    plt.figure(figsize=(10, 10))
    
    # Calculate main ROC curve
    roc_data = calculate_roc_curves(y_true, y_pred_proba)
    
    # Calculate confidence intervals
    ci_data = calculate_confidence_intervals(y_true, y_pred_proba)
    
    # Plot main curve
    plt.plot(roc_data['raw_data']['fpr']['micro'], 
            roc_data['raw_data']['tpr']['micro'],
            'b-', label=f'ROC curve (AUC = {ci_data["mean_auc"]:.3f} ± {(ci_data["ci_upper"]-ci_data["ci_lower"])/2:.3f})')
    
    # Add random baseline
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    
    # Customize plot
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_full_roc_analysis(result_dir, n_runs=5):
    """Generate comprehensive ROC analysis averaged across runs"""
    save_dir = os.path.join(result_dir, 'roc_analysis')
    os.makedirs(save_dir, exist_ok=True)
    
    print("Analyzing modalities...")
    individual_aucs = analyze_modality_combinations_multi_run(result_dir, save_dir, n_runs)
    
    print(f"\nROC analysis completed. Results saved in {save_dir}")
    return individual_aucs

# Usage
result_dir = r"C:\Users\90rez\OneDrive - University of Toronto\PhDUofT\ZivotData\Codes\MultimodalClassification\Phase_Specefic_Calssification_With_Generative_Augmentation\results_dir\checkpoints"
aucs = generate_full_roc_analysis(result_dir, n_runs=5)

# Print AUC scores
for modality, scores in aucs.items():
    print(f"{modality}: AUC = {scores['mean']:.3f} ± {scores['std']:.3f}")