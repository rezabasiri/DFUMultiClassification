
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import from original main
os.chdir(str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Import everything needed from main_original
# We'll import the module and modify its configs
import importlib.util
spec = importlib.util.spec_from_file_location("main_original", str(project_root / "src" / "main_original.py"))
main_orig = importlib.util.module_from_spec(spec)

# Execute the module to load all functions
spec.loader.exec_module(main_orig)

# Now call the function with our custom configs
# We need to replicate the logic from main_with_specialized_evaluation
# but with our custom modality list

configs = {
        'test_0': {'modalities': ['metadata']},
        'test_1': {'modalities': ['depth_rgb']},
        'test_2': {'modalities': ['depth_map']},
    }

print("Custom configs:", configs)
print("Starting original code execution...")

# Prepare dataset
data = main_orig.prepare_dataset(
    main_orig.depth_bb_file,
    main_orig.thermal_bb_file,
    main_orig.csv_file,
    list(set([mod for config in configs.values() for mod in config['modalities']]))
)

# Filter frequent misclassifications
data = main_orig.filter_frequent_misclassifications(
    data, main_orig.result_dir,
    thresholds={'I': 3, 'P': 2, 'R': 3}
)

if 10.0 < 100:
    data = data.sample(frac=10.0 / 100, random_state=42).reset_index(drop=True)

# Run cross-validation
print("\nStarting cross-validation...")
metrics, confusion_matrices, histories = main_orig.cross_validation_manual_split(
    data, configs, 0.8, 1
)

print("\nOriginal code execution complete!")
print("Metrics:", metrics)
