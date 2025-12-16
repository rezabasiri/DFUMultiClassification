# Testing Guide

This guide explains how to test the DFU Multi-Classification workflow with demo data.

## Quick Test

The `test_workflow.py` script provides a comprehensive test of the entire pipeline with minimal computational requirements.

### Prerequisites

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Prepare demo data:**
Ensure you have the following in `data/raw/`:
- Depth_RGB/ (with at least 20 images)
- Depth_Map_IMG/ (with at least 20 images)
- Thermal_RGB/ (with at least 20 images)
- Thermal_Map_IMG/ (with at least 20 images)
- DataMaster_Processed_V12_WithMissing.csv
- bounding_box_depth.csv
- bounding_box_thermal.csv

### Running the Test

```bash
python test_workflow.py
```

### What the Test Does

The test script performs the following steps with extensive debugging output:

1. **Environment Configuration**
   - Sets random seeds
   - Checks GPU availability
   - Configures paths

2. **Data Discovery**
   - Scans for image files
   - Checks CSV files
   - Reports data statistics

3. **Dataset Preparation**
   - Creates best matching dataset
   - Reports phase distribution
   - Shows sample information

4. **Test Configuration**
   - Uses minimal computational requirements:
     - 2 modalities (metadata + depth_rgb)
     - Batch size: 4
     - Epochs: 5
     - Image size: 64x64
     - No augmentation

5. **Data Splitting**
   - Splits by patient (80/20)
   - Reports train/val distribution
   - Shows patient IDs

6. **Dataset Creation**
   - Creates TensorFlow datasets
   - Computes class weights
   - Tests dataset iteration

7. **Model Building**
   - Builds multimodal model
   - Reports architecture details
   - Shows parameter counts

8. **Model Compilation**
   - Compiles with focal ordinal loss
   - Uses Adam optimizer

9. **Training**
   - Trains for 5 epochs
   - Shows progress for each epoch
   - Uses early stopping

10. **Evaluation**
    - Evaluates on validation set
    - Generates predictions
    - Shows classification report
    - Displays confusion matrix

11. **Save Results**
    - Saves test results to file

### Expected Output

The script will print detailed information at each step. Look for:
- ✓ marks indicating successful steps
- ✗ marks indicating errors
- ℹ marks indicating informational messages
- ⚠ marks indicating warnings

### Test Results

Results are saved to: `results/test_workflow_results.txt`

### Troubleshooting

#### Out of Memory Errors

If you get OOM errors:
```python
# Edit test_workflow.py and reduce:
TEST_CONFIG = {
    'batch_size': 2,  # Reduce from 4 to 2
    'image_size': 32,  # Reduce from 64 to 32
}
```

#### Missing Data Errors

Ensure your CSV files have:
- `Healing_Phase_cat` column with values: 'I', 'P', or 'R'
- `Patient#`, `Appt#`, `DFU#` columns
- Corresponding image filenames

#### Import Errors

Run from project root:
```bash
cd /path/to/DFUMultiClassification
python test_workflow.py
```

### Next Steps After Successful Test

1. **Review results:**
```bash
cat results/test_workflow_results.txt
```

2. **Run with more modalities:**
Edit `test_workflow.py`:
```python
TEST_CONFIG = {
    'selected_modalities': ['metadata', 'depth_rgb', 'thermal_rgb'],
    # ...
}
```

3. **Run full training:**
```bash
python src/main.py
```

## Testing Individual Modules

### Test Data Loading
```python
from src.data.image_processing import create_best_matching_dataset
from src.utils.config import get_data_paths

data_paths = get_data_paths()
df = create_best_matching_dataset(
    data_paths['bb_depth_csv'],
    data_paths['bb_thermal_csv'],
    data_paths['csv_file'],
    data_paths['depth_folder'],
    data_paths['thermal_folder'],
    'test_matching.csv'
)
print(f"Loaded {len(df)} samples")
```

### Test Model Building
```python
from src.models.builders import create_multimodal_model

input_shapes = {
    'metadata': (65,),
    'depth_rgb': (64, 64, 3)
}
selected_modalities = ['metadata', 'depth_rgb']
class_weights = {0: 1.0, 1: 1.0, 2: 1.0}

model = create_multimodal_model(input_shapes, selected_modalities, class_weights)
print(f"Model parameters: {model.count_params():,}")
```

## Performance Benchmarks

Expected performance on demo data (20 images per modality):

- **Data loading**: < 10 seconds
- **Dataset creation**: < 5 seconds
- **Model building**: < 2 seconds
- **Training (5 epochs)**: 1-5 minutes (depending on hardware)
- **Evaluation**: < 1 second

### Hardware Recommendations for Testing

**Minimum:**
- CPU: 2 cores
- RAM: 8GB
- No GPU required for demo data

**Recommended:**
- CPU: 4 cores
- RAM: 16GB
- GPU: Any CUDA-capable GPU
