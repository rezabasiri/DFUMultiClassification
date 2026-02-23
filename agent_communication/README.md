# Agent Communication Directory

## ESSENTIAL CONTEXT

### Environment
- **Virtual Environment**: `/home/rezab/projects/enviroments/multimodal/bin`
- **Project Directory**: `/home/rezab/projects/DFUMultiClassification`
- **Activate Command**: `source /home/rezab/projects/enviroments/multimodal/bin/activate`
- **Platform**: Linux (Ubuntu/WSL)

### Project Overview
**DFU Multi-Classification**: Diabetic Foot Ulcer healing phase classification using multimodal data (metadata, depth images, thermal images).

**Classes**:
- 0 = I (Inflammation)
- 1 = P (Proliferation)
- 2 = R (Remodeling)

**Dataset**:
- Raw file: `data/raw/DataMaster_Processed_V12_WithMissing.csv`
- Loaded via `prepare_dataset()` function (creates `best_matching.csv`)
- ~600 samples across 3 classes
- Multiple modalities: metadata, depth_rgb, depth_map, thermal_map

---

## HOW TO RUN INVESTIGATION

### Setup (First Time)
```bash
# Activate environment
source /home/rezab/projects/enviroments/multimodal/bin/activate

# Navigate to project
cd /home/rezab/projects/DFUMultiClassification

## File Locations

### Input Files (Required)
- `balanced_combined_healing_phases.csv` - Main dataset (project root)
- `data/raw/DataMaster_Processed_V12_WithMissing.csv` - Raw data
- `data/raw/bb_depth_annotation.csv` - Depth bounding boxes
- `data/raw/bb_thermal_annotation.csv` - Thermal bounding boxes

### Code Being Tested
- `src/main.py` - Main training orchestration
- `src/training/training_utils.py` - Training loop implementation
- `src/data/dataset_utils.py` - Data preparation and batching
- `src/data/image_processing.py` - Data loading (`prepare_dataset()`)
- `src/models/losses.py` - Custom metrics and losses

---