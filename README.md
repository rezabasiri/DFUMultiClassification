# DFU Multi-Classification

Multimodal deep learning for Diabetic Foot Ulcer (DFU) healing phase classification using the GAMAN (Generative Adaptive Multimodal Attention Network) architecture.

## Overview

This project implements a multimodal classification system for DFU healing phases (Inflammatory, Proliferative, Remodeling) using:
- Clinical metadata
- RGB imaging
- Thermal imaging
- Depth sensing

## Key Features

- **GAMAN Architecture**: Dynamic late fusion with hierarchical attention mechanisms
- **Phase-Specific Classification**: Three-class healing phase classification
- **Multimodal Integration**: Combines multiple imaging modalities with clinical data
- **Ensemble Framework**: Adaptive phase weighting for improved performance

## Project Structure

```
DFUMultiClassification/
├── src/                    # Source code
│   ├── main.py            # Main training script
│   ├── data/              # Data processing modules
│   ├── models/            # Model architectures & losses
│   ├── training/          # Training utilities
│   ├── evaluation/        # Metrics & visualization
│   └── utils/             # Configuration & utilities
├── data/                  # Data directory
│   ├── raw/               # Raw images & CSV files
│   └── processed/         # Processed data
├── paper/                 # Research paper (LaTeX)
├── docs/                  # Documentation
├── scripts/               # Standalone scripts
└── archive/               # Archived/unused code
```

## Quick Start

1. Configure paths in `src/utils/config.py` for your environment
2. Run training: `python src/main.py`

## Paper

The research paper is available in the `paper/` directory. See `paper/main.tex` for the full manuscript.

## Citation

If you use this code, please cite the associated paper on Multimodal Healing Phase Classification of DFU.

## License

See LICENSE file for details.
