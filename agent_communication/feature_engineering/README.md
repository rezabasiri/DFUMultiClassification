# Feature Engineering Task

## Overview
This directory contains planning, implementation, and results for comprehensive feature engineering to improve DFU healing phase classification beyond the current ~47-50% baseline.

## Critical Context

### Current Performance Baseline
- **Accuracy**: ~47-50% (patient-level CV, metadata only)
- **Previous "97.6%" result**: **Data leakage** from sample-level split (invalidated)
- **True challenge**: Generalizing to unseen patients (233 patients total)

### Why Feature Engineering?
Current raw features alone achieve only ~50% accuracy. Clinical domain knowledge suggests temperature gradients, temporal patterns, and composite indices may better capture healing phase dynamics than raw measurements.

## Directory Structure

```
feature_engineering/
├── README.md           # This file - overview and context
├── PLAN.md            # Comprehensive feature engineering & selection plan
├── PROGRESS.md        # Task tracking (checkboxes updated as work proceeds)
├── results/           # Feature importance plots, selection reports, experiment logs
└── test_*.py          # Testing scripts for feature engineering strategies
```

## Key Documents

### PLAN.md (Main Document)
- **9 Feature Engineering Strategies**: Temperature, Temporal, Composite Scores, Biomechanical, Patient Risk, Wound Ratios, Polynomial, Missing Data, Clinical Signatures
- **5 Feature Selection Approaches**: Filter, Wrapper, Embedded, Stability, Clinical Validation
- **6 Implementation Phases**: Setup → Engineering → Selection → Validation → Training → Integration
- **Success Criteria**: >55-60% accuracy, Min F1 > 0.20, clinical interpretability

### PROGRESS.md (Task Tracker)
- Checkboxes for all tasks (updated as work completes)
- Current status and blockers
- Results summary

## Collaboration Protocol

### Remote Agent (Claude Code)
- Strategic planning and code review
- Domain knowledge integration from paper
- Mathematical validation of feature formulas
- Documentation and result interpretation

### Local Agent
- Feature engineering code implementation (src/features/)
- Testing and validation experiments
- Performance benchmarking
- Results visualization

### Handoff
1. Implement one strategy at a time
2. Test independently with patient-level CV
3. Commit with clear message describing what was added
4. Update PROGRESS.md checkboxes
5. Generate feature importance report
6. Move to next strategy

## Critical Requirements

### Data Leakage Prevention
- **ALL experiments**: Use patient-level CV (src/training/training_utils.py:create_patient_folds)
- **Feature engineering**: Apply AFTER CV split (inside fold loop)
- **Temporal features**: Only use current/past information, never future
- **Scaling**: Fit on train, transform on validation

### Validation Standards
- Report per-fold metrics: Accuracy, F1 Macro, F1 per class [I, P, R], Min F1
- Track variance across folds (high variance = overfitting)
- **Min F1 must be > 0.15** (ensure minority class R is learned)
- Compare against baseline ~47-50%

## Quick Start

1. Read PLAN.md for comprehensive strategy
2. Check PROGRESS.md for current status
3. Implement features in src/features/feature_engineering.py
4. Test with patient-level CV
5. Update PROGRESS.md
6. Commit and push

## References

- **Paper**: paper/main.tex (clinical context, GAMAN architecture)
- **Raw Data**: data/raw/DataMaster_Processed_V12_WithMissing.csv (65+ features)
- **Baseline Results**: agent_communication/test_metadata_phase9_cv_output.txt (47.3% accuracy)
- **Investigation Summaries**: agent_communication/metadata_investigation_summary.txt

---

**Created**: 2025-12-24
**Status**: Planning complete, ready for implementation
