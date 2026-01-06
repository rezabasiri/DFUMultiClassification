# Local Agent Guide: Multimodal Outlier Detection

**Project:** DFU Classification with Combination-Specific Outlier Detection
**Context:** See `PROJECT_DESCRIPTION.md` and `../fusion_fix/FUSION_FIX_GUIDE.md`

---

## Quick Start

### 1. Fix Current Bugs (Required First)

**Bug 1: Path operator error** (`src/utils/outlier_detection.py`)
```python
# Add after EACH line: _, _, root = get_project_paths()
# At lines: 41, 172, 232, 305, 456, 626
root = Path(root)  # Convert string to Path object
```

**Bug 2: Checkpoint extension** (`src/training/training_utils.py:161`)
```python
# Change:
checkpoint_name = f'{modality_str}_{run}_{config_name}.ckpt'
# To:
checkpoint_name = f'{modality_str}_{run}_{config_name}.weights.h5'
```

### 2. Build Feature Cache

```bash
python scripts/precompute_outlier_features.py --image-size 32 --modalities all --device-mode single
```

**Expected output:**
- 5 `.npy` files in `cache_outlier/`
- Runtime: ~5-10 minutes
- Cache size: ~1.8 MB total

### 3. Run Training Pipeline

```bash
python src/main.py --mode search --device-mode single --resume_mode fresh
```

**Config check:** Verify `src/utils/production_config.py`:
```python
OUTLIER_REMOVAL = True
OUTLIER_CONTAMINATION = 0.15
INCLUDED_COMBINATIONS = [('metadata', 'thermal_map')]
```

---

## Expected Performance

**Target:** Kappa 0.2714 ± 0.08 (metadata + thermal_map with 15% cleaning)

| Metric | Baseline (no cleaning) | With 15% Outlier Removal | Improvement |
|--------|------------------------|--------------------------|-------------|
| **Kappa** | 0.1664 ± 0.05 | **0.2714 ± 0.08** | **+63%** |
| Fold 1 | 0.1857 | 0.2132 | +15% |
| Fold 2 | 0.1091 | 0.2366 | +117% |
| Fold 3 | 0.2043 | 0.3644 | +78% |

**Success criteria:** Final Kappa ≥ 0.26 (acceptable range)

---

## Monitoring Progress

### Key Checkpoints

**1. Feature Cache Build:**
```
✓ Saved to: thermal_map_features_32.npy
  Shape: (3107, 32), Size: 0.4 MB
```

**2. Outlier Detection:**
```
[1/1] metadata_thermal_map
  ✓ Applied for metadata_thermal_map: 2642 samples
    Removed 465 outliers
```

**3. Training (per fold):**
```
Fold 1/3
Training metadata+thermal_map with modalities: ['metadata', 'thermal_map']
Epoch 50/300 - val_loss: 0.85 - val_weighted_f1_score: 0.65
```

**4. Final Results:**
```
metadata + thermal_map:
  Kappa: 0.27 (±0.08)
  Weighted F1: 0.67
```

---

## Bug Fixing Protocol

### Small Bugs (Fix Locally)

**Examples:**
- Import errors → Add missing imports
- Path issues → Check file exists, use absolute paths
- Deprecation warnings → Update syntax (e.g., `tf.keras` → `keras`)
- Type mismatches → Add type conversion

**Action:** Fix immediately, test, continue

### Major Bugs (Report to Cloud)

**Examples:**
- Training crashes after 20+ minutes
- Kappa < 0.20 (significantly below target)
- Memory errors (OOM)
- Model architecture incompatibilities
- Incorrect feature dimensions

**Action:** Create bug report (template below)

---

## Bug Report Template

Create: `agent_communication/outlier_detection/BUG_REPORT_YYYYMMDD.md`

```markdown
# Bug Report: [Brief Title]

**Date:** YYYY-MM-DD HH:MM
**Severity:** Critical / Major / Minor

## Issue

[1-2 sentence description]

## Error Message

```
[Exact error traceback]
```

## Context

- **Command:** [exact command run]
- **Config:** [relevant production_config.py values]
- **Progress:** [how far into pipeline: cache/outlier/training/fold X]

## Attempted Fixes

1. [What you tried]
2. [Result]

## Current State

- Can continue: Yes/No
- Workaround: [if any]
- Blocks: [what's blocked]

## Logs

[Attach last 50 lines of output]
```

**Then:** Commit and push to branch `claude/run-dataset-polishing-X1NHe`

---

## Common Warnings (Safe to Ignore)

✅ **TensorFlow INFO messages:**
```
I0000 ... Loaded cuDNN version 91002
I0000 ... Omitted potentially buggy algorithm eng14{}
```
→ Informational only, XLA working correctly

✅ **Sklearn deprecation warnings:**
```
FutureWarning: `BaseEstimator._check_n_features` is deprecated
```
→ Library version mismatch, non-critical

✅ **Cache dataset warnings:**
```
W ... The calling iterator did not fully read the dataset being cached
```
→ TensorFlow optimization message, expected

---

## Progress Checklist

**Phase 1: Setup**
- [ ] Bugs fixed (Path conversion, checkpoint extension)
- [ ] Feature cache built (5 files in `cache_outlier/`)
- [ ] Config verified (`OUTLIER_REMOVAL = True`)

**Phase 2: Execution**
- [ ] Outlier detection ran (cleaned dataset in `data/cleaned/`)
- [ ] Fold 1/3 completed
- [ ] Fold 2/3 completed
- [ ] Fold 3/3 completed

**Phase 3: Validation**
- [ ] Results saved to `results/csv/modality_combination_results.csv`
- [ ] Kappa ≥ 0.26 (success) or < 0.26 (report issue)
- [ ] No crashes or major errors

**Phase 4: Report**
- [ ] If success: Note final Kappa, commit results
- [ ] If issues: Create bug report, commit and push

---

## Quick Commands

```bash
# Check git status
git status

# View last 30 lines of output
tail -30 /path/to/log.txt

# Monitor GPU usage
nvidia-smi

# Check cache files
ls -lh cache_outlier/

# Check cleaned datasets
ls -lh data/cleaned/

# Commit bug report
git add agent_communication/outlier_detection/BUG_REPORT_*.md
git commit -m "docs: Report bug in [component]"
git push -u origin claude/run-dataset-polishing-X1NHe
```

---

## Timeline Estimate

| Phase | Time | Cumulative |
|-------|------|------------|
| Bug fixes | 5 min | 5 min |
| Feature cache | 8 min | 13 min |
| Outlier detection | 2 min | 15 min |
| Training (3 folds) | 15 min | 30 min |
| **Total** | **~30 min** | - |

**If issues arise:** Add debugging time, create bug report

---

## Success Output

```
================================================================================
FINAL SUMMARY - BEST MODALITY COMBINATIONS
================================================================================

Top 3 Combinations by Kappa:
1. metadata + thermal_map: Kappa 0.27 (±0.08), Weighted F1 0.67 (±0.05)

Results saved to: results/csv/modality_combination_results.csv
```

**Next step:** Commit results, push to branch

---

**Questions?** See `PROJECT_DESCRIPTION.md` for detailed context
**Reference:** `../fusion_fix/FUSION_FIX_GUIDE.md` for Phase 7 baseline results
