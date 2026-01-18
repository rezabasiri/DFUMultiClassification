# Generative Augmentation Test

## Quick Start

### Phase 1: Error Check (Run First!)
```bash
cd /home/user/DFUMultiClassification
python agent_communication/generative_augmentation/test_generative_aug.py --quick --fresh
```
- Tests with 10% data, 2 epochs (~10-30 min)
- Verifies no errors before production run
- If errors: check gengen_test.log and report to cloud agent

### Phase 2: Production Run (After Phase 1 Passes)
```bash
python agent_communication/generative_augmentation/test_generative_aug.py --fresh
```
- Full test with 100% data, 300 epochs (~4-8 hours)
- Generates effectiveness comparison report

### After Completion
Commit these files:
```bash
git add agent_communication/generative_augmentation/GENGEN_*.* agent_communication/generative_augmentation/gengen_test.log
git commit -m "test: Generative augmentation effectiveness results"
git push
```

---

## What It Tests

**Modalities:** metadata, depth_rgb, thermal_map, depth_map (fixed)

**Test 1:** Baseline (no generative augmentation)
**Test 2:** With generative augmentation (Stable Diffusion on depth_rgb)

**Output:**
- `gengen_test.log` - Detailed execution log
- `GENGEN_PROGRESS.json` - Resumable progress tracker
- `GENGEN_REPORT.txt` - Final comparison with recommendation

---

## Resumability

If interrupted, resume with:
```bash
python agent_communication/generative_augmentation/test_generative_aug.py
```

---

## Configuration

All settings in `src/utils/production_config.py` (lines 70-80):
- `USE_GENERATIVE_AUGMENTATION` - Master switch (False/True)
- `GENERATIVE_AUG_MODEL_PATH` - Model directory
- `GENERATIVE_AUG_PROB` - Application probability (0.50)
- `GENERATIVE_AUG_MIX_RATIO` - Synthetic/real mix (0.01-0.05)

See INVESTIGATION_NOTES.md for technical details.
See MODEL_INSPECTION_REPORT.txt for model specifications.
