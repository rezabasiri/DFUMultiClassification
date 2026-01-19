# Local Agent Test Instructions
# End-to-End Validation of High-Quality Wound Generation Training System

**Purpose**: Validate the newly implemented training system before production runs. Catch and fix easy errors, report systemic issues back to cloud agent.

**Expected Time**: 15-30 minutes for quick test, 1-2 hours if running optional training test

---

## Overview

You need to test a new high-quality generative model training system that will replace the current underperforming generative augmentation (which currently hurts performance by -10.4% kappa).

The system includes:
- Multi-GPU training with LoRA adaptation
- Quality metrics (FID, SSIM, LPIPS, IS)
- Perceptual loss and EMA for maximum quality
- Full checkpoint management with resume capability
- Highly configurable YAML-based settings

Your job: Run comprehensive tests, fix obvious issues (missing dependencies, typos, simple bugs), and report back any systemic problems or unexpected behavior.

---

## Prerequisites

### 1. Environment Setup

**Location**: `/home/user/DFUMultiClassification/agent_communication/generative_augmentation`

```bash
cd /home/user/DFUMultiClassification/agent_communication/generative_augmentation
```

### 2. Install Dependencies

```bash
pip install torch torchvision diffusers transformers accelerate peft torchmetrics lpips pandas pyyaml tqdm pillow xformers tensorboard
```

**Note**: If `xformers` fails to install, that's okay - it's optional. The system will work without it (just slower).

### 3. Verify Data Availability

Check that wound image datasets exist:
```bash
ls -la /home/user/DFUMultiClassification/datasets/polished_data/depth_rgb/phase_I/
ls -la /home/user/DFUMultiClassification/datasets/polished_data/depth_rgb/phase_R/
```

Expected: At least 10 images in each directory (*.png or *.jpg files).

---

## Test Procedure

### Step 1: Run Comprehensive System Test

This validates all components without actually training:

```bash
cd /home/user/DFUMultiClassification/agent_communication/generative_augmentation
python scripts/test_training_system.py
```

**What it tests:**
1. ✓ Dependencies (torch, diffusers, transformers, etc.)
2. ✓ GPU availability (checks CUDA, counts GPUs)
3. ✓ Configuration loading (YAML parsing)
4. ✓ Dataset availability (checks images exist)
5. ✓ Custom module imports (quality_metrics, data_loader, etc.)
6. ✓ Base model loading (Stable Diffusion v2.1 Base)
7. ✓ Dataset loading (creates dataloaders, loads batches)
8. ✓ Quality metrics (SSIM, LPIPS computation)
9. ✓ Checkpoint management (save/load cycle)
10. ⚠ Quick training test (skipped by default)

**Expected Output:**
```
================================================================================
TRAINING SYSTEM END-TO-END TEST
================================================================================
Start time: 2026-01-19 ...

================================================================================
TEST 1: DEPENDENCY CHECKS
================================================================================
✓ PASS: Import PyTorch
✓ PASS: Import TorchVision
✓ PASS: Import Diffusers
...

================================================================================
TEST SUMMARY
================================================================================
Total tests: 25-30
Passed: 25-30 ✓
Failed: 0 ✗
Warnings: 0-2 ⚠

✓ ALL TESTS PASSED
System is ready for training!
================================================================================
```

**If tests fail:**
- Read the error messages carefully
- Check `reports/test_results.json` for details
- See "Common Issues" section below
- Fix obvious problems (typos, missing dependencies, etc.)
- Re-run the test
- If you can't fix it, report to cloud agent (see "Reporting Back" section)

### Step 2: Optional - Run Quick Training Test

This actually runs 1 epoch of training to verify the full pipeline works:

```bash
cd /home/user/DFUMultiClassification/agent_communication/generative_augmentation
RUN_TRAINING_TEST=true python scripts/test_training_system.py
```

**Warning**: This will take 5-15 minutes and requires GPU. Only run if Step 1 passed completely.

**Expected**: Training should complete without errors. You should see:
- Model loading messages
- Training progress bars
- Loss decreasing
- Checkpoint saved successfully
- Validation samples generated

### Step 3: Inspect Generated Files

After tests complete, check that these files were created:

```bash
ls -la reports/test_results.json
ls -la checkpoints/test_checkpoint/  # If you ran optional training test
```

**View test results:**
```bash
cat reports/test_results.json | python -m json.tool
```

---

## What to Check

### ✓ Expected Behavior

1. **All tests pass** (or only warnings, no failures)
2. **GPU detected**: Should see "CUDA Available: True, 2 GPU(s) detected"
3. **Dataset found**: Should find images in phase_I and phase_R directories
4. **Models load**: Stable Diffusion v2.1 Base downloads and loads (may take time on first run)
5. **Metrics work**: SSIM and LPIPS compute successfully on dummy data
6. **Checkpoints work**: Save/load cycle completes without errors

### ⚠ Warnings (Acceptable)

1. **xFormers not installed** - Optional optimization, system works without it
2. **Quick training test skipped** - Expected if you didn't set RUN_TRAINING_TEST=true
3. **Small dataset warning** - If < 10 images, acceptable for testing

### ✗ Failures (Must Fix or Report)

1. **Missing dependencies** - Install them with pip
2. **No CUDA GPUs** - System will work but be very slow; report this
3. **Dataset not found** - Check data paths in quick_test_config.yaml
4. **Import errors** - Check for typos in code, report if you can't fix
5. **Model loading failures** - Check internet connection (downloads models)
6. **OOM errors** - Reduce batch_size in quick_test_config.yaml

---

## Common Issues and Fixes

### Issue 1: Missing Dependencies
```
✗ FAIL: Import Diffusers
  ModuleNotFoundError: No module named 'diffusers'
```

**Fix:**
```bash
pip install diffusers transformers accelerate peft
```

### Issue 2: Dataset Not Found
```
✗ FAIL: Dataset directory exists
  Directory not found: /path/to/dataset
```

**Fix:** Check the data path in `configs/quick_test_config.yaml`:
```yaml
data:
  data_root: "/home/user/DFUMultiClassification/datasets/polished_data"
```

Make sure this path exists and contains the expected structure:
```
polished_data/
  depth_rgb/
    phase_I/
      *.png or *.jpg
    phase_R/
      *.png or *.jpg
```

### Issue 3: CUDA Out of Memory
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Fix:** Edit `configs/quick_test_config.yaml`:
```yaml
training:
  batch_size_per_gpu: 1  # Reduce from 2
```

Also try:
```yaml
training:
  gradient_checkpointing: true
  mixed_precision: "fp16"
```

### Issue 4: Model Download Fails
```
✗ FAIL: Load base model
  Connection error: ...
```

**Fix:**
1. Check internet connection
2. Retry (may be temporary Hugging Face server issue)
3. If persistent, report to cloud agent

### Issue 5: Import Errors in Custom Modules
```
✗ FAIL: Import quality_metrics
  ImportError: cannot import name 'QualityMetrics'
```

**Fix:**
1. Check for typos in `scripts/utils/quality_metrics.py`
2. Ensure all files are in correct locations:
   ```
   scripts/
     utils/
       quality_metrics.py
       data_loader.py
       training_utils.py
       checkpoint_utils.py
   ```
3. Check Python path: `sys.path.insert(0, str(Path(__file__).parent / "utils"))`
4. If you can't find the issue, report to cloud agent with full error traceback

---

## Inspecting Code Behavior vs Expected

### Check 1: Configuration Loading
```bash
python -c "import yaml; print(yaml.safe_load(open('configs/quick_test_config.yaml')))"
```

**Expected**: Should print valid YAML dict with keys: `model`, `lora`, `data`, `training`, `prompts`, `quality`, etc.

### Check 2: Module Imports
```bash
python -c "import sys; from pathlib import Path; sys.path.insert(0, str(Path('scripts/utils'))); from quality_metrics import QualityMetrics; print('OK')"
```

**Expected**: Should print "OK" without errors.

### Check 3: Dataset Loading
```bash
python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path('scripts/utils')))
from data_loader import create_dataloaders
import yaml
from transformers import CLIPTokenizer

config = yaml.safe_load(open('configs/quick_test_config.yaml'))
tokenizer = CLIPTokenizer.from_pretrained(config['model']['base_model'], subfolder='tokenizer')
train_loader, val_loader, train_size, val_size = create_dataloaders(
    data_root=config['data']['data_root'],
    modality=config['data']['modality'],
    phase=config['data']['phase'],
    resolution=config['model']['resolution'],
    tokenizer=tokenizer,
    batch_size=2,
    train_val_split=config['data']['train_val_split'],
    split_seed=config['data']['split_seed'],
    prompt=config['prompts']['positive'],
    augmentation=config['data']['augmentation'],
    num_workers=0
)
print(f'Train: {train_size}, Val: {val_size}')
"
```

**Expected**: Should print something like "Train: 120, Val: 20" (depends on dataset size).

### Check 4: Quality Metrics
```bash
python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path('scripts/utils')))
from quality_metrics import QualityMetrics
import torch

qm = QualityMetrics(device='cuda' if torch.cuda.is_available() else 'cpu')
img1 = torch.rand(1, 3, 128, 128)
img2 = torch.rand(1, 3, 128, 128)
ssim = qm.compute_ssim(img1, img2)
print(f'SSIM: {ssim:.4f}')
"
```

**Expected**: Should print a SSIM score between 0 and 1, e.g., "SSIM: 0.1234".

---

## Reporting Back

### If All Tests Pass ✓

Report to cloud agent:
```
✓ End-to-end test PASSED
- All 25+ tests passed successfully
- GPUs detected: 2× [GPU names]
- Dataset: Phase I ([N] images), Phase R ([M] images)
- All modules imported correctly
- Quality metrics working
- Checkpoint save/load working
- [Optional] Quick training test completed (1 epoch)

System is ready for production training.

Test results saved to: reports/test_results.json
```

### If Some Tests Failed ✗

For each failure, provide:
1. **Test name** (e.g., "Import Diffusers")
2. **Error message** (full traceback if available)
3. **What you tried** to fix it
4. **Whether you fixed it** or need help

Example report:
```
⚠ End-to-end test had 2 failures:

1. ✗ FAIL: Import xFormers
   Error: ModuleNotFoundError: No module named 'xformers'
   Tried: pip install xformers
   Result: Installation failed due to CUDA version mismatch
   Action needed: Is xFormers required? Config says it's optional.

2. ✗ FAIL: Dataset directory exists
   Error: Directory not found: /home/user/DFUMultiClassification/datasets/polished_data/depth_rgb/phase_P
   Tried: ls -la datasets/polished_data/depth_rgb/
   Result: Only phase_I and phase_R exist, no phase_P
   Action needed: Should config be updated to not expect phase_P?

Other tests: 23/25 passed

Need guidance on how to proceed.
```

### If Unexpected Behavior

Report anything that seems wrong:
```
⚠ Unexpected behavior detected:

1. GPU memory usage very high (15.5 GB / 16 GB) even with batch_size=2
   Expected: Should be ~8-10 GB based on documentation

2. Training loss not decreasing in quick training test
   Epoch 1: loss=0.543
   Epoch 2: loss=0.542
   Expected: Loss should decrease noticeably

3. Generated samples are all black/white
   Location: generated_samples/test/
   Expected: Should show colorful wound images
```

---

## Files to Review After Testing

If you want to understand the system better, review these key files:

1. **TRAINING_README.md** - User guide with usage examples
2. **IMPLEMENTATION_SUMMARY.md** - System overview and architecture
3. **ACTION_PLANS.md** - Improvement strategies (background context)
4. **configs/phase_I_config.yaml** - Production configuration for Phase I
5. **scripts/train_lora_model.py** - Main training script
6. **scripts/utils/quality_metrics.py** - Quality metrics implementation

---

## Next Steps After Successful Test

1. **Report success** to cloud agent
2. **Do NOT run production training yet** - wait for user's approval
3. **Review configurations** if you see potential improvements
4. **Ask questions** if anything is unclear

---

## Summary Checklist

Before reporting back, verify:

- [ ] Ran `python scripts/test_training_system.py`
- [ ] Checked test results (passed/failed/warnings)
- [ ] Fixed obvious issues (missing dependencies, etc.)
- [ ] Inspected `reports/test_results.json`
- [ ] Noted any unexpected behavior
- [ ] Optional: Ran quick training test (`RUN_TRAINING_TEST=true`)
- [ ] Prepared report with: results, errors (if any), attempts to fix, system specs

---

**Good luck!** The cloud agent is confident this system will substantially improve generative augmentation performance from the current -10.4% to expected +5% to +15% kappa improvement.
