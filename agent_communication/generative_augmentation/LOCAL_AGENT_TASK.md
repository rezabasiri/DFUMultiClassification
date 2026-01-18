# LOCAL AGENT TASK

## Objective
Verify the generative augmentation test script runs error-free before production use.

## Phase 1: Quick Error Check ⚡

Run this command first:
```bash
python agent_communication/generative_augmentation/test_generative_aug.py --quick --fresh
```

**What it does:**
- Tests with 10% data, 2 epochs, 64x64 images (~10-30 min)
- Verifies no errors in the test pipeline
- Creates log file with detailed output

**Expected outcome:**
- Completes without errors
- Generates GENGEN_REPORT.txt with test results
- You should see "ALL TESTS COMPLETE" message

**If errors occur:**
- Check `gengen_test.log` for details
- Report error to cloud agent

## Phase 2: Production Run 🚀 (After Phase 1 passes)

Run this command:
```bash
python agent_communication/generative_augmentation/test_generative_aug.py --fresh
```

**What it does:**
- Full test with 100% data, 300 epochs, 128x128 images (~4-8 hours)
- Compares baseline vs generative augmentation
- Creates comprehensive effectiveness report

**After completion:**
- Commit GENGEN_PROGRESS.json, GENGEN_REPORT.txt, and gengen_test.log
- Log file is synced to git so cloud agent can review details

---

**Full details:** See RUN_INSTRUCTIONS.txt
