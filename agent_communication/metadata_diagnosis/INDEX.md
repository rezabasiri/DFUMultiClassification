# Metadata Diagnosis - File Index

## Quick Start
1. Read: `README.md`
2. Run: `bash run_all_tests.sh`
3. Fill: `REPORT_TEMPLATE.md`

## File Structure

### Instructions
- `README.md` - Main instructions
- `AGENT_CHECKLIST.md` - Step-by-step checklist
- `EXPECTED_VS_ACTUAL.md` - Expected behavior vs current failure

### Test Scripts (run in order)
1. `test_1_data_loading.py` - Data loading validation
2. `test_2_feature_engineering.py` - Feature creation checks
3. `test_3_preprocessing.py` - Imputation + normalization
4. `test_4_rf_training.py` - **CRITICAL** - RF ordinal training
5. `test_5_end_to_end.py` - Integration with actual code

### Utilities
- `run_all_tests.sh` - Execute all tests
- `REPORT_TEMPLATE.md` - Report results here

## Test Purpose

| Test | What It Checks | Why Important |
|------|----------------|---------------|
| 1 | Data loads, labels correct | Foundation |
| 2 | Features engineered properly | Data quality |
| 3 | Imputation/norm work correctly | Preprocessing |
| 4 | RF training achieves >30% acc | **ROOT CAUSE** |
| 5 | Integration with caching.py | Code path |

## Expected Runtime
- Test 1-3: <10 seconds each
- Test 4: 30-60 seconds (trains 2 RFs)
- Test 5: 60-120 seconds (full pipeline)
- **Total: ~2-3 minutes**

## Critical Thresholds

**Test 4 must achieve:**
- Validation accuracy: >30%
- Cohen's Kappa: >0.1
- All classes: F1 >0.15

**If Test 4 fails these thresholds:**
→ Found the root cause!
→ Report full results immediately
