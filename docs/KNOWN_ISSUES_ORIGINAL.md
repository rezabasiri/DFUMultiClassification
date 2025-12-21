# Known Issues in main_original.py

This document tracks known bugs in `src/main_original.py` that were fixed in the refactored `src/main.py`. These issues prevent direct comparison testing.

## Issue 1: UnboundLocalError - alpha_value not defined

**Error:**
```
Error during training: cannot access local variable 'alpha_value' where it is not associated with a value
```

**Location:** `src/main_original.py` (training function)

**Description:**
The `alpha_value` variable is referenced before being assigned in certain code paths. This causes the training to fail with an UnboundLocalError.

**Root Cause:**
Similar to the bug fixed in refactored code (see tracker.md 2025-12-17 entry about alpha_value), the original code has a scope issue where `alpha_value`, `class_weights_dict`, and `class_weights` may not be initialized before being used.

**Impact:**
- Prevents successful training with metadata modality
- Causes training to fail after 3 retry attempts
- Results in "No valid predictions to save" error

**Fixed In:**
This was fixed in `src/training/training_utils.py` lines 818-833 by:
1. Adding default values before conditional blocks
2. Changing nested `if` statements to `elif` to prevent fall-through

**Status:**
❌ **NOT FIXED in main_original.py** (per user requirement: don't modify main_original.py)
✅ **FIXED in main.py** (refactored version)

**Workaround for comparison testing:**
Cannot test main_original.py until this issue is fixed. The refactored code can be tested independently using:
```bash
python scripts/simple_comparison_test.py metadata --skip_original
```

## Issue 2: Missing misclassification filter output

**Observation:**
Original code doesn't output "No misclassification file found" message at verbosity level 1.

**Location:** Misclassification filtering logic

**Impact:** Minor - just a difference in output verbosity

**Status:**
- Not a bug, just a difference in verbosity implementation
- Refactored code has better verbosity system

---

## Testing Strategy Given These Issues

Since main_original.py has known bugs that prevent it from running:

### Option 1: Test refactored code independently
```bash
# Verify refactored code works correctly
python scripts/simple_comparison_test.py metadata --skip_original
python scripts/simple_comparison_test.py depth_rgb --skip_original
python scripts/simple_comparison_test.py depth_map --skip_original
```

### Option 2: Compare against known good results
If you have previous successful runs from main_original.py (before the bug was introduced), you can:
1. Save those metrics as reference
2. Run refactored code
3. Compare against saved reference

### Option 3: Fix main_original.py temporarily for testing
If comparison is critical, create a temporary patched version:
```bash
cp src/main_original.py src/main_original_patched.py
# Apply the alpha_value fix to main_original_patched.py
# Run comparison between patched version and refactored version
```

## Recommendation

✅ **Recommended approach:**
1. Test refactored code independently to verify it works
2. Document that original code has known bugs
3. Focus validation on ensuring refactored code:
   - Produces sensible results
   - Doesn't crash
   - Has expected behavior for each modality

The refactoring process has actually **fixed bugs** that exist in the original code, making direct comparison difficult. This is a positive outcome - the refactored code is more robust.
