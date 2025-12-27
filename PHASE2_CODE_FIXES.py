"""
Phase 2 Code Fixes for auto_polish_dataset_v2.py

This file contains the exact code changes needed to fix Phase 2 failures.
Apply these changes to scripts/auto_polish_dataset_v2.py
"""

# ============================================================================
# FIX #1: Add batch size calculation method
# ============================================================================
# Location: After __init__ method (around line 156)
# Add this entire method:

FIX_1_BATCH_SIZE_CALCULATOR = '''
    def _calculate_phase2_batch_size(self):
        """
        Calculate appropriate batch size for Phase 2 based on number of image modalities.

        Phase 1 tests ONE modality at a time with GLOBAL_BATCH_SIZE.
        Phase 2 tests MULTIPLE modalities simultaneously, requiring proportional reduction.

        Returns:
            int: Adjusted batch size for Phase 2
        """
        from src.utils.production_config import GLOBAL_BATCH_SIZE

        # Count image modalities (exclude metadata which is small)
        image_modalities = [m for m in self.modalities if m != 'metadata']
        num_image_modalities = len(image_modalities)

        if num_image_modalities == 0:
            # Metadata only - use full batch size
            return GLOBAL_BATCH_SIZE

        # Divide batch size by number of image modalities to maintain similar memory usage
        adjusted_batch_size = max(16, GLOBAL_BATCH_SIZE // num_image_modalities)

        print(f"\\n{'='*80}")
        print("BATCH SIZE ADJUSTMENT FOR PHASE 2")
        print(f"{'='*80}")
        print(f"Phase 1 batch size (1 modality at a time): {GLOBAL_BATCH_SIZE}")
        print(f"Phase 2 modalities: {self.modalities}")
        print(f"  Image modalities: {image_modalities} (count: {num_image_modalities})")
        print(f"  Adjusted batch size: {GLOBAL_BATCH_SIZE} / {num_image_modalities} = {adjusted_batch_size}")
        print(f"  Memory reduction: ~{num_image_modalities}x less per batch")
        print(f"{'='*80}\\n")

        return adjusted_batch_size
'''

# ============================================================================
# FIX #2: Add batch size override in train_with_thresholds
# ============================================================================
# Location: In train_with_thresholds method, around line 1145
# Replace the config modification section

FIX_2_CONFIG_OVERRIDE_OLD = '''
        try:
            import re
            # Build the modality tuple string, e.g., "('metadata', 'depth_rgb')"
            modality_tuple = "(" + ", ".join(f"'{m}'" for m in self.modalities) + ",)"
            modified_config = re.sub(
                r'INCLUDED_COMBINATIONS\\s*=\\s*\\[[\\s\\S]*?\\n\\]',
                f"INCLUDED_COMBINATIONS = [\\n    {modality_tuple},  # Temporary: Phase 2 evaluation\\n]",
                original_config
            )
            with open(config_path, 'w') as f:
                f.write(modified_config)
'''

FIX_2_CONFIG_OVERRIDE_NEW = '''
        try:
            import re

            # Calculate adjusted batch size for Phase 2
            adjusted_batch_size = self._calculate_phase2_batch_size()

            # Build the modality tuple string, e.g., "('metadata', 'depth_rgb')"
            modality_tuple = "(" + ", ".join(f"'{m}'" for m in self.modalities) + ",)"

            # Modify INCLUDED_COMBINATIONS
            modified_config = re.sub(
                r'INCLUDED_COMBINATIONS\\s*=\\s*\\[[\\s\\S]*?\\n\\]',
                f"INCLUDED_COMBINATIONS = [\\n    {modality_tuple},  # Temporary: Phase 2 evaluation\\n]",
                original_config
            )

            # Modify GLOBAL_BATCH_SIZE for Phase 2
            modified_config = re.sub(
                r'GLOBAL_BATCH_SIZE\\s*=\\s*\\d+',
                f"GLOBAL_BATCH_SIZE = {adjusted_batch_size}",
                modified_config
            )

            with open(config_path, 'w') as f:
                f.write(modified_config)
'''

# ============================================================================
# FIX #3: Add error logging to subprocess failures
# ============================================================================
# Location: In train_with_thresholds method, around line 1180
# Replace the subprocess error handling

FIX_3_ERROR_LOGGING_OLD = '''
            print(f"⏳ Training with cv_folds={self.phase2_cv_folds} (fresh mode)...")
            result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)

            if result.returncode != 0:
                return None
'''

FIX_3_ERROR_LOGGING_NEW = '''
            print(f"⏳ Training with cv_folds={self.phase2_cv_folds} (fresh mode)...")
            result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"\\n{'='*80}")
                print("❌ TRAINING SUBPROCESS FAILED")
                print(f"{'='*80}")
                print(f"Return code: {result.returncode}")
                print(f"\\nCommand that failed:")
                print(' '.join(cmd))

                if result.stderr:
                    print(f"\\n{'─'*80}")
                    print("STDERR (last 3000 chars):")
                    print(f"{'─'*80}")
                    print(result.stderr[-3000:])

                if result.stdout:
                    print(f"\\n{'─'*80}")
                    print("STDOUT (last 3000 chars):")
                    print(f"{'─'*80}")
                    print(result.stdout[-3000:])

                print(f"{'='*80}\\n")
                return None
'''

# ============================================================================
# INSTRUCTIONS FOR LOCAL AGENT
# ============================================================================

INSTRUCTIONS = """
TO APPLY THESE FIXES:

1. Open scripts/auto_polish_dataset_v2.py in an editor

2. Apply FIX #1:
   - Find the __init__ method (ends around line 156)
   - After the __init__ method, add the _calculate_phase2_batch_size method
   - Copy the code from FIX_1_BATCH_SIZE_CALCULATOR above

3. Apply FIX #2:
   - Find the train_with_thresholds method (around line 1145)
   - Find the "try:" block with "import re"
   - Replace the old code with FIX_2_CONFIG_OVERRIDE_NEW

4. Apply FIX #3:
   - Find the subprocess.run call (around line 1180)
   - Replace the error handling with FIX_3_ERROR_LOGGING_NEW

5. Save the file

6. Run tests from PHASE2_INVESTIGATION_GUIDE.md

VERIFICATION:
After applying fixes, search for these strings in the file to verify:
- "_calculate_phase2_batch_size" - should exist once
- "adjusted_batch_size = self._calculate_phase2_batch_size()" - should exist in train_with_thresholds
- "TRAINING SUBPROCESS FAILED" - should exist in error handling
- "STDERR (last 3000 chars):" - should exist in error logging
"""

if __name__ == "__main__":
    print(INSTRUCTIONS)
    print("\n" + "="*80)
    print("FIX #1: Batch Size Calculator")
    print("="*80)
    print(FIX_1_BATCH_SIZE_CALCULATOR)
    print("\n" + "="*80)
    print("FIX #2: Config Override (REPLACE OLD WITH NEW)")
    print("="*80)
    print("OLD CODE TO FIND:")
    print(FIX_2_CONFIG_OVERRIDE_OLD)
    print("\nNEW CODE TO USE:")
    print(FIX_2_CONFIG_OVERRIDE_NEW)
    print("\n" + "="*80)
    print("FIX #3: Error Logging (REPLACE OLD WITH NEW)")
    print("="*80)
    print("OLD CODE TO FIND:")
    print(FIX_3_ERROR_LOGGING_OLD)
    print("\nNEW CODE TO USE:")
    print(FIX_3_ERROR_LOGGING_NEW)
