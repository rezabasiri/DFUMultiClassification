# TODO: Fix CSV Duplicate Rows in Modality Combination Results

## Problem

When running multi-fold CV with `MODALITY_SEARCH_MODE = 'all'`, the file `results/csv/modality_combination_results.csv` accumulates **duplicate rows** for the same modality combination instead of updating existing rows in-place.

### Example

After fold 1 completes all 15 combos, the CSV has 15 rows (all with Std=0.0 since only 1 fold).
When fold 2 completes metadata, instead of updating the existing metadata row, a **new row is appended** with the 2-fold aggregate (now showing non-zero Std):

```
metadata,0.762,0.0,...        ← fold 1 only (stale)
depth_rgb,0.477,0.0,...       ← fold 1 only
...all 15 combos...
metadata,0.638,0.139,...      ← 2-fold aggregate (correct, but duplicate)
```

The log confirms this behavior:
```
Results for metadata appended to results/csv/modality_combination_results.csv
```

### Impact

- CSV grows with stale duplicate entries (15 rows per fold pass = 45 rows for 3 folds instead of 15)
- Any tool reading the CSV may pick up the wrong (stale) row
- The first row for each combo always shows Std=0.0 which is misleading

## Root Cause

The CSV writing logic uses **append mode** rather than **read-update-write**. It does not check if a row for the same modality combination already exists before writing.

## Solution

In the code that writes to `modality_combination_results.csv` (search for the "appended to" log message):

1. Before writing, read the existing CSV into a DataFrame
2. Check if a row with the same `Modalities` value already exists
3. If yes: **update** that row with the new aggregated values
4. If no: append the new row
5. Write the full DataFrame back to the CSV

Pseudocode:
```python
import pandas as pd

results_path = "results/csv/modality_combination_results.csv"
new_row = {...}  # the aggregated results dict

if os.path.exists(results_path):
    df = pd.read_csv(results_path)
    mask = df['Modalities'] == new_row['Modalities']
    if mask.any():
        # Update existing row
        for col, val in new_row.items():
            df.loc[mask, col] = val
    else:
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
else:
    df = pd.DataFrame([new_row])

df.to_csv(results_path, index=False)
```

## Files to Investigate

- Search for `modality_combination_results` and `appended to` in `src/main.py` (or wherever the CSV write happens)
- The write likely uses `mode='a'` or `df.to_csv(..., mode='a')` — change to read-update-write pattern above
