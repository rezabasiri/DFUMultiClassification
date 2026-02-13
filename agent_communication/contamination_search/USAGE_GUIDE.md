# Multi-Parameter Search - Usage Guide

## Answer to Your Questions

### 1. Is it saving results to JSON?
**✅ YES** - Results saved to `results/multi_param_search_results.json`

### 2. Is it saved incrementally or at the end?
**✅ INCREMENTAL** - Saved **after each trial completes** (as you requested)

### 3. Can it resume if interrupted?
**✅ YES** - Uses Optuna's SQLite database for persistent storage

### 4. Can I start fresh?
**✅ YES** - Use `--fresh` flag to delete study and start over

---

## Quick Commands

### Start New Search
```bash
# Start fresh 20-trial search
python agent_communication/contamination_search/search_multi_param.py --n-trials 20 --fresh
```

### Resume Interrupted Search
```bash
# Continue from where it left off
python agent_communication/contamination_search/search_multi_param.py --n-trials 20
```

### Check Progress
```bash
# View current status and best params
python agent_communication/contamination_search/check_progress.py
```

### Pre-Flight Check
```bash
# Verify config is ready before starting
python agent_communication/contamination_search/preflight_check.py
```

---

## How Resume Works

### Files Created
```
results/
├── multi_param_search_results.json  # Updated after EACH trial
├── optuna_study.db                  # Optuna persistence (SQLite)
└── optuna_study.db-journal          # SQLite journal (temp)

logs_multi/
├── ep100_img128_s1_15_cont0.29_bs64.log  # Trial 0 training log
├── trial_0_results.csv                   # Trial 0 CSV backup
├── ep120_img256_s1_18_cont0.32_bs32.log  # Trial 1 training log
├── trial_1_results.csv                   # Trial 1 CSV backup
└── ...
```

### What Happens After Each Trial

1. **Trial completes** (3-fold CV finishes)
2. **Kappa extracted** from `results/csv/modality_results_averaged.csv`
3. **Results saved** to JSON immediately:
   ```json
   {
     "best_params": {...},
     "best_kappa": 0.3809,
     "n_trials_completed": 5,
     "n_trials_total": 20,
     "all_trials": [...]
   }
   ```
4. **Study saved** to SQLite database automatically by Optuna
5. **Console prints**: "💾 Progress saved to: multi_param_search_results.json"

### What Happens If Interrupted

**Scenario 1: Interrupted during trial execution**
- Current trial is marked as FAILED
- Previous completed trials are preserved in database
- Re-running continues from next trial

**Scenario 2: Interrupted between trials**
- All completed trials preserved
- Re-running continues immediately from next trial

**Example:**
```bash
# Start search
python search_multi_param.py --n-trials 20 --fresh

# ... completes 5 trials, then interrupted (Ctrl+C or crash)

# Resume - automatically continues from trial 6
python search_multi_param.py --n-trials 20

# Output shows:
# 📂 Resuming existing study with 5 completed trials
#    Best kappa so far: 0.3809
# Trials remaining: 15/20
```

---

## Typical Workflow

### Day 1: Start Search (Friday evening)
```bash
# 1. Verify config
python agent_communication/contamination_search/preflight_check.py

# 2. Start fresh search (20 trials)
python agent_communication/contamination_search/search_multi_param.py --n-trials 20 --fresh

# 3. Let it run overnight...
```

### Day 2: Check Progress (Saturday morning)
```bash
# Check how many trials completed
python agent_communication/contamination_search/check_progress.py

# Output shows:
# Status: 8/20 trials completed
# Best Parameters So Far:
#   N_EPOCHS: 120
#   IMAGE_SIZE: 128
#   ...
#   Kappa: 0.3952
```

### Day 2: Continue (if needed)
```bash
# Resume automatically (runs remaining 12 trials)
python agent_communication/contamination_search/search_multi_param.py --n-trials 20

# Or extend to 30 trials total
python agent_communication/contamination_search/search_multi_param.py --n-trials 30
```

### Day 3: Analyze Results
```bash
# View final results
python agent_communication/contamination_search/check_progress.py

# Results in: results/multi_param_search_results.json
```

---

## Advanced Usage

### Extend Search Beyond Original Target
```bash
# Started with 20 trials, want to add 10 more
python search_multi_param.py --n-trials 30

# Only runs 10 new trials (30 - 20 already completed)
```

### Start Completely Fresh
```bash
# Deletes database and JSON, starts from scratch
python search_multi_param.py --n-trials 20 --fresh
```

### Different Study Names (for parallel experiments)
```bash
# Run multiple independent searches
python search_multi_param.py --n-trials 20 --study-name experiment1 --fresh
python search_multi_param.py --n-trials 20 --study-name experiment2 --fresh

# Each has its own database and results
```

---

## Monitoring During Execution

### Console Output
After each trial completes, you'll see:
```
================================================================================
Running: epochs=120, img_size=128, stage1=18, cont=0.29, batch=64
Running all 3 folds in subprocesses (may take ~30-90 min depending on config)
================================================================================

✓ Average Kappa (3 folds): 0.3952
  Saved trial results to: trial_5_results.csv
💾 Progress saved to: multi_param_search_results.json
   Completed trials: 6/20
   Best kappa so far: 0.3952
```

### Check JSON Anytime
```bash
# View latest results while search is running
cat agent_communication/contamination_search/results/multi_param_search_results.json | head -30

# Or use check_progress.py
python agent_communication/contamination_search/check_progress.py
```

---

## Troubleshooting

### Search won't resume
**Problem:** Re-running shows "No search results found"
**Solution:** Make sure you're not using `--fresh` flag

### Want to restart from scratch
**Problem:** Study has bad trials, want clean start
**Solution:** Use `--fresh` flag to delete database

### Database corruption
**Problem:** SQLite error when resuming
**Solution:**
```bash
# Delete corrupted database and start fresh
rm agent_communication/contamination_search/results/optuna_study.db*
python search_multi_param.py --n-trials 20 --fresh
```

### JSON not updating
**Problem:** Results file not changing after trials
**Solution:** Check if trials are actually completing (look at logs_multi/)

---

## File Locations

| File | Purpose | When Updated |
|------|---------|--------------|
| `results/multi_param_search_results.json` | Human-readable results | After each trial ✅ |
| `results/optuna_study.db` | Optuna persistence database | After each trial ✅ |
| `logs_multi/*.log` | Per-trial training logs | During trial execution |
| `logs_multi/trial_*.csv` | Per-trial CSV backups | After trial completes |
| `src/utils/production_config.py.backup_multi` | Original config backup | Once at start |

---

## Comparison: Fresh vs Resume

| Aspect | `--fresh` | No flag (resume) |
|--------|-----------|------------------|
| Database | Deleted, new study | Loaded, continues |
| JSON results | Deleted, starts empty | Loaded, appends |
| Trial numbering | Starts from 0 | Continues from last |
| Use case | New experiment | Continue interrupted search |
| Speed | Full duration | Only remaining trials |

---

## Best Practices

1. **Always run pre-flight check first**
   ```bash
   python agent_communication/contamination_search/preflight_check.py
   ```

2. **Use `--fresh` for new experiments**
   ```bash
   python search_multi_param.py --n-trials 20 --fresh
   ```

3. **Check progress periodically**
   ```bash
   python check_progress.py
   ```

4. **Monitor logs for errors**
   ```bash
   tail -f agent_communication/contamination_search/logs_multi/ep*.log
   ```

5. **Backup results directory before major changes**
   ```bash
   cp -r agent_communication/contamination_search/results results_backup
   ```

---

## Summary

✅ **Incremental saving** - JSON updated after each trial
✅ **Auto-resume** - Optuna SQLite database preserves progress
✅ **Fresh start** - `--fresh` flag clears everything
✅ **Progress monitoring** - `check_progress.py` shows status
✅ **Safe interruption** - Can stop/resume anytime with Ctrl+C
