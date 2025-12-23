# START HERE - Local Agent Instructions

## Quick Start

```bash
# 1. Pull latest code
git pull origin claude/restore-weighted-f1-metrics-5PNy8

# 2. Read the comprehensive README
cat agent_communication/README.md
# Or open it in your editor

# 3. Follow the instructions in README.md
```

## What You Need to Do

The **complete instructions** are in `agent_communication/README.md`.

That file contains:
- ✅ All environment setup (paths, activation commands)
- ✅ Complete project context (what we're doing and why)
- ✅ The current problem we're investigating
- ✅ Step-by-step instructions for running debug scripts
- ✅ How to commit and push results
- ✅ What to tell the user after each phase

**Read the README first**, then start with Phase 1.

## TL;DR

If you've already read the README and just need a reminder:

```bash
# Setup (once)
source /Users/rezabasiri/env/multimodal/bin/activate
cd /Users/rezabasiri/DFUMultiClassification

# Run Phase 1
python agent_communication/debug_01_data_sanity.py

# Commit and push
git add agent_communication/results_01_data_sanity.txt
git commit -m "Debug Phase 1: Data sanity check complete"
git push origin claude/restore-weighted-f1-metrics-5PNy8

# Tell user: "Phase 1 complete, results pushed"
# Then wait for next instructions
```

---

**Important**: The README has ALL the context you need. Refer to it whenever you're unsure about:
- Environment paths
- What the project does
- What problem we're solving
- How to run the investigation
- What to do with results
