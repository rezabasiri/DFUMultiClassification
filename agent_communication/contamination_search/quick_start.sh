#!/bin/bash
# Quick start script for contamination search

echo "=================================================================="
echo "CONTAMINATION SEARCH - QUICK START"
echo "=================================================================="
echo ""

# Check if optuna is installed
if ! python -c "import optuna" 2>/dev/null; then
    echo "Installing optuna..."
    pip install optuna
    echo ""
fi

# Ask user for number of trials
read -p "Number of trials (default: 15, recommended: 10-20): " n_trials
n_trials=${n_trials:-15}

echo ""
echo "Starting Bayesian search with $n_trials trials..."
echo "This will take approximately $((n_trials * 20 / 60)) hours on your system"
echo ""
echo "Press Ctrl+C to cancel, or Enter to continue..."
read

# Run search
python agent_communication/contamination_search/search_contamination.py --n-trials $n_trials

# Analyze results
echo ""
echo "Analyzing results..."
python agent_communication/contamination_search/analyze_results.py

echo ""
echo "=================================================================="
echo "COMPLETE! Check agent_communication/contamination_search/results/"
echo "=================================================================="
