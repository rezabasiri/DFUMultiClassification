#!/bin/bash
# Quick script to check Stable Diffusion model availability
# Expected location: Codes/MultimodalClassification/ImageGeneration/models_5_7/

echo "=== Generative Augmentation Model Availability Check ==="
echo "Date: $(date)"
echo ""

BASE_DIR="Codes/MultimodalClassification/ImageGeneration/models_5_7"

# Check if base directory exists
if [ -d "$BASE_DIR" ]; then
    echo "✓ Base directory exists: $BASE_DIR"
    echo ""

    # List all model directories
    echo "Available models:"
    for model_dir in "$BASE_DIR"/*; do
        if [ -d "$model_dir" ]; then
            model_name=$(basename "$model_dir")
            # Check for common SD files
            has_model=false

            if [ -f "$model_dir/model_index.json" ]; then
                has_model=true
                echo "  ✓ $model_name (diffusers format)"
            elif [ -f "$model_dir/config.json" ]; then
                has_model=true
                echo "  ✓ $model_name (config found)"
            else
                echo "  ? $model_name (directory exists, format unclear)"
            fi
        fi
    done
    echo ""

    # Count models
    model_count=$(find "$BASE_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
    echo "Total model directories: $model_count"
    echo "Expected: 9 (3 modalities × 3 phases)"

else
    echo "✗ Base directory NOT found: $BASE_DIR"
    echo ""
    echo "Searching for alternative locations..."
    find . -type d -name "models_5_7" 2>/dev/null
fi

echo ""
echo "=== End of Check ==="
