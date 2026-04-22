#!/bin/bash
# ============================================================
# VisionGuard-AD — Run All Categories
# Train and evaluate PatchCore on all 15 MVTec-AD categories
# ============================================================

set -e

METHOD="${1:-patchcore}"
DATA_ROOT="${2:-./data/mvtec}"
OUTPUT_DIR="${3:-./outputs}"
BACKBONE="${4:-wide_resnet50}"

CATEGORIES=(bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper)

echo "============================================================"
echo "  VisionGuard-AD — Full Pipeline"
echo "  Method: $METHOD"
echo "  Backbone: $BACKBONE"
echo "  Data: $DATA_ROOT"
echo "============================================================"

for category in "${CATEGORIES[@]}"; do
    echo ""
    echo "────────────────────────────────────────"
    echo "  Training: $category"
    echo "────────────────────────────────────────"

    python train.py \
        --method "$METHOD" \
        --category "$category" \
        --backbone "$BACKBONE" \
        --data_root "$DATA_ROOT" \
        --output_dir "$OUTPUT_DIR" \
        --device auto

    echo ""
    echo "  Evaluating: $category"
    echo "────────────────────────────────────────"

    if [ "$METHOD" == "patchcore" ]; then
        MODEL_PATH="$OUTPUT_DIR/$category/patchcore_memory_bank.pt"
    else
        MODEL_PATH="$OUTPUT_DIR/$category/fastflow_best.pt"
    fi

    python evaluate.py \
        --method "$METHOD" \
        --category "$category" \
        --model_path "$MODEL_PATH" \
        --backbone "$BACKBONE" \
        --data_root "$DATA_ROOT" \
        --output_dir "$OUTPUT_DIR/eval" \
        --visualize \
        --num_visualizations 10

done

echo ""
echo "============================================================"
echo "  All categories complete!"
echo "  Results: $OUTPUT_DIR/eval/"
echo "============================================================"
