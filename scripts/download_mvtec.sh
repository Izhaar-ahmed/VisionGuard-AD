#!/bin/bash
# ============================================================
# VisionGuard-AD — Download MVTec-AD Dataset
# ============================================================
# The MVTec-AD dataset must be downloaded from the official page:
#   https://www.mvtec.com/company/research/datasets/mvtec-ad
#
# After accepting the license, download and extract to ./data/mvtec/
#
# Expected structure after extraction:
#   data/mvtec/
#   ├── bottle/
#   ├── cable/
#   ├── capsule/
#   ├── carpet/
#   ├── grid/
#   ├── hazelnut/
#   ├── leather/
#   ├── metal_nut/
#   ├── pill/
#   ├── screw/
#   ├── tile/
#   ├── toothbrush/
#   ├── transistor/
#   ├── wood/
#   └── zipper/
# ============================================================

set -e

DATA_DIR="./data/mvtec"
mkdir -p "$DATA_DIR"

echo "============================================================"
echo "  MVTec-AD Dataset Download"
echo "============================================================"
echo ""
echo "The MVTec-AD dataset requires accepting a license agreement."
echo "Please download it manually from:"
echo ""
echo "  https://www.mvtec.com/company/research/datasets/mvtec-ad"
echo ""
echo "After downloading, extract the archive to: $DATA_DIR"
echo ""
echo "Expected command after manual download:"
echo "  tar xf mvtec_anomaly_detection.tar.xz -C $DATA_DIR"
echo ""

# Check if dataset already exists
CATEGORIES=(bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper)
FOUND=0
for cat in "${CATEGORIES[@]}"; do
    if [ -d "$DATA_DIR/$cat" ]; then
        FOUND=$((FOUND + 1))
    fi
done

if [ "$FOUND" -eq 15 ]; then
    echo "✅ All 15 MVTec-AD categories found in $DATA_DIR"
    echo ""
    for cat in "${CATEGORIES[@]}"; do
        TRAIN_COUNT=$(find "$DATA_DIR/$cat/train" -type f -name "*.png" 2>/dev/null | wc -l)
        TEST_COUNT=$(find "$DATA_DIR/$cat/test" -type f -name "*.png" 2>/dev/null | wc -l)
        echo "  $cat: train=$TRAIN_COUNT, test=$TEST_COUNT"
    done
elif [ "$FOUND" -gt 0 ]; then
    echo "⚠️  Found $FOUND/15 categories. Some categories may be missing."
else
    echo "❌ No categories found. Please download and extract the dataset."
fi
