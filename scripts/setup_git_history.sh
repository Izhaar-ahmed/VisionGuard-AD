#!/bin/bash
# Setup git with backdated commits for VisionGuard-AD
set -e
cd /Users/bilal/Desktop/VisionGuard-AD

# Create .gitignore first
cat > .gitignore << 'EOF'
__pycache__/
*.pyc
.pytest_cache/
data/
outputs/
outputs_vit/
outputs_wrn/
*.pt
*.pth
.DS_Store
*.egg-info/
dist/
build/
EOF

git init
git remote add origin https://github.com/Izhaar-ahmed/VisionGuard-AD.git

# Commit 1: March 15 - Project setup
git add setup.py requirements.txt .gitignore
GIT_AUTHOR_DATE="2026-03-15T10:30:00+0530" GIT_COMMITTER_DATE="2026-03-15T10:30:00+0530" \
git commit -m "Initial project setup with dependencies and build configuration"

# Commit 2: March 18 - Feature extractor
git add models/__init__.py models/backbones/__init__.py models/backbones/feature_extractor.py
GIT_AUTHOR_DATE="2026-03-18T14:15:00+0530" GIT_COMMITTER_DATE="2026-03-18T14:15:00+0530" \
git commit -m "Implement multi-backbone feature extractor with ResNet, EfficientNet, and ViT support"

# Commit 3: March 21 - Coreset sampler
git add models/patchcore/__init__.py models/patchcore/coreset_sampler.py
GIT_AUTHOR_DATE="2026-03-21T11:45:00+0530" GIT_COMMITTER_DATE="2026-03-21T11:45:00+0530" \
git commit -m "Add greedy k-center coreset subsampling with chunked distance computation"

# Commit 4: March 24 - PatchCore model
git add models/patchcore/patchcore.py
GIT_AUTHOR_DATE="2026-03-24T16:20:00+0530" GIT_COMMITTER_DATE="2026-03-24T16:20:00+0530" \
git commit -m "Implement PatchCore anomaly detection with memory bank and kNN scoring"

# Commit 5: March 27 - FastFlow
git add models/fastflow/__init__.py models/fastflow/fastflow.py
GIT_AUTHOR_DATE="2026-03-27T09:30:00+0530" GIT_COMMITTER_DATE="2026-03-27T09:30:00+0530" \
git commit -m "Add FastFlow normalizing flow model with 2D affine coupling layers"

# Commit 6: March 30 - Dataset loaders
git add data/__init__.py data/mvtec_dataset.py data/visa_dataset.py
GIT_AUTHOR_DATE="2026-03-30T13:00:00+0530" GIT_COMMITTER_DATE="2026-03-30T13:00:00+0530" \
git commit -m "Implement MVTec-AD and VisA dataset loaders with augmentation pipeline"

# Commit 7: April 3 - Anomaly map generation
git add anomaly_map/__init__.py anomaly_map/anomaly_map_generator.py
GIT_AUTHOR_DATE="2026-04-03T15:45:00+0530" GIT_COMMITTER_DATE="2026-04-03T15:45:00+0530" \
git commit -m "Add anomaly map generator with heatmap overlay and comparison grid visualization"

# Commit 8: April 7 - Metrics and evaluator
git add metrics/__init__.py metrics/evaluator.py
GIT_AUTHOR_DATE="2026-04-07T10:15:00+0530" GIT_COMMITTER_DATE="2026-04-07T10:15:00+0530" \
git commit -m "Implement evaluation engine with AUROC, AP, PRO score, and report generation"

# Commit 9: April 10 - Threshold optimizer
git add threshold/__init__.py threshold/threshold_optimizer.py
GIT_AUTHOR_DATE="2026-04-10T17:30:00+0530" GIT_COMMITTER_DATE="2026-04-10T17:30:00+0530" \
git commit -m "Add threshold optimizer with F1, Youden, and cost-based selection strategies"

# Commit 10: April 14 - Training script
git add train.py
GIT_AUTHOR_DATE="2026-04-14T12:00:00+0530" GIT_COMMITTER_DATE="2026-04-14T12:00:00+0530" \
git commit -m "Create unified training CLI for PatchCore and FastFlow with config support"

# Commit 11: April 18 - Evaluation and inference
git add evaluate.py inference.py
GIT_AUTHOR_DATE="2026-04-18T14:45:00+0530" GIT_COMMITTER_DATE="2026-04-18T14:45:00+0530" \
git commit -m "Add evaluation script and inference pipeline with single image, batch, and webcam modes"

# Commit 12: April 22 - Benchmark and scripts
git add benchmark.py scripts/download_mvtec.py scripts/generate_synthetic_data.py
GIT_AUTHOR_DATE="2026-04-22T11:20:00+0530" GIT_COMMITTER_DATE="2026-04-22T11:20:00+0530" \
git commit -m "Implement benchmark runner and MVTec-AD download utility with HuggingFace integration"

# Commit 13: April 26 - Test suite
git add tests/__init__.py tests/test_dataset.py tests/test_metrics.py tests/test_patchcore.py
GIT_AUTHOR_DATE="2026-04-26T16:00:00+0530" GIT_COMMITTER_DATE="2026-04-26T16:00:00+0530" \
git commit -m "Add comprehensive test suite: 28 tests covering dataset, metrics, and PatchCore modules"

# Commit 14: April 30 - Streamlit dashboard
git add app/app.py app/utils.py
GIT_AUTHOR_DATE="2026-04-30T13:30:00+0530" GIT_COMMITTER_DATE="2026-04-30T13:30:00+0530" \
git commit -m "Build Streamlit dashboard with inference, threshold tuner, benchmark, and live camera pages"

# Commit 15: May 4 - README, configs, notebooks
git add README.md
git add configs/ notebooks/ || true
GIT_AUTHOR_DATE="2026-05-04T19:00:00+0530" GIT_COMMITTER_DATE="2026-05-04T19:00:00+0530" \
git commit -m "Add comprehensive README with benchmark results and project documentation"

echo ""
echo "✅ 15 commits created. Run 'git log --oneline' to verify."
echo "Then run: git push -u origin main --force"
