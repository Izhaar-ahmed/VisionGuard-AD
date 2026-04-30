"""
VisionGuard-AD — Premium Streamlit Dashboard
==============================================
Dark-themed industrial anomaly detection dashboard with glassmorphism design.
4 pages: Inference, Threshold Tuner, Benchmark, Live Camera.

Launch: streamlit run app/app.py
"""

import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

_project_root = str(Path(__file__).parent.parent)
_app_dir = str(Path(__file__).parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

from utils import (
    find_model_files, load_model_cached, run_inference,
    create_overlay_pil, create_binary_mask_pil,
)

# ─── Page Config ───
st.set_page_config(
    page_title="VisionGuard-AD",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Premium Dark CSS ───
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #0c1220 50%, #0f172a 100%);
    font-family: 'Inter', sans-serif;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    border-right: 1px solid rgba(6, 182, 212, 0.15);
}
.glass-card {
    background: rgba(30, 41, 59, 0.6);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(6, 182, 212, 0.15);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
}
.metric-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
    border: 1px solid rgba(6, 182, 212, 0.2);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    transition: all 0.3s ease;
}
.metric-card:hover { border-color: rgba(6, 182, 212, 0.5); transform: translateY(-2px); }
.metric-value { font-size: 32px; font-weight: 700; color: #06b6d4; }
.metric-label { font-size: 13px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }
.metric-green .metric-value { color: #10b981; }
.metric-yellow .metric-value { color: #f59e0b; }
.metric-red .metric-value { color: #ef4444; }
.hero-title {
    font-size: 36px; font-weight: 700;
    background: linear-gradient(135deg, #06b6d4, #0ea5e9, #8b5cf6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
}
.hero-subtitle { color: #94a3b8; font-size: 16px; }
.status-defect {
    background: rgba(239, 68, 68, 0.15); color: #ef4444;
    border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 12px;
    padding: 16px; text-align: center; font-size: 24px; font-weight: 700;
}
.status-normal {
    background: rgba(16, 185, 129, 0.15); color: #10b981;
    border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 12px;
    padding: 16px; text-align: center; font-size: 24px; font-weight: 700;
}
.table-header { color: #06b6d4; font-weight: 600; }
div[data-testid="stMetric"] {
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid rgba(6, 182, 212, 0.15);
    border-radius: 12px; padding: 16px;
}
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ───
st.sidebar.markdown('<p class="hero-title" style="font-size:24px">🛡️ VisionGuard-AD</p>', unsafe_allow_html=True)
st.sidebar.markdown('<p class="hero-subtitle">Industrial Anomaly Detection</p>', unsafe_allow_html=True)
st.sidebar.divider()
page = st.sidebar.radio("Navigate", [
    "🔬 Inference", "⚙️ Threshold Tuner",
    "📊 Benchmark", "📷 Live Camera",
])

# Backbone comparison data
BACKBONE_RESULTS = {
    "ResNet-18":       {"image_auroc": 96.90, "pixel_auroc": 97.35, "pro_score": 89.01, "f1": 94.95, "speed": "12.0 img/s", "params": "11M"},
    "WideResNet-50-2": {"image_auroc": 98.38, "pixel_auroc": 97.85, "pro_score": 90.92, "f1": 96.97, "speed": "5.2 img/s",  "params": "69M"},
    "ViT-B/16":        {"image_auroc": 96.54, "pixel_auroc": 96.33, "pro_score": 83.23, "f1": 95.52, "speed": "7.7 img/s",  "params": "86M"},
}


# ═════════════════════════════════════════
# PAGE 1: Inference
# ═════════════════════════════════════════
if page == "🔬 Inference":
    st.markdown('<p class="hero-title">🔬 Inference Engine</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Upload an image · detect anomalies · pixel-level localization</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("⚙️ Configuration")

        # Scan all output directories for models
        all_models = {}
        for out_dir in ["./outputs", "./outputs_wrn", "./outputs_vit"]:
            all_models.update(find_model_files(out_dir))

        if not all_models:
            st.warning("No trained models found. Train a model first.")
            model_key = None
        else:
            model_key = st.selectbox("Model", list(all_models.keys()))

        method = st.selectbox("Method", ["patchcore", "fastflow"])
        backbone = st.selectbox("Backbone", ["wide_resnet50", "resnet18", "resnet50", "vit_b16", "efficientnet_b4"])
        threshold = st.slider("Anomaly Threshold", 0.0, 1.0, 0.42, 0.01)
        uploaded = st.file_uploader("📁 Upload Image", type=["png", "jpg", "jpeg", "bmp"])
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        if uploaded is not None:
            image = Image.open(uploaded).convert("RGB")

            if model_key and model_key in all_models:
                model_info = all_models[model_key]
                model_path = model_info["path"]

                with st.spinner("🔄 Loading model & running inference..."):
                    t0 = time.time()
                    model, device = load_model_cached(model_path, method, backbone, "auto")
                    score, amap = run_inference(model, method, image, device)
                    elapsed = (time.time() - t0) * 1000

                is_defect = score > threshold
                if is_defect:
                    st.markdown(f'<div class="status-defect">⚠️ DEFECTIVE — Score: {score:.4f}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="status-normal">✅ NORMAL — Score: {score:.4f}</div>', unsafe_allow_html=True)

                mcols = st.columns(3)
                mcols[0].metric("Anomaly Score", f"{score:.4f}")
                mcols[1].metric("Threshold", f"{threshold:.2f}")
                mcols[2].metric("Inference", f"{elapsed:.0f}ms")

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.image(image, caption="Original", use_container_width=True)
                with c2:
                    overlay = create_overlay_pil(image, amap, threshold, score)
                    st.image(overlay, caption="Heatmap Overlay", use_container_width=True)
                with c3:
                    mask = create_binary_mask_pil(amap, threshold)
                    st.image(mask, caption="Prediction Mask", use_container_width=True)
            else:
                st.image(image, caption="Uploaded Image", use_container_width=True)
                st.info("Select a trained model to run inference.")


# ═════════════════════════════════════════
# PAGE 2: Threshold Tuner
# ═════════════════════════════════════════
elif page == "⚙️ Threshold Tuner":
    st.markdown('<p class="hero-title">⚙️ Threshold Tuner</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Explore ROC/PR curves · find optimal operating point</p>', unsafe_allow_html=True)

    eval_dirs = []
    for out_root in ["./outputs", "./outputs_wrn", "./outputs_vit"]:
        p = Path(out_root)
        if p.exists():
            for d in p.rglob("eval_results.json"):
                eval_dirs.append(str(d.parent))

    if not eval_dirs:
        st.warning("No evaluation results found. Run `python evaluate.py` first.")
    else:
        selected_dir = st.selectbox("Select Evaluation Run", eval_dirs)

        with open(os.path.join(selected_dir, "eval_results.json")) as f:
            results = json.load(f)

        cols = st.columns(4)
        for col, (key, label) in zip(cols, [
            ("image_auroc", "Image AUROC"), ("pixel_auroc", "Pixel AUROC"),
            ("pro_score", "PRO Score"), ("image_f1", "F1 Score"),
        ]):
            val = results.get(key, results.get(f"image_{key}", 0))
            if isinstance(val, (int, float)):
                col.metric(label, f"{val:.1f}%")
            else:
                col.metric(label, val)

        c1, c2 = st.columns(2)
        roc_path = os.path.join(selected_dir, "roc_curve.png")
        pr_path = os.path.join(selected_dir, "pr_curve.png")
        if os.path.exists(roc_path):
            with c1:
                st.image(roc_path, caption="ROC Curve", use_container_width=True)
        if os.path.exists(pr_path):
            with c2:
                st.image(pr_path, caption="Precision-Recall Curve", use_container_width=True)

        # Visualizations gallery
        viz_dir = os.path.join(selected_dir, "visualizations")
        if os.path.exists(viz_dir):
            st.subheader("📸 Sample Visualizations")
            viz_files = sorted(Path(viz_dir).glob("*.png"))[:8]
            viz_cols = st.columns(4)
            for i, vf in enumerate(viz_files):
                with viz_cols[i % 4]:
                    st.image(str(vf), use_container_width=True)


# ═════════════════════════════════════════
# PAGE 3: Benchmark Results
# ═════════════════════════════════════════
elif page == "📊 Benchmark":
    st.markdown('<p class="hero-title">📊 Backbone Benchmark — MVTec-AD Carpet</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Real results on 127 test images · 3 backbone architectures</p>', unsafe_allow_html=True)

    # Backbone comparison table
    import pandas as pd
    df = pd.DataFrame(BACKBONE_RESULTS).T
    df.index.name = "Backbone"

    st.subheader("🏆 Head-to-Head Comparison")
    st.dataframe(
        df.style
          .format({"image_auroc": "{:.2f}%", "pixel_auroc": "{:.2f}%", "pro_score": "{:.2f}%", "f1": "{:.2f}%"})
          .background_gradient(subset=["image_auroc", "pixel_auroc", "pro_score"], cmap="RdYlGn", vmin=80, vmax=100)
          .highlight_max(subset=["image_auroc", "pixel_auroc", "pro_score", "f1"], color="rgba(6, 182, 212, 0.3)"),
        use_container_width=True,
    )

    # Key findings
    st.subheader("💡 Key Findings")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="metric-card metric-green">
            <div class="metric-value">98.38%</div>
            <div class="metric-label">Best AUROC — WideResNet-50</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">12 img/s</div>
            <div class="metric-label">Fastest — ResNet-18</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="metric-card metric-yellow">
            <div class="metric-value">90.92%</div>
            <div class="metric-label">Best PRO — WideResNet-50</div>
        </div>""", unsafe_allow_html=True)

    # Bar chart
    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        backbones = list(BACKBONE_RESULTS.keys())
        for metric, color in [("image_auroc", "#06b6d4"), ("pixel_auroc", "#8b5cf6"), ("pro_score", "#10b981")]:
            vals = [BACKBONE_RESULTS[b][metric] for b in backbones]
            fig.add_trace(go.Bar(name=metric.replace("_", " ").title(), x=backbones, y=vals, marker_color=color))
        fig.update_layout(
            barmode="group", template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=400, yaxis_range=[75, 100],
            title="Metric Comparison by Backbone",
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.info("Install plotly for interactive charts: pip install plotly")

    # Full benchmark if available
    benchmark_path = Path("./outputs/benchmark/benchmark_results.json")
    if benchmark_path.exists():
        st.subheader("📋 Full Category Benchmark")
        with open(benchmark_path) as f:
            full_results = json.load(f)
        full_df = pd.DataFrame(full_results)
        st.dataframe(full_df.style.background_gradient(subset=["image_auroc"], cmap="RdYlGn", vmin=80, vmax=100), use_container_width=True)


# ═════════════════════════════════════════
# PAGE 4: Live Camera
# ═════════════════════════════════════════
elif page == "📷 Live Camera":
    st.markdown('<p class="hero-title">📷 Live Camera Monitor</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Real-time anomaly detection from webcam feed</p>', unsafe_allow_html=True)

    st.info("💡 For best real-time experience, use the CLI: `python inference.py --input 0 --model_path <path> --threshold 0.4`")

    all_models = {}
    for out_dir in ["./outputs", "./outputs_wrn", "./outputs_vit"]:
        all_models.update(find_model_files(out_dir))

    if all_models:
        c1, c2 = st.columns([1, 2])
        with c1:
            model_key = st.selectbox("Model", list(all_models.keys()))
            backbone = st.selectbox("Backbone", ["wide_resnet50", "resnet18", "vit_b16"])
            threshold = st.slider("Threshold", 0.0, 1.0, 0.42, 0.01)

        with c2:
            camera_input = st.camera_input("Capture Frame")
            if camera_input is not None:
                image = Image.open(camera_input).convert("RGB")
                model_info = all_models[model_key]

                with st.spinner("Running inference..."):
                    model, device = load_model_cached(model_info["path"], "patchcore", backbone, "auto")
                    score, amap = run_inference(model, "patchcore", image, device)

                is_defect = score > threshold
                if is_defect:
                    st.markdown(f'<div class="status-defect">⚠️ DEFECT — {score:.4f}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="status-normal">✅ NORMAL — {score:.4f}</div>', unsafe_allow_html=True)

                overlay = create_overlay_pil(image, amap, threshold, score)
                st.image(overlay, caption="Analysis Result", use_container_width=True)
    else:
        st.warning("No trained models available. Train a model first.")
