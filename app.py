import os
import io
import time
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

import importlib
import model_utils
importlib.reload(model_utils)

# ─── STREAMLIT CONFIGURATION ──────────────────────────────────────────────────
st.set_page_config(
    page_title="OpenCardiac AI Studio | data.pkl",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS STYLING (Dark Glassmorphism & Clinical Modernity) ──────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 95% !important;
    }

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.95) 50%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
        display: flex;
        justify-content: space-between;
        align-items: center;
        backdrop-filter: blur(12px);
    }

    .hero-title {
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.88rem;
        font-weight: 400;
    }

    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(56, 189, 248, 0.12);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        padding: 0.3rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .badge-pill-success {
        background: rgba(16, 185, 129, 0.12);
        color: #10b981;
        border-color: rgba(16, 185, 129, 0.3);
    }

    /* Metric Card Styling */
    .card-metric {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .card-metric:hover {
        border-color: rgba(56, 189, 248, 0.35);
        transform: translateY(-2px);
    }

    .card-metric-title {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94a3b8;
        margin-bottom: 0.35rem;
    }

    .card-metric-value {
        font-size: 1.65rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #f8fafc;
    }

    .card-metric-sub {
        font-size: 0.72rem;
        color: #64748b;
        margin-top: 0.25rem;
    }

    /* Structure swatch pill */
    .structure-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.35rem 0.75rem;
        border-radius: 8px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .pill-lv { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }
    .pill-myo { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .pill-rv { background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }

    /* Custom Streamlit tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 1.5rem;
    }

    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
        font-size: 0.88rem;
        padding: 0 1.25rem;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        background-color: rgba(56, 189, 248, 0.1) !important;
        border-bottom: 2px solid #38bdf8 !important;
        color: #38bdf8 !important;
    }

    /* Enhanced Clinical Image Display Cards */
    .image-viewport-card {
        background: radial-gradient(circle at 50% 0%, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 14px;
        padding: 0.85rem 1rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45);
        margin-bottom: 0.75rem;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .image-viewport-card:hover {
        border-color: rgba(56, 189, 248, 0.4);
        box-shadow: 0 12px 36px rgba(56, 189, 248, 0.15);
    }
    .image-viewport-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .image-viewport-title {
        font-size: 0.98rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        color: #f8fafc;
        display: flex;
        align-items: center;
        gap: 0.45rem;
    }
    .image-viewport-badge {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        background: rgba(255, 255, 255, 0.08);
        color: #cbd5e1;
        border: 1px solid rgba(255, 255, 255, 0.12);
    }

    /* Make images render sharp, large, and beautifully bordered */
    [data-testid="stImage"] {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100% !important;
    }
    [data-testid="stImage"] img {
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        box-shadow: 0 10px 32px rgba(0, 0, 0, 0.6) !important;
        width: 100% !important;
        max-width: 100% !important;
        object-fit: contain !important;
        image-rendering: auto !important;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    [data-testid="stImage"] img:hover {
        box-shadow: 0 14px 40px rgba(56, 189, 248, 0.25) !important;
        border-color: rgba(56, 189, 248, 0.5) !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── CACHED RESOURCE LOADERS ──────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading data.pkl checkpoint architecture...")
def get_checkpoint_data():
    try:
        data = model_utils.load_data_pkl("data.pkl")
        meta = model_utils.parse_checkpoint_metadata(data)
        return data, meta, None
    except Exception as e:
        return None, None, str(e)

@st.cache_resource(show_spinner="Initializing AI Inference Engine...")
def get_model_session():
    try:
        engine = model_utils.get_inference_engine()
        return engine, None
    except Exception as e:
        return None, str(e)

# Load resources
raw_checkpoint, checkpoint_meta, checkpoint_err = get_checkpoint_data()
inference_engine, inference_err = get_model_session()

# ─── HERO BANNER ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-banner">
    <div>
        <div class="hero-title">🫀 OpenCardiac AI Studio</div>
        <div class="hero-subtitle">
            Cardiac Cine MRI Automated Segmentation & Clinical Biometrics Engine running on <code>data.pkl</code>
        </div>
    </div>
    <div style="display: flex; gap: 0.75rem; flex-wrap: wrap;">
        <div class="badge-pill">
            <span>Model:</span>
            <strong>{checkpoint_meta['model'] if checkpoint_meta else 'UnetPlusPlus'}</strong>
        </div>
        <div class="badge-pill badge-pill-success">
            <span>Val Dice:</span>
            <strong>{(checkpoint_meta['best_val_dice']*100):.2f}%</strong>
        </div>
        <div class="badge-pill">
            <span>Engine:</span>
            <strong>{inference_engine['type'].upper() if inference_engine else 'OFFLINE'}</strong>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if checkpoint_err:
    st.error(f"⚠️ Error reading `data.pkl`: {checkpoint_err}")
if inference_err:
    st.warning(f"⚠️ Inference engine warning: {inference_err}")

# ─── SIDEBAR CONTROLS ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Scanner & Model Settings")

    st.markdown("#### 📏 Voxel Resolution")
    col_px, col_py = st.columns(2)
    with col_px:
        pixel_spacing_x = st.number_input("Pixel X (mm)", value=1.4, min_value=0.1, max_value=10.0, step=0.1)
    with col_py:
        pixel_spacing_y = st.number_input("Pixel Y (mm)", value=1.4, min_value=0.1, max_value=10.0, step=0.1)

    st.markdown("#### ✨ Ultra-HD & Edge Refinement")
    enable_hd = st.checkbox(
        "🌟 Ultra-HD Sub-Pixel Refinement",
        value=True,
        help="Continuous bicubic probability upsampling eliminates jagged stair-steps, producing razor-sharp, organic boundaries covering all edges and corners."
    )
    draw_contours = st.checkbox(
        "✏️ Razor-Sharp Boundary Outlines",
        value=True,
        help="Renders crisp 1.5–2px anatomical contour lines around cavity borders for clinical PACS-grade edge clarity."
    )
    col_w, col_boost = st.columns(2)
    with col_w:
        contour_width = st.selectbox("Contour Width", [1, 2, 3], index=1)
    with col_boost:
        contrast_boost = st.slider("Scan Clarity", min_value=1.0, max_value=1.8, value=1.25, step=0.05)

    st.markdown("#### 🎨 Color & Visualization")
    palette_choice = st.selectbox(
        "Color Palette",
        [
            "Classic Clinical (RV Red · MYO Green · LV Blue)",
            "Hemodynamic (LV Red · MYO Green · RV Blue)",
            "Radiology Neon (LV Cyan · MYO Green · RV Magenta)"
        ],
        index=0
    )
    palette_key = "cardiac" if "Hemodynamic" in palette_choice else ("neon" if "Neon" in palette_choice else "classic")
    overlay_alpha = st.slider("Mask Opacity", min_value=0.1, max_value=1.0, value=0.52, step=0.04)

    show_lv = st.checkbox("Left Ventricle (LV)", value=True)
    show_myo = st.checkbox("Myocardium (MYO)", value=True)
    show_rv = st.checkbox("Right Ventricle (RV)", value=True)

    st.markdown("---")
    st.markdown("#### 📦 Checkpoint Summary")
    if checkpoint_meta:
        st.caption(f"**Epoch:** {checkpoint_meta['epoch']} (Fold {checkpoint_meta['fold']})")
        st.caption(f"**Encoder:** `{checkpoint_meta['encoder']}`")
        st.caption(f"**Parameters:** {checkpoint_meta['total_parameters']:,}")
        st.caption(f"**Learning Rate:** {checkpoint_meta['learning_rate']}")
        st.caption(f"**Batch Size:** {checkpoint_meta['batch_size']}")

    st.markdown("---")
    st.caption("OpenCardiac AI • Research & Clinical Prototyping Only")

# ─── MAIN TABS ────────────────────────────────────────────────────────────────
tab_segment, tab_pkl, tab_cine, tab_docs = st.tabs([
    "🫀 Live Segmentation & Biometrics",
    "🔬 data.pkl Architecture Explorer",
    "🎬 4D Cine Cycle & EF Curve",
    "📖 Clinical Specs & Protocol"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: UPLOAD CMRI -> SEND TO MODEL -> SEND & EXPORT SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_segment:
    st.markdown("### 🫀 Upload CMRI ➔ Send to Model ➔ Receive Segmentation")
    st.caption("Upload a Cardiac MRI (DICOM `.dcm`, NIfTI `.nii`/`.nii.gz`, or image), dispatch to the `data.pkl` deep learning model, and inspect / download / send segmentations.")

    # Step 1: Input source selection
    st.markdown("#### Step 1: Upload or Select CMRI Scan")
    input_source = st.radio(
        "Choose CMRI Source:",
        ["📁 Upload CMRI File (.dcm, .nii, .nii.gz, .png, .jpg)", "🧪 Use Preloaded Clinical Demo Scans"],
        horizontal=True
    )

    selected_raw_image = None
    input_file_name = ""
    parsed_cmri = None

    if "Upload CMRI File" in input_source:
        uploaded_file = st.file_uploader(
            "Drop Cardiac MRI scan here (DICOM, NIfTI 3D/4D, or PNG/JPG)",
            type=["dcm", "nii", "gz", "png", "jpg", "jpeg"]
        )
        if uploaded_file is not None:
            input_file_name = uploaded_file.name
            with st.spinner("Parsing CMRI volume header & voxel metadata..."):
                file_bytes = uploaded_file.read()
                parsed_cmri = model_utils.parse_cmri(file_bytes, filename=uploaded_file.name)

            slices = parsed_cmri["slices"]
            total_slices = len(slices)

            # Metadata info badge
            st.success(f"✓ Successfully loaded **{input_file_name}** ({parsed_cmri['mode'].upper()} mode — {total_slices} slice{'s' if total_slices > 1 else ''})")

            if total_slices > 1:
                slice_col1, slice_col2 = st.columns([2, 1])
                with slice_col1:
                    target_slice_idx = st.slider("Select Slice to Segment (Z-Axis / Time Phase):", 0, total_slices - 1, total_slices // 2)
                with slice_col2:
                    st.metric("Total Slices/Frames", total_slices, f"Selected: Slice #{target_slice_idx + 1}")
                selected_raw_image = slices[target_slice_idx]
            else:
                selected_raw_image = slices[0]

            # Spacing
            if parsed_cmri["pixdim"][0] > 0:
                pixel_spacing_x = parsed_cmri["pixdim"][0]
                pixel_spacing_y = parsed_cmri["pixdim"][1]
    else:
        demo_col1, demo_col2 = st.columns([1.5, 2])
        with demo_col1:
            demo_choice = st.selectbox(
                "Select Cardiac Case / Pathology:",
                [
                    "End-Diastole (Relaxed LV, High Volume)",
                    "End-Systole (Contracted LV, Ejected)",
                    "Myocardial Hypertrophy (Thickened Wall)"
                ]
            )

        if "End-Diastole" in demo_choice:
            sample_path = "samples/sample_diastole.png"
            input_file_name = "sample_diastole.png"
        elif "End-Systole" in demo_choice:
            sample_path = "samples/sample_systole.png"
            input_file_name = "sample_systole.png"
        else:
            sample_path = "samples/sample_hypertrophy.png"
            input_file_name = "sample_hypertrophy.png"

        if os.path.exists(sample_path):
            selected_raw_image = np.array(Image.open(sample_path).convert("L"))
            st.info(f"Loaded clinical demo: `{input_file_name}` (224×224 normalized Short-Axis view)")

    st.markdown("---")

    # Step 2: Send to Model Action Button
    st.markdown("#### Step 2: Send to Model")
    btn_col1, btn_col2 = st.columns([1.2, 3])
    with btn_col1:
        run_segmentation = st.button("🚀 Send to Model & Run Segmentation", type="primary", use_container_width=True)

    if selected_raw_image is not None and (run_segmentation or st.session_state.get("auto_segment", True)):
        st.session_state["auto_segment"] = True

        with st.spinner("🤖 Sending CMRI tensor to UnetPlusPlus model for inference..."):
            # Preprocess image
            input_tensor, norm_img_224 = model_utils.preprocess_medical_image(selected_raw_image)

            # Run inference
            t0 = time.time()
            pred_mask, probs = model_utils.run_inference(inference_engine, input_tensor)
            inference_time_ms = (time.time() - t0) * 1000

            # Determine active anatomical classes
            active_classes = []
            if show_rv:
                active_classes.append(1)
            if show_myo:
                active_classes.append(2)
            if show_lv:
                active_classes.append(3)

            # Compute clinical metrics (anatomically correct: RV=1, MYO=2, LV=3)
            metrics = model_utils.compute_clinical_metrics(pred_mask, (pixel_spacing_x, pixel_spacing_y))

            # Ultra-HD Edge & Corner Refinement vs Standard Rendering
            if enable_hd:
                hd_res = model_utils.create_high_res_color_overlay(
                    base_img_orig=selected_raw_image,
                    probs_224=probs,
                    alpha=overlay_alpha,
                    target_size=(896, 896),
                    active_classes=tuple(active_classes),
                    draw_contours=draw_contours,
                    contour_width=contour_width,
                    enhance_contrast=True,
                    contrast_factor=contrast_boost,
                    sharpness_factor=contrast_boost + 0.1,
                    palette=palette_key
                )
                overlay_display = hd_res["composite_rgb"]
                base_display = hd_res["base_rgb"]
                active_mask_for_pkg = hd_res["mask_high"]
                overlay_uint8_for_pkg = hd_res["composite_uint8"]
                disp_w = hd_res["mask_high"].shape[1]
                disp_h = hd_res["mask_high"].shape[0]
                badge_text = f"✨ Ultra-HD {disp_w}×{disp_h} · Sub-Pixel Curved"
                badge_style = "background:rgba(56,189,248,0.18); color:#38bdf8; border:1px solid rgba(56,189,248,0.35);"
            else:
                filtered_mask = pred_mask.copy()
                for c in [1, 2, 3]:
                    if c not in active_classes:
                        filtered_mask[filtered_mask == c] = 0
                overlay_display = model_utils.create_color_overlay(norm_img_224, filtered_mask, alpha=overlay_alpha, palette=palette_key)
                base_display = norm_img_224
                active_mask_for_pkg = filtered_mask
                overlay_uint8_for_pkg = (overlay_display * 255).astype(np.uint8)
                badge_text = "Standard 224×224"
                badge_style = "background:rgba(100,116,139,0.18); color:#94a3b8; border:1px solid rgba(100,116,139,0.3);"

            # Generate export package (with HD files included)
            pkg = model_utils.generate_segmentation_package(
                raw_norm_224=norm_img_224,
                mask=pred_mask,
                metrics=metrics,
                alpha=overlay_alpha,
                high_res_overlay_uint8=overlay_uint8_for_pkg,
                high_res_mask=active_mask_for_pkg,
                palette=palette_key
            )

        # Step 3: Segmentation Results & Biometrics
        st.markdown("#### Step 3: Clinical Biometrics & Segmentation Results")

        # Color badges per active palette
        if palette_key == "classic":
            col_rv_dot, col_myo_dot, col_lv_dot = "🔴 Right Ventricle (RV)", "🟢 Myocardium (MYO)", "🔵 Left Ventricle (LV)"
        elif palette_key == "cardiac":
            col_lv_dot, col_myo_dot, col_rv_dot = "🔴 Left Ventricle (LV)", "🟢 Myocardium (MYO)", "🔵 Right Ventricle (RV)"
        else:
            col_rv_dot, col_myo_dot, col_lv_dot = "🟣 Right Ventricle (RV)", "🟢 Myocardium (MYO)", "🔵 Left Ventricle (LV)"

        # Top Metric Row
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        with m_col1:
            st.markdown(f"""
            <div class="card-metric">
                <div class="card-metric-title">{col_lv_dot}</div>
                <div class="card-metric-value">{metrics['lv_area_cm2']:.2f} <span style="font-size:0.9rem; font-weight:500;">cm²</span></div>
                <div class="card-metric-sub">{metrics['lv_pixels']} pixels</div>
            </div>
            """, unsafe_allow_html=True)

        with m_col2:
            st.markdown(f"""
            <div class="card-metric">
                <div class="card-metric-title">{col_myo_dot}</div>
                <div class="card-metric-value">{metrics['myo_area_cm2']:.2f} <span style="font-size:0.9rem; font-weight:500;">cm²</span></div>
                <div class="card-metric-sub">{metrics['myo_pixels']} pixels</div>
            </div>
            """, unsafe_allow_html=True)

        with m_col3:
            st.markdown(f"""
            <div class="card-metric">
                <div class="card-metric-title">{col_rv_dot}</div>
                <div class="card-metric-value">{metrics['rv_area_cm2']:.2f} <span style="font-size:0.9rem; font-weight:500;">cm²</span></div>
                <div class="card-metric-sub">{metrics['rv_pixels']} pixels</div>
            </div>
            """, unsafe_allow_html=True)

        with m_col4:
            ratio_val = metrics['myo_lv_ratio']
            ratio_color = "#10b981" if (0.6 <= ratio_val <= 1.8) else "#ef4444"
            st.markdown(f"""
            <div class="card-metric">
                <div class="card-metric-title">⚖️ MYO / LV Ratio</div>
                <div class="card-metric-value" style="color: {ratio_color};">{ratio_val:.2f}</div>
                <div class="card-metric-sub">Ref: 0.60 – 1.80</div>
            </div>
            """, unsafe_allow_html=True)

        with m_col5:
            st.markdown(f"""
            <div class="card-metric">
                <div class="card-metric-title">⚡ Latency</div>
                <div class="card-metric-value">{inference_time_ms:.1f} <span style="font-size:0.9rem; font-weight:500;">ms</span></div>
                <div class="card-metric-sub">{inference_engine['type'].upper()} Runtime</div>
            </div>
            """, unsafe_allow_html=True)

        # Clinical Impression Banner
        diag_color = "#10b981" if metrics['status_color'] == "green" else ("#ef4444" if metrics['status_color'] == "red" else "#f59e0b")
        st.markdown(f"""
        <div style="margin: 1.25rem 0; padding: 1rem 1.25rem; background: rgba(15, 23, 42, 0.85); border-left: 4px solid {diag_color}; border-radius: 8px;">
            <div style="font-size: 0.75rem; text-transform: uppercase; color: #94a3b8; font-weight: 600; letter-spacing: 0.05em;">Clinical Impression</div>
            <div style="font-size: 1.15rem; font-weight: 700; color: {diag_color}; margin-top: 0.2rem;">{metrics['diagnosis']}</div>
        </div>
        """, unsafe_allow_html=True)

        # Multi-Stage Image Visualization (High-Resolution Large Viewer)
        st.markdown("#### 🖼️ Visual Multi-Stage Segmentation")

        # Display Controls Bar
        disp_col1, disp_col2 = st.columns([3, 1])
        with disp_col1:
            view_layout = st.radio(
                "Display Layout:",
                [
                    "🖼️ Dual-View (Extra Large Side-by-Side)",
                    "🔲 2×2 Clinical Quad (Large Grid)",
                    "🔍 Focus Cinematic View (Full Width)",
                    "📊 Multi-Strip Overview (4 Columns)"
                ],
                horizontal=True,
                index=0
            )
        with disp_col2:
            st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
            show_extra_channels = st.checkbox("Show Isolated Cavities", value=False)

        # ── Mode 1: Dual-View (Extra Large Side-by-Side) ─────────────────────
        if "Dual-View" in view_layout:
            dv_col1, dv_col2 = st.columns(2, gap="medium")
            with dv_col1:
                st.markdown(f"""
                <div class="image-viewport-card">
                    <div class="image-viewport-header">
                        <span class="image-viewport-title">🏥 1. Original CMRI Scan</span>
                        <span class="image-viewport-badge" style="{badge_style}">{badge_text}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.image(base_display, use_container_width=True, clamp=True)

            with dv_col2:
                st.markdown(f"""
                <div class="image-viewport-card">
                    <div class="image-viewport-header">
                        <span class="image-viewport-title">🎯 2. Segmentation Color Overlay</span>
                        <span class="image-viewport-badge" style="{badge_style}">
                            Opacity: {overlay_alpha:.2f} · Contour: {contour_width}px
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.image(overlay_display, use_container_width=True)

            if show_extra_channels:
                st.markdown("##### 🔬 Individual Segmentation Channels (Large)")
                ch_col1, ch_col2, ch_col3, ch_col4 = st.columns(4, gap="medium")
                with ch_col1:
                    st.markdown("""
                    <div class="image-viewport-card">
                        <div class="image-viewport-header">
                            <span class="image-viewport-title">3. Full Mask</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.image(pkg["mask_bytes"], use_container_width=True)
                with ch_col2:
                    st.markdown("""
                    <div class="image-viewport-card">
                        <div class="image-viewport-header">
                            <span class="image-viewport-title" style="color:#60a5fa;">4. Left Ventricle (LV)</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.image(pkg["lv_mask_bytes"], use_container_width=True)
                with ch_col3:
                    st.markdown("""
                    <div class="image-viewport-card">
                        <div class="image-viewport-header">
                            <span class="image-viewport-title" style="color:#34d399;">5. Myocardium (MYO)</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.image(pkg["myo_mask_bytes"], use_container_width=True)
                with ch_col4:
                    st.markdown("""
                    <div class="image-viewport-card">
                        <div class="image-viewport-header">
                            <span class="image-viewport-title" style="color:#f87171;">6. Right Ventricle (RV)</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.image(pkg["rv_mask_bytes"], use_container_width=True)

        # ── Mode 2: 2x2 Clinical Quad (Large Grid) ───────────────────────────
        elif "2×2 Clinical Quad" in view_layout:
            r1_col1, r1_col2 = st.columns(2, gap="medium")
            with r1_col1:
                st.markdown(f"""
                <div class="image-viewport-card">
                    <div class="image-viewport-header">
                        <span class="image-viewport-title">🏥 1. Original CMRI Scan</span>
                        <span class="image-viewport-badge" style="{badge_style}">{badge_text}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.image(base_display, use_container_width=True, clamp=True)

            with r1_col2:
                st.markdown(f"""
                <div class="image-viewport-card">
                    <div class="image-viewport-header">
                        <span class="image-viewport-title">🎯 2. Color Overlay</span>
                        <span class="image-viewport-badge" style="{badge_style}">Opacity: {overlay_alpha:.2f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.image(overlay_display, use_container_width=True)

            r2_col1, r2_col2 = st.columns(2, gap="medium")
            with r2_col1:
                st.markdown("""
                <div class="image-viewport-card">
                    <div class="image-viewport-header">
                        <span class="image-viewport-title">🎨 3. Segmentation Mask (Color)</span>
                        <span class="image-viewport-badge">HD Labels</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.image(pkg["mask_bytes"], use_container_width=True)

            with r2_col2:
                st.markdown("""
                <div class="image-viewport-card">
                    <div class="image-viewport-header">
                        <span class="image-viewport-title" style="color:#60a5fa;">🔵 4. Isolated Left Ventricle (LV)</span>
                        <span class="image-viewport-badge">Binary Cavity</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.image(pkg["lv_mask_bytes"], use_container_width=True)

        # ── Mode 3: Focus Cinematic View (Full Width) ────────────────────────
        elif "Focus Cinematic View" in view_layout:
            focus_choice = st.selectbox(
                "Select Channel to Display in Cinematic Full-Width:",
                [
                    "🎯 Ultra-HD Color Overlay (Scan + AI Contours)",
                    "🏥 Enhanced CMRI Input Scan",
                    "🎨 Color Segmentation Mask (All Classes)",
                    "🔵 Isolated Left Ventricle (LV Cavity)",
                    "🟢 Isolated Myocardial Wall (MYO)",
                    "🔴 Isolated Right Ventricle (RV Crescent)"
                ]
            )
            _, fc_col, _ = st.columns([0.5, 5, 0.5])
            with fc_col:
                if "Color Overlay" in focus_choice:
                    st.image(overlay_display, caption=f"Cinematic Overlay ({badge_text} · Opacity: {overlay_alpha:.2f})", use_container_width=True)
                elif "Enhanced CMRI" in focus_choice:
                    st.image(base_display, caption=f"Enhanced High-Definition CMRI Scan ({badge_text})", use_container_width=True, clamp=True)
                elif "Color Segmentation Mask" in focus_choice:
                    st.image(pkg["mask_bytes"], caption="Full Color Segmentation Mask", use_container_width=True)
                elif "Isolated Left Ventricle" in focus_choice:
                    st.image(pkg["lv_mask_bytes"], caption="Isolated Left Ventricle (LV)", use_container_width=True)
                elif "Isolated Myocardial" in focus_choice:
                    st.image(pkg["myo_mask_bytes"], caption="Isolated Myocardium Wall (MYO)", use_container_width=True)
                else:
                    st.image(pkg["rv_mask_bytes"], caption="Isolated Right Ventricle (RV)", use_container_width=True)

        # ── Mode 4: Multi-Strip Overview (4 Columns) ─────────────────────────
        else:
            v_col1, v_col2, v_col3, v_col4 = st.columns(4, gap="small")
            with v_col1:
                st.image(base_display, caption="1. Original CMRI Scan", use_container_width=True, clamp=True)
            with v_col2:
                st.image(overlay_display, caption=f"2. Color Overlay ({badge_text})", use_container_width=True)
            with v_col3:
                st.image(pkg["mask_bytes"], caption="3. Segmentation Mask", use_container_width=True)
            with v_col4:
                st.image(pkg["lv_mask_bytes"], caption="4. Isolated LV", use_container_width=True)

        # Structure Legend Badges (Dynamic matching palette)
        if palette_key == "classic":
            st.markdown("""
            <div style="margin-top: 0.5rem; margin-bottom: 1.25rem;">
                <span class="structure-pill" style="background:rgba(239,68,68,0.15); color:#f87171; border:1px solid rgba(239,68,68,0.3);">● Right Ventricle (RV Crescent)</span>
                <span class="structure-pill" style="background:rgba(16,185,129,0.15); color:#34d399; border:1px solid rgba(16,185,129,0.3);">● Myocardium (MYO Muscle)</span>
                <span class="structure-pill" style="background:rgba(59,130,246,0.15); color:#60a5fa; border:1px solid rgba(59,130,246,0.3);">● Left Ventricle (LV Cavity)</span>
            </div>
            """, unsafe_allow_html=True)
        elif palette_key == "cardiac":
            st.markdown("""
            <div style="margin-top: 0.5rem; margin-bottom: 1.25rem;">
                <span class="structure-pill" style="background:rgba(239,68,68,0.15); color:#f87171; border:1px solid rgba(239,68,68,0.3);">● Left Ventricle (LV Cavity)</span>
                <span class="structure-pill" style="background:rgba(16,185,129,0.15); color:#34d399; border:1px solid rgba(16,185,129,0.3);">● Myocardium (MYO Muscle)</span>
                <span class="structure-pill" style="background:rgba(59,130,246,0.15); color:#60a5fa; border:1px solid rgba(59,130,246,0.3);">● Right Ventricle (RV Crescent)</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="margin-top: 0.5rem; margin-bottom: 1.25rem;">
                <span class="structure-pill" style="background:rgba(236,72,153,0.15); color:#f472b6; border:1px solid rgba(236,72,153,0.3);">● Right Ventricle (RV Magenta)</span>
                <span class="structure-pill" style="background:rgba(16,185,129,0.15); color:#34d399; border:1px solid rgba(16,185,129,0.3);">● Myocardium (MYO Emerald)</span>
                <span class="structure-pill" style="background:rgba(6,182,212,0.15); color:#38bdf8; border:1px solid rgba(6,182,212,0.3);">● Left Ventricle (LV Cyan)</span>
            </div>
            """, unsafe_allow_html=True)

        # Polar Chart
        polar_col1, polar_col2 = st.columns([1.2, 1])
        with polar_col1:
            sector_names = [
                "01: Anterior", "02: Anterolateral", "03: Lateral", "04: Posterolateral",
                "05: Inferior", "06: Inferoseptal", "07: Septal", "08: Anteroseptal",
                "09: Anterior Inner", "10: Lateral Inner", "11: Inferior Inner", "12: Septal Inner"
            ]
            fig_polar = go.Figure()
            fig_polar.add_trace(go.Scatterpolar(
                r=metrics['wall_thicknesses'] + [metrics['wall_thicknesses'][0]],
                theta=sector_names + [sector_names[0]],
                fill='toself',
                fillcolor='rgba(16, 185, 129, 0.25)',
                line=dict(color='#10b981', width=2.5),
                name='Thickness (mm)'
            ))
            fig_polar.update_layout(
                title="12-Sector Myocardial Wall Thickness Polar Map",
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, max(max(metrics['wall_thicknesses'])*1.2, 15)], gridcolor='rgba(255,255,255,0.15)'),
                    angularaxis=dict(gridcolor='rgba(255,255,255,0.15)', linecolor='rgba(255,255,255,0.2)')
                ),
                showlegend=False,
                margin=dict(l=40, r=40, t=40, b=30),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=340,
                font=dict(color='#94a3b8')
            )
            st.plotly_chart(fig_polar, use_container_width=True)

        with polar_col2:
            st.markdown("##### Sector Wall Thickness Readings (mm)")
            thickness_df = pd.DataFrame({
                "Sector": [f"Sector {i+1} ({sector_names[i].split(': ')[1]})" for i in range(12)],
                "Thickness (mm)": metrics['wall_thicknesses'],
                "Evaluation": [
                    "Normal" if 6.0 <= t <= 12.0 else ("Hypertrophic" if t > 12.0 else "Thin / Akinetic")
                    for t in metrics['wall_thicknesses']
                ]
            })
            st.dataframe(thickness_df, height=300, use_container_width=True)

        st.markdown("---")

        # Step 4: Send & Export Segmentation Action Center
        st.markdown("#### Step 4: Send & Download Segmentation")
        st.caption("Download the generated segmentations directly or send them to an external PACS / hospital webhook endpoint.")

        dl_col1, dl_col2, dl_col3, dl_col4 = st.columns(4)
        with dl_col1:
            st.download_button(
                label="📥 Download Mask (PNG)",
                data=pkg["mask_bytes"],
                file_name=f"mask_{input_file_name}.png",
                mime="image/png",
                use_container_width=True
            )
        with dl_col2:
            st.download_button(
                label="📥 Download Overlay (PNG)",
                data=pkg["overlay_bytes"],
                file_name=f"overlay_{input_file_name}.png",
                mime="image/png",
                use_container_width=True
            )
        with dl_col3:
            st.download_button(
                label="📥 Download Full Bundle (.ZIP)",
                data=pkg["zip_bytes"],
                file_name=f"segmentation_bundle_{input_file_name}.zip",
                mime="application/zip",
                use_container_width=True
            )
        with dl_col4:
            st.download_button(
                label="📥 Export Clinical Report (JSON)",
                data=pkg["report_json_bytes"],
                file_name=f"report_{input_file_name}.json",
                mime="application/json",
                use_container_width=True
            )

        # External Send Webhook Section
        st.markdown("##### 🌐 Send Segmentation to External Webhook / PACS Server")
        send_col1, send_col2 = st.columns([3, 1])
        with send_col1:
            target_webhook = st.text_input(
                "Target Endpoint URL:",
                value="https://httpbin.org/post",
                placeholder="https://your-hospital-pacs.local/api/receive-segmentation"
            )
        with send_col2:
            st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
            send_btn = st.button("📡 Send Segmentation Now", use_container_width=True)

        if send_btn:
            with st.spinner(f"Sending segmentation package to {target_webhook}..."):
                try:
                    mask_b64 = base64.b64encode(pkg["mask_bytes"]).decode("utf-8")
                    overlay_b64 = base64.b64encode(pkg["overlay_bytes"]).decode("utf-8")
                    status_code, resp_text = model_utils.send_segmentation_to_endpoint(
                        target_webhook,
                        metrics,
                        mask_base64=mask_b64,
                        overlay_base64=overlay_b64
                    )
                    if 200 <= status_code < 300:
                        st.success(f"✓ Segmentation successfully sent! Server returned HTTP {status_code}.")
                    else:
                        st.warning(f"Server responded with HTTP {status_code}: {resp_text[:200]}")
                except Exception as ex:
                    st.error(f"Error sending segmentation: {str(ex)}")

        # Step 5: Developer API Integration
        st.markdown("---")
        with st.expander("💻 Developer API: Send CMRI & Get Segmentation via REST API"):
            st.markdown("""
            You can also upload CMRI and get segmentations programmatically via the **OpenCardiac AI REST API** (`api_server.py`):

            **Endpoint:** `POST http://localhost:8000/api/segment`

            **Python Example:**
            ```python
            import requests

            url = "http://localhost:8000/api/segment"
            files = {"file": open("my_cardiac_mri.dcm", "rb")}
            response = requests.post(url, files=files)
            result = response.json()

            print("LV Area:", result["metrics"]["lv_area_cm2"])
            print("Diagnosis:", result["metrics"]["diagnosis"])
            # result["segmentation"]["mask_png_base64"] contains the mask PNG!
            ```

            **cURL Command:**
            ```bash
            curl -X POST "http://localhost:8000/api/segment" -F "file=@my_scan.dcm"
            ```
            """)

    elif selected_raw_image is None:
        st.info("👆 Please upload a CMRI file or choose a preloaded demo scan above to begin.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: DATA.PKL CHECKPOINT & ARCHITECTURE EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab_pkl:
    st.markdown("### 🔬 `data.pkl` Deep Learning Checkpoint Anatomy")
    st.caption("Complete breakdown of hyperparameters, weights, layer tensors, and training metrics serialized in `data.pkl`.")

    if checkpoint_meta is not None:
        # Checkpoint KPI row
        pk_col1, pk_col2, pk_col3, pk_col4, pk_col5 = st.columns(5)
        with pk_col1:
            st.markdown(f"""
            <div class="card-metric">
                <div class="card-metric-title">🏆 Best Val Dice</div>
                <div class="card-metric-value" style="color:#10b981;">{(checkpoint_meta['best_val_dice']*100):.2f}%</div>
                <div class="card-metric-sub">Loss: {checkpoint_meta['best_val_loss']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)

        with pk_col2:
            st.markdown(f"""
            <div class="card-metric">
                <div class="card-metric-title">⏳ Epoch & Fold</div>
                <div class="card-metric-value">Epoch {checkpoint_meta['epoch']}</div>
                <div class="card-metric-sub">Cross-Validation Fold {checkpoint_meta['fold']}</div>
            </div>
            """, unsafe_allow_html=True)

        with pk_col3:
            st.markdown(f"""
            <div class="card-metric">
                <div class="card-metric-title">🧠 Total Parameters</div>
                <div class="card-metric-value">{(checkpoint_meta['total_parameters']/1e6):.2f}M</div>
                <div class="card-metric-sub">{checkpoint_meta['total_parameters']:,} parameters</div>
            </div>
            """, unsafe_allow_html=True)

        with pk_col4:
            st.markdown(f"""
            <div class="card-metric">
                <div class="card-metric-title">🏗️ Model & Encoder</div>
                <div class="card-metric-value" style="font-size:1.15rem;">{checkpoint_meta['model']}</div>
                <div class="card-metric-sub">{checkpoint_meta['encoder']}</div>
            </div>
            """, unsafe_allow_html=True)

        with pk_col5:
            st.markdown(f"""
            <div class="card-metric">
                <div class="card-metric-title">⚡ Training Specs</div>
                <div class="card-metric-value" style="font-size:1.3rem;">LR: {checkpoint_meta['learning_rate']}</div>
                <div class="card-metric-sub">Batch Size: {checkpoint_meta['batch_size']}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Parameter distribution chart & architecture breakdown
        ch_col1, ch_col2 = st.columns([1, 1])

        with ch_col1:
            st.markdown("#### 📊 Parameter Distribution by Component")
            dist_data = pd.DataFrame({
                "Component": ["Encoder (timm-tf_efficientnet_lite0)", "Decoder (Nested Dense Skip Pathways)", "Segmentation Head (4 Classes)"],
                "Parameters": [checkpoint_meta["encoder_parameters"], checkpoint_meta["decoder_parameters"], checkpoint_meta["head_parameters"]],
                "Share": [
                    f"{(checkpoint_meta['encoder_parameters']/checkpoint_meta['total_parameters']*100):.1f}%",
                    f"{(checkpoint_meta['decoder_parameters']/checkpoint_meta['total_parameters']*100):.1f}%",
                    f"{(checkpoint_meta['head_parameters']/checkpoint_meta['total_parameters']*100):.2f}%"
                ]
            })
            fig_pie = px.pie(
                dist_data,
                names="Component",
                values="Parameters",
                color="Component",
                color_discrete_map={
                    "Encoder (timm-tf_efficientnet_lite0)": "#38bdf8",
                    "Decoder (Nested Dense Skip Pathways)": "#818cf8",
                    "Segmentation Head (4 Classes)": "#34d399"
                },
                hole=0.45
            )
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=20, b=20),
                font=dict(color="#94a3b8"),
                height=320
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with ch_col2:
            st.markdown("#### 📐 Architecture Technical Specs")
            st.markdown(f"""
            - **Root File:** `data.pkl` (Serialized PyTorch checkpoint state)
            - **Input Channel:** `1` (Grayscale Short-Axis Cine MRI, normalized $[0, 1]$)
            - **Input Resolution:** `224 × 224`
            - **Output Classes:** `4` (Background, Left Ventricle, Myocardium, Right Ventricle)
            - **Encoder:** `timm-tf_efficientnet_lite0` (Depthwise separable convolutions for high throughput)
            - **Decoder:** `UnetPlusPlus` with nested, dense skip connections bridging semantic gap
            - **Total Serialized Layers:** `{checkpoint_meta['total_layers']}` tensors
            - **Checkpoint Validation Score:** **`{checkpoint_meta['best_val_dice']:.4f}` Dice**
            """)

        # Interactive Layer Explorer
        st.markdown("---")
        st.markdown("#### 🔍 Interactive State Dict & Layer Tensor Explorer")
        st.caption("Search, filter, and inspect tensor shapes and parameters stored inside `data.pkl`.")

        f_col1, f_col2, f_col3 = st.columns([2, 1, 1])
        with f_col1:
            search_query = st.text_input("Filter by Layer Name (e.g., 'conv_stem', 'decoder.blocks', 'attention', 'head')", value="")
        with f_col2:
            category_filter = st.selectbox("Filter by Category", ["All", "Encoder", "Decoder", "Head", "Other"])
        with f_col3:
            st.markdown(f"<div style='margin-top:1.8rem; color:#94a3b8; font-size:0.85rem;'>Total Tensors: <strong>{checkpoint_meta['total_layers']}</strong></div>", unsafe_allow_html=True)

        all_layers = checkpoint_meta["layers"]
        if search_query:
            all_layers = [l for l in all_layers if search_query.lower() in l["layer_name"].lower()]
        if category_filter != "All":
            all_layers = [l for l in all_layers if l["category"] == category_filter]

        df_layers = pd.DataFrame(all_layers)
        if not df_layers.empty:
            df_display = df_layers[["category", "layer_name", "shape", "num_elements", "dtype"]].copy()
            df_display.columns = ["Category", "Tensor Key", "Shape", "Parameters", "Data Type"]
            st.dataframe(
                df_display,
                use_container_width=True,
                height=450,
                column_config={
                    "Parameters": st.column_config.NumberColumn(format="%d"),
                }
            )
        else:
            st.info("No layers matched the filter criteria.")

    else:
        st.error("No checkpoint data loaded.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: 4D CINE CYCLE & EJECTION FRACTION (EF)
# ══════════════════════════════════════════════════════════════════════════════
with tab_cine:
    st.markdown("### 🎬 4D Multi-Frame Cine MRI & Ejection Fraction Analysis")
    st.caption("Process a full cardiac cycle cine loop (End-Diastole to End-Systole) to automatically compute ventricular volume curves and Ejection Fraction.")

    # Load 4D Cine sample
    cine_path = "samples/cine_series_4d.npz"
    if os.path.exists(cine_path):
        cine_npz = np.load(cine_path)
        cine_frames = cine_npz["frames"] # (10, 224, 224)
        num_frames = len(cine_frames)

        c_top1, c_top2 = st.columns([2, 1])
        with c_top1:
            frame_idx = st.slider("Select Cine Frame (Phase)", min_value=0, max_value=num_frames-1, value=0)
        with c_top2:
            st.caption(f"Loaded: `{cine_path}` ({num_frames} frames in cardiac cycle)")

        # Run inference across all frames for curve
        all_lv_areas = []
        all_myo_areas = []
        all_masks = []

        if inference_engine is not None:
            for f_i in range(num_frames):
                frame_data = cine_frames[f_i]
                t_in, norm_f = model_utils.preprocess_medical_image(frame_data)
                m, p = model_utils.run_inference(inference_engine, t_in)
                all_masks.append(m)
                m_stats = model_utils.compute_clinical_metrics(m, (pixel_spacing_x, pixel_spacing_y))
                all_lv_areas.append(m_stats["lv_area_cm2"])
                all_myo_areas.append(m_stats["myo_area_cm2"])

            # Compute EF
            ed_idx = int(np.argmax(all_lv_areas))
            es_idx = int(np.argmin(all_lv_areas))
            edv = all_lv_areas[ed_idx]
            esv = all_lv_areas[es_idx]
            ef = ((edv - esv) / edv * 100.0) if edv > 0 else 0.0

            # EF Metrics Banner
            ef_col1, ef_col2, ef_col3, ef_col4 = st.columns(4)
            with ef_col1:
                ef_color = "#10b981" if ef >= 50 else ("#f59e0b" if ef >= 40 else "#ef4444")
                st.markdown(f"""
                <div class="card-metric">
                    <div class="card-metric-title">🫀 Ejection Fraction (EF)</div>
                    <div class="card-metric-value" style="color: {ef_color};">{ef:.1f}%</div>
                    <div class="card-metric-sub">Ref Normal: ≥ 50-55%</div>
                </div>
                """, unsafe_allow_html=True)

            with ef_col2:
                st.markdown(f"""
                <div class="card-metric">
                    <div class="card-metric-title">📈 End-Diastolic Area (EDA)</div>
                    <div class="card-metric-value">{edv:.2f} <span style="font-size:0.9rem;">cm²</span></div>
                    <div class="card-metric-sub">Frame {ed_idx + 1} (Max LV Area)</div>
                </div>
                """, unsafe_allow_html=True)

            with ef_col3:
                st.markdown(f"""
                <div class="card-metric">
                    <div class="card-metric-title">📉 End-Systolic Area (ESA)</div>
                    <div class="card-metric-value">{esv:.2f} <span style="font-size:0.9rem;">cm²</span></div>
                    <div class="card-metric-sub">Frame {es_idx + 1} (Min LV Area)</div>
                </div>
                """, unsafe_allow_html=True)

            with ef_col4:
                stroke_area = edv - esv
                st.markdown(f"""
                <div class="card-metric">
                    <div class="card-metric-title">🔄 Stroke Area Change</div>
                    <div class="card-metric-value">{stroke_area:.2f} <span style="font-size:0.9rem;">cm²</span></div>
                    <div class="card-metric-sub">Δ (EDA − ESA)</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # Large Side-by-Side Cine Phase Images
            st.markdown(f"#### 🎬 Cine Phase Frame #{frame_idx + 1} of {num_frames} (Dual-View)")
            cine_img_col1, cine_img_col2 = st.columns(2, gap="medium")

            curr_frame = cine_frames[frame_idx]
            if enable_hd:
                t_in, _ = model_utils.preprocess_medical_image(curr_frame)
                _, p_curr = model_utils.run_inference(inference_engine, t_in)
                hd_cine = model_utils.create_high_res_color_overlay(
                    base_img_orig=curr_frame,
                    probs_224=p_curr,
                    alpha=overlay_alpha,
                    target_size=(896, 896),
                    draw_contours=draw_contours,
                    contour_width=contour_width,
                    palette=palette_key
                )
                curr_display_scan = hd_cine["base_rgb"]
                curr_overlay = hd_cine["composite_rgb"]
                cine_badge = "✨ Ultra-HD 896×896 · Sub-Pixel"
                cine_badge_style = "background:rgba(56,189,248,0.18); color:#38bdf8; border:1px solid rgba(56,189,248,0.35);"
            else:
                curr_display_scan = curr_frame
                curr_mask = all_masks[frame_idx]
                curr_overlay = model_utils.create_color_overlay(curr_frame, curr_mask, alpha=overlay_alpha, palette=palette_key)
                cine_badge = "Standard 224×224"
                cine_badge_style = "background:rgba(100,116,139,0.18); color:#94a3b8; border:1px solid rgba(100,116,139,0.3);"

            with cine_img_col1:
                st.markdown(f"""
                <div class="image-viewport-card">
                    <div class="image-viewport-header">
                        <span class="image-viewport-title">🏥 Original Cine Frame Scan</span>
                        <span class="image-viewport-badge" style="{cine_badge_style}">{cine_badge}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.image(curr_display_scan, use_container_width=True, clamp=True)

            with cine_img_col2:
                st.markdown(f"""
                <div class="image-viewport-card">
                    <div class="image-viewport-header">
                        <span class="image-viewport-title">🎯 Segmented Cine Overlay</span>
                        <span class="image-viewport-badge" style="{cine_badge_style}">
                            LV Area: {all_lv_areas[frame_idx]:.2f} cm² · Frame #{frame_idx + 1}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.image(curr_overlay, use_container_width=True)

            # Full-Width Plotly Phase Volume Curve
            st.markdown("---")
            # Plotly Phase Curve
            curve_df = pd.DataFrame({
                "Frame": list(range(1, num_frames + 1)),
                "LV Cavity Area (cm²)": all_lv_areas,
                "Myocardium Area (cm²)": all_myo_areas
            })
            fig_curve = go.Figure()
            fig_curve.add_trace(go.Scatter(
                x=curve_df["Frame"],
                y=curve_df["LV Cavity Area (cm²)"],
                mode="lines+markers",
                name="LV Cavity Area",
                line=dict(color="#ef4444", width=3),
                marker=dict(size=8)
            ))
            fig_curve.add_trace(go.Scatter(
                x=curve_df["Frame"],
                y=curve_df["Myocardium Area (cm²)"],
                mode="lines+markers",
                name="Myocardium Area",
                line=dict(color="#10b981", width=2.5, dash="dot"),
                marker=dict(size=6)
            ))
            # Add vertical marker for current frame
            fig_curve.add_vline(x=frame_idx + 1, line_width=2, line_dash="dash", line_color="#38bdf8", annotation_text="Selected Frame")

            fig_curve.update_layout(
                title="Cardiac Phase Volume / Area Curve (Cine)",
                xaxis_title="Cardiac Frame Index (Time)",
                yaxis_title="Area (cm²)",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8"),
                margin=dict(l=40, r=40, t=50, b=40),
                height=360,
                legend=dict(orientation="h", y=1.12)
            )
            st.plotly_chart(fig_curve, use_container_width=True)

    else:
        st.warning("Sample cine 4D series not found. Please verify the `samples/` folder.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: CLINICAL DOCUMENTATION & ARCHITECTURAL PROTOCOL
# ══════════════════════════════════════════════════════════════════════════════
with tab_docs:
    st.markdown("### 📖 OpenCardiac AI Clinical & Architectural Documentation")
    st.markdown("""
    #### 1. Background & Clinical Intent
    Cardiac Magnetic Resonance (CMR) imaging is the gold standard for non-invasive assessment of myocardial structure and ventricular function.
    Quantifying left ventricular ejection fraction (LVEF), ventricular volumes, and regional wall thickness is critical in:
    - **Heart Failure Assessment** (HFrEF vs HFpEF)
    - **Myocardial Infarction & Ischemic Heart Disease**
    - **Hypertrophic Cardiomyopathy (HCM)**
    - **Dilated Cardiomyopathy (DCM)**

    #### 2. Deep Learning Segmentation Pipeline
    - **Model Architecture:** **UnetPlusPlus (Nested U-Net)**
      - Introduces nested, dense skip connections bridging semantic features across encoder and decoder levels.
      - Mitigates fine structural boundary loss compared to standard UNet.
    - **Backbone Encoder:** **EfficientNet Lite-0 (`timm-tf_efficientnet_lite0`)**
      - Mobile/edge-optimized neural network with inverted residual blocks.
      - Low compute latency (< 25 ms per slice) without sacrificing receptive field.
    - **Segmentation Classes:**
      - `Class 0`: Background
      - `Class 1`: Left Ventricle (LV) Cavity / Endocardium (Red)
      - `Class 2`: Myocardium (MYO) Wall (Emerald)
      - `Class 3`: Right Ventricle (RV) Cavity (Blue)

    #### 3. Ejection Fraction Formulation
    $$EF = \\frac{\\text{EDV} - \\text{ESV}}{\\text{EDV}} \\times 100\\%$$
    - **Normal:** $EF \\ge 50\\%$
    - **Mildly Impaired:** $40\\% \\le EF < 50\\%$
    - **Moderately Impaired:** $30\\% \\le EF < 40\\%$
    - **Severely Impaired:** $EF < 30\\%$

    #### 4. Disclaimer
    *This software is intended strictly for research and educational purposes. It is not approved as a medical device for primary diagnostic use.*
    """)
