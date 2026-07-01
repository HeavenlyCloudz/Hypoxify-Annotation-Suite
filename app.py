"""Hypoxify Annotation Suite - Streamlit Application

This app allows researchers to:
1. Select their imaging configuration (4-antenna microwave, 8-antenna, thermoacoustic, etc.)
2. Upload their data (images or raw CSV/MAT files)
3. Segment using SAM with click prompting
4. Export to COCO, YOLO, MONAI, or PNG
5. Download the annotated data
"""

import streamlit as st
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from pathlib import Path
import tempfile
import json
import io
import base64
import time

# Import hypoxify_annotate
from hypoxify_annotate import (
    SUPPORTED_CONFIGURATIONS,
    EXPORT_FORMATS,
    SAMWrapper,
    PhysicsGuidedSegmenter,
    delay_and_sum_reconstruction,
    auto_load,
    load_multi_angle_scans,
    apply_background_subtraction_to_paths,
    to_coco_format,
    to_yolo_format,
    to_monai_format,
    save_coco,
    save_yolo,
    save_png_mask,
)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Hypoxify Annotation Suite",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #0d47a1;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .feature-card {
        background-color: #f5f5f5;
        border-radius: 10px;
        padding: 15px;
        margin: 5px 0;
        border-left: 4px solid #0d47a1;
    }
    .stButton button {
        font-weight: 600;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0d47a1;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

if "image_loaded" not in st.session_state:
    st.session_state.image_loaded = False
if "current_image" not in st.session_state:
    st.session_state.current_image = None
if "current_mask" not in st.session_state:
    st.session_state.current_mask = None
if "foreground_clicks" not in st.session_state:
    st.session_state.foreground_clicks = []
if "background_clicks" not in st.session_state:
    st.session_state.background_clicks = []
if "segmenter" not in st.session_state:
    st.session_state.segmenter = None
if "raw_data" not in st.session_state:
    st.session_state.raw_data = None
if "frequencies" not in st.session_state:
    st.session_state.frequencies = None
if "selected_config" not in st.session_state:
    st.session_state.selected_config = "microwave_4_antenna"
if "reconstructed_image" not in st.session_state:
    st.session_state.reconstructed_image = None
if "mode" not in st.session_state:
    st.session_state.mode = "direct_image"  # or "raw_reconstruct"
if "masks_history" not in st.session_state:
    st.session_state.masks_history = []
if "export_ready" not in st.session_state:
    st.session_state.export_ready = False

# =============================================================================
# SIDEBAR - CONFIGURATION AND CONTROLS
# =============================================================================

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # Configuration selection
    config_names = list(SUPPORTED_CONFIGURATIONS.keys())
    config_labels = {
        k: f"{SUPPORTED_CONFIGURATIONS[k]['modality'].title()} - {SUPPORTED_CONFIGURATIONS[k]['description']}"
        for k in config_names
    }
    
    selected_config = st.selectbox(
        "Select System Configuration",
        options=config_names,
        format_func=lambda x: config_labels[x],
        index=config_names.index(st.session_state.selected_config) if st.session_state.selected_config in config_names else 0,
        help="Select the imaging system configuration matching your data"
    )
    
    if selected_config != st.session_state.selected_config:
        st.session_state.selected_config = selected_config
        st.session_state.foreground_clicks = []
        st.session_state.background_clicks = []
        st.session_state.current_mask = None
        st.session_state.reconstructed_image = None
        st.session_state.image_loaded = False
    
    st.markdown("---")
    
    # Mode selection
    st.markdown("### 📥 Input Mode")
    
    mode = st.radio(
        "How are you uploading your data?",
        options=["Upload Image (PNG/JPG/TIFF)", "Upload Raw Data (CSV/S2P/MAT)"],
        index=0,
        help="Direct image upload works for any image. Raw data will be reconstructed first."
    )
    
    if "Raw" in mode:
        st.session_state.mode = "raw_reconstruct"
    else:
        st.session_state.mode = "direct_image"
    
    st.markdown("---")
    
    # SAM Settings
    st.markdown("### 🎯 Segmentation Settings")
    
    sam_model = st.selectbox(
        "SAM Model",
        options=["vit_b", "vit_l", "vit_h"],
        index=0,
        help="vit_b is fastest, vit_h is most accurate but slower"
    )
    
    use_physics_guidance = st.checkbox(
        "🧬 Physics-Guided Segmentation",
        value=False,
        help="Use physical features (dielectric contrast, acoustic pressure) to guide SAM"
    )
    
    st.markdown("---")
    
    # Export
    st.markdown("### 📤 Export")
    
    export_format = st.selectbox(
        "Export Format",
        options=list(EXPORT_FORMATS.keys()),
        format_func=lambda x: f"{x.upper()} - {EXPORT_FORMATS[x]['description']}"
    )
    
    if st.button("📥 Download Annotation", type="primary", use_container_width=True):
        if st.session_state.current_mask is not None:
            st.session_state.export_ready = True
            st.rerun()
        else:
            st.warning("Please generate a mask first.")

# =============================================================================
# MAIN CONTENT AREA
# =============================================================================

st.markdown('<p class="main-header">🔬 Hypoxify Annotation Suite</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Physics-informed segmentation for microwave and thermoacoustic imaging</p>', unsafe_allow_html=True)

# Display current configuration info
config = SUPPORTED_CONFIGURATIONS[st.session_state.selected_config]
st.info(
    f"**Active Configuration:** {config['modality'].title()} - {config['description']}\n\n"
    f"**Supported Formats:** {', '.join(config['file_formats'])}"
)

# =============================================================================
# TWO-COLUMN LAYOUT
# =============================================================================

col1, col2 = st.columns([1, 1])

# =============================================================================
# LEFT COLUMN - IMAGE UPLOAD & PREPROCESSING
# =============================================================================

with col1:
    st.markdown("## 📷 Input Data")
    
    # File upload based on mode
    if st.session_state.mode == "direct_image":
        uploaded_file = st.file_uploader(
            "Upload Image",
            type=["png", "jpg", "jpeg", "tiff", "bmp", "dcm"],
            help="Upload a reconstructed image for annotation"
        )
        
        if uploaded_file is not None:
            try:
                # Read image
                image = Image.open(uploaded_file)
                image_array = np.array(image)
                
                # Convert to RGB if grayscale
                if image_array.ndim == 2:
                    image_array = np.stack([image_array] * 3, axis=-1)
                
                st.session_state.current_image = image_array
                st.session_state.image_loaded = True
                st.session_state.reconstructed_image = None
                st.session_state.foreground_clicks = []
                st.session_state.background_clicks = []
                st.session_state.current_mask = None
                
                # Display image info
                st.write(f"**Image shape:** {image_array.shape}")
                st.write(f"**Data type:** {image_array.dtype}")
                
            except Exception as e:
                st.error(f"Error loading image: {e}")
    
    else:  # raw_reconstruct mode
        uploaded_files = st.file_uploader(
            "Upload Raw Data Files",
            type=["csv", "s2p", "mat"],
            accept_multiple_files=True,
            help="Upload S21 parameter files. For multi-angle data, upload all files."
        )
        
        if uploaded_files:
            st.write(f"**Files uploaded:** {len(uploaded_files)}")
            
            # Determine if multi-angle
            has_angle = any("angle" in f.name.lower() for f in uploaded_files)
            
            if has_angle:
                st.info("Multi-angle data detected. Files will be averaged across rotations.")
            
            # Reconstruction button
            if st.button("🔨 Reconstruct Image from Raw Data", type="primary"):
                with st.spinner("Reconstructing image from raw data..."):
                    try:
                        # Load and process files
                        temp_dir = tempfile.mkdtemp()
                        file_paths = []
                        
                        for f in uploaded_files:
                            temp_path = Path(temp_dir) / f.name
                            temp_path.write_bytes(f.read())
                            file_paths.append(temp_path)
                        
                        # Load data
                        if len(file_paths) == 1:
                            frequencies, s21_data = auto_load(file_paths[0])
                            
                            # For single file, need to know which path
                            # Assume it's path 1
                            s21_data_dict = {1: s21_data}
                        else:
                            # Multiple files - load each
                            s21_data_dict = {}
                            for fp in file_paths:
                                # Try to extract path number from filename
                                import re
                                match = re.search(r'path(\d+)', fp.name)
                                if match:
                                    path_num = int(match.group(1))
                                    _, s21 = auto_load(fp)
                                    s21_data_dict[path_num] = s21
                            
                            frequencies, _ = auto_load(file_paths[0])
                        
                        # Apply background subtraction (if baseline available)
                        # For now, we'll skip or use estimate
                        
                        # Reconstruct
                        image = delay_and_sum_reconstruction(
                            s21_data_dict,
                            frequencies,
                            grid_size=80,
                            grid_extent=100.0
                        )
                        
                        st.session_state.reconstructed_image = image
                        st.session_state.current_image = image
                        st.session_state.image_loaded = True
                        st.session_state.foreground_clicks = []
                        st.session_state.background_clicks = []
                        st.session_state.current_mask = None
                        st.session_state.frequencies = frequencies
                        st.session_state.raw_data = s21_data_dict
                        
                        st.success("Reconstruction complete!")
                        
                    except Exception as e:
                        st.error(f"Reconstruction error: {e}")
                        import traceback
                        st.code(traceback.format_exc())
    
    # =========================================================================
    # DISPLAY CURRENT IMAGE
    # =========================================================================
    
    st.markdown("### 🖼️ Current Image")
    
    if st.session_state.current_image is not None:
        # Resize for display
        img = st.session_state.current_image
        if img.dtype != np.uint8:
            img = (img / img.max() * 255).astype(np.uint8)
        
        # Display image with click handling
        st.image(img, use_container_width=True, clamp=True)
        
        # Show clicks
        if st.session_state.foreground_clicks:
            st.caption(f"🔵 Foreground clicks: {len(st.session_state.foreground_clicks)}")
        if st.session_state.background_clicks:
            st.caption(f"🔴 Background clicks: {len(st.session_state.background_clicks)}")
        
        # Mask overlay option
        if st.session_state.current_mask is not None:
            if st.checkbox("Show mask overlay"):
                mask = st.session_state.current_mask
                # Create overlay
                overlay = img.copy()
                if overlay.ndim == 2:
                    overlay = np.stack([overlay] * 3, axis=-1)
                overlay[mask > 0, 0] = overlay[mask > 0, 0] * 0.5 + 200 * 0.5
                st.image(overlay, use_container_width=True)
        
        # Interactive click instructions
        st.caption("💡 Click on image below to add foreground points. Right-click for background.")
        st.caption(f"Click mode: {'Foreground' if not st.session_state.background_clicks else 'Mixed'}")
        
    else:
        st.info("👆 Upload data to begin")

# =============================================================================
# RIGHT COLUMN - SEGMENTATION & EXPORT
# =============================================================================

with col2:
    st.markdown("## 🎯 Segmentation")
    
    # Click controls
    col_reset, col_undo = st.columns(2)
    
    with col_reset:
        if st.button("🔄 Reset All Clicks", use_container_width=True):
            st.session_state.foreground_clicks = []
            st.session_state.background_clicks = []
            st.session_state.current_mask = None
            st.rerun()
    
    with col_undo:
        if st.button("↩️ Undo Last Click", use_container_width=True):
            if st.session_state.background_clicks:
                st.session_state.background_clicks.pop()
            elif st.session_state.foreground_clicks:
                st.session_state.foreground_clicks.pop()
            st.session_state.current_mask = None
            st.rerun()
    
    # Click mode toggle
    click_mode = st.radio(
        "Click Mode",
        ["Foreground (tumor)", "Background (remove)"],
        horizontal=True,
        key="click_mode_radio"
    )
    
    # Generate button
    if st.session_state.image_loaded:
        if len(st.session_state.foreground_clicks) > 0:
            if st.button("⚡ Generate Mask", type="primary", use_container_width=True):
                with st.spinner("Segmenting..."):
                    try:
                        # Initialize SAM
                        if st.session_state.segmenter is None:
                            # Try to load SAM from common paths
                            import os
                            sam_paths = [
                                Path("sam_vit_b.pth"),
                                Path.home() / ".cache" / "sam" / "sam_vit_b.pth",
                                Path.home() / "sam" / "sam_vit_b.pth",
                            ]
                            
                            # Check if SAM checkpoint exists
                            sam_path = None
                            for p in sam_paths:
                                if p.exists():
                                    sam_path = p
                                    break
                            
                            if sam_path:
                                segmenter = SAMWrapper(
                                    model_type=sam_model,
                                    checkpoint_path=sam_path
                                )
                            else:
                                st.warning("SAM checkpoint not found. Using simulation mode.")
                                # Create a wrapper that simulates SAM
                                class MockSAM:
                                    def set_image(self, img):
                                        self.image = img
                                    def from_clicks(self, fg, bg=None):
                                        # Create circular mask for demo
                                        h, w = self.image.shape[:2]
                                        mask = np.zeros((h, w))
                                        if fg:
                                            cx, cy = fg[0]
                                            for i in range(h):
                                                for j in range(w):
                                                    if (i-cy)**2 + (j-cx)**2 < 1600:
                                                        mask[i,j] = 1
                                        return mask
                                segmenter = MockSAM()
                            
                            st.session_state.segmenter = segmenter
                        
                        # Set image
                        st.session_state.segmenter.set_image(st.session_state.current_image)
                        
                        # Generate mask
                        if use_physics_guidance and st.session_state.raw_data is not None:
                            # Physics-guided segmentation
                            pg_segmenter = PhysicsGuidedSegmenter(
                                st.session_state.segmenter,
                                raw_rf_data=st.session_state.raw_data
                            )
                            mask = pg_segmenter.segment_from_physics_clicks(
                                st.session_state.foreground_clicks,
                                st.session_state.background_clicks if st.session_state.background_clicks else None
                            )
                        else:
                            # Standard SAM segmentation
                            mask = st.session_state.segmenter.from_clicks(
                                st.session_state.foreground_clicks,
                                st.session_state.background_clicks if st.session_state.background_clicks else None
                            )
                        
                        st.session_state.current_mask = mask
                        st.session_state.masks_history.append(mask)
                        st.success("Mask generated!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Segmentation error: {e}")
                        import traceback
                        st.code(traceback.format_exc())
        else:
            st.info("Add at least one foreground click (blue dot) to generate a mask.")
    else:
        st.info("Load an image first.")
    
    # =========================================================================
    # MASK DISPLAY
    # =========================================================================
    
    st.markdown("### 🧬 Mask Result")
    
    if st.session_state.current_mask is not None:
        mask = st.session_state.current_mask
        
        # Display mask
        st.image(mask * 255, use_container_width=True, clamp=True)
        
        # Mask statistics
        mask_area = np.sum(mask)
        mask_percent = (mask_area / mask.size) * 100
        st.caption(f"Mask area: {mask_area} pixels ({mask_percent:.2f}% of image)")
        
        # Export options
        st.markdown("### 📤 Export")
        
        # Show export preview for the selected format
        if export_format == "coco":
            coco_data = to_coco_format(
                [mask],
                image_ids=[1],
                image_shapes=[mask.shape]
            )
            st.json(coco_data)
            
        elif export_format == "yolo":
            yolo_data = to_yolo_format([mask], [mask.shape])
            st.code(yolo_data[0] if yolo_data else "No data")
            
        else:
            st.info("Mask is ready for export. Click 'Download' in the sidebar.")
        
        # Download button
        if st.session_state.export_ready:
            # Generate download based on format
            if export_format == "coco":
                coco_data = to_coco_format([mask], image_ids=[1], image_shapes=[mask.shape])
                json_str = json.dumps(coco_data, indent=2)
                st.download_button(
                    "📥 Download COCO JSON",
                    json_str,
                    file_name="annotation_coco.json",
                    mime="application/json"
                )
            
            elif export_format == "yolo":
                yolo_data = to_yolo_format([mask], [mask.shape])
                txt_str = "\n".join(yolo_data)
                st.download_button(
                    "📥 Download YOLO TXT",
                    txt_str,
                    file_name="annotation_yolo.txt",
                    mime="text/plain"
                )
            
            elif export_format == "png":
                import io
                mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                buf = io.BytesIO()
                mask_img.save(buf, format="PNG")
                st.download_button(
                    "📥 Download PNG Mask",
                    buf.getvalue(),
                    file_name="mask.png",
                    mime="image/png"
                )
            
            elif export_format == "monai":
                monai_data = to_monai_format([mask], image_ids=[1])
                json_str = json.dumps(monai_data, indent=2)
                st.download_button(
                    "📥 Download MONAI JSON",
                    json_str,
                    file_name="annotation_monai.json",
                    mime="application/json"
                )
            
            # Reset export flag
            st.session_state.export_ready = False
            st.rerun()
    
    else:
        st.info("🔄 No mask generated yet. Add clicks and click 'Generate Mask'.")

# =============================================================================
# BOTTOM - TABS FOR ADDITIONAL FEATURES
# =============================================================================

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📊 Data Info", "🧬 Physics Features", "📋 History"])

with tab1:
    if st.session_state.current_image is not None:
        st.write(f"**Image shape:** {st.session_state.current_image.shape}")
        st.write(f"**Data type:** {st.session_state.current_image.dtype}")
        if st.session_state.frequencies is not None:
            st.write(f"**Frequency range:** {st.session_state.frequencies[0]:.2f} - {st.session_state.frequencies[-1]:.2f} GHz")
            st.write(f"**Number of frequency points:** {len(st.session_state.frequencies)}")
        
        if st.session_state.raw_data is not None:
            st.write(f"**Raw data paths:** {list(st.session_state.raw_data.keys())}")
    else:
        st.info("No data loaded yet.")

with tab2:
    if st.session_state.raw_data is not None and st.session_state.foreground_clicks:
        st.write("Physical features extracted at click positions:")
        
        from hypoxify_annotate.segment.physics_guided import extract_physical_features_at_clicks
        
        features = extract_physical_features_at_clicks(
            np.array(list(st.session_state.raw_data.values()))[0],
            st.session_state.foreground_clicks
        )
        
        feature_names = [
            "Dielectric Contrast",
            "Acoustic Pressure",
            "SNR",
            "Energy",
            "Local Variance",
            "Peak/Average",
        ]
        
        df = pd.DataFrame(features, columns=feature_names)
        st.dataframe(df)
    else:
        st.info("Load raw data and add clicks to see physical features.")

with tab3:
    st.write(f"**Masks generated:** {len(st.session_state.masks_history)}")
    if st.session_state.masks_history:
        for i, mask in enumerate(st.session_state.masks_history[-5:]):
            st.caption(f"Mask {i+1}: {np.sum(mask)} pixels")
    else:
        st.info("No masks generated yet.")

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.caption(
    "🔬 Hypoxify Annotation Suite v0.1.0 | "
    "Physics-informed segmentation for biomedical imaging | "
    "Data stored locally; no data uploaded to servers."
)
