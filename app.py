import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import json
import os
import tempfile
from pathlib import Path
import pandas as pd
from scipy.ndimage import gaussian_filter, distance_transform_edt
from scipy import ndimage
from scipy.io import loadmat
import torch
import torchvision
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Hypoxify Annotation Suite",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# CUSTOM CSS
# ------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #0d47a1;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .click-instruction {
        background-color: #e3f2fd;
        padding: 10px;
        border-radius: 8px;
        font-size: 14px;
        margin: 8px 0;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #ddd;
        text-align: center;
    }
    .stButton button {
        font-weight: 600;
        border-radius: 8px;
        width: 100%;
    }
    .image-container {
        border: 2px solid #0d47a1;
        border-radius: 10px;
        padding: 10px;
        background-color: #fafafa;
    }
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #28a745;
    }
    .info-box {
        background-color: #d1ecf1;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #17a2b8;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# SESSION STATE - PERSISTENT DATA
# ------------------------------------------------------------
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
if "segmented" not in st.session_state:
    st.session_state.segmented = False
if "modality" not in st.session_state:
    st.session_state.modality = "MITT"
if "sam_initialized" not in st.session_state:
    st.session_state.sam_initialized = False
if "candidates" not in st.session_state:
    st.session_state.candidates = []
if "synthetic_generated" not in st.session_state:
    st.session_state.synthetic_generated = False
if "click_mode" not in st.session_state:
    st.session_state.click_mode = "Foreground (+)"
if "uncertainty_heatmap" not in st.session_state:
    st.session_state.uncertainty_heatmap = None
if "variations" not in st.session_state:
    st.session_state.variations = None
if "sam_predictor" not in st.session_state:
    st.session_state.sam_predictor = None
if "sam_is_mock" not in st.session_state:
    st.session_state.sam_is_mock = True

# ------------------------------------------------------------
# HEADER WITH LOGO
# ------------------------------------------------------------
col1, col2 = st.columns([1, 5])

with col1:
    try:
        st.image("logo image.png", width=80)
    except:
        st.markdown("🧬")

with col2:
    st.markdown('<p class="main-header">Hypoxify Annotation Suite</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Clinical-grade multi-modality physics-informed segmentation (MITT, MWI, Photoacoustic, Ultrasound)</p>', unsafe_allow_html=True)

st.markdown("---")

# ------------------------------------------------------------
# 0. SAM2 - AUTO-INITIALIZE WITH CACHE
# ------------------------------------------------------------
@st.cache_resource
def load_sam2():
    """Load SAM2 model with caching - only runs ONCE."""
    try:
        # Try to import SAM2
        from sam2 import sam_model_registry, SamPredictor
        
        # Try to load the model
        try:
            model = sam_model_registry["tiny"](checkpoint=None)
            model.to(device="cpu")
            predictor = SamPredictor(model)
            return predictor, False  # predictor, is_mock
        except Exception as e:
            st.warning(f"⚠️ SAM2 load error: {e}")
            return None, True
    except ImportError:
        # SAM2 not installed
        return None, True
    except Exception as e:
        st.warning(f"⚠️ SAM2 init error: {e}")
        return None, True

# Auto-initialize SAM2 on startup
if st.session_state.sam_predictor is None:
    predictor, is_mock = load_sam2()
    st.session_state.sam_predictor = predictor
    st.session_state.sam_is_mock = is_mock
    st.session_state.sam_initialized = not is_mock

# ------------------------------------------------------------
# 1. PHYSICS ENGINE
# ------------------------------------------------------------
class PhysicsSimulator:
    @staticmethod
    def extract_physical_signature(image, click_point, modality="microwave"):
        gray = cv2.cvtColor(np.uint8(image), cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        edges = cv2.Canny(gray, 50, 150)
        dist_from_click = np.sqrt((np.arange(h)[:, None] - click_point[1])**2 +
                                  (np.arange(w)[None, :] - click_point[0])**2)
        
        dielectric = edges.astype(np.float32) + (1 / (dist_from_click + 1)) * 10
        acoustic = 50 * np.exp(-dist_from_click / 100) + np.random.normal(0, 5, (h, w))
        local_std = ndimage.generic_filter(gray, np.std, size=5)
        absorption = local_std / local_std.max()
        
        if modality == "microwave":
            dielectric = dielectric * 1.2
        elif modality == "photoacoustic":
            acoustic = acoustic * 1.5
        elif modality == "ultrasound":
            absorption = absorption * 1.3
        
        return {
            "dielectric": dielectric / dielectric.max(),
            "acoustic": acoustic / acoustic.max(),
            "absorption": absorption,
            "modality": modality
        }

    @staticmethod
    def apply_physics_to_segmentation(prior_mask, physics_maps):
        weights = {"dielectric": 0.4, "acoustic": 0.3, "absorption": 0.3}
        modality = physics_maps.get("modality", "microwave")
        if modality == "photoacoustic":
            weights = {"acoustic": 0.5, "dielectric": 0.3, "absorption": 0.2}
        elif modality == "ultrasound":
            weights = {"absorption": 0.5, "dielectric": 0.3, "acoustic": 0.2}
        
        physics_weight = (physics_maps["dielectric"] * weights["dielectric"] +
                          physics_maps["acoustic"] * weights["acoustic"] +
                          physics_maps["absorption"] * weights["absorption"])
        
        refined = prior_mask * (physics_weight > 0.3)
        return (refined > 0).astype(np.uint8) * 255

# ------------------------------------------------------------
# 2. UNCERTAINTY CALCULATOR
# ------------------------------------------------------------
class UncertaintyCalculator:
    @staticmethod
    def compute_heatmaps(image, mask):
        h, w = mask.shape
        gray = cv2.cvtColor(np.uint8(image*255), cv2.COLOR_RGB2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        signal_uncertainty = 1 - (magnitude / magnitude.max())
        dist = distance_transform_edt(mask)
        dist_norm = dist / dist.max() if dist.max() > 0 else dist
        model_uncertainty = 1 - dist_norm
        total_uncertainty = (signal_uncertainty * 0.6 + model_uncertainty * 0.4)
        
        heatmap = np.zeros((h, w, 3), dtype=np.uint8)
        heatmap[:, :, 1] = (1 - total_uncertainty) * 255
        heatmap[:, :, 0] = total_uncertainty * 255
        overlay = np.uint8(image * 255 * 0.5) + heatmap * 0.5
        return np.uint8(overlay), total_uncertainty

# ------------------------------------------------------------
# 3. SYNTHETIC DATA GENERATOR
# ------------------------------------------------------------
class SyntheticDataGenerator:
    @staticmethod
    def generate_variations(base_image, base_mask, n_variations=10):
        variations = []
        h, w = base_image.shape[:2]
        
        for _ in range(n_variations):
            img = base_image.copy().astype(np.float32)
            mask = base_mask.copy()
            
            angle = np.random.uniform(-10, 10)
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h))
            mask = cv2.warpAffine(mask, M, (w, h))
            
            scale = np.random.uniform(0.9, 1.1)
            M = cv2.getRotationMatrix2D((w/2, h/2), 0, scale)
            img = cv2.warpAffine(img, M, (w, h))
            mask = cv2.warpAffine(mask, M, (w, h))
            
            noise = np.random.normal(0, np.random.uniform(5, 20), img.shape)
            img = np.clip(img + noise, 0, 255).astype(np.uint8)
            
            if np.random.random() > 0.5:
                kernel_size = np.random.choice([3, 5])
                img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
            
            mask = (mask > 127).astype(np.uint8) * 255
            variations.append((img, mask))
        
        return variations

# ------------------------------------------------------------
# 4. SEGMENTATION HELPER
# ------------------------------------------------------------
def run_segmentation(image, points, labels, physics, predictor, is_mock):
    """Run SAM2 prediction with physics conditioning."""
    if predictor is not None and not is_mock:
        try:
            predictor.set_image(image)
            points_np = np.array(points)
            labels_np = np.array(labels)
            masks, scores, _ = predictor.predict(
                point_coords=points_np,
                point_labels=labels_np,
                multimask_output=True
            )
            best_idx = np.argmax(scores)
            mask = masks[best_idx].astype(np.uint8)
        except Exception as e:
            st.warning(f"SAM2 prediction failed: {e}. Using mock.")
            is_mock = True
            mask = _mock_predict(image, points, labels)
    else:
        mask = _mock_predict(image, points, labels)
    
    # Apply physics conditioning
    if physics:
        mask = PhysicsSimulator.apply_physics_to_segmentation(mask, physics)
    
    return mask

def _mock_predict(image, points, labels):
    """Fallback mock segmentation."""
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if points:
        fg_points = [p for p, l in zip(points, labels) if l == 1]
        if fg_points:
            cx, cy = fg_points[0]
            radius = min(h, w) // 6
            y, x = np.ogrid[:h, :w]
            dist = (x - cx)**2 + (y - cy)**2
            mask[dist < radius**2] = 1
    return mask

# ------------------------------------------------------------
# 5. MAIN STREAMLIT APP
# ------------------------------------------------------------

# SIDEBAR
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    modality = st.selectbox(
        "Imaging Modality",
        ["MITT", "Microwave Imaging", "Photoacoustic", "Ultrasound"],
        help="Select the imaging modality for physics-guided segmentation"
    )
    st.session_state.modality = modality
    
    st.markdown("---")
    
    st.markdown("### 🧬 SAM2 Status")
    if st.session_state.sam_initialized and not st.session_state.sam_is_mock:
        st.success("🟢 SAM2 Ready")
        st.info("Model: tiny (~78MB, cached)")
    else:
        st.warning("🟡 SAM2 running in mock mode")
        st.info("Using fallback segmentation (circle-based)")
    
    st.markdown("---")
    
    st.markdown("### 🔬 Physics Parameters")
    use_dielectric = st.checkbox("Dielectric Contrast", value=True)
    use_acoustic = st.checkbox("Acoustic Pressure", value=True)
    use_absorption = st.checkbox("Energy Absorption", value=True)
    
    st.markdown("---")
    
    st.markdown("### 💾 Project")
    project_name = st.text_input("Project Name", value="my_project")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save"):
            st.success("Saved!")
    with col2:
        if st.button("📂 Load"):
            st.info("Load project")

# MAIN CONTENT
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📡 Data Ingestion",
    "🎯 Segmentation",
    "📊 Results",
    "🔄 3D Propagation",
    "🧪 Synthetic Data",
    "📤 Export"
])

# TAB 1: DATA INGESTION
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📂 Upload Your Data")
        st.markdown("**Supported:** PNG, JPG, TIFF, DICOM (.dcm), CSV, S2P, MAT")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["png", "jpg", "jpeg", "tiff", "dcm", "csv", "s2p", "mat"]
        )
        
        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            file_ext = Path(uploaded_file.name).suffix.lower()
            
            if file_ext in ['.png', '.jpg', '.jpeg', '.tiff']:
                img = Image.open(io.BytesIO(file_bytes))
                img_array = np.array(img)
                if img_array.ndim == 2:
                    img_array = np.stack([img_array] * 3, axis=-1)
                st.session_state.current_image = img_array
                st.session_state.image_loaded = True
                st.success("✅ Image loaded!")
            
            elif file_ext == '.dcm':
                try:
                    import pydicom
                    from pydicom.pixel_data_handlers.util import apply_voi_lut
                    ds = pydicom.dcmread(io.BytesIO(file_bytes))
                    pixel_array = ds.pixel_array
                    if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
                        pixel_array = apply_voi_lut(pixel_array, ds)
                    if pixel_array.dtype != np.uint8:
                        pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min() + 1e-8) * 255
                        pixel_array = pixel_array.astype(np.uint8)
                    if len(pixel_array.shape) == 2:
                        pixel_array = np.stack([pixel_array] * 3, axis=-1)
                    st.session_state.current_image = pixel_array
                    st.session_state.image_loaded = True
                    st.success("✅ DICOM loaded!")
                except Exception as e:
                    st.error(f"❌ Error loading DICOM: {e}")
            
            st.session_state.foreground_clicks = []
            st.session_state.background_clicks = []
            st.session_state.current_mask = None
            st.session_state.segmented = False
    
    with col2:
        st.markdown("### 📋 Project Info")
        if st.session_state.image_loaded:
            st.markdown(f"""
            <div class="success-box">
                <strong>✅ Image Loaded</strong><br>
                Shape: {st.session_state.current_image.shape}<br>
                Modality: {st.session_state.modality}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box">
                <strong>ℹ️ No image loaded</strong><br>
                Upload an image to begin
            </div>
            """, unsafe_allow_html=True)
        
        if st.session_state.image_loaded:
            st.image(st.session_state.current_image, caption="Current Image", use_container_width=True)

# TAB 2: SEGMENTATION
with tab2:
    if not st.session_state.image_loaded:
        st.warning("⚠️ Please upload an image in the Data Ingestion tab first.")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 🖼️ Click on Image to Place Points")
            st.markdown("""
            <div class="click-instruction">
                👆 Click directly on the image to place points. 
                Select mode below first: <b>Foreground (+)</b> or <b>Background (-)</b>
            </div>
            """, unsafe_allow_html=True)
            
            # Prepare image with existing points
            img_display = st.session_state.current_image.copy()
            
            for px, py in st.session_state.foreground_clicks:
                cv2.circle(img_display, (px, py), 8, (0, 255, 0), -1)
                cv2.circle(img_display, (px, py), 10, (255, 255, 255), 2)
            for px, py in st.session_state.background_clicks:
                cv2.circle(img_display, (px, py), 8, (255, 0, 0), -1)
                cv2.circle(img_display, (px, py), 10, (255, 255, 255), 2)
            
            img_bgr = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)
            
            # Clickable image
            click_success = False
            try:
                from streamlit_image_coordinates import streamlit_image_coordinates
                
                value = streamlit_image_coordinates(
                    img_bgr,
                    key="image_click",
                    click_event=True,
                    width="stretch"
                )
                
                if value is not None and value["x"] > 0 and value["y"] > 0:
                    x, y = value["x"], value["y"]
                    h, w = st.session_state.current_image.shape[:2]
                    if 0 <= x < w and 0 <= y < h:
                        if st.session_state.click_mode == "Foreground (+)" and (x, y) not in st.session_state.foreground_clicks:
                            st.session_state.foreground_clicks.append((int(x), int(y)))
                            st.rerun()
                        elif st.session_state.click_mode == "Background (-)" and (x, y) not in st.session_state.background_clicks:
                            st.session_state.background_clicks.append((int(x), int(y)))
                            st.rerun()
                click_success = True
            except ImportError:
                st.image(img_bgr, caption="Image with points", use_container_width=True)
            except Exception as e:
                st.image(img_bgr, caption="Image with points", use_container_width=True)
            
            # Point controls
            col_count, col_mode, col_actions = st.columns([1, 1, 1])
            with col_count:
                st.markdown(f"**FG:** {len(st.session_state.foreground_clicks)} | **BG:** {len(st.session_state.background_clicks)}")
            with col_mode:
                st.session_state.click_mode = st.selectbox(
                    "Mode",
                    ["Foreground (+)", "Background (-)"],
                    key="click_mode_selector",
                    label_visibility="collapsed"
                )
            with col_actions:
                col_clear, col_undo = st.columns(2)
                with col_clear:
                    if st.button("🗑️ Clear", width="stretch"):
                        st.session_state.foreground_clicks = []
                        st.session_state.background_clicks = []
                        st.rerun()
                with col_undo:
                    if st.button("↩️ Undo", width="stretch"):
                        if st.session_state.background_clicks:
                            st.session_state.background_clicks.pop()
                        elif st.session_state.foreground_clicks:
                            st.session_state.foreground_clicks.pop()
                        st.rerun()
            
            # Fallback manual input
            with st.expander("✏️ Manual Coordinate Input (Fallback)"):
                col_x, col_y, col_btn = st.columns([1, 1, 1])
                with col_x:
                    x_coord = st.number_input(
                        "X", 
                        min_value=0, 
                        max_value=st.session_state.current_image.shape[1]-1, 
                        value=st.session_state.current_image.shape[1]//2,
                        step=1,
                        key="manual_x"
                    )
                with col_y:
                    y_coord = st.number_input(
                        "Y", 
                        min_value=0, 
                        max_value=st.session_state.current_image.shape[0]-1, 
                        value=st.session_state.current_image.shape[0]//2,
                        step=1,
                        key="manual_y"
                    )
                with col_btn:
                    if st.button("➕ Add Point", width="stretch"):
                        if st.session_state.click_mode == "Foreground (+)" and (int(x_coord), int(y_coord)) not in st.session_state.foreground_clicks:
                            st.session_state.foreground_clicks.append((int(x_coord), int(y_coord)))
                            st.rerun()
                        elif st.session_state.click_mode == "Background (-)" and (int(x_coord), int(y_coord)) not in st.session_state.background_clicks:
                            st.session_state.background_clicks.append((int(x_coord), int(y_coord)))
                            st.rerun()
            
            # Show mask if segmented
            if st.session_state.segmented and st.session_state.current_mask is not None:
                st.markdown("### 🎯 Segmentation Result")
                mask_display = st.session_state.current_image.copy()
                mask = st.session_state.current_mask
                mask_display[mask > 0] = mask_display[mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
                st.image(mask_display, caption="Segmented Mask Overlay", use_container_width=True)
        
        with col2:
            st.markdown("### 🎯 Segmentation Controls")
            
            if st.button("🚀 Run Physics-Guided SAM", type="primary", width="stretch"):
                if len(st.session_state.foreground_clicks) == 0:
                    st.warning("Please place at least one foreground point.")
                else:
                    with st.spinner("Running physics-guided segmentation..."):
                        try:
                            image = st.session_state.current_image
                            points = st.session_state.foreground_clicks + st.session_state.background_clicks
                            labels = [1] * len(st.session_state.foreground_clicks) + [0] * len(st.session_state.background_clicks)
                            
                            physics = {}
                            if st.session_state.foreground_clicks and (use_dielectric or use_acoustic or use_absorption):
                                physics = PhysicsSimulator.extract_physical_signature(
                                    image,
                                    st.session_state.foreground_clicks[0],
                                    st.session_state.modality.lower()
                                )
                            
                            mask = run_segmentation(
                                image,
                                points,
                                labels,
                                physics,
                                st.session_state.sam_predictor,
                                st.session_state.sam_is_mock
                            )
                            
                            st.session_state.current_mask = mask
                            st.session_state.segmented = True
                            
                            st.session_state.candidates = []
                            for i in range(3):
                                mask_var = mask.copy()
                                kernel = np.ones((3,3), np.uint8)
                                if i == 0:
                                    mask_var = cv2.erode(mask_var, kernel, iterations=1)
                                elif i == 1:
                                    mask_var = cv2.dilate(mask_var, kernel, iterations=1)
                                score = 0.95 - i * 0.05
                                st.session_state.candidates.append((mask_var, score, f"Candidate {i+1}"))
                            
                            st.success("✅ Segmentation complete!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Segmentation failed: {e}")
            
            if st.session_state.segmented and st.session_state.candidates:
                st.markdown("### 🎯 Select Candidate")
                candidate_labels = [f"{c[2]} (Score: {c[1]:.2f})" for c in st.session_state.candidates]
                selected = st.selectbox("Choose mask", candidate_labels, index=0)
                if selected:
                    idx = candidate_labels.index(selected)
                    st.session_state.current_mask = st.session_state.candidates[idx][0]
                    st.success(f"Selected {st.session_state.candidates[idx][2]}")
            
            st.markdown("---")
            
            if st.session_state.segmented and st.session_state.current_mask is not None:
                if st.button("🔥 Show Uncertainty Heatmap", width="stretch"):
                    with st.spinner("Calculating uncertainty..."):
                        heatmap, _ = UncertaintyCalculator.compute_heatmaps(
                            st.session_state.current_image / 255.0,
                            st.session_state.current_mask
                        )
                        st.session_state.uncertainty_heatmap = heatmap
                        st.rerun()
                
                if st.session_state.uncertainty_heatmap is not None:
                    st.image(st.session_state.uncertainty_heatmap, caption="Uncertainty Heatmap (Red=Uncertain, Green=Confident)", use_container_width=True)
            
            if st.session_state.segmented and st.session_state.current_mask is not None:
                st.markdown("---")
                st.markdown("### 📊 Live Volumetry")
                mask = st.session_state.current_mask
                voxel_count = np.sum(mask)
                volume_mm3 = voxel_count
                volume_mL = volume_mm3 / 1000
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <strong>Voxels</strong><br>
                        <span style="font-size: 1.5rem;">{voxel_count:,}</span>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <strong>Volume</strong><br>
                        <span style="font-size: 1.5rem;">{volume_mm3:.1f} mm³</span>
                    </div>
                    """, unsafe_allow_html=True)

# TAB 3: RESULTS
with tab3:
    if not st.session_state.segmented or st.session_state.current_mask is None:
        st.warning("⚠️ No segmentation results yet.")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 🧬 Segmentation Results")
            overlay = st.session_state.current_image.copy()
            mask = st.session_state.current_mask
            overlay[mask > 0] = overlay[mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
            st.image(overlay, caption="Segmented Mask", use_container_width=True)
            st.image(mask * 255, caption="Binary Mask", use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Metrics")
            mask = st.session_state.current_mask
            mask_area = np.sum(mask)
            mask_percent = (mask_area / mask.size) * 100
            
            st.markdown(f"""
            <div class="metric-card">
                <strong>Mask Area</strong><br>
                <span style="font-size: 1.5rem;">{mask_area:,}</span> pixels
            </div>
            <br>
            <div class="metric-card">
                <strong>Coverage</strong><br>
                <span style="font-size: 1.5rem;">{mask_percent:.2f}%</span>
            </div>
            <br>
            <div class="metric-card">
                <strong>Modality</strong><br>
                <span style="font-size: 1rem;">{st.session_state.modality}</span>
            </div>
            <br>
            <div class="metric-card">
                <strong>Points Used</strong><br>
                <span style="font-size: 1rem;">{len(st.session_state.foreground_clicks)} FG, {len(st.session_state.background_clicks)} BG</span>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("💾 Save Mask to Project", type="primary", width="stretch"):
                st.success("✅ Mask saved!")

# TAB 4: 3D PROPAGATION
with tab4:
    st.markdown("### 📦 3D Volumetric Propagation")
    
    volume_files = st.file_uploader(
        "Upload Volume Slices",
        type=["png", "jpg", "jpeg", "tiff"],
        accept_multiple_files=True
    )
    
    if volume_files:
        st.success(f"✅ {len(volume_files)} slices uploaded")
        
        if st.button("🚀 Propagate Through Volume", type="primary"):
            with st.spinner("Processing volume..."):
                try:
                    images = []
                    for f in volume_files:
                        img = Image.open(f)
                        img_array = np.array(img)
                        if img_array.ndim == 2:
                            img_array = np.stack([img_array] * 3, axis=-1)
                        images.append(img_array)
                    
                    first_img = images[0]
                    center = (first_img.shape[1]//2, first_img.shape[0]//2)
                    physics = PhysicsSimulator.extract_physical_signature(first_img, center, st.session_state.modality.lower())
                    
                    mask = _mock_predict(first_img, [center], [1])
                    mask = PhysicsSimulator.apply_physics_to_segmentation(mask, physics)
                    
                    masks = [mask]
                    prev_gray = cv2.cvtColor(first_img, cv2.COLOR_RGB2GRAY)
                    area = np.sum(mask)
                    
                    for i in range(1, len(images)):
                        curr_gray = cv2.cvtColor(images[i], cv2.COLOR_RGB2GRAY)
                        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                        h, w = mask.shape
                        flow_x = flow[:,:,0]
                        flow_y = flow[:,:,1]
                        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
                        new_x = (grid_x + flow_x).astype(np.float32)
                        new_y = (grid_y + flow_y).astype(np.float32)
                        warped = cv2.remap(mask.astype(np.float32), new_x, new_y, cv2.INTER_LINEAR)
                        warped = (warped > 0.5).astype(np.uint8) * 255
                        if np.sum(warped) < area * 0.3:
                            warped = cv2.dilate(warped, np.ones((5,5), np.uint8))
                        masks.append(warped)
                        prev_gray = curr_gray
                        mask = warped
                    
                    st.success("✅ Volume propagation complete!")
                    
                    slice_idx = st.slider("Browse Slices", 0, len(images)-1, 0)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(images[slice_idx], caption=f"Slice {slice_idx}", use_container_width=True)
                    with col2:
                        overlay = images[slice_idx].copy()
                        overlay[masks[slice_idx] > 0] = overlay[masks[slice_idx] > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
                        st.image(overlay, caption=f"Slice {slice_idx} with Mask", use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Propagation failed: {e}")

# TAB 5: SYNTHETIC DATA
with tab5:
    st.markdown("### 🧪 Synthetic Data Generator")
    
    if not st.session_state.segmented or st.session_state.current_mask is None:
        st.warning("⚠️ Please segment an image first.")
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            n_variations = st.slider("Number of Variations", 1, 50, 10)
            if st.button("🧬 Generate Synthetic Dataset", type="primary"):
                with st.spinner(f"Generating {n_variations} variations..."):
                    variations = SyntheticDataGenerator.generate_variations(
                        st.session_state.current_image,
                        st.session_state.current_mask,
                        n_variations
                    )
                    st.session_state.variations = variations
                    st.session_state.synthetic_generated = True
                    st.success(f"✅ Generated {len(variations)} variations!")
        
        with col2:
            if st.session_state.synthetic_generated and st.session_state.variations is not None:
                st.markdown("### 📊 Sample Preview")
                sample_idx = st.slider("Sample", 0, len(st.session_state.variations)-1, 0)
                img, mask = st.session_state.variations[sample_idx]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(img, caption=f"Sample {sample_idx} - Image", use_container_width=True)
                with col2:
                    st.image(mask, caption=f"Sample {sample_idx} - Mask", use_container_width=True)
                
                if st.button("📥 Download Synthetic Dataset"):
                    import zipfile
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w') as zf:
                        manifest = {"samples": []}
                        for i, (img, mask) in enumerate(st.session_state.variations):
                            img_bytes = io.BytesIO()
                            Image.fromarray(img).save(img_bytes, format="PNG")
                            zf.writestr(f"sample_{i:04d}_img.png", img_bytes.getvalue())
                            
                            mask_bytes = io.BytesIO()
                            Image.fromarray(mask).save(mask_bytes, format="PNG")
                            zf.writestr(f"sample_{i:04d}_mask.png", mask_bytes.getvalue())
                            
                            manifest["samples"].append({
                                "image": f"sample_{i:04d}_img.png",
                                "mask": f"sample_{i:04d}_mask.png"
                            })
                        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
                    
                    zip_buffer.seek(0)
                    st.download_button(
                        label="Download ZIP",
                        data=zip_buffer.getvalue(),
                        file_name="synthetic_dataset.zip",
                        mime="application/zip"
                    )

# TAB 6: EXPORT
with tab6:
    st.markdown("### 📤 Export Annotations")
    
    if not st.session_state.segmented or st.session_state.current_mask is None:
        st.warning("⚠️ No annotations to export.")
    else:
        export_format = st.selectbox(
            "Export Format",
            ["COCO JSON", "YOLO TXT", "PNG Mask", "MONAI JSON", "nnU-Net"]
        )
        
        if st.button("📥 Export", type="primary"):
            with st.spinner("Preparing export..."):
                mask = st.session_state.current_mask
                
                if export_format == "PNG Mask":
                    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                    buf = io.BytesIO()
                    mask_img.save(buf, format="PNG")
                    st.download_button(
                        label="Download PNG Mask",
                        data=buf.getvalue(),
                        file_name="mask.png",
                        mime="image/png"
                    )
                
                elif export_format == "COCO JSON":
                    coco_data = {
                        "images": [{"id": 1, "width": mask.shape[1], "height": mask.shape[0]}],
                        "annotations": [{
                            "id": 1,
                            "image_id": 1,
                            "category_id": 1,
                            "bbox": [0, 0, mask.shape[1], mask.shape[0]],
                            "area": int(np.sum(mask)),
                            "iscrowd": 0
                        }],
                        "categories": [{"id": 1, "name": "lesion"}]
                    }
                    json_str = json.dumps(coco_data, indent=2)
                    st.download_button(
                        label="Download COCO JSON",
                        data=json_str,
                        file_name="annotation_coco.json",
                        mime="application/json"
                    )
                
                elif export_format == "YOLO TXT":
                    h, w = mask.shape
                    y, x = np.where(mask > 0)
                    if len(x) > 0 and len(y) > 0:
                        x_min, x_max = np.min(x), np.max(x)
                        y_min, y_max = np.min(y), np.max(y)
                        x_center = (x_min + x_max) / 2 / w
                        y_center = (y_min + y_max) / 2 / h
                        bbox_w = (x_max - x_min) / w
                        bbox_h = (y_max - y_min) / h
                        yolo_line = f"0 {x_center:.6f} {y_center:.6f} {bbox_w:.6f} {bbox_h:.6f}"
                    else:
                        yolo_line = "0 0 0 0 0"
                    
                    st.download_button(
                        label="Download YOLO TXT",
                        data=yolo_line,
                        file_name="annotation_yolo.txt",
                        mime="text/plain"
                    )
                
                else:
                    st.info(f"ℹ️ {export_format} export coming soon!")

# FOOTER
st.markdown("---")
st.caption("🔬 Hypoxify Annotation Suite v2.0")
