"""
Hypoxify Annotation Suite - Standalone App
Clean version with image display + manual X/Y coordinate input
"""

import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
from pathlib import Path
import tempfile
import json
import io
import re
from typing import Dict, List, Optional, Tuple
from scipy.ndimage import gaussian_filter

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Hypoxify Annotation Suite",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CUSTOM CSS
# =============================================================================

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
    .click-count {
        font-weight: bold;
        color: #0d47a1;
    }
    .coord-input {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 8px;
        margin: 8px 0;
        border: 1px solid #ddd;
    }
    .stButton button {
        font-weight: 600;
        border-radius: 8px;
    }
    .image-container {
        border: 2px solid #0d47a1;
        border-radius: 10px;
        padding: 10px;
        background-color: #fafafa;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONSTANTS
# =============================================================================

MICROWAVE_CONFIG = {
    "start_freq_ghz": 2.0,
    "stop_freq_ghz": 3.0,
    "num_points": 201,
    "antenna_positions": {
        1: (-75, 0),
        2: (75, 0),
        3: (0, -75),
        4: (0, 75),
    },
    "path_to_antenna_pair": {
        1: (1, 3),
        2: (1, 4),
        3: (2, 3),
        4: (2, 4),
    },
}

SUPPORTED_CONFIGURATIONS = {
    "microwave_4_antenna": {
        "modality": "microwave",
        "description": "4-antenna microwave imaging system (S21)",
        "num_antennas": 4,
        "file_formats": [".csv", ".s2p", ".mat"],
    },
    "microwave_8_antenna": {
        "modality": "microwave",
        "description": "8-antenna microwave imaging system",
        "num_antennas": 8,
        "file_formats": [".csv", ".s2p", ".mat"],
    },
    "thermoacoustic_ring": {
        "modality": "thermoacoustic",
        "description": "Ring-array thermoacoustic system",
        "num_detectors": 128,
        "file_formats": [".h5", ".mat"],
    },
    "mri": {
        "modality": "mri",
        "description": "MRI DICOM/NIfTI",
        "file_formats": [".dcm", ".nii", ".nii.gz"],
    },
    "ct": {
        "modality": "ct",
        "description": "CT DICOM/NIfTI",
        "file_formats": [".dcm", ".nii", ".nii.gz"],
    },
    "histology": {
        "modality": "histology",
        "description": "Histology slides",
        "file_formats": [".svs", ".ndpi", ".tiff", ".png", ".jpg"],
    },
}

EXPORT_FORMATS = {
    "coco": {"extension": ".json", "description": "COCO format"},
    "yolo": {"extension": ".txt", "description": "YOLO format"},
    "monai": {"extension": ".json", "description": "MONAI format"},
    "png": {"extension": ".png", "description": "PNG mask"},
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def db_to_linear(db):
    """Convert dB to linear magnitude."""
    return 10 ** (np.asarray(db) / 10)

def linear_to_db(linear):
    """Convert linear magnitude to dB."""
    linear = np.maximum(np.asarray(linear), 1e-12)
    return 10 * np.log10(linear)

def delay_and_sum_reconstruction(
    s21_data: Dict[int, np.ndarray],
    frequencies: np.ndarray,
    baseline_data: Optional[Dict[int, np.ndarray]] = None,
    grid_size: int = 80,
    grid_extent: float = 100.0,
    start_freq: float = 2.0,
    stop_freq: float = 3.0,
    num_points: int = 201,
    sigma: float = 2.0
) -> np.ndarray:
    """
    Reconstruct image using delay-and-sum beamforming.
    This is the EXACT algorithm from your Pi code.
    """
    antenna_positions = {
        1: (-75, 0),
        2: (75, 0),
        3: (0, -75),
        4: (0, 75),
    }
    
    path_to_antenna_pair = {
        1: (1, 3),
        2: (1, 4),
        3: (2, 3),
        4: (2, 4),
    }
    
    x_grid = np.linspace(-grid_extent, grid_extent, grid_size)
    y_grid = np.linspace(-grid_extent, grid_extent, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    image = np.zeros((grid_size, grid_size))
    c = 3e8
    
    START_FREQ_HZ = start_freq * 1e9
    STOP_FREQ_HZ = stop_freq * 1e9
    
    num_paths_used = 0
    
    for path_num, s21_db in s21_data.items():
        if path_num not in path_to_antenna_pair:
            continue
        
        tx_ant, rx_ant = path_to_antenna_pair[path_num]
        tx_pos = antenna_positions[tx_ant]
        rx_pos = antenna_positions[rx_ant]
        
        s21_linear = db_to_linear(s21_db)
        
        if baseline_data and path_num in baseline_data:
            baseline_linear = db_to_linear(baseline_data[path_num])
            s21_linear = s21_linear - baseline_linear
            s21_linear = np.maximum(s21_linear, 1e-12)
        
        for i in range(grid_size):
            for j in range(grid_size):
                point = (X[i, j], Y[i, j])
                
                d_tx = np.sqrt((tx_pos[0] - point[0])**2 + (tx_pos[1] - point[1])**2)
                d_rx = np.sqrt((rx_pos[0] - point[0])**2 + (rx_pos[1] - point[1])**2)
                total_dist = (d_tx + d_rx) / 1000
                
                delay = total_dist / c
                
                freq_idx = int(np.clip(delay * 1e9 / (STOP_FREQ_HZ / 1e9) * num_points, 0, num_points - 1))
                freq_idx = min(freq_idx, len(s21_linear) - 1)
                freq_idx = max(freq_idx, 0)
                
                image[i, j] += s21_linear[freq_idx]
        
        num_paths_used += 1
    
    if num_paths_used > 0:
        image /= num_paths_used
    else:
        raise ValueError("No valid paths found in s21_data")
    
    image = gaussian_filter(image, sigma=sigma)
    
    if image.max() > 0:
        image = np.clip(image, 0, np.percentile(image, 95))
        image = (image / image.max()) * 255
    
    return image.astype(np.uint8)

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_s21_csv(filepath):
    """Load S21 data from CSV format."""
    path = Path(filepath)
    df = pd.read_csv(path)
    
    freq_col = None
    for col in df.columns:
        if 'freq' in col.lower() or 'ghz' in col.lower():
            freq_col = col
            break
    if freq_col is None:
        raise ValueError(f"Frequency column not found. Available: {df.columns.tolist()}")
    
    s21_col = None
    for col in df.columns:
        if 's21' in col.lower() or 's_param' in col.lower():
            s21_col = col
            break
    if s21_col is None:
        raise ValueError(f"S21 column not found. Available: {df.columns.tolist()}")
    
    frequencies = df[freq_col].values.astype(np.float64)
    s21_db = df[s21_col].values.astype(np.float64)
    
    return frequencies, s21_db

def load_s2p(filepath):
    """Load S2P (Touchstone) file format."""
    path = Path(filepath)
    frequencies = []
    s21_mag_linear = []
    
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('!') or line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    freq_mhz = float(parts[0])
                    freq_ghz = freq_mhz / 1000.0
                    real = float(parts[1])
                    imag = float(parts[2])
                    mag = np.sqrt(real**2 + imag**2)
                    s21_mag_linear.append(mag)
                    frequencies.append(freq_ghz)
                except ValueError:
                    continue
    
    if not frequencies:
        raise ValueError(f"No valid data found in {path}")
    
    s21_db = np.array([20 * np.log10(m) if m > 0 else -100 for m in s21_mag_linear])
    return np.array(frequencies), s21_db

def load_mat(filepath):
    """Load MATLAB .mat file."""
    try:
        from scipy.io import loadmat
    except ImportError:
        raise ImportError("scipy.io required for .mat files")
    
    path = Path(filepath)
    mat_data = loadmat(path)
    
    freq_keys = ['frequencies', 'freq', 'f', 'Frequency_GHz']
    s21_keys = ['S21_dB', 's21_db', 'S21', 'data']
    
    frequencies = None
    s21_db = None
    
    for key in freq_keys:
        if key in mat_data:
            val = mat_data[key]
            if isinstance(val, np.ndarray):
                frequencies = val.flatten()
                break
    
    for key in s21_keys:
        if key in mat_data:
            val = mat_data[key]
            if isinstance(val, np.ndarray):
                s21_db = val.flatten()
                break
    
    if frequencies is None:
        raise ValueError(f"Frequency variable not found. Available: {list(mat_data.keys())}")
    if s21_db is None:
        raise ValueError(f"S21 variable not found. Available: {list(mat_data.keys())}")
    
    return frequencies.astype(np.float64), s21_db.astype(np.float64)

def auto_load(filepath):
    """Automatically detect file format and load S21 data."""
    path = Path(filepath)
    suffix = path.suffix.lower()
    
    if suffix == '.csv':
        return load_s21_csv(path)
    elif suffix == '.s2p':
        return load_s2p(path)
    elif suffix == '.mat':
        return load_mat(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def get_bbox_from_mask(mask):
    """Get bounding box from binary mask."""
    mask = np.asarray(mask)
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return (0, 0, 0, 0)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return (int(x_min), int(y_min), int(x_max), int(y_max))

def to_coco_format(masks, image_ids, category_ids=None, image_shapes=None):
    """Convert masks to COCO JSON format."""
    if category_ids is None:
        category_ids = [1] * len(masks)
    
    annotations = []
    for idx, mask in enumerate(masks):
        mask = np.asarray(mask)
        h, w = mask.shape[:2] if image_shapes is None else image_shapes[idx]
        
        x_min, y_min, x_max, y_max = get_bbox_from_mask(mask)
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        
        annotations.append({
            "id": idx,
            "image_id": image_ids[idx],
            "category_id": category_ids[idx],
            "bbox": [x_min, y_min, bbox_width, bbox_height],
            "area": int(np.sum(mask)),
            "iscrowd": 0,
        })
    
    return {
        "images": [
            {"id": img_id, "width": w, "height": h}
            for img_id, (h, w) in zip(image_ids, image_shapes or [])
        ],
        "annotations": annotations,
        "categories": [{"id": 1, "name": "lesion"}, {"id": 2, "name": "tumor"}],
    }

def to_yolo_format(masks, image_shapes, class_ids=None):
    """Convert masks to YOLO TXT format."""
    if class_ids is None:
        class_ids = [0] * len(masks)
    
    yolo_lines = []
    for mask, (h, w), class_id in zip(masks, image_shapes, class_ids):
        mask = np.asarray(mask)
        x_min, y_min, x_max, y_max = get_bbox_from_mask(mask)
        
        x_center = ((x_min + x_max) / 2) / w
        y_center = ((y_min + y_max) / 2) / h
        bbox_width = (x_max - x_min) / w
        bbox_height = (y_max - y_min) / h
        
        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")
    
    return yolo_lines

def to_monai_format(masks, image_ids, category_ids=None):
    """Convert masks to MONAI format."""
    if category_ids is None:
        category_ids = [1] * len(masks)
    
    monai_output = []
    for mask, img_id, cat_id in zip(masks, image_ids, category_ids):
        mask = np.asarray(mask)
        mask_flat = mask.flatten()
        rle = []
        prev = None
        count = 0
        for val in mask_flat:
            if val == prev:
                count += 1
            else:
                if prev is not None:
                    rle.append(count)
                prev = val
                count = 1
        rle.append(count)
        
        monai_output.append({
            "image_id": img_id,
            "label": cat_id,
            "segmentation": {"counts": rle, "size": list(mask.shape)},
        })
    
    return monai_output

# =============================================================================
# SAM WRAPPER (Mock mode)
# =============================================================================

SAM_AVAILABLE = False
try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    pass

class SAMWrapper:
    """SAM wrapper with mock fallback."""
    
    def __init__(self, model_type="vit_b", checkpoint_path=None, device="auto"):
        self._image = None
        self._use_mock = True
        self._loaded = False
        
        if not SAM_AVAILABLE:
            self._use_mock = True
            self._loaded = True
            return
        
        if checkpoint_path is None:
            default_paths = [
                Path("sam_vit_b.pth"),
                Path.home() / ".cache" / "sam" / "sam_vit_b.pth",
            ]
            for p in default_paths:
                if p.exists():
                    checkpoint_path = p
                    break
        
        if checkpoint_path is None or not Path(checkpoint_path).exists():
            self._use_mock = True
            self._loaded = True
            return
        
        try:
            import torch
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
            sam.to(device=device)
            self.predictor = SamPredictor(sam)
            self._loaded = True
            self._use_mock = False
        except Exception as e:
            print(f"Failed to load SAM: {e}")
            self._use_mock = True
            self._loaded = True
    
    def set_image(self, image):
        image = np.asarray(image)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        self._image = image
        if not self._use_mock and hasattr(self, 'predictor'):
            self.predictor.set_image(image)
    
    def from_clicks(self, foreground, background=None, multimask=False):
        if self._use_mock:
            return self._mock_from_clicks(foreground, background)
        
        if not foreground:
            raise ValueError("At least one foreground click is required")
        
        points = []
        labels = []
        for x, y in foreground:
            points.append([x, y])
            labels.append(1)
        if background:
            for x, y in background:
                points.append([x, y])
                labels.append(0)
        
        points = np.array(points)
        labels = np.array(labels)
        
        masks, scores, logits = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=multimask
        )
        
        if multimask:
            return [mask.astype(np.uint8) for mask in masks]
        else:
            best_idx = np.argmax(scores)
            return masks[best_idx].astype(np.uint8)
    
    def _mock_from_clicks(self, foreground, background=None):
        """Mock segmentation: circular mask centered on first click."""
        if self._image is None:
            raise RuntimeError("No image set")
        
        h, w = self._image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if not foreground:
            return mask
        
        cx, cy = foreground[0]
        radius = min(h, w) // 6
        
        for i in range(h):
            for j in range(w):
                if (i - cy)**2 + (j - cx)**2 < radius**2:
                    mask[i, j] = 1
        
        return mask

# =============================================================================
# SESSION STATE
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
if "masks_history" not in st.session_state:
    st.session_state.masks_history = []
if "export_ready" not in st.session_state:
    st.session_state.export_ready = False
if "click_mode" not in st.session_state:
    st.session_state.click_mode = "foreground"
if "image_width" not in st.session_state:
    st.session_state.image_width = 512
if "image_height" not in st.session_state:
    st.session_state.image_height = 512

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    # Logo
    logo_path = Path("Hypoxify Logo.png")
    if logo_path.exists():
        try:
            logo = Image.open(logo_path)
            st.image(logo, use_container_width=True)
        except:
            st.markdown("## 🔬 Hypoxify")
    else:
        st.markdown("## 🔬 Hypoxify")
    
    st.markdown("### ⚙️ Configuration")
    
    config_names = list(SUPPORTED_CONFIGURATIONS.keys())
    config_labels = {
        k: f"{SUPPORTED_CONFIGURATIONS[k]['modality'].title()} - {SUPPORTED_CONFIGURATIONS[k]['description']}"
        for k in config_names
    }
    
    selected_config = st.selectbox(
        "Select System Configuration",
        options=config_names,
        format_func=lambda x: config_labels[x],
        index=0
    )
    
    st.markdown("---")
    
    st.markdown("### 📥 Input Mode")
    mode = st.radio(
        "How are you uploading your data?",
        options=["Upload Image (PNG/JPG/TIFF)", "Upload Raw Data (CSV/S2P/MAT)"],
        index=0
    )
    
    st.markdown("---")
    
    st.markdown("### 🎯 Segmentation Settings")
    sam_model = st.selectbox("SAM Model", options=["vit_b", "vit_l", "vit_h"], index=0)
    
    st.markdown("---")
    
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
    
    st.markdown("---")
    if SAM_AVAILABLE:
        st.success("✅ SAM installed")
    else:
        st.info("ℹ️ SAM not installed (using mock)")

# =============================================================================
# MAIN HEADER
# =============================================================================

st.markdown('<p class="main-header">🔬 Hypoxify Annotation Suite</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Physics-informed segmentation for microwave and thermoacoustic imaging</p>', unsafe_allow_html=True)

config = SUPPORTED_CONFIGURATIONS[st.session_state.selected_config]
st.info(
    f"**Active Configuration:** {config['modality'].title()} - {config['description']}\n\n"
    f"**Supported Formats:** {', '.join(config['file_formats'])}"
)

col1, col2 = st.columns([1, 1])

# =============================================================================
# LEFT COLUMN
# =============================================================================

with col1:
    st.markdown("## 📷 Input Data")
    
    if mode == "Upload Image (PNG/JPG/TIFF)":
        uploaded_file = st.file_uploader(
            "Upload Image",
            type=["png", "jpg", "jpeg", "tiff", "bmp"],
            help="Upload a reconstructed image for annotation"
        )
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                image_array = np.array(image)
                if image_array.ndim == 2:
                    image_array = np.stack([image_array] * 3, axis=-1)
                st.session_state.current_image = image_array
                st.session_state.image_loaded = True
                st.session_state.image_width = image_array.shape[1]
                st.session_state.image_height = image_array.shape[0]
                st.session_state.foreground_clicks = []
                st.session_state.background_clicks = []
                st.session_state.current_mask = None
                st.success(f"✅ Loaded: {uploaded_file.name} ({image_array.shape})")
            except Exception as e:
                st.error(f"Error loading image: {e}")
    
    else:
        uploaded_files = st.file_uploader(
            "Upload Raw Data Files",
            type=["csv", "s2p", "mat"],
            accept_multiple_files=True,
        )
        
        baseline_file = st.file_uploader(
            "Upload Baseline (Air) Scan (Optional)",
            type=["csv", "s2p", "mat"],
        )
        
        if uploaded_files:
            st.write(f"**Files uploaded:** {len(uploaded_files)}")
            
            if st.button("🔨 Reconstruct Image from Raw Data", type="primary"):
                with st.spinner("Reconstructing..."):
                    try:
                        temp_dir = tempfile.mkdtemp()
                        file_paths = []
                        for f in uploaded_files:
                            temp_path = Path(temp_dir) / f.name
                            temp_path.write_bytes(f.read())
                            file_paths.append(temp_path)
                        
                        baseline_data = None
                        if baseline_file:
                            baseline_path = Path(temp_dir) / baseline_file.name
                            baseline_path.write_bytes(baseline_file.read())
                            baseline_freq, baseline_s21 = auto_load(baseline_path)
                            baseline_data = {1: baseline_s21}
                        
                        if len(file_paths) == 1:
                            frequencies, s21_data = auto_load(file_paths[0])
                            s21_data_dict = {1: s21_data}
                        else:
                            s21_data_dict = {}
                            for fp in file_paths:
                                match = re.search(r'path(\d+)', fp.name)
                                if match:
                                    path_num = int(match.group(1))
                                    _, s21 = auto_load(fp)
                                    s21_data_dict[path_num] = s21
                            frequencies, _ = auto_load(file_paths[0])
                        
                        start_freq = float(frequencies[0])
                        stop_freq = float(frequencies[-1])
                        num_points = len(frequencies)
                        
                        image = delay_and_sum_reconstruction(
                            s21_data=s21_data_dict,
                            frequencies=frequencies,
                            baseline_data=baseline_data,
                            grid_size=80,
                            grid_extent=100.0,
                            start_freq=start_freq,
                            stop_freq=stop_freq,
                            num_points=num_points
                        )
                        
                        st.session_state.current_image = image
                        st.session_state.image_loaded = True
                        st.session_state.image_width = image.shape[1]
                        st.session_state.image_height = image.shape[0]
                        st.session_state.foreground_clicks = []
                        st.session_state.background_clicks = []
                        st.session_state.current_mask = None
                        st.session_state.frequencies = frequencies
                        st.session_state.raw_data = s21_data_dict
                        
                        st.success("✅ Reconstruction complete!")
                    except Exception as e:
                        st.error(f"Reconstruction error: {e}")
                        import traceback
                        st.code(traceback.format_exc())
    
    # =========================================================================
    # IMAGE DISPLAY WITH COORDINATE LABELS
    # =========================================================================
    
    st.markdown("### 🖼️ Image Display")
    
    if st.session_state.current_image is not None:
        img = st.session_state.current_image
        if img.dtype != np.uint8:
            img = (img / img.max() * 255).astype(np.uint8)
        
        # Display image with coordinate labels
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        
        # Show image with dimensions
        st.image(img, use_container_width=True, clamp=True)
        
        # Display image info below
        st.caption(f"Image size: {st.session_state.image_width} × {st.session_state.image_height} pixels")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # =========================================================================
        # MANUAL COORDINATE INPUT
        # =========================================================================
        
        st.markdown("### 📍 Add Click Points")
        
        st.markdown(f"""
        <div class="click-instruction">
            🔵 <b>Click Mode:</b> {st.session_state.click_mode.upper()}
            {' (Add foreground points)' if st.session_state.click_mode == 'foreground' else ' (Add background points)'}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="click-count">
            Foreground clicks: {len(st.session_state.foreground_clicks)} | 
            Background clicks: {len(st.session_state.background_clicks)}
        </div>
        """, unsafe_allow_html=True)
        
        # Coordinate input
        col_x, col_y = st.columns(2)
        
        with col_x:
            x_coord = st.number_input(
                "X (pixels)", 
                min_value=0, 
                max_value=st.session_state.image_width - 1, 
                value=st.session_state.image_width // 2, 
                step=1
            )
        with col_y:
            y_coord = st.number_input(
                "Y (pixels)", 
                min_value=0, 
                max_value=st.session_state.image_height - 1, 
                value=st.session_state.image_height // 2, 
                step=1
            )
        
        # Add buttons
        col_add_fg, col_add_bg, col_reset, col_undo = st.columns(4)
        
        with col_add_fg:
            if st.button("➕ Add FG", use_container_width=True, type="primary"):
                st.session_state.foreground_clicks.append((int(x_coord), int(y_coord)))
                st.session_state.current_mask = None
                st.rerun()
        
        with col_add_bg:
            if st.button("➖ Add BG", use_container_width=True):
                st.session_state.background_clicks.append((int(x_coord), int(y_coord)))
                st.session_state.current_mask = None
                st.rerun()
        
        with col_reset:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.foreground_clicks = []
                st.session_state.background_clicks = []
                st.session_state.current_mask = None
                st.rerun()
        
        with col_undo:
            if st.button("↩️ Undo", use_container_width=True):
                if st.session_state.background_clicks:
                    st.session_state.background_clicks.pop()
                elif st.session_state.foreground_clicks:
                    st.session_state.foreground_clicks.pop()
                st.session_state.current_mask = None
                st.rerun()
        
        # Show clicked points as list
        if st.session_state.foreground_clicks or st.session_state.background_clicks:
            st.markdown("**📍 Clicked Points:**")
            
            if st.session_state.foreground_clicks:
                fg_str = ", ".join([f"({x}, {y})" for x, y in st.session_state.foreground_clicks[-10:]])
                st.caption(f"🔵 FG: {fg_str}")
            
            if st.session_state.background_clicks:
                bg_str = ", ".join([f"({x}, {y})" for x, y in st.session_state.background_clicks[-10:]])
                st.caption(f"🔴 BG: {bg_str}")
        
        # Mask overlay
        if st.session_state.current_mask is not None:
            if st.checkbox("Show mask overlay"):
                mask = st.session_state.current_mask
                overlay = img.copy()
                if overlay.ndim == 2:
                    overlay = np.stack([overlay] * 3, axis=-1)
                overlay[mask > 0, 0] = overlay[mask > 0, 0] * 0.5 + 200 * 0.5
                st.image(overlay, use_container_width=True)
        
    else:
        st.info("👆 Upload data to begin")

# =============================================================================
# RIGHT COLUMN
# =============================================================================

with col2:
    st.markdown("## 🎯 Segmentation")
    
    # Mode toggle
    col_mode1, col_mode2 = st.columns(2)
    with col_mode1:
        if st.button("🔵 Foreground", use_container_width=True, 
                     type="primary" if st.session_state.click_mode == "foreground" else "secondary"):
            st.session_state.click_mode = "foreground"
            st.rerun()
    with col_mode2:
        if st.button("🔴 Background", use_container_width=True,
                     type="primary" if st.session_state.click_mode == "background" else "secondary"):
            st.session_state.click_mode = "background"
            st.rerun()
    
    st.markdown("---")
    
    # Generate mask
    if st.session_state.image_loaded:
        if len(st.session_state.foreground_clicks) > 0:
            if st.button("⚡ Generate Mask", type="primary", use_container_width=True):
                with st.spinner("Segmenting..."):
                    try:
                        if st.session_state.segmenter is None:
                            checkpoint_path = None
                            for p in [Path("sam_vit_b.pth"), Path.home() / ".cache" / "sam" / "sam_vit_b.pth"]:
                                if p.exists():
                                    checkpoint_path = p
                                    break
                            st.session_state.segmenter = SAMWrapper(
                                model_type=sam_model,
                                checkpoint_path=checkpoint_path
                            )
                        
                        st.session_state.segmenter.set_image(st.session_state.current_image)
                        mask = st.session_state.segmenter.from_clicks(
                            st.session_state.foreground_clicks,
                            st.session_state.background_clicks if st.session_state.background_clicks else None
                        )
                        st.session_state.current_mask = mask
                        st.session_state.masks_history.append(mask)
                        st.success("✅ Mask generated!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Segmentation error: {e}")
                        import traceback
                        st.code(traceback.format_exc())
        else:
            st.info("👉 Add at least one foreground click to generate a mask.")
    else:
        st.info("📤 Load an image first.")
    
    # Mask result
    st.markdown("### 🧬 Mask Result")
    
    if st.session_state.current_mask is not None:
        mask = st.session_state.current_mask
        st.image(mask * 255, use_container_width=True, clamp=True)
        
        mask_area = np.sum(mask)
        mask_percent = (mask_area / mask.size) * 100
        st.caption(f"Mask area: {mask_area} pixels ({mask_percent:.2f}%)")
        
        st.markdown("### 📤 Export")
        
        if st.session_state.export_ready:
            if export_format == "coco":
                coco_data = to_coco_format([mask], image_ids=[1], image_shapes=[mask.shape])
                json_str = json.dumps(coco_data, indent=2)
                st.download_button("📥 Download COCO JSON", json_str, file_name="annotation_coco.json", mime="application/json")
            elif export_format == "yolo":
                yolo_data = to_yolo_format([mask], [mask.shape])
                txt_str = "\n".join(yolo_data)
                st.download_button("📥 Download YOLO TXT", txt_str, file_name="annotation_yolo.txt", mime="text/plain")
            elif export_format == "png":
                mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                buf = io.BytesIO()
                mask_img.save(buf, format="PNG")
                st.download_button("📥 Download PNG Mask", buf.getvalue(), file_name="mask.png", mime="image/png")
            elif export_format == "monai":
                monai_data = to_monai_format([mask], image_ids=[1])
                json_str = json.dumps(monai_data, indent=2)
                st.download_button("📥 Download MONAI JSON", json_str, file_name="annotation_monai.json", mime="application/json")
            
            st.session_state.export_ready = False
            st.rerun()
    else:
        st.info("🔄 No mask generated yet.")

# =============================================================================
# FOOTER TABS
# =============================================================================

st.markdown("---")
tab1, tab2, tab3 = st.tabs(["📊 Data Info", "📋 History", "ℹ️ About"])

with tab1:
    if st.session_state.current_image is not None:
        st.write(f"**Image shape:** {st.session_state.current_image.shape}")
        st.write(f"**Data type:** {st.session_state.current_image.dtype}")
        if st.session_state.frequencies is not None:
            st.write(f"**Frequency range:** {st.session_state.frequencies[0]:.2f} - {st.session_state.frequencies[-1]:.2f} GHz")
            st.write(f"**Frequency points:** {len(st.session_state.frequencies)}")
        if st.session_state.raw_data is not None:
            st.write(f"**Raw data paths:** {list(st.session_state.raw_data.keys())}")
        st.write(f"**Foreground clicks:** {len(st.session_state.foreground_clicks)}")
        st.write(f"**Background clicks:** {len(st.session_state.background_clicks)}")
    else:
        st.info("No data loaded yet.")

with tab2:
    st.write(f"**Masks generated:** {len(st.session_state.masks_history)}")
    if st.session_state.masks_history:
        for i, mask in enumerate(st.session_state.masks_history[-5:]):
            area = np.sum(mask)
            st.caption(f"Mask {i+1}: {area} pixels")
    else:
        st.info("No masks generated yet.")

with tab3:
    st.markdown("""
    ### 🔬 Hypoxify Annotation Suite
    
    **Version:** 0.3.0
    
    **Features:**
    - Physics-informed segmentation for microwave/thermoacoustic imaging
    - Manual X/Y coordinate input for annotation
    - Raw data reconstruction (S21 parameters)
    - Multiple export formats (COCO, YOLO, MONAI, PNG)
    
    **Author:** Anie Udofia
    
    **License:** MIT
    """)

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.caption(
    "🔬 Hypoxify Annotation Suite v0.3.0 | "
    "Physics-informed segmentation | "
    "Data stored locally; no data uploaded to servers."
)
