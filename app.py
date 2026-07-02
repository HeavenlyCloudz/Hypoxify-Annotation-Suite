"""
Hypoxify Annotation Suite - Standalone App
Everything self-contained in one file for easy testing and deployment.
"""

import streamlit as st
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
import tempfile
import json
import io
import re
import time
from typing import Dict, List, Tuple, Optional, Union
from scipy.ndimage import gaussian_filter

# =============================================================================
# CONSTANTS - Directly defined here
# =============================================================================

# Microwave Imaging (2-3 GHz)
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

# Physical constants
C_MM_PER_PS = 0.3  # Speed of light in mm/ps
DEFAULT_GRID_SIZE = 80
GAUSSIAN_SIGMA = 2.0

# Supported configurations
SUPPORTED_CONFIGURATIONS = {
    "microwave_4_antenna": {
        "modality": "microwave",
        "description": "4-antenna microwave imaging system (S21)",
        "num_antennas": 4,
        "antenna_positions": MICROWAVE_CONFIG["antenna_positions"],
        "path_to_antenna_pair": MICROWAVE_CONFIG["path_to_antenna_pair"],
        "file_formats": [".csv", ".s2p", ".mat"],
    },
    "microwave_8_antenna": {
        "modality": "microwave",
        "description": "8-antenna microwave imaging system",
        "num_antennas": 8,
        "antenna_positions": {
            1: (-75, -40), 2: (-75, 40),
            3: (-40, -75), 4: (40, -75),
            5: (75, -40), 6: (75, 40),
            7: (-40, 75), 8: (40, 75),
        },
        "path_to_antenna_pair": {
            1: (1, 5), 2: (2, 6), 3: (3, 7), 4: (4, 8),
            5: (1, 3), 6: (2, 4), 7: (5, 7), 8: (6, 8),
        },
        "file_formats": [".csv", ".s2p", ".mat"],
    },
    "thermoacoustic_ring": {
        "modality": "thermoacoustic",
        "description": "Ring-array thermoacoustic system (PATATO compatible)",
        "num_detectors": 128,
        "detector_radius_mm": 50,
        "file_formats": [".h5", ".mat"],
    },
    "mri": {
        "modality": "mri",
        "description": "MRI DICOM/NIfTI (standard medical imaging)",
        "file_formats": [".dcm", ".nii", ".nii.gz"],
    },
    "ct": {
        "modality": "ct",
        "description": "CT DICOM/NIfTI (standard medical imaging)",
        "file_formats": [".dcm", ".nii", ".nii.gz"],
    },
    "histology": {
        "modality": "histology",
        "description": "Histology slides (whole-slide images)",
        "file_formats": [".svs", ".ndpi", ".tiff", ".png", ".jpg"],
    },
}

EXPORT_FORMATS = {
    "coco": {"extension": ".json", "description": "COCO format for object detection"},
    "yolo": {"extension": ".txt", "description": "YOLO format for object detection"},
    "monai": {"extension": ".json", "description": "MONAI format for medical imaging"},
    "nifti": {"extension": ".nii.gz", "description": "NIfTI format for volumetric data"},
    "png": {"extension": ".png", "description": "PNG mask format"},
    "dicom_seg": {"extension": ".dcm", "description": "DICOM SEG format"},
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def db_to_linear(db_values):
    """Convert dB to linear magnitude."""
    db_values = np.asarray(db_values)
    return 10 ** (db_values / 10)

def linear_to_db(linear_values):
    """Convert linear magnitude to dB."""
    linear_values = np.asarray(linear_values)
    linear_values = np.maximum(linear_values, 1e-12)
    return 10 * np.log10(linear_values)

def apply_background_subtraction(patient_s21_db, baseline_s21_db):
    """Remove direct antenna coupling by subtracting baseline in linear domain."""
    patient_s21_db = np.asarray(patient_s21_db)
    baseline_s21_db = np.asarray(baseline_s21_db)
    
    if len(patient_s21_db) != len(baseline_s21_db):
        raise ValueError(f"Length mismatch: patient {len(patient_s21_db)} != baseline {len(baseline_s21_db)}")
    
    patient_linear = db_to_linear(patient_s21_db)
    baseline_linear = db_to_linear(baseline_s21_db)
    corrected_linear = patient_linear - baseline_linear
    corrected_linear = np.maximum(corrected_linear, 1e-12)
    return linear_to_db(corrected_linear)

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_s21_csv(filepath):
    """Load S21 data from CSV format."""
    import pandas as pd
    path = Path(filepath)
    df = pd.read_csv(path)
    
    # Try to find frequency column
    freq_col = None
    for col in df.columns:
        if 'freq' in col.lower() or 'ghz' in col.lower():
            freq_col = col
            break
    if freq_col is None:
        raise ValueError(f"Frequency column not found. Available: {df.columns.tolist()}")
    
    # Try to find S21 column
    s21_col = None
    for col in df.columns:
        if 's21' in col.lower() or 's_param' in col.lower():
            s21_col = col
            break
    if s21_col is None:
        raise ValueError(f"S21 column not found. Available: {df.columns.tolist()}")
    
    frequencies = df[freq_col].values.astype(np.float64)
    s21_db = df[s21_col].values.astype(np.float64)
    
    if len(frequencies) != len(s21_db):
        raise ValueError(f"Length mismatch: frequencies {len(frequencies)} != s21 {len(s21_db)}")
    
    return frequencies, s21_db

def load_s2p(filepath):
    """Load S2P (Touchstone) file format."""
    path = Path(filepath)
    frequencies = []
    s21_mag_linear = []
    
    format_type = "ma"
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#') and 'MA' in line.upper():
                format_type = "ma"
            elif line.startswith('#') and 'RI' in line.upper():
                format_type = "ri"
            elif not line.startswith('!') and not line.startswith('#'):
                break
    
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('!') or line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                freq_mhz = float(parts[0])
                freq_ghz = freq_mhz / 1000.0
                if format_type == "ma":
                    mag = float(parts[1])
                else:
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
    """Load MATLAB .mat file containing S21 data."""
    try:
        from scipy.io import loadmat
    except ImportError:
        raise ImportError("scipy.io required for .mat files. Install scipy: pip install scipy")
    
    path = Path(filepath)
    mat_data = loadmat(path)
    
    freq_keys = ['frequencies', 'freq', 'f', 'Frequency_GHz', 'frequency']
    s21_keys = ['S21_dB', 's21_db', 'S21', 'data', 's_params']
    
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
        raise ValueError(f"Unsupported file format: {suffix}. Supported: .csv, .s2p, .mat")

def load_multi_angle_scans(directory, angles=None):
    """Load all rotation scans from a directory."""
    if angles is None:
        angles = [0, 120, 240]
    
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    all_data = {}
    for angle in angles:
        angle_data = {}
        for path_num in [1, 2, 3, 4]:
            pattern = f"path{path_num}_angle{angle}_*.csv"
            files = list(directory.glob(pattern))
            if files:
                latest = max(files, key=lambda f: f.stat().st_mtime)
                try:
                    _, s21 = load_s21_csv(latest)
                    angle_data[path_num] = s21
                except Exception as e:
                    print(f"Warning: Could not load {latest}: {e}")
        if angle_data:
            all_data[angle] = angle_data
    return all_data

# =============================================================================
# RECONSTRUCTION FUNCTIONS
# =============================================================================

def delay_and_sum_reconstruction(
    s21_data,
    frequencies,
    baseline_data=None,
    antenna_positions=None,
    path_to_antenna_pair=None,
    grid_size=DEFAULT_GRID_SIZE,
    grid_extent=100.0,
    sigma=GAUSSIAN_SIGMA
):
    """Reconstruct image using delay-and-sum beamforming."""
    if antenna_positions is None:
        antenna_positions = MICROWAVE_CONFIG["antenna_positions"]
    if path_to_antenna_pair is None:
        path_to_antenna_pair = MICROWAVE_CONFIG["path_to_antenna_pair"]
    
    x_grid = np.linspace(-grid_extent, grid_extent, grid_size)
    y_grid = np.linspace(-grid_extent, grid_extent, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    image = np.zeros((grid_size, grid_size))
    num_paths = 0
    
    for path_num, s21_db in s21_data.items():
        if path_num not in path_to_antenna_pair:
            continue
        
        tx_ant, rx_ant = path_to_antenna_pair[path_num]
        tx_pos = antenna_positions[tx_ant]
        rx_pos = antenna_positions[rx_ant]
        
        s21_linear = db_to_linear(s21_db)
        
        if baseline_data and path_num in baseline_data:
            corrected_db = apply_background_subtraction(s21_db, baseline_data[path_num])
            s21_linear = db_to_linear(corrected_db)
        
        s21_linear = np.maximum(s21_linear, 1e-12)
        
        for i in range(grid_size):
            for j in range(grid_size):
                point = (X[i, j], Y[i, j])
                d_tx = np.sqrt((tx_pos[0] - point[0])**2 + (tx_pos[1] - point[1])**2)
                d_rx = np.sqrt((rx_pos[0] - point[0])**2 + (rx_pos[1] - point[1])**2)
                total_distance = (d_tx + d_rx) / 1000
                
                delay_ps = total_distance / C_MM_PER_PS
                freq_range_ghz = frequencies[-1] - frequencies[0]
                freq_idx = int(delay_ps * freq_range_ghz / 1000)
                freq_idx = np.clip(freq_idx, 0, len(s21_linear) - 1)
                
                image[i, j] += s21_linear[freq_idx]
        
        num_paths += 1
    
    if num_paths > 0:
        image /= num_paths
    else:
        raise ValueError("No valid paths found in s21_data")
    
    image = gaussian_filter(image, sigma=sigma)
    
    if image.max() > 0:
        image = np.clip(image, 0, np.percentile(image, 95))
        image = (image / image.max()) * 255
    
    return image.astype(np.uint8)

def reconstruct_from_multi_angle(angle_data, frequencies, baseline_data=None, grid_size=DEFAULT_GRID_SIZE):
    """Reconstruct image by averaging across multiple rotation angles."""
    images = []
    for angle, s21_data in angle_data.items():
        image = delay_and_sum_reconstruction(
            s21_data=s21_data,
            frequencies=frequencies,
            baseline_data=baseline_data,
            grid_size=grid_size
        )
        images.append(image)
    
    if not images:
        raise ValueError("No images were reconstructed from angle_data")
    
    avg_image = np.mean(images, axis=0)
    return avg_image.astype(np.uint8)

# =============================================================================
# SAM WRAPPER (Mock for testing without SAM installed)
# =============================================================================

class MockSAMWrapper:
    """Mock SAM wrapper that generates simple circular masks for testing."""
    
    def __init__(self, model_type="vit_b", checkpoint_path=None, device="auto"):
        self.model_type = model_type
        self.device = device
        self._image = None
        self._loaded = True
        print(f"Mock SAM initialized (simulation mode)")
    
    def set_image(self, image):
        """Store image for reference."""
        image = np.asarray(image)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        self._image = image
        print(f"Image set: shape={self._image.shape}")
    
    def from_clicks(self, foreground, background=None, multimask=False):
        """Generate a circular mask centered on the first foreground click."""
        if self._image is None:
            raise RuntimeError("No image set. Call set_image first.")
        
        if not foreground:
            raise ValueError("At least one foreground click is required")
        
        h, w = self._image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Create circular mask centered on first foreground click
        cx, cy = foreground[0]
        radius = min(h, w) // 5
        
        for i in range(h):
            for j in range(w):
                if (i - cy)**2 + (j - cx)**2 < radius**2:
                    mask[i, j] = 1
        
        return mask
    
    def from_box(self, box, multimask=False):
        """Generate mask from bounding box."""
        if self._image is None:
            raise RuntimeError("No image set. Call set_image first.")
        
        x_min, y_min, x_max, y_max = box
        h, w = self._image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for i in range(max(0, y_min), min(h, y_max)):
            for j in range(max(0, x_min), min(w, x_max)):
                mask[i, j] = 1
        
        return mask

# Try to import real SAM, fallback to mock
try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
    print("SAM loaded successfully")
except ImportError:
    SAM_AVAILABLE = False
    print("SAM not installed. Using mock segmentation.")

class SAMWrapper:
    """Real or mock SAM wrapper based on availability."""
    
    def __init__(self, model_type="vit_b", checkpoint_path=None, device="auto"):
        if not SAM_AVAILABLE:
            self._mock = MockSAMWrapper(model_type, checkpoint_path, device)
            self._use_mock = True
            self._loaded = True
            return
        
        self._use_mock = False
        self.model_type = model_type
        self.device = device
        self.predictor = None
        self._image = None
        self._loaded = False
        
        # Try to load real SAM
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
            print(f"SAM checkpoint not found. Using mock segmentation.")
            self._mock = MockSAMWrapper(model_type, checkpoint_path, device)
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
            print(f"SAM loaded: {model_type} on {device}")
        except Exception as e:
            print(f"Failed to load SAM: {e}. Using mock segmentation.")
            self._mock = MockSAMWrapper(model_type, checkpoint_path, device)
            self._use_mock = True
            self._loaded = True
    
    @property
    def loaded(self):
        return self._loaded
    
    def set_image(self, image):
        if self._use_mock:
            return self._mock.set_image(image)
        
        image = np.asarray(image)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        self._image = image
        self.predictor.set_image(image)
    
    def from_clicks(self, foreground, background=None, multimask=False):
        if self._use_mock:
            return self._mock.from_clicks(foreground, background, multimask)
        
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
    
    def from_box(self, box, multimask=False):
        if self._use_mock:
            return self._mock.from_box(box, multimask)
        
        box = np.array(box)
        masks, scores, logits = self.predictor.predict(
            box=box,
            multimask_output=multimask
        )
        
        if multimask:
            return [mask.astype(np.uint8) for mask in masks]
        else:
            best_idx = np.argmax(scores)
            return masks[best_idx].astype(np.uint8)

# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def get_bbox_from_mask(mask):
    """Get bounding box from binary mask."""
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D, got shape {mask.shape}")
    
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return (0, 0, 0, 0)
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return (int(x_min), int(y_min), int(x_max), int(y_max))

def mask_to_polygon(mask, tolerance=2.0):
    """Convert binary mask to polygon using OpenCV."""
    try:
        import cv2
    except ImportError:
        return []
    
    mask = np.asarray(mask)
    if mask.ndim != 2:
        return []
    
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        return []
    
    polygon = cv2.approxPolyDP(contours[0], tolerance, True)
    polygon_list = polygon.squeeze().tolist()
    if isinstance(polygon_list, list) and isinstance(polygon_list[0], (int, float)):
        polygon_list = [polygon_list]
    return polygon_list

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
        polygon = mask_to_polygon(mask)
        
        annotations.append({
            "id": idx,
            "image_id": image_ids[idx],
            "category_id": category_ids[idx],
            "segmentation": [polygon] if polygon else [],
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
    """Convert masks to MONAI format (simplified RLE)."""
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
    .main-header { font-size: 2.5rem; font-weight: 700; color: #0d47a1; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.1rem; color: #555; margin-bottom: 1.5rem; }
    .feature-card { background-color: #f5f5f5; border-radius: 10px; padding: 15px; margin: 5px 0; border-left: 4px solid #0d47a1; }
    .stButton button { font-weight: 600; border-radius: 8px; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; padding: 10px 20px; font-weight: 600; }
    .stTabs [aria-selected="true"] { background-color: #0d47a1; color: white; }
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
    st.session_state.mode = "direct_image"
if "masks_history" not in st.session_state:
    st.session_state.masks_history = []
if "export_ready" not in st.session_state:
    st.session_state.export_ready = False

# =============================================================================
# SIDEBAR - CONFIGURATION AND CONTROLS
# =============================================================================

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
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
    
    st.markdown("### 🎯 Segmentation Settings")
    sam_model = st.selectbox(
        "SAM Model",
        options=["vit_b", "vit_l", "vit_h"],
        index=0,
        help="vit_b is fastest, vit_h is most accurate but slower"
    )
    
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
    
    # Show SAM status
    st.markdown("---")
    if SAM_AVAILABLE:
        st.success("✅ SAM installed (real mode)")
    else:
        st.info("ℹ️ SAM not installed (using mock segmentation)")

# =============================================================================
# MAIN CONTENT AREA
# =============================================================================

st.markdown('<p class="main-header">🔬 Hypoxify Annotation Suite</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Physics-informed segmentation for microwave and thermoacoustic imaging</p>', unsafe_allow_html=True)

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
    
    if st.session_state.mode == "direct_image":
        uploaded_file = st.file_uploader(
            "Upload Image",
            type=["png", "jpg", "jpeg", "tiff", "bmp", "dcm"],
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
                st.session_state.reconstructed_image = None
                st.session_state.foreground_clicks = []
                st.session_state.background_clicks = []
                st.session_state.current_mask = None
                st.write(f"**Image shape:** {image_array.shape}")
                st.write(f"**Data type:** {image_array.dtype}")
            except Exception as e:
                st.error(f"Error loading image: {e}")
    
    else:
        uploaded_files = st.file_uploader(
            "Upload Raw Data Files",
            type=["csv", "s2p", "mat"],
            accept_multiple_files=True,
            help="Upload S21 parameter files."
        )
        
        if uploaded_files:
            st.write(f"**Files uploaded:** {len(uploaded_files)}")
            
            if st.button("🔨 Reconstruct Image from Raw Data", type="primary"):
                with st.spinner("Reconstructing image from raw data..."):
                    try:
                        temp_dir = tempfile.mkdtemp()
                        file_paths = []
                        for f in uploaded_files:
                            temp_path = Path(temp_dir) / f.name
                            temp_path.write_bytes(f.read())
                            file_paths.append(temp_path)
                        
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
    
    # Display current image
    st.markdown("### 🖼️ Current Image")
    
    if st.session_state.current_image is not None:
        img = st.session_state.current_image
        if img.dtype != np.uint8:
            img = (img / img.max() * 255).astype(np.uint8)
        st.image(img, use_container_width=True, clamp=True)
        
        if st.session_state.foreground_clicks:
            st.caption(f"🔵 Foreground clicks: {len(st.session_state.foreground_clicks)}")
        if st.session_state.background_clicks:
            st.caption(f"🔴 Background clicks: {len(st.session_state.background_clicks)}")
        
        if st.session_state.current_mask is not None:
            if st.checkbox("Show mask overlay"):
                mask = st.session_state.current_mask
                overlay = img.copy()
                if overlay.ndim == 2:
                    overlay = np.stack([overlay] * 3, axis=-1)
                overlay[mask > 0, 0] = overlay[mask > 0, 0] * 0.5 + 200 * 0.5
                st.image(overlay, use_container_width=True)
        
        st.caption("💡 Click on image below to add foreground points. Right-click for background.")
    else:
        st.info("👆 Upload data to begin")

# =============================================================================
# RIGHT COLUMN - SEGMENTATION & EXPORT
# =============================================================================

with col2:
    st.markdown("## 🎯 Segmentation")
    
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
    
    click_mode = st.radio(
        "Click Mode",
        ["Foreground (tumor)", "Background (remove)"],
        horizontal=True,
        key="click_mode_radio"
    )
    
    if st.session_state.image_loaded:
        if len(st.session_state.foreground_clicks) > 0:
            if st.button("⚡ Generate Mask", type="primary", use_container_width=True):
                with st.spinner("Segmenting..."):
                    try:
                        if st.session_state.segmenter is None:
                            # Auto-detect checkpoint
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
    
    # Mask display
    st.markdown("### 🧬 Mask Result")
    
    if st.session_state.current_mask is not None:
        mask = st.session_state.current_mask
        st.image(mask * 255, use_container_width=True, clamp=True)
        
        mask_area = np.sum(mask)
        mask_percent = (mask_area / mask.size) * 100
        st.caption(f"Mask area: {mask_area} pixels ({mask_percent:.2f}% of image)")
        
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
        st.info("🔄 No mask generated yet. Add clicks and click 'Generate Mask'.")

# =============================================================================
# BOTTOM TABS
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
    st.info("Physical features will be extracted from raw data when available.")
    if st.session_state.raw_data is not None and st.session_state.foreground_clicks:
        st.write("Click positions recorded for physical feature extraction.")

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
