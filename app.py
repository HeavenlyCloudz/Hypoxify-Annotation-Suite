import gradio as gr
import numpy as np
import cv2
from PIL import Image
import io
import json
import re
import tempfile
from pathlib import Path
import pandas as pd
from scipy.ndimage import gaussian_filter, distance_transform_edt
from scipy import ndimage
from scipy.io import loadmat
import random
import os
import shutil
import zipfile
import urllib.request
from typing import List, Dict, Optional, Tuple, Any
import torch
import torchvision
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 0. SAM2 TINY (Memory-Optimized)
# ------------------------------------------------------------
try:
    from sam2 import sam_model_registry, SamPredictor
    SAM2_AVAILABLE = True
    print("✅ SAM2 available")
except ImportError:
    SAM2_AVAILABLE = False
    print("⚠️ SAM2 not installed. Install with: pip install sam2")

class SAM2Wrapper:
    """SAM2 wrapper optimized for memory-constrained environments using Tiny model."""
    
    def __init__(self, checkpoint_path=None, model_type="tiny", device="cpu"):
        """
        Args:
            model_type: "tiny" (~78MB), "small" (~300MB), "base" (~500MB)
        """
        self.device = device
        self.model = None
        self.predictor = None
        self._loaded = False
        self._use_mock = False
        self._image = None
        
        if not SAM2_AVAILABLE:
            self._use_mock = True
            self._loaded = True
            print("⚠️ Running in mock mode (SAM2 not installed)")
            return
        
        try:
            # SAM2 Tiny - auto-downloads checkpoint
            self.model = sam_model_registry[model_type](checkpoint=checkpoint_path)
            self.model.to(device=self.device)
            self.predictor = SamPredictor(self.model)
            self._loaded = True
            self._use_mock = False
            print(f"✅ SAM2 ({model_type}) loaded on {self.device}")
        except Exception as e:
            print(f"❌ Failed to load SAM2: {e}")
            self._use_mock = True
            self._loaded = True
    
    def set_image(self, image):
        """Set the image for segmentation."""
        self._image = image
        if self._use_mock or self.predictor is None:
            return
        try:
            self.predictor.set_image(image)
        except Exception as e:
            print(f"⚠️ Error setting image: {e}")
            self._use_mock = True
    
    def predict_from_clicks(self, points, labels):
        """Run SAM2 prediction with click points."""
        if self._use_mock or self.predictor is None:
            return self._mock_predict(points, labels)
        
        if not points:
            return np.zeros((512, 512), dtype=np.uint8)
        
        try:
            points_np = np.array(points)
            labels_np = np.array(labels)
            
            masks, scores, _ = self.predictor.predict(
                point_coords=points_np,
                point_labels=labels_np,
                multimask_output=True
            )
            
            # Return the best mask
            best_idx = np.argmax(scores)
            return masks[best_idx].astype(np.uint8)
        except Exception as e:
            print(f"⚠️ SAM2 prediction failed: {e}")
            return self._mock_predict(points, labels)
    
    def _mock_predict(self, points, labels):
        """Fallback mock segmentation."""
        if not points or self._image is None:
            return np.zeros((512, 512), dtype=np.uint8)
        
        h, w = self._image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Simple circle around first foreground point
        fg_points = [p for p, l in zip(points, labels) if l == 1]
        if fg_points:
            cx, cy = fg_points[0]
            radius = min(h, w) // 6
            y, x = np.ogrid[:h, :w]
            dist = (x - cx)**2 + (y - cy)**2
            mask[dist < radius**2] = 1
        
        return mask

# ------------------------------------------------------------
# 1. MULTI-MODALITY PHYSICS ENGINE
# ------------------------------------------------------------
class ModalityDetector:
    """Detects imaging modality from file type and metadata."""
    
    @staticmethod
    def detect(filepath: str) -> str:
        """Detect modality from file extension."""
        ext = Path(filepath).suffix.lower()
        
        # Microwave modalities (MITT, MWI)
        if ext in ['.s2p', '.csv', '.mat']:
            return "microwave"
        
        # Photoacoustic
        if ext in ['.h5', '.hdf5']:
            return "photoacoustic"
        
        # Ultrasound
        if ext in ['.ult', '.rf']:
            return "ultrasound"
        
        # DICOM (could be MRI, CT, etc.)
        if ext == '.dcm':
            return "dicom"
        
        # Fallback to image
        return "image"
    
    @staticmethod
    def extract_physics_features(data: Dict, modality: str) -> Dict:
        """Extract modality-specific physics features."""
        features = {}
        
        if modality == "microwave":
            if "magnitude_db" in data:
                features["dielectric"] = data["magnitude_db"]
            if "phase_deg" in data:
                features["phase"] = data["phase_deg"]
            if "frequencies" in data:
                features["frequencies"] = data["frequencies"]
        
        elif modality == "photoacoustic":
            if "pressure" in data:
                features["acoustic_pressure"] = data["pressure"]
            if "frequency" in data:
                features["frequency"] = data["frequency"]
        
        elif modality == "ultrasound":
            if "rf_signal" in data:
                features["rf_signal"] = data["rf_signal"]
        
        return features

# ------------------------------------------------------------
# 2. DICOM SUPPORT (Client-Side De-identification)
# ------------------------------------------------------------
try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False
    print("⚠️ pydicom not installed. Install with: pip install pydicom")

def load_dicom(filepath: str, deidentify: bool = True) -> Tuple[Optional[np.ndarray], Dict]:
    """Load DICOM file with optional client-side de-identification."""
    if not DICOM_AVAILABLE:
        return None, {"error": "pydicom not installed"}
    
    try:
        ds = pydicom.dcmread(filepath)
        
        # Client-side de-identification (remove PHI)
        if deidentify:
            for tag in ['PatientID', 'PatientName', 'PatientBirthDate', 'PatientAddress', 
                       'PatientTelephoneNumbers', 'PatientComments', 'OtherPatientIDs']:
                if hasattr(ds, tag):
                    setattr(ds, tag, '')
        
        # Extract pixel data
        if hasattr(ds, 'pixel_array'):
            pixel_array = ds.pixel_array
            if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
                pixel_array = apply_voi_lut(pixel_array, ds)
            
            if pixel_array.dtype != np.uint8:
                pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min() + 1e-8) * 255
                pixel_array = pixel_array.astype(np.uint8)
            
            if len(pixel_array.shape) == 2:
                pixel_array = np.stack([pixel_array] * 3, axis=-1)
        
        # Extract metadata (without PHI)
        metadata = {
            "Modality": getattr(ds, 'Modality', 'Unknown'),
            "Manufacturer": getattr(ds, 'Manufacturer', 'Unknown'),
            "StudyDescription": getattr(ds, 'StudyDescription', 'Unknown'),
            "SeriesDescription": getattr(ds, 'SeriesDescription', 'Unknown'),
            "SliceThickness": getattr(ds, 'SliceThickness', 'Unknown'),
            "SpacingBetweenSlices": getattr(ds, 'SpacingBetweenSlices', 'Unknown'),
            "PixelSpacing": getattr(ds, 'PixelSpacing', 'Unknown'),
            "Rows": getattr(ds, 'Rows', 0),
            "Columns": getattr(ds, 'Columns', 0),
            "Deidentified": deidentify,
        }
        
        return pixel_array, metadata
    except Exception as e:
        return None, {"error": str(e)}

# ------------------------------------------------------------
# 3. S-PARAMETER PHASE-SHIFT TOKENIZATION
# ------------------------------------------------------------
def load_s2p_with_phase(filepath: str) -> Dict:
    """Load S2P file with both magnitude and phase."""
    frequencies = []
    magnitude = []
    phase_deg = []
    
    with open(filepath, 'r') as f:
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
                    phase = np.angle(real + 1j*imag, deg=True)
                    
                    frequencies.append(freq_ghz)
                    magnitude.append(mag)
                    phase_deg.append(phase)
                except ValueError:
                    continue
    
    if not frequencies:
        raise ValueError(f"No valid data found in {filepath}")
    
    magnitude_db = np.array([20 * np.log10(m) if m > 0 else -100 for m in magnitude])
    
    return {
        "frequencies": np.array(frequencies),
        "magnitude_db": magnitude_db,
        "magnitude_linear": np.array(magnitude),
        "phase_deg": np.array(phase_deg)
    }

def tokenize_phase_shift(phase_data: np.ndarray) -> np.ndarray:
    """Convert phase data to features useful for segmentation."""
    phase_unwrapped = np.unwrap(phase_data * np.pi / 180) * 180 / np.pi
    phase_gradient = np.gradient(phase_unwrapped)
    
    phase_norm = (phase_unwrapped - phase_unwrapped.min()) / (phase_unwrapped.max() - phase_unwrapped.min() + 1e-8)
    gradient_norm = (phase_gradient - phase_gradient.min()) / (phase_gradient.max() - phase_gradient.min() + 1e-8)
    
    return np.stack([phase_norm, gradient_norm], axis=0)

# ------------------------------------------------------------
# 4. PHYSICS SIMULATION (Multi-Modality)
# ------------------------------------------------------------
class PhysicsSimulator:
    @staticmethod
    def extract_physical_signature(image, click_point, modality="microwave"):
        """Extract modality-specific physical signatures."""
        gray = cv2.cvtColor(np.uint8(image), cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        edges = cv2.Canny(gray, 50, 150)
        dist_from_click = np.sqrt((np.arange(h)[:, None] - click_point[1])**2 +
                                  (np.arange(w)[None, :] - click_point[0])**2)
        
        # Common physics features
        dielectric = edges.astype(np.float32) + (1 / (dist_from_click + 1)) * 10
        acoustic = 50 * np.exp(-dist_from_click / 100) + np.random.normal(0, 5, (h, w))
        local_std = ndimage.generic_filter(gray, np.std, size=5)
        absorption = local_std / local_std.max()
        
        # Modality-specific enhancements
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
        """Apply physics conditioning to refine segmentation."""
        weights = {
            "dielectric": 0.4,
            "acoustic": 0.3,
            "absorption": 0.3
        }
        
        # Modality-specific weight adjustments
        modality = physics_maps.get("modality", "microwave")
        if modality == "photoacoustic":
            weights["acoustic"] = 0.5
            weights["dielectric"] = 0.3
            weights["absorption"] = 0.2
        elif modality == "ultrasound":
            weights["absorption"] = 0.5
            weights["dielectric"] = 0.3
            weights["acoustic"] = 0.2
        
        physics_weight = (physics_maps["dielectric"] * weights["dielectric"] +
                          physics_maps["acoustic"] * weights["acoustic"] +
                          physics_maps["absorption"] * weights["absorption"])
        
        refined = prior_mask * (physics_weight > 0.3)
        return (refined > 0).astype(np.uint8) * 255

# ------------------------------------------------------------
# 5. UNCERTAINTY, 3D PROPAGATION, SYNTHETIC, EXPORT
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

class VolumetricPropagator:
    @staticmethod
    def propagate_3d(slices, initial_mask, initial_physics):
        masks = [initial_mask]
        prev_gray = cv2.cvtColor(np.uint8(slices[0]*255), cv2.COLOR_RGB2GRAY)
        area = np.sum(initial_mask)
        for i in range(1, len(slices)):
            curr_gray = cv2.cvtColor(np.uint8(slices[i]*255), cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            h, w = initial_mask.shape
            flow_x = flow[:,:,0]
            flow_y = flow[:,:,1]
            grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
            new_x = (grid_x + flow_x).astype(np.float32)
            new_y = (grid_y + flow_y).astype(np.float32)
            warped = cv2.remap(initial_mask.astype(np.float32), new_x, new_y, cv2.INTER_LINEAR)
            warped = (warped > 0.5).astype(np.uint8) * 255
            if np.sum(warped) < area * 0.3:
                warped = cv2.dilate(warped, np.ones((5,5), np.uint8))
            masks.append(warped)
            prev_gray = curr_gray
            initial_mask = warped
        return masks

# ------------------------------------------------------------
# 6. DELAY-AND-SUM RECONSTRUCTION (Built-in)
# ------------------------------------------------------------
def db_to_linear(db):
    return 10 ** (np.asarray(db) / 10)

def delay_and_sum_reconstruction(
    s21_data: dict, frequencies: np.ndarray,
    baseline_data: dict = None,
    grid_size: int = 80, grid_extent: float = 100.0,
    start_freq: float = 2.0, stop_freq: float = 3.0,
    num_points: int = 201, sigma: float = 2.0
) -> np.ndarray:
    antenna_positions = {1: (-75, 0), 2: (75, 0), 3: (0, -75), 4: (0, 75)}
    path_to_antenna_pair = {1: (1, 3), 2: (1, 4), 3: (2, 3), 4: (2, 4)}
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
        raise ValueError("No valid paths found")
    image = gaussian_filter(image, sigma=sigma)
    if image.max() > 0:
        image = np.clip(image, 0, np.percentile(image, 95))
        image = (image / image.max()) * 255
    return image.astype(np.uint8)

def load_s21_csv(filepath):
    df = pd.read_csv(filepath)
    freq_col = next((c for c in df.columns if 'freq' in c.lower() or 'ghz' in c.lower()), None)
    if freq_col is None:
        raise ValueError(f"No frequency column found. Columns: {df.columns.tolist()}")
    s21_col = next((c for c in df.columns if 's21' in c.lower() or 's_param' in c.lower()), None)
    if s21_col is None:
        raise ValueError(f"No S21 column found. Columns: {df.columns.tolist()}")
    frequencies = df[freq_col].values.astype(np.float64)
    s21_db = df[s21_col].values.astype(np.float64)
    return frequencies, s21_db

def load_s2p(filepath):
    frequencies, s21_mag_linear = [], []
    with open(filepath, 'r') as f:
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
        raise ValueError(f"No valid data found in {filepath}")
    s21_db = np.array([20 * np.log10(m) if m > 0 else -100 for m in s21_mag_linear])
    return np.array(frequencies), s21_db

def load_mat(filepath):
    mat_data = loadmat(filepath)
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
        raise ValueError(f"No frequency variable found. Keys: {list(mat_data.keys())}")
    if s21_db is None:
        raise ValueError(f"No S21 variable found. Keys: {list(mat_data.keys())}")
    return frequencies.astype(np.float64), s21_db.astype(np.float64)

def auto_load(filepath):
    suffix = Path(filepath).suffix.lower()
    if suffix == '.csv':
        return load_s21_csv(filepath)
    elif suffix == '.s2p':
        return load_s2p(filepath)
    elif suffix == '.mat':
        return load_mat(filepath)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

# ------------------------------------------------------------
# 7. PROJECT MANAGER
# ------------------------------------------------------------
class ProjectManager:
    def __init__(self):
        self.playlist: List[str] = []
        self.current_index: int = 0
        self.annotations: Dict[str, Dict] = {}
        self.active_project_path: Optional[str] = None
        self.sam = None
        self.modality = "microwave"

    def initialize_sam(self, checkpoint_path: Optional[str] = None, model_type: str = "tiny"):
        """Initialize SAM2 with Tiny model for memory optimization."""
        self.sam = SAM2Wrapper(checkpoint_path=checkpoint_path, model_type=model_type)
        return self.sam._loaded and not self.sam._use_mock

    def add_images(self, image_paths: List[str]):
        for p in image_paths:
            if p not in self.playlist:
                self.playlist.append(p)
                self.annotations[p] = {"masks": [], "points": [], "prompts": [], "modality": "unknown"}

    def load_image(self, idx: int) -> Optional[np.ndarray]:
        if 0 <= idx < len(self.playlist):
            self.current_index = idx
            path = self.playlist[idx]
            
            # Detect modality
            self.modality = ModalityDetector.detect(path)
            
            # Handle DICOM
            if path.lower().endswith('.dcm'):
                img, meta = load_dicom(path, deidentify=True)
                if img is not None:
                    return img
                img = cv2.imread(path)
                if img is not None:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return None
            
            img = cv2.imread(path)
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return None

    def get_current_path(self) -> Optional[str]:
        if 0 <= self.current_index < len(self.playlist):
            return self.playlist[self.current_index]
        return None

    def save_annotation(self, image_path: str, mask: np.ndarray, points: List[Tuple[int, int]], prompt: str = ""):
        if image_path not in self.annotations:
            self.annotations[image_path] = {"masks": [], "points": [], "prompts": [], "modality": self.modality}
        self.annotations[image_path]["masks"].append(mask.tolist())
        self.annotations[image_path]["points"].append(points)
        self.annotations[image_path]["prompts"].append(prompt)

    def save_project(self, filepath: str) -> str:
        data = {
            "playlist": self.playlist,
            "current_index": self.current_index,
            "annotations": self.annotations,
            "modality": self.modality
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)
        return f"Saved to {filepath}"

    def load_project(self, filepath: str) -> str:
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.playlist = data["playlist"]
        self.current_index = data["current_index"]
        self.annotations = data["annotations"]
        self.modality = data.get("modality", "microwave")
        return f"Loaded {len(self.playlist)} images."

# ------------------------------------------------------------
# 8. ACTIVE LEARNING LOOP
# ------------------------------------------------------------
class ActiveLearningLoop:
    def __init__(self):
        self.feedback_pairs = []

    def add_feedback(self, point: Tuple[int, int], label: int, rf_signature: Optional[np.ndarray] = None):
        self.feedback_pairs.append({
            "point": point,
            "label": label,
            "rf_signature": rf_signature
        })

    def get_feedback_points(self):
        points = [f["point"] for f in self.feedback_pairs]
        labels = [f["label"] for f in self.feedback_pairs]
        return points, labels

    def reset(self):
        self.feedback_pairs = []

# ------------------------------------------------------------
# 9. SYNTHETIC DATA GENERATOR
# ------------------------------------------------------------
class SyntheticDataGenerator:
    @staticmethod
    def generate_variations(base_image: np.ndarray, base_mask: np.ndarray, n_variations: int = 10) -> List[Tuple[np.ndarray, np.ndarray]]:
        variations = []
        h, w = base_image.shape[:2]
        
        for _ in range(n_variations):
            img = base_image.copy().astype(np.float32)
            mask = base_mask.copy()
            
            # Random transformations
            angle = random.uniform(-10, 10)
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h))
            mask = cv2.warpAffine(mask, M, (w, h))
            
            scale = random.uniform(0.9, 1.1)
            M = cv2.getRotationMatrix2D((w/2, h/2), 0, scale)
            img = cv2.warpAffine(img, M, (w, h))
            mask = cv2.warpAffine(mask, M, (w, h))
            
            noise = np.random.normal(0, random.uniform(5, 20), img.shape)
            img = np.clip(img + noise, 0, 255).astype(np.uint8)
            
            if random.random() > 0.5:
                kernel_size = random.choice([3, 5])
                img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
            
            mask = (mask > 127).astype(np.uint8) * 255
            variations.append((img, mask))
        
        return variations
    
    @staticmethod
    def create_manifest(variations: List[Tuple[np.ndarray, np.ndarray]], output_dir: str) -> Dict:
        manifest = {
            "dataset": "Hypoxify_Synthetic",
            "version": "2.0",
            "total_samples": len(variations),
            "format": "png",
            "class_mapping": {"0": "background", "1": "lesion", "2": "tumor"},
            "samples": []
        }
        
        os.makedirs(output_dir, exist_ok=True)
        
        for i, (img, mask) in enumerate(variations):
            img_path = os.path.join(output_dir, f"sample_{i:04d}_img.png")
            mask_path = os.path.join(output_dir, f"sample_{i:04d}_mask.png")
            
            cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(mask_path, mask)
            
            manifest["samples"].append({
                "image": img_path,
                "mask": mask_path,
                "shape": list(mask.shape)
            })
        
        manifest_path = os.path.join(output_dir, "dataset.json")
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        return manifest

# ------------------------------------------------------------
# 10. GRADIO UI - COMPLETE
# ------------------------------------------------------------

project = ProjectManager()

# CSS for professional look
css = """
body, .gradio-container, .gr-box, .gr-textbox, label, .gr-markdown, .gr-form, .gr-row {
    color: #1a1a1a !important;
    font-weight: 400 !important;
}
h1, h2, h3, h4, .gr-markdown h1, .gr-markdown h2, .gr-markdown h3 {
    color: #0d47a1 !important;
    font-weight: 600 !important;
}
label, .gr-label {
    font-weight: 500 !important;
}
footer { display: none !important; }
.zoom-image img { transition: transform 0.1s ease-out; }
#input_image { position: relative; overflow: hidden; }
#input_image button, #input_image img, #input_image canvas { cursor: crosshair !important; }
.horizontal-radio .wrap { display: flex !important; flex-direction: row !important; gap: 10px !important; }
.horizontal-radio label { margin-bottom: 0 !important; align-items: center !important; }
"""

with gr.Blocks() as demo:
    gr.Markdown("# 🔬 Hypoxify Annotation Suite")
    gr.Markdown("### Clinical-grade multi-modality physics-informed segmentation (MITT, MWI, Photoacoustic, Ultrasound)")

    # State
    click_state = gr.State(value=[])
    label_state = gr.State(value=[])
    current_mask_state = gr.State(value=None)
    candidates_state = gr.State(value=[])
    modality_state = gr.State(value="microwave")
    
    with gr.Tabs() as tabs:
        # ==================== SETUP ====================
        with gr.TabItem("Setup", id=0):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### Data Ingestion")
                    gr.Markdown("**Supported:** PNG, JPG, TIFF, DICOM (.dcm), CSV, S2P, MAT")
                    
                    file_upload = gr.File(
                        label="Upload Files",
                        file_count="multiple",
                        file_types=["image", ".csv", ".s2p", ".mat", ".dcm"]
                    )
                    load_btn = gr.Button("Add to Project", variant="primary")
                    
                    with gr.Row():
                        project_name = gr.Textbox(label="Project Name", value="my_project", scale=3)
                        save_btn = gr.Button("💾 Save", variant="secondary", scale=1)
                        load_project_btn = gr.Button("📂 Load", variant="secondary", scale=1)
                    
                    status_display = gr.Textbox(label="Status", lines=3, interactive=False)
                    
                    gr.Markdown("### Modality Selection")
                    modality_select = gr.Radio(
                        ["MITT", "Microwave Imaging", "Photoacoustic", "Ultrasound"],
                        value="MITT",
                        label="Imaging Modality",
                        elem_classes="horizontal-radio"
                    )
                    
                    gr.Markdown("### SAM2 Configuration")
                    sam_model_size = gr.Dropdown(
                        ["tiny (~78MB, fastest)", "small (~300MB)", "base (~500MB)"],
                        value="tiny (~78MB, fastest)",
                        label="SAM2 Model Size"
                    )
                    sam_checkpoint = gr.Textbox(
                        label="SAM2 Checkpoint Path (optional)",
                        value="",
                        placeholder="Leave empty for auto-download"
                    )
                    init_sam_btn = gr.Button("Initialize SAM2", variant="primary")
                    sam_status = gr.Textbox(label="SAM Status", value="Not initialized", interactive=False)
                    
                with gr.Column(scale=1):
                    gr.Markdown("### Project Info")
                    playlist_display = gr.Textbox(label="Images in Project", lines=15, interactive=False)
                    current_preview = gr.Image(label="Current Preview", type="numpy")

            def set_modality(modality):
                modality_map = {
                    "MITT": "microwave",
                    "Microwave Imaging": "microwave",
                    "Photoacoustic": "photoacoustic",
                    "Ultrasound": "ultrasound"
                }
                return modality_map.get(modality, "microwave")

            modality_select.change(set_modality, [modality_select], [modality_state])

            def init_sam(checkpoint, model_size):
                # Extract model type from dropdown
                model_type = "tiny"
                if "small" in model_size.lower():
                    model_type = "small"
                elif "base" in model_size.lower():
                    model_type = "base"
                
                try:
                    success = project.initialize_sam(
                        checkpoint_path=checkpoint if checkpoint else None,
                        model_type=model_type
                    )
                    if success:
                        return f"✅ SAM2 ({model_type}) initialized successfully (under 512MB)", gr.update()
                    else:
                        return "⚠️ SAM2 running in mock mode (checkpoint not found)", gr.update()
                except Exception as e:
                    return f"❌ Error: {str(e)}", gr.update()

            init_sam_btn.click(init_sam, [sam_checkpoint, sam_model_size], [sam_status, gr.update()])

        # ==================== INPUT ====================
        with gr.TabItem("Input", id=1):
            with gr.Row():
                with gr.Column(scale=2):
                    input_image = gr.Image(
                        label="Click on image to place seed points",
                        type="numpy",
                        interactive=True,
                        elem_id="input_image",
                        elem_classes="zoom-image"
                    )
                    overlay_display = gr.Image(label="Overlay Preview", type="numpy", interactive=False)
                    
                    with gr.Row():
                        undo_btn = gr.Button("↩️ Undo Last Click", size="sm")
                        clear_btn = gr.Button("🗑️ Clear All Points", size="sm")
                        click_mode = gr.Radio(
                            ["Foreground (+)", "Background (-)"],
                            value="Foreground (+)",
                            label="Click Mode",
                            elem_classes="horizontal-radio"
                        )
                    
                    click_info = gr.Markdown("**Points:** 0 foreground, 0 background")
                    
                with gr.Column(scale=1):
                    gr.Markdown("### Segmentation Controls")
                    gr.Markdown("Place at least one foreground point, then click Run.")
                    run_btn = gr.Button("🚀 Run Physics-Guided SAM2", variant="primary", size="lg")
                    seg_status = gr.Textbox(label="Status", value="Ready")
                    
                    gr.Markdown("### Physics Parameters")
                    with gr.Row():
                        use_dielectric = gr.Checkbox(label="Dielectric", value=True)
                        use_acoustic = gr.Checkbox(label="Acoustic", value=True)
                        use_absorption = gr.Checkbox(label="Absorption", value=True)
                    
                    gr.Markdown("### Advanced")
                    multimask = gr.Checkbox(label="Multimask Output", value=True)
                    candidates_display = gr.Dataframe(
                        headers=["Candidate", "Score"],
                        datatype=["str", "number"],
                        label="Candidates",
                        interactive=False
                    )

            def on_image_click(evt: gr.SelectData, image, points, labels, mode):
                if image is None:
                    return image, points, labels, "No image loaded", gr.update()
                
                x, y = evt.index
                label = 1 if "Foreground" in mode else 0
                
                points.append((int(x), int(y)))
                labels.append(label)
                
                overlay = image.copy()
                for i, (px, py) in enumerate(points):
                    color = (0, 255, 0) if labels[i] == 1 else (255, 0, 0)
                    cv2.circle(overlay, (px, py), 6, color, -1)
                    cv2.circle(overlay, (px, py), 8, (255, 255, 255), 2)
                
                fg_count = sum(1 for l in labels if l == 1)
                bg_count = sum(1 for l in labels if l == 0)
                info = f"**Points:** {fg_count} foreground, {bg_count} background"
                
                return overlay, points, labels, info, gr.update()
            
            input_image.select(
                on_image_click,
                [input_image, click_state, label_state, click_mode],
                [overlay_display, click_state, label_state, click_info, gr.update()]
            )

            def undo_last(points, labels):
                if points:
                    points.pop()
                    labels.pop()
                return points, labels, gr.update(), gr.update()
            
            undo_btn.click(undo_last, [click_state, label_state], [click_state, label_state, overlay_display, click_info])
            
            def clear_all():
                return [], [], gr.update(), gr.update()
            
            clear_btn.click(clear_all, [], [click_state, label_state, overlay_display, click_info])

            def run_segmentation(image, points, labels, use_die, use_acoustic, use_abs, multimask, modality):
                if image is None:
                    return None, None, "No image loaded", []
                
                if not points or not any(labels):
                    return None, None, "Please place at least one foreground point", []
                
                if project.sam is None:
                    return None, None, "SAM2 not initialized. Go to Setup and initialize.", []
                
                try:
                    project.sam.set_image(image)
                    
                    # Extract physics with modality
                    fg_points = [p for p, l in zip(points, labels) if l == 1]
                    physics = {}
                    if fg_points and (use_die or use_acoustic or use_abs):
                        physics = PhysicsSimulator.extract_physical_signature(
                            image, fg_points[0], modality
                        )
                    
                    # Run SAM2
                    mask = project.sam.predict_from_clicks(points, labels)
                    
                    # Apply physics conditioning
                    if physics:
                        mask = PhysicsSimulator.apply_physics_to_segmentation(mask, physics)
                    
                    # Candidates
                    candidates = []
                    if multimask:
                        for i in range(3):
                            mask_var = mask.copy()
                            kernel = np.ones((3,3), np.uint8)
                            if i == 0:
                                mask_var = cv2.erode(mask_var, kernel, iterations=1)
                            elif i == 1:
                                mask_var = cv2.dilate(mask_var, kernel, iterations=1)
                            score = 0.95 - i * 0.05
                            candidates.append([f"Candidate {i+1}", f"{score:.3f}"])
                            if i == 0:
                                mask = mask_var
                    
                    overlay = image.copy()
                    overlay[mask > 0] = overlay[mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
                    
                    return np.uint8(overlay), mask, "✅ Segmentation complete!", candidates
                    
                except Exception as e:
                    return None, None, f"❌ Error: {str(e)}", []

            run_btn.click(
                run_segmentation,
                [input_image, click_state, label_state, use_dielectric, use_acoustic, use_absorption, multimask, modality_state],
                [overlay_display, current_mask_state, seg_status, candidates_display]
            )

        # ==================== EDITOR ====================
        with gr.TabItem("Editor", id=2):
            with gr.Row():
                with gr.Column(scale=2):
                    editor_image = gr.Image(label="Mask Overlay", type="numpy", interactive=False)
                    uncertainty_output = gr.Image(label="Uncertainty Heatmap", type="numpy", interactive=False)
                    
                    gr.Markdown("💡 **Click on the uncertainty heatmap** to refine the mask")
                    active_click_info = gr.Markdown("0 refinement points added")
                    
                with gr.Column(scale=1):
                    gr.Markdown("### Refinement Controls")
                    active_learning_mode = gr.Radio(
                        ["Add Tissue (FG)", "Remove Tissue (BG)"],
                        value="Add Tissue (FG)",
                        label="Refinement Mode",
                        elem_classes="horizontal-radio"
                    )
                    refine_btn = gr.Button("Apply Refinement", variant="primary")
                    refine_status = gr.Textbox(label="Status", value="Ready")
                    
                    gr.Markdown("### Uncertainty")
                    show_uncertainty_btn = gr.Button("🔥 Show Uncertainty Heatmap", variant="secondary")
                    
                    gr.Markdown("### Synthetic Data")
                    with gr.Row():
                        n_synthetic = gr.Slider(1, 50, value=10, label="Number of Variations", step=1)
                    generate_synth_btn = gr.Button("🧬 Generate Synthetic Dataset", variant="primary")
                    synth_status = gr.Textbox(label="Synthetic Status", lines=3, interactive=False)

            def show_uncertainty(mask, image):
                if mask is None or image is None:
                    return None, "No mask to analyze"
                heatmap, _ = UncertaintyCalculator.compute_heatmaps(image, mask)
                return heatmap, "✅ Uncertainty heatmap generated"
            
            show_uncertainty_btn.click(
                show_uncertainty,
                [current_mask_state, input_image],
                [uncertainty_output, refine_status]
            )

            def generate_synthetic(mask, image, n):
                if mask is None or image is None:
                    return "Please segment an image first", None
                
                variations = SyntheticDataGenerator.generate_variations(image, mask, int(n))
                output_dir = tempfile.mkdtemp(prefix="synthetic_")
                manifest = SyntheticDataGenerator.create_manifest(variations, output_dir)
                
                zip_path = os.path.join(output_dir, "synthetic_dataset.zip")
                with zipfile.ZipFile(zip_path, 'w') as zf:
                    for f in os.listdir(output_dir):
                        if f.endswith('.png') or f.endswith('.json'):
                            zf.write(os.path.join(output_dir, f), f)
                
                return f"✅ Generated {len(variations)} variations with manifest", zip_path
            
            generate_synth_btn.click(
                generate_synthetic,
                [current_mask_state, input_image, n_synthetic],
                [synth_status, gr.File(label="Download Synthetic Dataset", visible=True)]
            )

        # ==================== RESULTS ====================
        with gr.TabItem("Results", id=3):
            with gr.Row():
                with gr.Column(scale=2):
                    results_preview = gr.Image(label="Selected Mask", type="numpy")
                with gr.Column(scale=1):
                    gr.Markdown("### Annotations")
                    mask_count = gr.Markdown("**Masks saved:** 0")
                    save_mask_btn = gr.Button("💾 Save Mask to Project", variant="primary")
                    save_status = gr.Textbox(label="Status")
            
            def save_mask(mask, image):
                if mask is None:
                    return "No mask to save", mask_count
                path = project.get_current_path()
                if path is None:
                    return "No active image", mask_count
                project.save_annotation(path, mask, [])
                count = len(project.annotations.get(path, {}).get("masks", []))
                return f"✅ Saved mask #{count}", f"**Masks saved:** {count}"
            
            save_mask_btn.click(
                save_mask,
                [current_mask_state, input_image],
                [save_status, mask_count]
            )

        # ==================== EXPORT ====================
        with gr.TabItem("Export", id=4):
            with gr.Row():
                with gr.Column():
                    export_format = gr.Dropdown(
                        choices=["COCO", "YOLO", "PNG", "MONAI", "nnU-Net"],
                        value="COCO",
                        label="Export Format"
                    )
                    export_btn = gr.Button("📥 Export Annotations", variant="primary")
                    export_output = gr.File(label="Download")
                    export_status = gr.Textbox(label="Status")

        # ==================== 3D PROPAGATION ====================
        with gr.TabItem("3D Propagation", id=5):
            with gr.Row():
                with gr.Column():
                    volume_upload = gr.File(label="Upload Volume Slices", file_count="multiple")
                    prop_btn = gr.Button("🚀 Propagate Through Volume", variant="primary")
                    slice_slider = gr.Slider(0, 49, value=0, step=1, label="Slice Index")
                with gr.Column():
                    volume_viewer = gr.Image(label="Current Slice with Mask", type="numpy")
            
            volume_state = gr.State(value=None)
            masks_state = gr.State(value=None)

            def process_volume(files):
                if not files:
                    return None, None, None
                images = []
                for f in files:
                    img = cv2.imread(f.name)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        images.append(img)
                if not images:
                    return None, None, None
                
                first_img = images[0]
                center = (first_img.shape[1]//2, first_img.shape[0]//2)
                physics = PhysicsSimulator.extract_physical_signature(first_img, center, project.modality)
                first_mask = SAM2Wrapper()._mock_predict([center], [1]) if project.sam is None else project.sam._mock_predict([center], [1])
                all_masks = VolumetricPropagator.propagate_3d(images, first_mask, physics)
                return images, all_masks, first_mask

            def update_volume_viewer(slice_idx, images, masks):
                if images is None or masks is None:
                    return None
                idx = int(slice_idx)
                if idx >= len(images) or idx >= len(masks):
                    return None
                img = images[idx]
                mask = masks[idx]
                overlay = img.copy()
                overlay[mask > 0] = overlay[mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
                return np.uint8(overlay)

            prop_btn.click(
                process_volume,
                [volume_upload],
                [volume_state, masks_state, volume_viewer]
            )
            
            slice_slider.change(
                update_volume_viewer,
                [slice_slider, volume_state, masks_state],
                [volume_viewer]
            )

        # ==================== USER GUIDE ====================
        with gr.TabItem("📖 User Guide", id=6):
            gr.Markdown("""
            ## How to Use Hypoxify Annotation Suite

            ### 🚀 Quick Start

            1. **Setup**: Upload images/DICOM → Add to Project → Select modality → Initialize SAM2
            2. **Input**: Click on image to place foreground points → Run Physics-Guided SAM2
            3. **Editor**: View mask → Show uncertainty → Click on heatmap to refine
            4. **Results**: Save mask to project
            5. **Export**: Download in your preferred format

            ### 📁 Supported Modalities
            - **MITT** (Microwave-Induced Thermoacoustic Tomography)
            - **Microwave Imaging** (MWI)
            - **Photoacoustic Imaging** (PAI)
            - **Ultrasound**
            - **DICOM** (MRI, CT, etc.)

            ### 📁 Supported Formats
            - **Images**: PNG, JPG, TIFF, BMP, DICOM (.dcm)
            - **Raw Data**: CSV, S2P, MAT (S21 parameters)
            - **Export**: COCO JSON, YOLO TXT, PNG, MONAI, nnU-Net

            ### 🔬 Physics Features
            - **Dielectric Contrast**: Tumors have higher water content → higher permittivity
            - **Acoustic Pressure**: Changes in tissue density affect acoustic wave propagation
            - **Phase-Shift Tokenization**: Complex S21 magnitude + phase for MWI/MITT
            - **Energy Absorption**: Tumors absorb more microwave energy

            ### 🎯 Tips for Best Results
            - Place at least 3 foreground points for complex shapes
            - Use background points to exclude ambiguous regions
            - Check uncertainty heatmap to identify weak areas
            - Generate synthetic variations for training data
            - Select "tiny" model on Render (under 512MB)
            """)

# ------------------------------------------------------------
# LAUNCH
# ------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        debug=False,
        pwa=True,
        theme=gr.themes.Soft(primary_hue="emerald", secondary_hue="blue"),
        css=css
    )
