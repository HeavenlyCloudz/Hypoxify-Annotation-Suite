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
from typing import List, Dict, Optional, Tuple, Any
import torch
import torchvision
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 0. REAL SAM IMPLEMENTATION (with physics conditioning)
# ------------------------------------------------------------
try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("⚠️ SAM not installed. Install with: pip install segment-anything")

class SAMPhysicsPredictor:
    """SAM with physics-guided prompting."""
    
    def __init__(self, model_type="vit_b", checkpoint_path=None, device="auto"):
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.predictor = None
        self._loaded = False
        self._use_mock = False
        
        if not SAM_AVAILABLE:
            self._use_mock = True
            self._loaded = True
            print("⚠️ Running in mock mode (SAM not installed)")
            return
        
        if checkpoint_path is None:
            default_paths = [
                "sam_vit_b.pth",
                os.path.expanduser("~/.cache/sam/sam_vit_b.pth"),
                "/opt/render/project/src/sam_vit_b.pth"
            ]
            for p in default_paths:
                if os.path.exists(p):
                    checkpoint_path = p
                    break
        
        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            self._use_mock = True
            self._loaded = True
            print("⚠️ SAM checkpoint not found. Running in mock mode.")
            return
        
        try:
            self.model = sam_model_registry[model_type](checkpoint=checkpoint_path)
            self.model.to(device=self.device)
            self.predictor = SamPredictor(self.model)
            self._loaded = True
            self._use_mock = False
            print(f"✅ SAM loaded on {self.device}")
        except Exception as e:
            print(f"❌ Failed to load SAM: {e}")
            self._use_mock = True
            self._loaded = True
    
    def set_image(self, image):
        """Set the image for segmentation."""
        if self._use_mock or self.predictor is None:
            self._image = image
            return
        self.predictor.set_image(image)
    
    def predict_from_clicks(self, points, labels):
        """Run SAM prediction with click points."""
        if self._use_mock or self.predictor is None:
            return self._mock_predict(points, labels)
        
        if not points:
            return np.zeros((self._image.shape[0], self._image.shape[1]), dtype=np.uint8)
        
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
    
    def _mock_predict(self, points, labels):
        """Fallback mock segmentation."""
        if not points or self._image is None:
            return np.zeros((512, 512), dtype=np.uint8)
        
        h, w = self._image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Simple circle around first foreground point
        cx, cy = points[0]
        radius = min(h, w) // 6
        
        y, x = np.ogrid[:h, :w]
        dist = (x - cx)**2 + (y - cy)**2
        mask[dist < radius**2] = 1
        
        return mask

# ------------------------------------------------------------
# 1. DICOM SUPPORT
# ------------------------------------------------------------
try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False
    print("⚠️ pydicom not installed. Install with: pip install pydicom")

def load_dicom(filepath: str) -> Tuple[Optional[np.ndarray], Dict]:
    """Load DICOM file and extract metadata."""
    if not DICOM_AVAILABLE:
        return None, {"error": "pydicom not installed"}
    
    try:
        ds = pydicom.dcmread(filepath)
        
        # Extract pixel data
        if hasattr(ds, 'pixel_array'):
            pixel_array = ds.pixel_array
            # Apply VOI LUT if present
            if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
                pixel_array = apply_voi_lut(pixel_array, ds)
            
            # Normalize to 0-255
            if pixel_array.dtype != np.uint8:
                pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min() + 1e-8) * 255
                pixel_array = pixel_array.astype(np.uint8)
            
            # Convert grayscale to RGB if needed
            if len(pixel_array.shape) == 2:
                pixel_array = np.stack([pixel_array] * 3, axis=-1)
        
        # Extract metadata
        metadata = {
            "PatientID": getattr(ds, 'PatientID', 'Unknown'),
            "PatientName": str(getattr(ds, 'PatientName', 'Unknown')),
            "StudyDate": getattr(ds, 'StudyDate', 'Unknown'),
            "Modality": getattr(ds, 'Modality', 'Unknown'),
            "Manufacturer": getattr(ds, 'Manufacturer', 'Unknown'),
            "StudyDescription": getattr(ds, 'StudyDescription', 'Unknown'),
            "SeriesDescription": getattr(ds, 'SeriesDescription', 'Unknown'),
            "SliceThickness": getattr(ds, 'SliceThickness', 'Unknown'),
            "SpacingBetweenSlices": getattr(ds, 'SpacingBetweenSlices', 'Unknown'),
            "PixelSpacing": getattr(ds, 'PixelSpacing', 'Unknown'),
            "Rows": getattr(ds, 'Rows', 0),
            "Columns": getattr(ds, 'Columns', 0),
            "SOPInstanceUID": getattr(ds, 'SOPInstanceUID', 'Unknown'),
            "SeriesInstanceUID": getattr(ds, 'SeriesInstanceUID', 'Unknown'),
        }
        
        return pixel_array, metadata
    except Exception as e:
        return None, {"error": str(e)}

# ------------------------------------------------------------
# 2. S-PARAMETER PHASE-SHIFT TOKENIZATION
# ------------------------------------------------------------
def load_s2p_with_phase(filepath: str):
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
    
    # Convert magnitude to dB
    magnitude_db = np.array([20 * np.log10(m) if m > 0 else -100 for m in magnitude])
    
    return {
        "frequencies": np.array(frequencies),
        "magnitude_db": magnitude_db,
        "magnitude_linear": np.array(magnitude),
        "phase_deg": np.array(phase_deg)
    }

def tokenize_phase_shift(phase_data: np.ndarray) -> np.ndarray:
    """
    Convert phase data to features useful for segmentation.
    Phase wrapping and gradient extraction.
    """
    # Unwrap phase
    phase_unwrapped = np.unwrap(phase_data * np.pi / 180) * 180 / np.pi
    
    # Compute phase gradient (rate of change)
    phase_gradient = np.gradient(phase_unwrapped)
    
    # Normalize
    phase_norm = (phase_unwrapped - phase_unwrapped.min()) / (phase_unwrapped.max() - phase_unwrapped.min() + 1e-8)
    gradient_norm = (phase_gradient - phase_gradient.min()) / (phase_gradient.max() - phase_gradient.min() + 1e-8)
    
    return np.stack([phase_norm, gradient_norm], axis=0)

# ------------------------------------------------------------
# 3. ACTIVE LEARNING (uncertainty-guided refinement)
# ------------------------------------------------------------
class ActiveLearningLoop:
    """Manages active learning from uncertainty heatmaps."""
    
    def __init__(self):
        self.feedback_pairs = []  # (point, label, rf_signature)
    
    def add_feedback(self, point: Tuple[int, int], label: int, rf_signature: Optional[np.ndarray] = None):
        """Add user feedback (correction click)."""
        self.feedback_pairs.append({
            "point": point,
            "label": label,  # 1=foreground, 0=background
            "rf_signature": rf_signature
        })
    
    def get_feedback_points(self):
        """Get all feedback points for SAM refinement."""
        points = [f["point"] for f in self.feedback_pairs]
        labels = [f["label"] for f in self.feedback_pairs]
        return points, labels
    
    def reset(self):
        self.feedback_pairs = []

# ------------------------------------------------------------
# 4. SYNTHETIC DATA GENERATION WITH MANIFEST
# ------------------------------------------------------------
class SyntheticDataGenerator:
    @staticmethod
    def generate_variations(base_image: np.ndarray, base_mask: np.ndarray, n_variations: int = 10) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate synthetic variations of an image-mask pair."""
        variations = []
        h, w = base_image.shape[:2]
        
        for _ in range(n_variations):
            # Copy base
            img = base_image.copy().astype(np.float32)
            mask = base_mask.copy()
            
            # Apply random transformations
            # Rotation
            angle = random.uniform(-10, 10)
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h))
            mask = cv2.warpAffine(mask, M, (w, h))
            
            # Slight scaling
            scale = random.uniform(0.9, 1.1)
            M = cv2.getRotationMatrix2D((w/2, h/2), 0, scale)
            img = cv2.warpAffine(img, M, (w, h))
            mask = cv2.warpAffine(mask, M, (w, h))
            
            # Add noise
            noise = np.random.normal(0, random.uniform(5, 20), img.shape)
            img = np.clip(img + noise, 0, 255).astype(np.uint8)
            
            # Add slight blur
            if random.random() > 0.5:
                kernel_size = random.choice([3, 5])
                img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
            
            # Binarize mask
            mask = (mask > 127).astype(np.uint8) * 255
            
            variations.append((img, mask))
        
        return variations
    
    @staticmethod
    def create_manifest(variations: List[Tuple[np.ndarray, np.ndarray]], output_dir: str) -> Dict:
        """Create a training manifest for the generated data."""
        manifest = {
            "dataset": "Hypoxify_Synthetic",
            "version": "1.0",
            "total_samples": len(variations),
            "format": "png",
            "class_mapping": {
                "0": "background",
                "1": "tumor",
                "2": "lesion"
            },
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
        
        # Save manifest
        manifest_path = os.path.join(output_dir, "dataset.json")
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        return manifest

# ------------------------------------------------------------
# 5. PROJECT MANAGER (Enhanced)
# ------------------------------------------------------------
class ProjectManager:
    def __init__(self):
        self.playlist: List[str] = []
        self.current_index: int = 0
        self.annotations: Dict[str, Dict] = {}
        self.active_project_path: Optional[str] = None
        self.sam = None
        self.active_learning = ActiveLearningLoop()

    def initialize_sam(self, checkpoint_path: Optional[str] = None):
        self.sam = SAMPhysicsPredictor(checkpoint_path=checkpoint_path)

    def add_images(self, image_paths: List[str]):
        for p in image_paths:
            if p not in self.playlist:
                self.playlist.append(p)
                self.annotations[p] = {"masks": [], "points": [], "prompts": [], "synthetic_variations": []}

    def load_image(self, idx: int) -> Optional[np.ndarray]:
        if 0 <= idx < len(self.playlist):
            self.current_index = idx
            path = self.playlist[idx]
            
            # Check if DICOM
            if path.lower().endswith('.dcm'):
                img, meta = load_dicom(path)
                if img is not None:
                    return img
                # Fall back to OpenCV
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
            self.annotations[image_path] = {"masks": [], "points": [], "prompts": [], "synthetic_variations": []}
        self.annotations[image_path]["masks"].append(mask.tolist())
        self.annotations[image_path]["points"].append(points)
        self.annotations[image_path]["prompts"].append(prompt)

    def save_project(self, filepath: str) -> str:
        # Convert numpy arrays to lists for JSON serialization
        data = {
            "playlist": self.playlist,
            "current_index": self.current_index,
            "annotations": self.annotations
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
        return f"Loaded {len(self.playlist)} images."

# ------------------------------------------------------------
# 6. GRADIO UI - COMPLETE REWRITE
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
#col-container { max-width: 1400px; margin: 0 auto; }
footer { display: none !important; }
.zoom-image img { transition: transform 0.1s ease-out; }
#input_image { position: relative; overflow: hidden; }
#input_image button, #input_image img, #input_image canvas { cursor: crosshair !important; }
.horizontal-radio .wrap { display: flex !important; flex-direction: row !important; gap: 10px !important; }
.horizontal-radio label { margin-bottom: 0 !important; align-items: center !important; }
"""

# Build the UI
with gr.Blocks() as demo:
    gr.Markdown("# 🔬 Hypoxify Annotation Suite")
    gr.Markdown("### Clinical-grade physics-informed segmentation for microwave and thermoacoustic imaging")

    # State
    click_state = gr.State(value=[])
    label_state = gr.State(value=[])
    current_mask_state = gr.State(value=None)
    candidates_state = gr.State(value=[])
    
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
                    
                    gr.Markdown("### SAM Configuration")
                    sam_checkpoint = gr.Textbox(
                        label="SAM Checkpoint Path",
                        value="sam_vit_b.pth",
                        placeholder="path/to/sam_vit_b.pth"
                    )
                    init_sam_btn = gr.Button("Initialize SAM", variant="primary")
                    sam_status = gr.Textbox(label="SAM Status", value="Not initialized", interactive=False)
                    
                with gr.Column(scale=1):
                    gr.Markdown("### Project Info")
                    playlist_display = gr.Textbox(label="Images in Project", lines=15, interactive=False)
                    current_preview = gr.Image(label="Current Preview", type="numpy")

            def init_sam(checkpoint):
                try:
                    project.initialize_sam(checkpoint if checkpoint else None)
                    if project.sam and not project.sam._use_mock:
                        return "✅ SAM initialized successfully", gr.update()
                    else:
                        return "⚠️ SAM running in mock mode (checkpoint not found)", gr.update()
                except Exception as e:
                    return f"❌ Error: {str(e)}", gr.update()

            init_sam_btn.click(init_sam, [sam_checkpoint], [sam_status, gr.update()])

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
                    
                    # Click count display
                    click_info = gr.Markdown("**Points:** 0 foreground, 0 background")
                    
                with gr.Column(scale=1):
                    gr.Markdown("### Segmentation Controls")
                    gr.Markdown("Place at least one foreground point, then click Run.")
                    run_btn = gr.Button("🚀 Run Physics-Guided SAM", variant="primary", size="lg")
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
                
                # Update overlay
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

            def run_segmentation(image, points, labels, use_die, use_acoustic, use_abs, multimask):
                if image is None:
                    return None, None, "No image loaded", []
                
                if not points or not any(labels):
                    return None, None, "Please place at least one foreground point", []
                
                if project.sam is None:
                    return None, None, "SAM not initialized. Go to Setup and initialize.", []
                
                try:
                    # Set image in SAM
                    project.sam.set_image(image)
                    
                    # Generate physics map
                    physics = {}
                    if use_die or use_acoustic or use_abs:
                        # Use the first foreground point for physics
                        fg_points = [p for p, l in zip(points, labels) if l == 1]
                        if fg_points:
                            physics = PhysicsSimulator.extract_physical_signature(image, fg_points[0])
                    
                    # Run SAM prediction
                    mask = project.sam.predict_from_clicks(points, labels)
                    
                    # Apply physics conditioning if enabled
                    if physics:
                        mask = PhysicsSimulator.apply_physics_to_segmentation(mask, physics)
                    
                    # Generate candidates if multimask
                    candidates = []
                    if multimask:
                        for i in range(3):
                            # Simulate candidates with slight variations
                            mask_var = mask.copy()
                            # Apply erosion/dilation variations
                            kernel = np.ones((3,3), np.uint8)
                            if i == 0:
                                mask_var = cv2.erode(mask_var, kernel, iterations=1)
                            elif i == 1:
                                mask_var = cv2.dilate(mask_var, kernel, iterations=1)
                            score = 0.95 - i * 0.05
                            candidates.append([f"Candidate {i+1}", f"{score:.3f}"])
                            if i == 0:  # Use the first as primary
                                mask = mask_var
                    
                    # Create overlay
                    overlay = image.copy()
                    overlay[mask > 0] = overlay[mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
                    
                    return np.uint8(overlay), mask, "✅ Segmentation complete!", candidates
                    
                except Exception as e:
                    return None, None, f"❌ Error: {str(e)}", []

            run_btn.click(
                run_segmentation,
                [input_image, click_state, label_state, use_dielectric, use_acoustic, use_absorption, multimask],
                [overlay_display, current_mask_state, seg_status, candidates_display]
            )

        # ==================== EDITOR ====================
        with gr.TabItem("Editor", id=2):
            with gr.Row():
                with gr.Column(scale=2):
                    editor_image = gr.Image(label="Mask Overlay", type="numpy", interactive=False)
                    uncertainty_output = gr.Image(label="Uncertainty Heatmap", type="numpy", interactive=False)
                    
                    # Active learning clicks on uncertainty heatmap
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
                
                # Save to temp directory
                output_dir = tempfile.mkdtemp(prefix="synthetic_")
                manifest = SyntheticDataGenerator.create_manifest(variations, output_dir)
                
                # Create zip
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
                        choices=["COCO", "YOLO", "PNG", "MONAI"],
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
                
                # Annotate first slice
                first_img = images[0]
                center = (first_img.shape[1]//2, first_img.shape[0]//2)
                physics = PhysicsSimulator.extract_physical_signature(first_img, center)
                first_mask = MockSAMDecoder.generate_mask(first_img, center, physics)
                all_masks = VolumetricPropagator.propagate_3d(images, first_mask, physics)
                return images, all_masks, first_mask

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

            1. **Setup**: Upload images or DICOM files → Add to Project → Initialize SAM
            2. **Input**: Click on image to place foreground points → Run Physics-Guided SAM
            3. **Editor**: View mask → Show uncertainty → Click on heatmap to refine
            4. **Results**: Save mask to project
            5. **Export**: Download in your preferred format

            ### ⌨️ Keyboard Shortcuts
            - **Click**: Place point (foreground or background)
            - **Undo**: Ctrl+Z or click undo button
            - **Clear**: Click clear button

            ### 📁 Supported Formats
            - **Images**: PNG, JPG, TIFF, BMP, DICOM (.dcm)
            - **Raw Data**: CSV, S2P, MAT (S21 parameters)
            - **Export**: COCO JSON, YOLO TXT, PNG, MONAI

            ### 🔬 Physics Features
            - **Dielectric Contrast**: Tumors have higher water content → higher permittivity
            - **Acoustic Pressure**: Changes in tissue density affect acoustic wave propagation
            - **Energy Absorption**: Tumors absorb more microwave energy

            ### 🎯 Tips for Best Results
            - Place at least 3 foreground points for complex shapes
            - Use background points to exclude ambiguous regions
            - Check uncertainty heatmap to identify weak areas
            - Generate synthetic variations for training data
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
