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
import random
import os
import shutil
from typing import List, Dict, Optional, Tuple, Any

# ------------------------------------------------------------
# 1. PROJECT MANAGER
# ------------------------------------------------------------
class ProjectManager:
    """Handles playlist, current index, annotations per image, and save/load."""
    def __init__(self):
        self.playlist: List[str] = []
        self.current_index: int = 0
        self.annotations: Dict[str, Dict] = {}
        self.active_project_path: Optional[str] = None

    def add_images(self, image_paths: List[str]):
        for p in image_paths:
            if p not in self.playlist:
                self.playlist.append(p)
                self.annotations[p] = {"masks": [], "points": [], "prompts": []}

    def load_image(self, idx: int) -> Optional[np.ndarray]:
        if 0 <= idx < len(self.playlist):
            self.current_index = idx
            path = self.playlist[idx]
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
            self.annotations[image_path] = {"masks": [], "points": [], "prompts": []}
        self.annotations[image_path]["masks"].append(mask.tolist())
        self.annotations[image_path]["points"].append(points)
        self.annotations[image_path]["prompts"].append(prompt)

    def clear_current_annotations(self):
        path = self.get_current_path()
        if path and path in self.annotations:
            self.annotations[path] = {"masks": [], "points": [], "prompts": []}

    def save_project(self, filepath: str) -> str:
        data = {
            "playlist": self.playlist,
            "current_index": self.current_index,
            "annotations": self.annotations
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        return f"Saved to {filepath}"

    def load_project(self, filepath: str) -> str:
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.playlist = data["playlist"]
        self.current_index = data["current_index"]
        self.annotations = data["annotations"]
        return f"Loaded {len(self.playlist)} images."

    def next_image(self) -> Optional[np.ndarray]:
        if self.current_index + 1 < len(self.playlist):
            self.current_index += 1
            return self.load_image(self.current_index)
        return None

    def prev_image(self) -> Optional[np.ndarray]:
        if self.current_index - 1 >= 0:
            self.current_index -= 1
            return self.load_image(self.current_index)
        return None

# ------------------------------------------------------------
# 2. RECONSTRUCTION FUNCTIONS (CSV, S2P, MAT supported)
# ------------------------------------------------------------
def db_to_linear(db):
    return 10 ** (np.asarray(db) / 10)

def linear_to_db(linear):
    linear = np.maximum(np.asarray(linear), 1e-12)
    return 10 * np.log10(linear)

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
    from scipy.io import loadmat
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
# 3. PHYSICS SIMULATION AND SEGMENTATION
# ------------------------------------------------------------
class PhysicsSimulator:
    @staticmethod
    def extract_physical_signature(image, click_point):
        gray = cv2.cvtColor(np.uint8(image), cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        edges = cv2.Canny(gray, 50, 150)
        dist_from_click = np.sqrt((np.arange(h)[:, None] - click_point[1])**2 +
                                  (np.arange(w)[None, :] - click_point[0])**2)
        dielectric = edges.astype(np.float32) + (1 / (dist_from_click + 1)) * 10
        acoustic = 50 * np.exp(-dist_from_click / 100) + np.random.normal(0, 5, (h, w))
        local_std = ndimage.generic_filter(gray, np.std, size=5)
        absorption = local_std / local_std.max()
        return {
            "dielectric": dielectric / dielectric.max(),
            "acoustic": acoustic / acoustic.max(),
            "absorption": absorption
        }

    @staticmethod
    def apply_physics_to_segmentation(prior_mask, physics_maps):
        physics_weight = (physics_maps["dielectric"] * 0.4 +
                          physics_maps["acoustic"] * 0.3 +
                          physics_maps["absorption"] * 0.3)
        refined = prior_mask * (physics_weight > 0.3)
        return (refined > 0).astype(np.uint8) * 255

class MockSAMDecoder:
    @staticmethod
    def generate_mask(image, click_point, physics_maps):
        img = np.uint8(image * 255) if image.max() <= 1.0 else np.uint8(image)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        h, w = img.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        x, y = click_point
        rect = (max(0, x-40), max(0, y-40), min(w, x+40), min(h, y+40))
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        refined = PhysicsSimulator.apply_physics_to_segmentation(mask, physics_maps)
        kernel = np.ones((3,3), np.uint8)
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel)
        return refined

# ------------------------------------------------------------
# 4. UNCERTAINTY, 3D PROPAGATION, SYNTHETIC, EXPORT
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

class SyntheticGenerator:
    @staticmethod
    def generate(hypoxia_score, tumor_size, roughness, snr):
        size = 256
        center = (size//2, size//2)
        radius = int(30 + tumor_size * 20)
        img = np.zeros((size, size, 3), dtype=np.uint8)
        for _ in range(100):
            angle = random.uniform(0, 2*np.pi)
            r = radius + random.randint(-int(roughness*10), int(roughness*10))
            x = int(center[0] + r * np.cos(angle))
            y = int(center[1] + r * np.sin(angle))
            if 0 <= x < size and 0 <= y < size:
                cv2.circle(img, (x, y), 2, (200, 50, 100), -1)
        hypoxia_factor = hypoxia_score / 100
        img[:, :, 0] = (img[:, :, 0] * (1 + hypoxia_factor * 0.5)).clip(0, 255).astype(np.uint8)
        img[:, :, 2] = (img[:, :, 2] * (1 - hypoxia_factor * 0.3)).clip(0, 255).astype(np.uint8)
        noise = np.random.normal(0, snr * 10, (size, size))
        img = img.astype(np.float32) + noise[:, :, None]
        img = np.clip(img, 0, 255).astype(np.uint8)
        if snr < 0.5:
            img = cv2.GaussianBlur(img, (7, 7), 2)
        return img

def get_bbox_from_mask(mask):
    mask = np.asarray(mask)
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return (0, 0, 0, 0)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return (int(x_min), int(y_min), int(x_max), int(y_max))

def to_coco_format(masks, image_ids=None, image_shapes=None):
    if image_ids is None:
        image_ids = [1] * len(masks)
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
            "category_id": 1,
            "bbox": [x_min, y_min, bbox_width, bbox_height],
            "area": int(np.sum(mask)),
            "iscrowd": 0,
        })
    return {
        "images": [{"id": img_id, "width": w, "height": h} for img_id, (h, w) in zip(image_ids, image_shapes or [])],
        "annotations": annotations,
        "categories": [{"id": 1, "name": "lesion"}],
    }

# ------------------------------------------------------------
# 5. GRADIO APP WITH FULL WORKFLOW
# ------------------------------------------------------------

# Global project instance
project = ProjectManager()
# Global variables for the current session
current_candidates = []  # list of (mask, score, prompt)
selected_indices = set()

# ------------------------------------------------------------
# Helper UI functions
# ------------------------------------------------------------
def reconstruct_from_files(files, baseline_files, grid_size, grid_extent, sigma):
    if not files:
        return None, "Please upload at least one data file."
    try:
        temp_dir = tempfile.mkdtemp()
        file_paths = []
        for f in files:
            src_path = Path(f.name)
            file_paths.append(src_path)

        baseline_data = {}
        if baseline_files:
            for bf in baseline_files:
                bf_path = Path(bf.name)
                match = re.search(r'path(\d+)', bf_path.stem, re.IGNORECASE)
                if match:
                    path_num = int(match.group(1))
                else:
                    path_num = 1
                try:
                    _, s21 = auto_load(bf_path)
                    baseline_data[path_num] = s21
                except Exception as e:
                    return None, f"Error loading baseline {bf_path.name}: {e}"

        if len(file_paths) == 1:
            frequencies, s21_data = auto_load(file_paths[0])
            s21_data_dict = {1: s21_data}
        else:
            s21_data_dict = {}
            for fp in file_paths:
                match = re.search(r'path(\d+)', fp.stem, re.IGNORECASE)
                if match:
                    path_num = int(match.group(1))
                else:
                    path_num = len(s21_data_dict) + 1
                try:
                    _, s21 = auto_load(fp)
                    s21_data_dict[path_num] = s21
                except Exception as e:
                    return None, f"Error loading {fp.name}: {e}"
            frequencies, _ = auto_load(file_paths[0])

        start_freq = float(frequencies[0])
        stop_freq = float(frequencies[-1])
        num_points = len(frequencies)

        image = delay_and_sum_reconstruction(
            s21_data=s21_data_dict,
            frequencies=frequencies,
            baseline_data=baseline_data if baseline_data else None,
            grid_size=int(grid_size),
            grid_extent=float(grid_extent),
            start_freq=start_freq,
            stop_freq=stop_freq,
            num_points=num_points,
            sigma=float(sigma)
        )
        return image, f"Reconstruction successful. Paths: {list(s21_data_dict.keys())}"
    except Exception as e:
        import traceback
        return None, f"Error: {e}\n{traceback.format_exc()}"

def on_image_click(evt: gr.SelectData, image):
    if image is None:
        return image, None, None
    x, y = evt.index
    img_copy = np.uint8(image * 255) if image.max() <= 1.0 else np.uint8(image)
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
    cv2.circle(img_copy, (x, y), 8, (0, 0, 255), 2)
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    return img_copy, (int(x), int(y)), f"Point set at ({x}, {y})"

def run_physics_segmentation(image, point):
    if image is None:
        return [], None, "Please load an image.", gr.update()
    if point is None:
        return [], None, "Please set a seed point.", gr.update()
    # generate multiple candidates (simulate multimask)
    candidates = []
    physics_maps = None
    for i in range(3):
        physics_maps = PhysicsSimulator.extract_physical_signature(image, point)
        mask = MockSAMDecoder.generate_mask(image, point, physics_maps)
        score = 0.9 - i * 0.08 + np.random.normal(0, 0.02)
        candidates.append((mask, float(np.clip(score, 0.5, 0.99)), f"Candidate {i+1}"))
    # show dielectric map
    if physics_maps is not None:
        phys_img = np.stack([physics_maps["dielectric"]]*3, axis=2)
        phys_img = np.uint8(phys_img * 255)
    else:
        phys_img = None
    return candidates, phys_img, "Generated 3 candidates.", gr.update(selected=2)  # switch to Editor (id=2)

def add_selected_to_project(candidates, selected_labels):
    if not candidates or not selected_labels:
        return "No candidates selected.", gr.update()
    # find masks
    selected_masks = []
    for label in selected_labels:
        for c in candidates:
            if label.startswith(c[2]):
                selected_masks.append(c[0])
                break
    if not selected_masks:
        return "No matching candidates.", gr.update()
    # get current image path from project
    img_path = project.get_current_path()
    if img_path is None:
        return "No image loaded in project.", gr.update()
    # save each mask
    for mask in selected_masks:
        project.save_annotation(img_path, mask, [])
    return f"Added {len(selected_masks)} masks to project.", gr.update(selected=3)  # switch to Results (id=3)

def init_editor():
    img_path = project.get_current_path()
    if img_path is None or img_path not in project.annotations:
        return None, "No annotations for this image."
    masks = project.annotations[img_path]["masks"]
    if not masks:
        return None, "No masks saved."
    # show the last mask with overlay
    last_mask = np.array(masks[-1]).astype(np.uint8) * 255
    # overlay on current image
    img = cv2.imread(img_path)
    if img is None:
        return None, "Could not load image."
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    overlay = img.copy()
    overlay[last_mask > 0] = overlay[last_mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
    return np.uint8(overlay), f"Showing mask {len(masks)}"

def show_uncertainty():
    img_path = project.get_current_path()
    if img_path is None or img_path not in project.annotations:
        return None
    masks = project.annotations[img_path]["masks"]
    if not masks:
        return None
    mask = np.array(masks[-1])
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    heatmap, _ = UncertaintyCalculator.compute_heatmaps(img / 255.0, mask)
    return heatmap

def export_annotations(fmt):
    # gather all masks from project
    all_masks = []
    for path, ann in project.annotations.items():
        for mask_list in ann["masks"]:
            mask = np.array(mask_list)
            all_masks.append(mask)
    if not all_masks:
        return None, "No annotations to export."
    if fmt == "COCO":
        coco = to_coco_format(all_masks, image_ids=list(range(len(all_masks))), image_shapes=[mask.shape for mask in all_masks])
        json_str = json.dumps(coco, indent=2)
        return json_str, "COCO JSON ready"
    elif fmt == "PNG":
        import zipfile
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zf:
            for i, mask in enumerate(all_masks):
                mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                buf = io.BytesIO()
                mask_img.save(buf, format="PNG")
                zf.writestr(f"mask_{i}.png", buf.getvalue())
        zip_buffer.seek(0)
        return zip_buffer.getvalue(), "PNG zip ready"
    elif fmt == "YOLO":
        lines = []
        for mask in all_masks:
            h, w = mask.shape
            x_min, y_min, x_max, y_max = get_bbox_from_mask(mask)
            x_center = ((x_min + x_max) / 2) / w
            y_center = ((y_min + y_max) / 2) / h
            bbox_width = (x_max - x_min) / w
            bbox_height = (y_max - y_min) / h
            lines.append(f"0 {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")
        txt = "\n".join(lines)
        return txt, "YOLO text ready"
    else:
        return None, "Unsupported format"

def add_images_to_project(files):
    if not files:
        return "No files selected.", "", None
    paths = [f.name for f in files]
    project.add_images(paths)
    if not project.playlist:
        return "No valid images.", "", None
    img = project.load_image(0)
    status = f"Added {len(paths)} images. Total: {len(project.playlist)}"
    playlist_str = "\n".join([str(i+1)+": "+os.path.basename(p) for i,p in enumerate(project.playlist)])
    return status, playlist_str, img

def save_project(name):
    if not project.playlist:
        return "No project to save."
    os.makedirs("saved_projects", exist_ok=True)
    path = f"saved_projects/{name}.json"
    msg = project.save_project(path)
    return msg

def reconstruct_and_add(data_files, baseline_files, gs, ge, sig):
    img, msg = reconstruct_from_files(data_files, baseline_files, gs, ge, sig)
    if img is not None:
        temp_path = tempfile.mktemp(suffix=".png")
        cv2.imwrite(temp_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        project.add_images([temp_path])
        proj_img = project.load_image(0)
        playlist_str = "\n".join([os.path.basename(p) for p in project.playlist])
        return msg, playlist_str, proj_img
    return msg, "", None

def process_volume(files):
    if not files:
        return None, None, None
    images = []
    for f in files:
        img = cv2.imread(f.name)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
    if not images:
        return None, None, None
    first_img = images[0]
    click = (first_img.shape[1]//2, first_img.shape[0]//2)
    physics = PhysicsSimulator.extract_physical_signature(first_img, click)
    first_mask = MockSAMDecoder.generate_mask(first_img, click, physics)
    all_masks = VolumetricPropagator.propagate_3d(images, first_mask, physics)
    return images, all_masks, first_mask

def update_volume_viewer(slice_idx, images, masks):
    if images is None or masks is None:
        return None
    idx = int(slice_idx)
    img = images[idx]
    mask = masks[idx]
    overlay = img.copy()
    overlay[mask > 0] = overlay[mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
    return np.uint8(overlay)

# ------------------------------------------------------------
# CSS (darker font, PWA ready – no logo styling needed)
# ------------------------------------------------------------
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
"""

# ------------------------------------------------------------
# BUILD UI
# ------------------------------------------------------------
with gr.Blocks() as demo:
    # --- Simple Header (no logo) ---
    gr.Markdown("# 🔬 Hypoxify Annotation Suite")
    gr.Markdown("### Physics‑informed segmentation for microwave and thermoacoustic imaging")

    # State variables
    st_current_image_path = gr.State(value=None)
    st_candidates = gr.State(value=[])
    st_selected_choices = gr.State(value=[])
    click_state = gr.State(value=None)

    with gr.Tabs() as tabs:
        # ==================== TAB 0: SETUP ====================
        with gr.TabItem("Setup", id=0):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### Load Images")
                    img_upload = gr.File(label="Upload images (PNG, JPG, etc.)", file_count="multiple", file_types=["image"])
                    load_imgs_btn = gr.Button("Add to Project", variant="primary")
                    with gr.Row():
                        project_name = gr.Textbox(label="Project Name", value="my_project", scale=3)
                        save_project_btn = gr.Button("Save Project", variant="secondary", scale=1)
                    project_status = gr.Textbox(label="Status", lines=3, interactive=False)
                    # Raw data reconstruction
                    gr.Markdown("### Raw Data Reconstruction")
                    gr.Markdown("**Supported file types:** CSV, S2P, MAT")
                    data_files = gr.File(label="Upload S21 data files", file_count="multiple")
                    baseline_files = gr.File(label="Baseline (air) files (optional)", file_count="multiple")
                    grid_size = gr.Slider(30, 150, value=80, label="Grid size")
                    grid_extent = gr.Slider(50, 200, value=100.0, label="Grid extent (mm)")
                    sigma = gr.Slider(0.5, 5.0, value=2.0, label="Smoothing")
                    recon_btn = gr.Button("Reconstruct and Add to Project", variant="primary")
                with gr.Column(scale=1):
                    gr.Markdown("### Project Info")
                    playlist_display = gr.Textbox(label="Images in Project", lines=10, interactive=False)
                    current_img_display = gr.Image(label="Current Image Preview", type="numpy")

            load_imgs_btn.click(
                add_images_to_project,
                [img_upload],
                [project_status, playlist_display, current_img_display]
            )
            save_project_btn.click(
                save_project,
                [project_name],
                [project_status]
            )
            recon_btn.click(
                reconstruct_and_add,
                [data_files, baseline_files, grid_size, grid_extent, sigma],
                [project_status, playlist_display, current_img_display]
            )

        # ==================== TAB 1: INPUT ====================
        with gr.TabItem("Input", id=1):
            with gr.Row():
                with gr.Column(scale=2):
                    input_image = gr.Image(label="Current Image (click to set seed)", type="numpy", interactive=True, elem_id="input_image")
                    click_display = gr.Image(label="Click point preview", type="numpy", interactive=False)
                    input_image.select(
                        on_image_click,
                        [input_image],
                        [click_display, click_state, project_status]
                    )
                with gr.Column(scale=1):
                    gr.Markdown("### Seed Points")
                    gr.Markdown("Click on the image to place a **foreground** seed point.")
                    run_inference_btn = gr.Button("Run Physics-Guided Segmentation", variant="primary")
                    inference_status = gr.Textbox(label="Status")

        # ==================== TAB 2: EDITOR ====================
        with gr.TabItem("Editor", id=2):
            with gr.Row():
                with gr.Column(scale=2):
                    editor_image = gr.Image(label="Selected Mask Overlay", type="numpy", interactive=False)
                    uncertainty_btn = gr.Button("Show Uncertainty Heatmap", variant="secondary")
                    uncertainty_output = gr.Image(label="Uncertainty Heatmap (Red=Uncertain, Green=Confident)", type="numpy")
                with gr.Column(scale=1):
                    gr.Markdown("### Refine Mask")
                    refine_status = gr.Textbox(label="Status")
                    gr.Markdown("💡 **Tip:** The uncertainty heatmap shows where the model is guessing (red). You can add more seed points in the **Input** tab to refine.")

            uncertainty_btn.click(
                show_uncertainty,
                None,
                [uncertainty_output]
            )

        # ==================== TAB 3: RESULTS ====================
        with gr.TabItem("Results", id=3):
            with gr.Row():
                with gr.Column(scale=2):
                    results_preview = gr.Image(label="Dielectric Map Preview", type="numpy")
                with gr.Column(scale=1):
                    candidates_list = gr.CheckboxGroup(label="Select Candidates", choices=[])
                    select_all_btn = gr.Button("Select All", size="sm")
                    deselect_all_btn = gr.Button("Deselect All", size="sm")
                    add_selected_btn = gr.Button("Add Selected to Project", variant="primary")
                    results_status = gr.Textbox(label="Status")

            run_inference_btn.click(
                run_physics_segmentation,
                [input_image, click_state],
                [st_candidates, results_preview, inference_status, tabs]
            ).then(
                lambda candidates: ([f"{c[2]} (score {c[1]:.2f})" for c in candidates], candidates),
                [st_candidates],
                [candidates_list, st_candidates]
            )

            select_all_btn.click(
                lambda choices: choices,
                [candidates_list],
                [candidates_list]
            )
            deselect_all_btn.click(
                lambda: [],
                None,
                [candidates_list]
            )
            add_selected_btn.click(
                add_selected_to_project,
                [st_candidates, candidates_list],
                [results_status, tabs]
            ).then(
                init_editor,
                None,
                [editor_image, refine_status]
            )

        # ==================== TAB 4: EXPORT ====================
        with gr.TabItem("Export", id=4):
            with gr.Row():
                with gr.Column():
                    export_format = gr.Dropdown(choices=["COCO", "YOLO", "PNG"], value="COCO", label="Export format")
                    export_btn = gr.Button("📥 Export Annotations", variant="primary")
                    export_output = gr.File(label="Download")
                    export_status = gr.Textbox(label="Status")

            export_btn.click(
                export_annotations,
                [export_format],
                [export_output, export_status]
            )

        # ==================== TAB 5: 3D PROPAGATION ====================
        with gr.TabItem("3D Propagation", id=5):
            with gr.Row():
                with gr.Column():
                    volume_upload = gr.File(label="Upload volume slices (multiple images)", file_count="multiple")
                    prop_btn = gr.Button("Propagate from first slice", variant="primary")
                    slice_slider = gr.Slider(0, 49, value=0, step=1, label="Slice index")
                with gr.Column():
                    volume_viewer = gr.Image(label="Current slice with propagated mask", type="numpy")
            volume_state = gr.State(value=None)
            masks_state = gr.State(value=None)

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

        # ==================== TAB 6: USER GUIDE ====================
        with gr.TabItem("📖 User Guide", id=6):
            gr.Markdown("""
            ## How to Use the Hypoxify Annotation Suite

            ### 1️⃣ Setup Tab – Load Your Data
            - **Upload Images**: Click "Upload images" to add PNG, JPG, or other image files. Then click "Add to Project".
            - **Raw Data Reconstruction**: If you have microwave S21 data:
                - Upload your `.csv`, `.s2p`, or `.mat` files.
                - Optionally upload **baseline (air) files** for background subtraction.
                - Adjust grid size and smoothing, then click "Reconstruct and Add to Project".
            - **Save Project**: Give your project a name and click "Save Project". This saves all your annotations as a JSON file in the `saved_projects/` folder.

            ---

            ### 2️⃣ Input Tab – Place Seed Points
            - Click anywhere on the image to place a **foreground** seed point (red dot).
            - This tells the model where the object of interest is.
            - Click **"Run Physics‑Guided Segmentation"** to generate candidate masks.

            ---

            ### 3️⃣ Editor Tab – Review & Uncertainty
            - View the selected mask overlaid on the image.
            - Click **"Show Uncertainty Heatmap"** to see:
                - 🟢 **Green** = Model is confident.
                - 🔴 **Red** = Model is guessing.
            - If the mask needs improvement, go back to the **Input** tab and add more seed points.

            ---

            ### 4️⃣ Results Tab – Select Best Candidate
            - The model generates 3 candidate masks (simulating SAM's multimask output).
            - Check the box next to the candidate(s) you want to keep.
            - Click **"Add Selected to Project"** to save them.

            ---

            ### 5️⃣ Export Tab – Download Annotations
            - Choose your format: **COCO** (JSON), **YOLO** (TXT), or **PNG** (zip of masks).
            - Click "Export Annotations" to download.

            ---

            ### 🧊 3D Propagation Tab – Volume Annotation
            - Upload a stack of slices (multiple image files).
            - Click "Propagate from first slice" – the app automatically annotates the first slice and propagates the mask through the entire volume using optical flow.
            - Use the slider to browse slices.

            ---

            ### 💾 Saving & Loading
            - Your annotations are automatically stored in the project memory.
            - To resume later, click **"Save Project"** in the Setup tab, then load the JSON file from the `saved_projects/` directory.

            ---

            ### 📁 Supported File Types
            - **Images**: PNG, JPG, JPEG, TIFF, BMP
            - **Raw Data**: CSV, S2P, MAT (for microwave S21 parameters)
            - **Export**: COCO JSON, YOLO TXT, PNG mask zip

            ### 🚀 Tips
            - For best results, place the seed point near the center of the target.
            - Use the uncertainty heatmap to identify weak areas and add more seed points there.
            - The 3D propagation works best when slices are ordered sequentially.

            **Enjoy annotating with Hypoxify!**
            """)

# ------------------------------------------------------------
# LAUNCH (Render‑ready)
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
