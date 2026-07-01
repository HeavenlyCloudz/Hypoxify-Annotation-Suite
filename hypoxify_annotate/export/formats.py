"""Export annotations to various formats (COCO, YOLO, MONAI, etc.)"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from PIL import Image
import base64

from ..utils.validators import validate_array


def get_bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Get bounding box from binary mask.
    
    Args:
        mask: Binary mask (2D array)
    
    Returns:
        (x_min, y_min, x_max, y_max)
    """
    mask = np.asarray(mask)
    validate_array(mask, name="mask")
    
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D, got shape {mask.shape}")
    
    coords = np.argwhere(mask > 0)
    
    if len(coords) == 0:
        return (0, 0, 0, 0)
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    return (int(x_min), int(y_min), int(x_max), int(y_max))


def mask_to_polygon(mask: np.ndarray, tolerance: float = 2.0) -> List[List[float]]:
    """
    Convert binary mask to polygon using OpenCV.
    
    Args:
        mask: Binary mask (2D array)
        tolerance: Polygon simplification tolerance
    
    Returns:
        List of polygon vertices as [x, y] pairs
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python required for polygon conversion")
    
    mask = np.asarray(mask)
    validate_array(mask, name="mask")
    
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D, got shape {mask.shape}")
    
    # Find contours
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        return []
    
    # Simplify contour
    polygon = cv2.approxPolyDP(contours[0], tolerance, True)
    
    # Convert to list of [x, y]
    polygon_list = polygon.squeeze().tolist()
    
    # Handle single point case
    if isinstance(polygon_list, list) and isinstance(polygon_list[0], (int, float)):
        polygon_list = [polygon_list]
    
    return polygon_list


def to_coco_format(
    masks: List[np.ndarray],
    image_ids: List[int],
    category_ids: Optional[List[int]] = None,
    image_shapes: Optional[List[Tuple[int, int]]] = None
) -> Dict:
    """
    Convert masks to COCO JSON format.
    
    Args:
        masks: List of binary masks
        image_ids: List of image IDs
        category_ids: List of category IDs (default: all 1)
        image_shapes: List of (height, width) tuples
    
    Returns:
        COCO JSON dictionary
    """
    if category_ids is None:
        category_ids = [1] * len(masks)
    
    annotations = []
    
    for idx, mask in enumerate(masks):
        mask = np.asarray(mask)
        h, w = mask.shape[:2] if image_shapes is None else image_shapes[idx]
        
        # Get bounding box
        x_min, y_min, x_max, y_max = get_bbox_from_mask(mask)
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        
        # Get polygon
        polygon = mask_to_polygon(mask)
        
        annotation = {
            "id": idx,
            "image_id": image_ids[idx],
            "category_id": category_ids[idx],
            "segmentation": [polygon] if polygon else [],
            "bbox": [x_min, y_min, bbox_width, bbox_height],
            "area": int(np.sum(mask)),
            "iscrowd": 0,
        }
        annotations.append(annotation)
    
    coco_output = {
        "images": [
            {"id": img_id, "width": w, "height": h} 
            for img_id, (h, w) in zip(image_ids, image_shapes or [])
        ],
        "annotations": annotations,
        "categories": [
            {"id": 1, "name": "lesion"},
            {"id": 2, "name": "tumor"},
        ]
    }
    
    return coco_output


def to_yolo_format(
    masks: List[np.ndarray],
    image_shapes: List[Tuple[int, int]],
    class_ids: Optional[List[int]] = None
) -> List[str]:
    """
    Convert masks to YOLO TXT format.
    
    Args:
        masks: List of binary masks
        image_shapes: List of (height, width) tuples
        class_ids: List of class IDs (default: all 0)
    
    Returns:
        List of YOLO format strings (one per mask)
    """
    if class_ids is None:
        class_ids = [0] * len(masks)
    
    yolo_lines = []
    
    for mask, (h, w), class_id in zip(masks, image_shapes, class_ids):
        mask = np.asarray(mask)
        
        x_min, y_min, x_max, y_max = get_bbox_from_mask(mask)
        
        # Normalize coordinates
        x_center = ((x_min + x_max) / 2) / w
        y_center = ((y_min + y_max) / 2) / h
        bbox_width = (x_max - x_min) / w
        bbox_height = (y_max - y_min) / h
        
        # YOLO format: class x_center y_center width height
        line = f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}"
        yolo_lines.append(line)
    
    return yolo_lines


def to_monai_format(
    masks: List[np.ndarray],
    image_ids: List[int],
    category_ids: Optional[List[int]] = None
) -> Dict:
    """
    Convert masks to MONAI format (for medical imaging AI).
    
    Args:
        masks: List of binary masks
        image_ids: List of image IDs
        category_ids: List of category IDs
    
    Returns:
        MONAI JSON format
    """
    if category_ids is None:
        category_ids = [1] * len(masks)
    
    monai_output = []
    
    for mask, img_id, cat_id in zip(masks, image_ids, category_ids):
        mask = np.asarray(mask)
        
        # RLE encoding for MONAI
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
            "segmentation": {"counts": rle, "size": mask.shape},
        })
    
    return monai_output


def save_coco(
    coco_dict: Dict,
    output_path: Union[str, Path]
) -> None:
    """
    Save COCO JSON to file.
    
    Args:
        coco_dict: COCO dictionary
        output_path: Path to output file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(coco_dict, f, indent=2)


def save_yolo(
    yolo_lines: List[str],
    output_dir: Union[str, Path],
    filename_prefix: str = "annotation"
) -> None:
    """
    Save YOLO format to TXT files.
    
    Args:
        yolo_lines: List of YOLO format strings
        output_dir: Output directory
        filename_prefix: Prefix for output filenames
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, line in enumerate(yolo_lines):
        output_path = output_dir / f"{filename_prefix}_{idx}.txt"
        with open(output_path, 'w') as f:
            f.write(line + '\n')


def save_png_mask(
    mask: np.ndarray,
    output_path: Union[str, Path]
) -> None:
    """
    Save mask as PNG image.
    
    Args:
        mask: Binary mask
        output_path: Path to output file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    mask = (np.asarray(mask) * 255).astype(np.uint8)
    Image.fromarray(mask).save(output_path)
