"""Segment Anything Model (SAM) wrapper with click prompting"""

import numpy as np
from typing import List, Tuple, Optional, Union
from pathlib import Path

from ..utils.validators import validate_array


class SAMWrapper:
    """
    Wrapper for Meta's Segment Anything Model (SAM) with click prompting.
    
    This provides a clean interface for SAM with support for:
    - Foreground and background clicks
    - Multiple masks
    - Physics-guided conditioning (for extension)
    
    Usage:
        from hypoxify_annotate.segment import SAMWrapper
        
        segmenter = SAMWrapper(model_type="vit_b")
        segmenter.set_image(image)
        mask = segmenter.from_clicks(
            foreground=[(100, 150), (120, 160)],
            background=[(10, 10)]
        )
    """
    
    def __init__(
        self,
        model_type: str = "vit_b",
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: str = "auto"
    ):
        """
        Initialize SAM wrapper.
        
        Args:
            model_type: 'vit_b', 'vit_l', or 'vit_h'
            checkpoint_path: Path to SAM checkpoint file
            device: 'cuda', 'cpu', or 'auto'
        """
        self.model_type = model_type
        self.device = device
        self.predictor = None
        self._loaded = False
        self._image = None
        
        # Try to load SAM
        self._load_model(checkpoint_path)
    
    def _load_model(self, checkpoint_path: Optional[Union[str, Path]] = None):
        """Load SAM model"""
        try:
            from segment_anything import sam_model_registry, SamPredictor
        except ImportError:
            raise ImportError(
                "segment-anything not installed. "
                "Install: pip install segment-anything"
            )
        
        # Determine device
        if self.device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model
        if checkpoint_path is None:
            # Try default paths
            default_paths = [
                Path("sam_vit_b.pth"),
                Path.home() / ".cache" / "sam" / "sam_vit_b.pth",
            ]
            for p in default_paths:
                if p.exists():
                    checkpoint_path = p
                    break
        
        if checkpoint_path is None:
            raise ValueError(
                "SAM checkpoint not found. Please provide checkpoint_path."
            )
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint_path}")
        
        # Load model
        try:
            sam = sam_model_registry[self.model_type](checkpoint=str(checkpoint_path))
            sam.to(device=self.device)
            self.predictor = SamPredictor(sam)
            self._loaded = True
            print(f"SAM loaded: {self.model_type} on {self.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load SAM: {e}")
    
    @property
    def loaded(self) -> bool:
        """Check if model is loaded"""
        return self._loaded
    
    def set_image(self, image: np.ndarray):
        """
        Set image for segmentation.
        
        Args:
            image: Image as numpy array (H, W) or (H, W, C)
        """
        if not self._loaded:
            raise RuntimeError("SAM not loaded. Call _load_model first.")
        
        image = np.asarray(image)
        validate_array(image, name="image")
        
        # Handle grayscale images
        if image.ndim == 2:
            # Convert to 3-channel
            image = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3 and image.shape[-1] == 4:
            # RGBA to RGB
            image = image[:, :, :3]
        
        self._image = image
        self.predictor.set_image(image)
    
    def from_clicks(
        self,
        foreground: List[Tuple[int, int]],
        background: Optional[List[Tuple[int, int]]] = None,
        multimask: bool = False
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Generate mask(s) from foreground and background clicks.
        
        Args:
            foreground: List of (x, y) foreground click positions
            background: List of (x, y) background click positions
            multimask: If True, return multiple masks
        
        Returns:
            Binary mask(s) as uint8 array(s)
        """
        if not self._loaded:
            raise RuntimeError("SAM not loaded. Call _load_model first.")
        
        if self._image is None:
            raise RuntimeError("No image set. Call set_image first.")
        
        if not foreground:
            raise ValueError("At least one foreground click is required")
        
        # Prepare points and labels
        points = []
        labels = []
        
        for x, y in foreground:
            points.append([x, y])
            labels.append(1)  # Foreground
        
        if background:
            for x, y in background:
                points.append([x, y])
                labels.append(0)  # Background
        
        points = np.array(points)
        labels = np.array(labels)
        
        # Generate mask
        masks, scores, logits = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=multimask
        )
        
        if multimask:
            # Return all masks as list
            return [mask.astype(np.uint8) for mask in masks]
        else:
            # Return best mask (highest IoU)
            best_idx = np.argmax(scores)
            return masks[best_idx].astype(np.uint8)
    
    def from_box(
        self,
        box: Tuple[int, int, int, int],
        multimask: bool = False
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Generate mask from bounding box.
        
        Args:
            box: (x_min, y_min, x_max, y_max)
            multimask: If True, return multiple masks
        
        Returns:
            Binary mask(s) as uint8 array(s)
        """
        if not self._loaded:
            raise RuntimeError("SAM not loaded. Call _load_model first.")
        
        if self._image is None:
            raise RuntimeError("No image set. Call set_image first.")
        
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
    
    def batch_process(
        self,
        images: List[np.ndarray],
        foreground: List[Tuple[int, int]],
        background: Optional[List[Tuple[int, int]]] = None
    ) -> List[np.ndarray]:
        """
        Process multiple images with the same click prompts.
        
        Args:
            images: List of images
            foreground: Foreground click positions
            background: Background click positions
        
        Returns:
            List of masks
        """
        masks = []
        
        for image in images:
            self.set_image(image)
            mask = self.from_clicks(foreground, background)
            masks.append(mask)
        
        return masks
