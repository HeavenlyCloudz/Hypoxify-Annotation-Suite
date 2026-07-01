"""Physics-guided prompting for SAM (novel contribution)"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from ..utils.constants import ANTENNA_POSITIONS


def extract_physical_features_at_click(
    raw_rf_data: np.ndarray,
    x: int,
    y: int,
    window_size: int = 5
) -> Dict[str, float]:
    """
    Extract physical features at a click position from raw RF data.
    
    These features represent:
    - Dielectric contrast (microwave absorption)
    - Acoustic pressure amplitude
    - Signal-to-noise ratio at that location
    
    Args:
        raw_rf_data: Raw radiofrequency data
        x: X coordinate of click
        y: Y coordinate of click
        window_size: Size of local window for feature extraction
    
    Returns:
        Dictionary of physical features
    """
    # Ensure coordinates are within bounds
    h, w = raw_rf_data.shape
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)
    
    # Extract local patch
    half = window_size // 2
    x_start = max(0, x - half)
    x_end = min(w, x + half + 1)
    y_start = max(0, y - half)
    y_end = min(h, y + half + 1)
    
    patch = raw_rf_data[y_start:y_end, x_start:x_end]
    
    # Physical features
    features = {
        'dielectric_contrast': float(np.mean(patch)),
        'acoustic_pressure': float(np.max(np.abs(patch))),
        'snr': float(np.mean(np.abs(patch)) / (np.std(np.abs(patch)) + 1e-6)),
        'energy': float(np.sum(patch ** 2)),
        'local_variance': float(np.var(patch)),
        'peak_to_average': float(np.max(patch) / (np.mean(patch) + 1e-6)),
    }
    
    return features


def extract_physical_features_at_clicks(
    raw_rf_data: np.ndarray,
    clicks: List[Tuple[int, int]],
    window_size: int = 5
) -> np.ndarray:
    """
    Extract physical features at multiple click positions.
    
    Args:
        raw_rf_data: Raw radiofrequency data
        clicks: List of (x, y) click positions
        window_size: Size of local window
    
    Returns:
        Array of physical feature vectors, shape (num_clicks, 6)
    """
    features = []
    
    for x, y in clicks:
        f = extract_physical_features_at_click(raw_rf_data, x, y, window_size)
        features.append([
            f['dielectric_contrast'],
            f['acoustic_pressure'],
            f['snr'],
            f['energy'],
            f['local_variance'],
            f['peak_to_average'],
        ])
    
    return np.array(features)


class PhysicsGuidedSegmenter:
    """
    SAM wrapper that incorporates physical features into segmentation.
    
    This is a novel approach that conditions SAM's mask generation on
    physical properties of the tissue (dielectric contrast, acoustic pressure)
    rather than just visual features.
    """
    
    def __init__(self, sam_wrapper, raw_rf_data: Optional[np.ndarray] = None):
        """
        Initialize physics-guided segmenter.
        
        Args:
            sam_wrapper: Instance of SAMWrapper
            raw_rf_data: Raw RF data for physical feature extraction
        """
        self.sam = sam_wrapper
        self.raw_rf_data = raw_rf_data
        self.physical_features = None
    
    def set_raw_rf_data(self, raw_rf_data: np.ndarray):
        """Set raw RF data for physical feature extraction"""
        self.raw_rf_data = raw_rf_data
    
    def segment_from_physics_clicks(
        self,
        foreground: List[Tuple[int, int]],
        background: Optional[List[Tuple[int, int]]] = None,
        apply_physics_weighting: bool = True
    ) -> np.ndarray:
        """
        Segment using clicks enhanced with physical features.
        
        Args:
            foreground: Foreground click positions
            background: Background click positions
            apply_physics_weighting: If True, weight SAM outputs by physical features
        
        Returns:
            Binary mask
        """
        if self.raw_rf_data is None:
            raise ValueError("Raw RF data not set. Call set_raw_rf_data first.")
        
        # Extract physical features at foreground clicks
        foreground_features = extract_physical_features_at_clicks(
            self.raw_rf_data, foreground
        )
        
        # Extract physical features at background clicks
        background_features = None
        if background:
            background_features = extract_physical_features_at_clicks(
                self.raw_rf_data, background
            )
        
        # Store for potential use
        self.physical_features = {
            'foreground': foreground_features,
            'background': background_features,
        }
        
        # Generate mask using SAM
        mask = self.sam.from_clicks(foreground, background, multimask=False)
        
        # Apply physics weighting if enabled
        if apply_physics_weighting and len(foreground) > 0:
            # Use dielectric contrast to refine mask
            # High contrast regions = more likely to be tumor
            mask = self._apply_physics_weighting(mask, foreground_features)
        
        return mask
    
    def _apply_physics_weighting(
        self,
        mask: np.ndarray,
        foreground_features: np.ndarray
    ) -> np.ndarray:
        """
        Refine mask based on physical features.
        
        High dielectric contrast regions are weighted higher.
        """
        # Simple weighting: average dielectric contrast of foreground clicks
        avg_contrast = np.mean(foreground_features[:, 0])
        
        # Threshold based on physical features
        if self.raw_rf_data is not None:
            # Map contrast to pixel weights
            contrast_map = self.raw_rf_data / (np.max(self.raw_rf_data) + 1e-6)
            
            # Only keep mask pixels with sufficient physical contrast
            physical_threshold = avg_contrast * 0.5
            mask_physical = contrast_map > physical_threshold
            
            # Combine with SAM mask
            refined_mask = mask * mask_physical
            
            return refined_mask.astype(np.uint8)
        
        return mask
