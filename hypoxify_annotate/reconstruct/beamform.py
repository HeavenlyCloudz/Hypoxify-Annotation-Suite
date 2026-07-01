"""Microwave-induced thermoacoustic image reconstruction"""

import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Dict, Optional, Tuple, List
from ..utils.constants import (
    ANTENNA_POSITIONS,
    PATH_TO_ANTENNA_PAIR,
    C_MM_PER_PS,
    DEFAULT_GRID_SIZE,
    GAUSSIAN_SIGMA
)
from ..preprocess.bg_subtract import db_to_linear, apply_background_subtraction
from ..utils.validators import validate_array


def delay_and_sum_reconstruction(
    s21_data: Dict[int, np.ndarray],
    frequencies: np.ndarray,
    baseline_data: Optional[Dict[int, np.ndarray]] = None,
    antenna_positions: Optional[Dict[int, Tuple[float, float]]] = None,
    path_to_antenna_pair: Optional[Dict[int, Tuple[int, int]]] = None,
    grid_size: int = DEFAULT_GRID_SIZE,
    grid_extent: float = 100.0,
    sigma: float = GAUSSIAN_SIGMA
) -> np.ndarray:
    """
    Reconstruct image using delay-and-sum beamforming.
    
    This is your cross-coherence algorithm from Project Oracle.
    
    Args:
        s21_data: Dict {path_num: s21_db_array}
        frequencies: Frequency array in GHz
        baseline_data: Optional baseline for background subtraction
        antenna_positions: Dict {antenna_id: (x, y)} in mm
        path_to_antenna_pair: Dict {path_num: (tx_id, rx_id)}
        grid_size: Reconstruction grid size (grid_size x grid_size)
        grid_extent: Grid extent in mm (±grid_extent)
        sigma: Gaussian filter sigma for smoothing
    
    Returns:
        2D reconstructed image as uint8 (0-255)
    """
    # Use defaults if not provided
    if antenna_positions is None:
        antenna_positions = ANTENNA_POSITIONS
    
    if path_to_antenna_pair is None:
        path_to_antenna_pair = PATH_TO_ANTENNA_PAIR
    
    # Grid for reconstruction
    x_grid = np.linspace(-grid_extent, grid_extent, grid_size)
    y_grid = np.linspace(-grid_extent, grid_extent, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    image = np.zeros((grid_size, grid_size))
    num_paths = 0
    
    # Validate inputs
    validate_array(frequencies, name="frequencies")
    
    for path_num, s21_db in s21_data.items():
        if path_num not in path_to_antenna_pair:
            continue
        
        # Get antenna positions
        tx_ant, rx_ant = path_to_antenna_pair[path_num]
        tx_pos = antenna_positions[tx_ant]
        rx_pos = antenna_positions[rx_ant]
        
        # Convert to linear domain
        s21_linear = db_to_linear(s21_db)
        
        # Apply background subtraction if available
        if baseline_data and path_num in baseline_data:
            s21_linear = db_to_linear(
                s21_db - baseline_data[path_num]
            )  # Note: subtraction in dB is approximate
            # Better: use the proper bg_subtract function
            
            # Proper linear domain subtraction
            from ..preprocess.bg_subtract import apply_background_subtraction
            corrected_db = apply_background_subtraction(s21_db, baseline_data[path_num])
            s21_linear = db_to_linear(corrected_db)
        
        s21_linear = np.maximum(s21_linear, 1e-12)
        
        # Delay-and-sum for each point in grid
        for i in range(grid_size):
            for j in range(grid_size):
                point = (X[i, j], Y[i, j])
                
                # Calculate distances from transmitter and receiver to point
                d_tx = np.sqrt((tx_pos[0] - point[0])**2 + (tx_pos[1] - point[1])**2)
                d_rx = np.sqrt((rx_pos[0] - point[0])**2 + (rx_pos[1] - point[1])**2)
                total_distance = (d_tx + d_rx) / 1000  # Convert to meters
                
                # Calculate time delay in picoseconds
                delay_ps = total_distance / C_MM_PER_PS
                
                # Convert delay to frequency index
                freq_range_ghz = frequencies[-1] - frequencies[0]
                freq_idx = int(delay_ps * freq_range_ghz / 1000)
                freq_idx = np.clip(freq_idx, 0, len(s21_linear) - 1)
                
                # Add contribution (coherent summation)
                image[i, j] += s21_linear[freq_idx]
        
        num_paths += 1
    
    # Average across paths
    if num_paths > 0:
        image /= num_paths
    else:
        raise ValueError("No valid paths found in s21_data")
    
    # Apply smoothing
    image = gaussian_filter(image, sigma=sigma)
    
    # Normalize to 0-255
    if image.max() > 0:
        # Clip to 95th percentile to handle outliers
        image = np.clip(image, 0, np.percentile(image, 95))
        image = (image / image.max()) * 255
    
    return image.astype(np.uint8)


def reconstruct_from_multi_angle(
    angle_data: Dict[int, Dict[int, np.ndarray]],
    frequencies: np.ndarray,
    baseline_data: Optional[Dict[int, np.ndarray]] = None,
    antenna_positions: Optional[Dict[int, Tuple[float, float]]] = None,
    path_to_antenna_pair: Optional[Dict[int, Tuple[int, int]]] = None,
    grid_size: int = DEFAULT_GRID_SIZE
) -> np.ndarray:
    """
    Reconstruct image by averaging across multiple rotation angles.
    
    Args:
        angle_data: Dict {angle: {path_num: s21_db}}
        frequencies: Frequency array in GHz
        baseline_data: Optional baseline for subtraction
        antenna_positions: Dict {antenna_id: (x, y)}
        path_to_antenna_pair: Dict {path_num: (tx_id, rx_id)}
        grid_size: Reconstruction grid size
    
    Returns:
        Averaged reconstructed image as uint8
    """
    images = []
    
    for angle, s21_data in angle_data.items():
        image = delay_and_sum_reconstruction(
            s21_data=s21_data,
            frequencies=frequencies,
            baseline_data=baseline_data,
            antenna_positions=antenna_positions,
            path_to_antenna_pair=path_to_antenna_pair,
            grid_size=grid_size
        )
        images.append(image)
    
    if not images:
        raise ValueError("No images were reconstructed from angle_data")
    
    # Average across angles
    avg_image = np.mean(images, axis=0)
    return avg_image.astype(np.uint8)


def rotation_to_angle_matrix(
    angle_deg: float,
    x_positions: np.ndarray,
    y_positions: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotate positions by given angle.
    
    Args:
        angle_deg: Rotation angle in degrees
        x_positions: X coordinates
        y_positions: Y coordinates
    
    Returns:
        Rotated (x, y) positions
    """
    angle_rad = np.radians(angle_deg)
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    
    x_rot = x_positions * cos_theta - y_positions * sin_theta
    y_rot = x_positions * sin_theta + y_positions * cos_theta
    
    return x_rot, y_rot
