"""Time-Gain Compensation for deep tissue signal amplification"""

import numpy as np
from typing import Optional, Tuple
from ..utils.constants import SOUND_SPEED_M_S, TGC_ATTENUATION_COEFF
from ..utils.validators import validate_array


def time_gain_compensation(
    signal: np.ndarray,
    sampling_rate: float,
    center_frequency_mhz: float,
    depth_start_m: float = 0.0,
    attenuation_coeff: float = TGC_ATTENUATION_COEFF,
    speed_of_sound: float = SOUND_SPEED_M_S
) -> np.ndarray:
    """
    Apply time-gain compensation to amplify deeper signals.
    
    Thermoacoustic waves attenuate exponentially with depth. This function
    applies a gain that increases with time (depth) to compensate.
    
    Args:
        signal: Input signal array (1D or 2D)
        sampling_rate: Sampling rate in Hz
        center_frequency_mhz: Center frequency in MHz
        depth_start_m: Starting depth in meters
        attenuation_coeff: Attenuation coefficient in dB/cm/MHz
        speed_of_sound: Speed of sound in m/s
    
    Returns:
        Compensated signal
    """
    signal = np.asarray(signal)
    validate_array(signal, name="signal")
    
    num_samples = len(signal)
    
    # Time vector
    t = np.arange(num_samples) / sampling_rate
    
    # Depth vector (assuming round-trip for ultrasound, one-way for thermoacoustic)
    depth_m = t * speed_of_sound / 2  # half for round-trip
    
    # Calculate gain in dB
    depth_cm = depth_m * 100  # Convert to cm
    attenuation_db = depth_cm * attenuation_coeff * center_frequency_mhz
    
    # Convert to linear gain
    gain_linear = 10 ** (attenuation_db / 20)
    
    # Apply gain (avoid amplifying early noise)
    if depth_start_m > 0:
        start_idx = int(depth_start_m * 2 / speed_of_sound * sampling_rate)
        gain_linear[:start_idx] = 1.0
    
    # Apply gain to signal
    if signal.ndim == 1:
        compensated = signal * gain_linear
    elif signal.ndim == 2:
        # Each row is a separate receiver
        compensated = signal * gain_linear[np.newaxis, :]
    else:
        raise ValueError(f"Signal must be 1D or 2D, got shape {signal.shape}")
    
    return compensated


def time_gain_compensation_2d(
    image: np.ndarray,
    depth_axis: int = 0,
    center_frequency_mhz: float = 2.25,
    attenuation_coeff: float = TGC_ATTENUATION_COEFF
) -> np.ndarray:
    """
    Apply TGC to a 2D image along the depth axis.
    
    Args:
        image: 2D image array
        depth_axis: Axis index for depth (0 or 1)
        center_frequency_mhz: Center frequency in MHz
        attenuation_coeff: Attenuation coefficient in dB/cm/MHz
    
    Returns:
        TGC-compensated image
    """
    image = np.asarray(image)
    validate_array(image, name="image")
    
    if image.ndim != 2:
        raise ValueError(f"Image must be 2D, got shape {image.shape}")
    
    num_depth = image.shape[depth_axis]
    
    # Depth in cm (assuming pixel spacing of 1 mm)
    depth_cm = np.arange(num_depth) / 10  # mm to cm
    
    # Gain profile
    attenuation_db = depth_cm * attenuation_coeff * center_frequency_mhz
    gain_linear = 10 ** (attenuation_db / 20)
    
    # Apply gain along depth axis
    if depth_axis == 0:
        compensated = image * gain_linear[:, np.newaxis]
    else:
        compensated = image * gain_linear[np.newaxis, :]
    
    return compensated


def calculate_snr_improvement(
    signal: np.ndarray,
    compensated: np.ndarray,
    depth_range: Tuple[float, float]
) -> float:
    """
    Calculate SNR improvement after TGC.
    
    Args:
        signal: Original signal
        compensated: TGC-compensated signal
        depth_range: (start_depth_m, end_depth_m) for SNR calculation
    
    Returns:
        SNR improvement in dB
    """
    signal = np.asarray(signal)
    compensated = np.asarray(compensated)
    
    start_idx = int(depth_range[0] * 2 / SOUND_SPEED_M_S * 16000)
    end_idx = int(depth_range[1] * 2 / SOUND_SPEED_M_S * 16000)
    
    # Calculate SNR as mean/std
    signal_region = signal[start_idx:end_idx]
    comp_region = compensated[start_idx:end_idx]
    
    snr_original = np.mean(signal_region) / np.std(signal_region)
    snr_compensated = np.mean(comp_region) / np.std(comp_region)
    
    if snr_original > 0:
        return 20 * np.log10(snr_compensated / snr_original)
    else:
        return float('inf')
