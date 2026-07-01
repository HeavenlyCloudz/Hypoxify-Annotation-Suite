"""Signal denoising utilities for microwave and thermoacoustic data"""

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter
from typing import Optional, Tuple
from ..utils.validators import validate_array


def savgol_filter_1d(
    data: np.ndarray,
    window_length: int = 11,
    polyorder: int = 3
) -> np.ndarray:
    """
    Apply Savitzky-Golay filter for smoothing.
    
    Args:
        data: 1D signal array
        window_length: Window length (must be odd)
        polyorder: Polynomial order
    
    Returns:
        Filtered signal
    """
    data = np.asarray(data)
    validate_array(data, name="data")
    
    if data.ndim != 1:
        raise ValueError(f"Data must be 1D, got shape {data.shape}")
    
    try:
        from scipy.signal import savgol_filter
        return savgol_filter(data, window_length, polyorder)
    except ImportError:
        # Fallback to simple moving average
        window = np.ones(window_length) / window_length
        return np.convolve(data, window, mode='same')


def butter_bandpass_filter(
    data: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 4
) -> np.ndarray:
    """
    Apply Butterworth bandpass filter.
    
    Args:
        data: Input signal
        lowcut: Low cutoff frequency
        highcut: High cutoff frequency
        fs: Sampling frequency
        order: Filter order
    
    Returns:
        Filtered signal
    """
    data = np.asarray(data)
    validate_array(data, name="data")
    
    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    
    if low <= 0 or high >= 1:
        raise ValueError(f"Invalid cutoff frequencies: low={low}, high={high}")
    
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)


def wavelet_denoise(
    data: np.ndarray,
    wavelet: str = 'db4',
    level: int = 4,
    threshold_method: str = 'universal'
) -> np.ndarray:
    """
    Denoise using wavelet thresholding.
    
    Args:
        data: Input signal
        wavelet: Wavelet type (e.g., 'db4', 'sym8')
        level: Decomposition level
        threshold_method: 'universal' or 'scaled'
    
    Returns:
        Denoised signal
    """
    try:
        import pywt
    except ImportError:
        raise ImportError("pywt required for wavelet denoising. Install: pip install PyWavelets")
    
    data = np.asarray(data)
    validate_array(data, name="data")
    
    # Decompose
    coeffs = pywt.wavedec(data, wavelet, level=level)
    
    # Calculate threshold
    if threshold_method == 'universal':
        sigma = np.median(np.abs(coeffs[-level])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(data)))
    else:
        threshold = np.std(coeffs[-level]) * 0.8
    
    # Apply soft thresholding to detail coefficients
    coeffs_thresh = [coeffs[0]]  # Keep approximation
    
    for i in range(1, len(coeffs)):
        coeffs_thresh.append(pywt.threshold(coeffs[i], threshold, mode='soft'))
    
    # Reconstruct
    return pywt.waverec(coeffs_thresh, wavelet)


def gaussian_smooth_2d(
    image: np.ndarray,
    sigma: float = 2.0
) -> np.ndarray:
    """
    Apply 2D Gaussian smoothing to image.
    
    Args:
        image: 2D image array
        sigma: Standard deviation of Gaussian kernel
    
    Returns:
        Smoothed image
    """
    image = np.asarray(image)
    validate_array(image, name="image")
    
    if image.ndim != 2:
        raise ValueError(f"Image must be 2D, got shape {image.shape}")
    
    return gaussian_filter(image, sigma=sigma)


def remove_baseline_wander(
    data: np.ndarray,
    fs: float,
    cutoff_freq: float = 0.5
) -> np.ndarray:
    """
    Remove baseline wander using high-pass filter.
    
    Args:
        data: Input signal
        fs: Sampling frequency
        cutoff_freq: High-pass cutoff frequency
    
    Returns:
        Signal with baseline removed
    """
    data = np.asarray(data)
    validate_array(data, name="data")
    
    nyquist = fs / 2
    cutoff = cutoff_freq / nyquist
    
    if cutoff >= 1:
        raise ValueError(f"Invalid cutoff frequency: {cutoff_freq} > {nyquist}")
    
    b, a = signal.butter(2, cutoff, btype='high')
    return signal.filtfilt(b, a, data)
