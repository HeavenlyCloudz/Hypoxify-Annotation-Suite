"""Background subtraction for microwave data (linear domain, not dB)"""

import numpy as np
from typing import Dict, Optional, Union, Tuple
from ..utils.validators import validate_array


def db_to_linear(db_values: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert dB to linear magnitude (power ratio).
    
    Args:
        db_values: Values in decibels
    
    Returns:
        Linear magnitude values
    """
    db_values = np.asarray(db_values)
    return 10 ** (db_values / 10)


def linear_to_db(linear_values: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert linear magnitude to dB.
    
    Args:
        linear_values: Linear magnitude values
    
    Returns:
        Values in decibels
    """
    linear_values = np.asarray(linear_values)
    linear_values = np.maximum(linear_values, 1e-12)  # Avoid log(0)
    return 10 * np.log10(linear_values)


def apply_background_subtraction(
    patient_s21_db: np.ndarray,
    baseline_s21_db: np.ndarray
) -> np.ndarray:
    """
    Remove direct antenna coupling by subtracting baseline in linear domain.
    
    IMPORTANT: Subtraction MUST be in linear (power) domain, not dB!
    
    Why linear domain?
    - Subtraction in dB = division in linear (WRONG for coupling removal)
    - Linear domain subtraction properly removes the additive coupling signal
    
    Args:
        patient_s21_db: Patient measurement in dB
        baseline_s21_db: Baseline (air/reference) measurement in dB
    
    Returns:
        Corrected S21 in dB with coupling removed
    
    Raises:
        ValueError: If arrays have different lengths
    """
    patient_s21_db = np.asarray(patient_s21_db)
    baseline_s21_db = np.asarray(baseline_s21_db)
    
    if len(patient_s21_db) != len(baseline_s21_db):
        raise ValueError(
            f"Length mismatch: patient {len(patient_s21_db)} != baseline {len(baseline_s21_db)}"
        )
    
    validate_array(patient_s21_db, name="patient_s21_db")
    validate_array(baseline_s21_db, name="baseline_s21_db")
    
    # Convert to linear domain
    patient_linear = db_to_linear(patient_s21_db)
    baseline_linear = db_to_linear(baseline_s21_db)
    
    # Subtract in linear domain
    corrected_linear = patient_linear - baseline_linear
    corrected_linear = np.maximum(corrected_linear, 1e-12)
    
    # Convert back to dB
    return linear_to_db(corrected_linear)


def apply_background_subtraction_to_paths(
    patient_data: Dict[int, np.ndarray],
    baseline_data: Dict[int, np.ndarray]
) -> Dict[int, np.ndarray]:
    """
    Apply background subtraction to all antenna paths.
    
    Args:
        patient_data: Dict {path_num: s21_db_array}
        baseline_data: Dict {path_num: s21_db_array}
    
    Returns:
        Corrected data dict with same structure
    """
    corrected_data = {}
    
    for path_num, patient_s21 in patient_data.items():
        if path_num in baseline_data:
            try:
                corrected_data[path_num] = apply_background_subtraction(
                    patient_s21, baseline_data[path_num]
                )
            except ValueError as e:
                raise ValueError(f"Path {path_num}: {e}")
        else:
            # No baseline for this path, keep original
            corrected_data[path_num] = patient_s21
    
    return corrected_data


def estimate_baseline_from_data(
    s21_data: Dict[int, np.ndarray],
    method: str = "min"
) -> Dict[int, np.ndarray]:
    """
    Estimate baseline from data when no explicit baseline scan exists.
    
    Args:
        s21_data: Dict {path_num: s21_db_array}
        method: 'min' for minimum values, 'median' for median across paths
    
    Returns:
        Estimated baseline dict
    """
    baseline = {}
    
    if method == "min":
        for path_num, data in s21_data.items():
            # Use minimum value (least attenuation = closest to air)
            baseline[path_num] = np.ones_like(data) * np.min(data)
    
    elif method == "median":
        # Collect all data and take median per frequency point
        # Assumes data is aligned across paths
        all_data = np.array(list(s21_data.values()))
        median_data = np.median(all_data, axis=0)
        
        for path_num in s21_data.keys():
            baseline[path_num] = median_data
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'min' or 'median'")
    
    return baseline
