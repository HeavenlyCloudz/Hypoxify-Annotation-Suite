"""Load microwave data from various formats (CSV, MAT, S2P)"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Union
import json
import re

from ..utils.validators import validate_file_path, validate_array


def load_s21_csv(
    filepath: Union[str, Path],
    freq_column: str = "Frequency_GHz",
    s21_column: str = "S21_dB"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load S21 data from CSV format.
    
    Args:
        filepath: Path to CSV file
        freq_column: Column name for frequency
        s21_column: Column name for S21 values
    
    Returns:
        frequencies: Array of frequencies in GHz
        s21_db: Array of S21 values in dB
    
    Raises:
        ValueError: If columns not found
    """
    path = validate_file_path(filepath, must_exist=True, expected_extensions=[".csv"])
    
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise ValueError(f"Could not read CSV file {path}: {e}")
    
    # Try to find frequency column
    if freq_column in df.columns:
        frequencies = df[freq_column].values
    else:
        # Try alternative column names
        alt_freq = [c for c in df.columns if 'freq' in c.lower() or 'ghz' in c.lower()]
        if alt_freq:
            frequencies = df[alt_freq[0]].values
        else:
            raise ValueError(f"Frequency column not found. Available: {df.columns.tolist()}")
    
    # Try to find S21 column
    if s21_column in df.columns:
        s21_db = df[s21_column].values
    else:
        # Try alternative column names
        alt_s21 = [c for c in df.columns if 's21' in c.lower() or 's_param' in c.lower()]
        if alt_s21:
            s21_db = df[alt_s21[0]].values
        else:
            raise ValueError(f"S21 column not found. Available: {df.columns.tolist()}")
    
    # Validate arrays
    validate_array(frequencies, name="frequencies")
    validate_array(s21_db, name="s21_db")
    
    if len(frequencies) != len(s21_db):
        raise ValueError(f"Length mismatch: frequencies {len(frequencies)} != s21 {len(s21_db)}")
    
    return frequencies.astype(np.float64), s21_db.astype(np.float64)


def load_s2p(filepath: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load S2P (Touchstone) file format.
    
    Supports both magnitude/phase and real/imag formats.
    
    Args:
        filepath: Path to .s2p file
    
    Returns:
        frequencies: Array of frequencies in GHz
        s21_db: Array of S21 values in dB
    """
    path = validate_file_path(filepath, must_exist=True, expected_extensions=[".s2p"])
    
    frequencies = []
    s21_mag_linear = []
    
    # Parse header for format
    format_type = "ma"  # default: magnitude/angle
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#') and 'MA' in line.upper():
                format_type = "ma"
            elif line.startswith('#') and 'RI' in line.upper():
                format_type = "ri"
            elif not line.startswith('!') and not line.startswith('#'):
                break
    
    # Parse data
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('!') or line.startswith('#'):
                continue
            
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            
            try:
                freq_mhz = float(parts[0])
                freq_ghz = freq_mhz / 1000.0
                
                if format_type == "ma":
                    mag = float(parts[1])
                    s21_mag_linear.append(mag)
                else:  # ri
                    real = float(parts[1])
                    imag = float(parts[2])
                    mag = np.sqrt(real**2 + imag**2)
                    s21_mag_linear.append(mag)
                
                frequencies.append(freq_ghz)
                
            except ValueError:
                continue
    
    if not frequencies:
        raise ValueError(f"No valid data found in {path}")
    
    # Convert linear magnitude to dB
    s21_db = np.array([20 * np.log10(m) if m > 0 else -100 for m in s21_mag_linear])
    
    return np.array(frequencies), s21_db


def load_mat(filepath: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load MATLAB .mat file containing S21 data.
    
    Args:
        filepath: Path to .mat file
    
    Returns:
        frequencies: Array of frequencies in GHz
        s21_db: Array of S21 values in dB
    """
    path = validate_file_path(filepath, must_exist=True, expected_extensions=[".mat"])
    
    try:
        from scipy.io import loadmat
    except ImportError:
        raise ImportError("scipy.io required for .mat files. Install scipy: pip install scipy")
    
    try:
        mat_data = loadmat(path)
    except Exception as e:
        raise ValueError(f"Could not load .mat file {path}: {e}")
    
    # Try common variable names
    freq_keys = ['frequencies', 'freq', 'f', 'Frequency_GHz', 'frequency']
    s21_keys = ['S21_dB', 's21_db', 'S21', 'data', 's_params']
    
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
        raise ValueError(f"Frequency variable not found. Available: {list(mat_data.keys())}")
    
    if s21_db is None:
        raise ValueError(f"S21 variable not found. Available: {list(mat_data.keys())}")
    
    validate_array(frequencies, name="frequencies")
    validate_array(s21_db, name="s21_db")
    
    return frequencies.astype(np.float64), s21_db.astype(np.float64)


def load_multi_angle_scans(
    directory: Union[str, Path],
    angles: List[int] = None
) -> Dict[int, Dict[int, np.ndarray]]:
    """
    Load all rotation scans from a directory.
    
    Args:
        directory: Directory containing CSV files
        angles: List of rotation angles (default: [0, 120, 240])
    
    Returns:
        Dictionary: {angle: {path_num: s21_db_array}}
    """
    if angles is None:
        angles = [0, 120, 240]
    
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    all_data = {}
    
    for angle in angles:
        angle_data = {}
        for path_num in [1, 2, 3, 4]:
            pattern = f"path{path_num}_angle{angle}_*.csv"
            files = list(directory.glob(pattern))
            
            if files:
                latest = max(files, key=lambda f: f.stat().st_mtime)
                try:
                    _, s21 = load_s21_csv(latest)
                    angle_data[path_num] = s21
                except Exception as e:
                    print(f"Warning: Could not load {latest}: {e}")
        
        if angle_data:
            all_data[angle] = angle_data
    
    return all_data


def auto_load(filepath: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Automatically detect file format and load S21 data.
    
    Args:
        filepath: Path to CSV, S2P, or MAT file
    
    Returns:
        frequencies: Array of frequencies in GHz
        s21_db: Array of S21 values in dB
    """
    path = Path(filepath)
    suffix = path.suffix.lower()
    
    if suffix == '.csv':
        return load_s21_csv(path)
    elif suffix == '.s2p':
        return load_s2p(path)
    elif suffix == '.mat':
        return load_mat(path)
    elif suffix == '.json':
        # Try loading from JSON config + data
        with open(path, 'r') as f:
            config = json.load(f)
            frequencies = np.array(config['frequencies'])
            s21_db = np.array(config['s21_db'])
            return frequencies, s21_db
    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. "
            f"Supported: .csv, .s2p, .mat, .json"
        )
