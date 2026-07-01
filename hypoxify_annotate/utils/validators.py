"""Input validation utilities"""

import numpy as np
from pathlib import Path
from typing import Union, Dict, Optional, Tuple


def validate_array(
    data: np.ndarray, 
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    dtype: Optional[type] = None,
    shape: Optional[Tuple[int, ...]] = None,
    name: str = "array"
) -> None:
    """
    Validate array properties with clear error messages.
    
    Args:
        data: Array to validate
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        dtype: Expected data type
        shape: Expected shape
        name: Name for error messages
    
    Raises:
        ValueError: With descriptive message
    """
    if not isinstance(data, np.ndarray):
        raise ValueError(f"{name} must be a numpy array, got {type(data).__name__}")
    
    if data.size == 0:
        raise ValueError(f"{name} is empty")
    
    if dtype is not None and data.dtype != dtype:
        raise ValueError(f"{name} dtype {data.dtype} != expected {dtype}")
    
    if shape is not None and data.shape != shape:
        raise ValueError(f"{name} shape {data.shape} != expected {shape}")
    
    if min_value is not None and np.min(data) < min_value:
        raise ValueError(f"{name} has values below {min_value}: min={np.min(data)}")
    
    if max_value is not None and np.max(data) > max_value:
        raise ValueError(f"{name} has values above {max_value}: max={np.max(data)}")


def validate_file_path(
    path: Union[str, Path],
    must_exist: bool = True,
    expected_extensions: Optional[list] = None
) -> Path:
    """
    Validate file path.
    
    Args:
        path: File path
        must_exist: Whether file must exist
        expected_extensions: List of allowed extensions
    
    Returns:
        Path object
    
    Raises:
        FileNotFoundError: If file doesn't exist and must_exist is True
        ValueError: If extension not in expected_extensions
    """
    path = Path(path)
    
    if must_exist and not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    if expected_extensions is not None:
        ext = path.suffix.lower()
        if ext not in expected_extensions:
            raise ValueError(
                f"File extension {ext} not supported. "
                f"Supported: {', '.join(expected_extensions)}"
            )
    
    return path


def validate_configuration(
    config_name: str,
    supported_configs: Dict[str, Dict]
) -> Dict:
    """
    Validate that configuration name exists.
    
    Args:
        config_name: Name of configuration
        supported_configs: Dictionary of configurations
    
    Returns:
        Configuration dictionary
    
    Raises:
        ValueError: If config_name not found
    """
    if config_name not in supported_configs:
        raise ValueError(
            f"Configuration '{config_name}' not found. "
            f"Supported: {', '.join(supported_configs.keys())}"
        )
    
    return supported_configs[config_name]
