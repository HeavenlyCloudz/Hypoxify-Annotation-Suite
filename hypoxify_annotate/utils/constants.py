"""Constants and configuration for microwave/thermoacoustic data processing"""

# =============================================================================
# DEFAULT CONFIGURATIONS
# =============================================================================

# Microwave Imaging (2-3 GHz)
MICROWAVE_CONFIG = {
    "start_freq_ghz": 2.0,
    "stop_freq_ghz": 3.0,
    "num_points": 201,
    "antenna_positions": {
        1: (-75, 0),
        2: (75, 0),
        3: (0, -75),
        4: (0, 75),
    },
    "path_to_antenna_pair": {
        1: (1, 3),
        2: (1, 4),
        3: (2, 3),
        4: (2, 4),
    },
}

# Thermoacoustic Imaging (for PATATO compatibility)
THERMOACOUSTIC_CONFIG = {
    "detector_positions": {
        "ring": "360_degrees",
        "num_detectors": 128,
    },
    "center_frequency_mhz": 2.25,
    "bandwidth_mhz": 3.0,
}

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

C_MM_PER_PS = 0.3  # Speed of light in mm/ps
SOUND_SPEED_M_S = 1540  # Speed of sound in soft tissue
ATTENUATION_COEFF_DB_CM_MHZ = 0.6  # For abdominal fat

# =============================================================================
# RECONSTRUCTION DEFAULTS
# =============================================================================

DEFAULT_GRID_SIZE = 80
GAUSSIAN_SIGMA = 2.0
TGC_ATTENUATION_COEFF = 0.6  # dB/cm/MHz

# =============================================================================
# SUPPORTED CONFIGURATIONS (for Streamlit dropdown)
# =============================================================================

SUPPORTED_CONFIGURATIONS = {
    "microwave_4_antenna": {
        "modality": "microwave",
        "description": "4-antenna microwave imaging system (S21)",
        "num_antennas": 4,
        "antenna_positions": MICROWAVE_CONFIG["antenna_positions"],
        "path_to_antenna_pair": MICROWAVE_CONFIG["path_to_antenna_pair"],
        "file_formats": [".csv", ".s2p", ".mat"],
    },
    "microwave_8_antenna": {
        "modality": "microwave",
        "description": "8-antenna microwave imaging system",
        "num_antennas": 8,
        "antenna_positions": {
            1: (-75, -40), 2: (-75, 40),
            3: (-40, -75), 4: (40, -75),
            5: (75, -40), 6: (75, 40),
            7: (-40, 75), 8: (40, 75),
        },
        "path_to_antenna_pair": {
            1: (1, 5), 2: (2, 6), 3: (3, 7), 4: (4, 8),
            5: (1, 3), 6: (2, 4), 7: (5, 7), 8: (6, 8),
        },
        "file_formats": [".csv", ".s2p", ".mat"],
    },
    "thermoacoustic_ring": {
        "modality": "thermoacoustic",
        "description": "Ring-array thermoacoustic system (PATATO compatible)",
        "num_detectors": 128,
        "detector_radius_mm": 50,
        "file_formats": [".h5", ".mat"],
    },
    "thermoacoustic_linear": {
        "modality": "thermoacoustic",
        "description": "Linear-array thermoacoustic system",
        "num_detectors": 64,
        "detector_length_mm": 100,
        "file_formats": [".h5", ".mat"],
    },
    "mri": {
        "modality": "mri",
        "description": "MRI DICOM/NIfTI (standard medical imaging)",
        "file_formats": [".dcm", ".nii", ".nii.gz"],
    },
    "ct": {
        "modality": "ct",
        "description": "CT DICOM/NIfTI (standard medical imaging)",
        "file_formats": [".dcm", ".nii", ".nii.gz"],
    },
    "histology": {
        "modality": "histology",
        "description": "Histology slides (whole-slide images)",
        "file_formats": [".svs", ".ndpi", ".tiff", ".png", ".jpg"],
    },
}

# =============================================================================
# EXPORT FORMATS
# =============================================================================

EXPORT_FORMATS = {
    "coco": {
        "extension": ".json",
        "description": "COCO format for object detection",
        "use_case": "PyTorch detection models",
    },
    "yolo": {
        "extension": ".txt",
        "description": "YOLO format for object detection",
        "use_case": "YOLO, Ultralytics",
    },
    "monai": {
        "extension": ".json",
        "description": "MONAI format for medical imaging",
        "use_case": "MONAI, 3D medical AI",
    },
    "nifti": {
        "extension": ".nii.gz",
        "description": "NIfTI format for volumetric data",
        "use_case": "3D medical imaging",
    },
    "png": {
        "extension": ".png",
        "description": "PNG mask format",
        "use_case": "Visualization, simple masks",
    },
    "dicom_seg": {
        "extension": ".dcm",
        "description": "DICOM SEG format",
        "use_case": "Clinical PACS integration",
    },
}
