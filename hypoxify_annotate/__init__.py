"""
Hypoxify Annotation Suite - Physics-informed annotation for biomedical imaging.

Hypoxify helps researchers segment, annotate, and prepare imaging data for AI training,
with specialized support for microwave and thermoacoustic modalities.

Modules:
    - io: Load data from various formats (CSV, S2P, MAT, DICOM, H5)
    - preprocess: Background subtraction, denoising, time-gain compensation
    - reconstruct: Delay-and-sum beamforming, thermoacoustic reconstruction
    - segment: SAM wrapper, physics-guided prompting
    - features: Time-domain and frequency-domain feature extraction
    - export: COCO, YOLO, MONAI, PNG formats
    - utils: Constants, validators
"""

# Version
from .version import __version__, __author__, __license__

# IO
from .io.microwave import (
    auto_load,
    load_s21_csv,
    load_s2p,
    load_mat,
    load_multi_angle_scans,
)

# Preprocess
from .preprocess.bg_subtract import (
    apply_background_subtraction,
    apply_background_subtraction_to_paths,
    db_to_linear,
    linear_to_db,
    estimate_baseline_from_data,
)

from .preprocess.denoise import (
    savgol_filter_1d,
    butter_bandpass_filter,
    wavelet_denoise,
    gaussian_smooth_2d,
    remove_baseline_wander,
)

from .preprocess.tgc import (
    time_gain_compensation,
    time_gain_compensation_2d,
)

# Reconstruction
from .reconstruct.beamform import (
    delay_and_sum_reconstruction,
    reconstruct_from_multi_angle,
)

# Segment
from .segment.sam_wrapper import SAMWrapper
from .segment.physics_guided import (
    PhysicsGuidedSegmenter,
    extract_physical_features_at_click,
    extract_physical_features_at_clicks,
)

# Export
from .export.formats import (
    to_coco_format,
    to_yolo_format,
    to_monai_format,
    save_coco,
    save_yolo,
    save_png_mask,
    get_bbox_from_mask,
    mask_to_polygon,
)

# Constants
from .utils.constants import (
    SUPPORTED_CONFIGURATIONS,
    EXPORT_FORMATS,
    MICROWAVE_CONFIG,
    THERMOACOUSTIC_CONFIG,
)

__all__ = [
    # Version
    "__version__",
    "__author__",
    "__license__",
    # IO
    "auto_load",
    "load_s21_csv",
    "load_s2p",
    "load_mat",
    "load_multi_angle_scans",
    # Preprocess
    "apply_background_subtraction",
    "apply_background_subtraction_to_paths",
    "db_to_linear",
    "linear_to_db",
    "estimate_baseline_from_data",
    "savgol_filter_1d",
    "butter_bandpass_filter",
    "wavelet_denoise",
    "gaussian_smooth_2d",
    "remove_baseline_wander",
    "time_gain_compensation",
    "time_gain_compensation_2d",
    # Reconstruction
    "delay_and_sum_reconstruction",
    "reconstruct_from_multi_angle",
    # Segment
    "SAMWrapper",
    "PhysicsGuidedSegmenter",
    "extract_physical_features_at_click",
    "extract_physical_features_at_clicks",
    # Export
    "to_coco_format",
    "to_yolo_format",
    "to_monai_format",
    "save_coco",
    "save_yolo",
    "save_png_mask",
    "get_bbox_from_mask",
    "mask_to_polygon",
    # Constants
    "SUPPORTED_CONFIGURATIONS",
    "EXPORT_FORMATS",
    "MICROWAVE_CONFIG",
    "THERMOACOUSTIC_CONFIG",
]
