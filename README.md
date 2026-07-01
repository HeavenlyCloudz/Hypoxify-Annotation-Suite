# Hypoxify-Annotation-Suite
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
