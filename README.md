🔬 Hypoxify Annotation Suite
Physics-informed segmentation for microwave and thermoacoustic imaging

📖 The Problem We Solve
Biomedical researchers studying emerging imaging modalities—microwave-induced thermoacoustic tomography (MITT), microwave imaging, and thermoacoustic imaging—face a critical bottleneck:

Manual annotation takes 30-48 minutes per case because:

Standard AI models (SAM, UNet) fail on blurry, artifact-ridden MITT images

Existing tools don't understand the physics of microwave/acoustic signals

Researchers spend hours reformatting data between incompatible tools

There's no unified platform for reconstruction → segmentation → export

Hypoxify changes this. We embed the physics of microwave and thermoacoustic waves directly into the AI's neural pathways, enabling accurate annotation on data where standard models fail.

✨ Key Features
Feature	Description
🧬 Physics-Guided Segmentation	SAM conditioned on dielectric contrast and acoustic pressure—not just pixels
📡 Raw Data Support	Load CSV, S2P, MAT, DICOM, H5 files directly
🔨 Signal Processing	Linear-domain background subtraction (correct!), time-gain compensation, denoising
📊 Multi-Angle Reconstruction	Average across rotations for improved SNR
🎯 8+ System Configurations	4-antenna microwave, 8-antenna, thermoacoustic ring/linear, MRI, CT, histology
📤 Export Any Format	COCO, YOLO, MONAI, PNG, DICOM SEG—ready for AI training
🎨 No-Code Interface	Streamlit dashboard—upload, click, export. No coding required
🔬 Python API	Full package for integration into existing workflows
🚀 Quick Start
Option 1: Run the Streamlit App (No Code)
bash
# Clone the repository
git clone https://github.com/HeavenlyCloudz/Hypoxify-Annotation-Suite.git
cd Hypoxify-Annotation-Suite

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
The app will open in your browser at http://localhost:8501.

Option 2: Use the Python Package
python
from hypoxify_annotate import (
    auto_load,
    delay_and_sum_reconstruction,
    SAMWrapper,
    to_coco_format
)

# 1. Load your data
frequencies, s21_data = auto_load("patient_scan.csv")

# 2. Reconstruct image
s21_dict = {1: s21_data}  # Map path number to data
image = delay_and_sum_reconstruction(s21_dict, frequencies)

# 3. Segment with physics-guided SAM
segmenter = SAMWrapper()
segmenter.set_image(image)
mask = segmenter.from_clicks(foreground=[(100, 150)])

# 4. Export for AI training
coco = to_coco_format([mask], image_ids=[1], image_shapes=[image.shape])
📦 Installation
Prerequisites
Python 3.8 or higher

pip package manager

From PyPI (Coming Soon)
bash
pip install hypoxify-annotate
From Source
bash
git clone https://github.com/HeavenlyCloudz/Hypoxify-Annotation-Suite.git
cd Hypoxify-Annotation-Suite
pip install -e .
Dependencies
text
numpy>=1.21.0      # Numerical computing
scipy>=1.7.0       # Scientific computing
pandas>=1.3.0      # Data handling
pillow>=9.0.0      # Image I/O
streamlit>=1.25.0  # Web interface
segment-anything   # Meta's SAM (optional, fallback mock mode)
opencv-python      # Polygon conversion
h5py               # HDF5 support
pydicom            # DICOM support
🧠 Novel Contributions
1. Physics-Guided Prompting
Instead of passing just pixel coordinates to SAM, we extract raw RF signal characteristics at click locations—dielectric contrast, acoustic pressure, energy absorption coefficients—and pass them into the mask decoder.

2. Linear-Domain Background Subtraction
Most researchers subtract in dB (incorrect). We subtract in linear power domain, properly removing direct antenna coupling that can be 40+ dB stronger than tissue signal.

3. Time-Domain Feature Extraction
Converting frequency-domain S21 to time-domain impulse responses reveals scattering properties that frequency-domain alone cannot capture.

4. Volumetric Propagation (Planned)
Using SAM 2's memory architecture to propagate annotations across 3D slices—from hours to seconds.

🗂️ Project Structure
text
Hypoxify-Annotation-Suite/
├── app.py                          # Streamlit application (standalone)
├── hypoxify_annotate/              # Python package
│   ├── io/                         # Data loaders (CSV, S2P, MAT, DICOM, H5)
│   ├── preprocess/                 # Background subtraction, denoising, TGC
│   ├── reconstruct/                # Delay-and-sum beamforming
│   ├── segment/                    # SAM wrapper, physics-guided prompting
│   ├── features/                   # Time/frequency domain feature extraction
│   ├── export/                     # COCO, YOLO, MONAI, PNG formats
│   └── utils/                      # Constants, validators
├── requirements.txt                # Python dependencies
├── setup.py                        # Package installation
└── README.md                       # This file
🖥️ Streamlit App Overview
Configuration Sidebar
Select your imaging system (4-antenna microwave, 8-antenna, thermoacoustic, MRI, CT, histology)

Choose input mode (direct image or raw data reconstruction)

Toggle physics-guided segmentation

Export in your preferred format

Annotation Workflow
Upload an image or raw data files

Click on the image to add foreground (tumor) points

Optionally add background points to refine

Click "Generate Mask"

Export to COCO, YOLO, MONAI, or PNG

📚 Modules
Module	Description
io	Load CSV, S2P, MAT, DICOM, H5 with auto-format detection
preprocess	Linear-domain background subtraction, Savitzky-Golay filtering, wavelet denoising, time-gain compensation
reconstruct	Delay-and-sum beamforming, multi-angle averaging
segment	SAM wrapper with click/box prompting, physics-guided conditioning
features	Time-domain impulse response features, frequency-domain S21 features
export	COCO JSON, YOLO TXT, MONAI JSON, PNG masks
utils	Constants, validation, antenna configurations
🤝 Contributing
We welcome contributions! Here's how:

Fork the repository

Create a feature branch (git checkout -b feature/AmazingFeature)

Commit your changes (git commit -m 'Add AmazingFeature')

Push to the branch (git push origin feature/AmazingFeature)

Open a Pull Request

Development Setup
bash
git clone https://github.com/HeavenlyCloudz/Hypoxify-Annotation-Suite.git
cd Hypoxify-Annotation-Suite
pip install -e .[dev]
pytest tests/
📄 License
Distributed under the MIT License. See LICENSE for more information.

🙏 Acknowledgments
Dr. Elise Fear (UCalgary) - Microwave imaging expertise and mentorship

CWSF - Silver medal project that started this journey

Pfizer Oncology - Science Award recognition

Meta AI - Segment Anything Model (SAM)

📬 Contact
Anie Udofia - anie.udofia@hypoxify.ai

Project Link: https://github.com/HeavenlyCloudz/Hypoxify-Annotation-Suite
