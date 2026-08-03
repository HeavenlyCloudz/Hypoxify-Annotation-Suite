# Hypoxify Annotation Suite

Clinical-grade, physics-informed medical image annotation platform for microwave,
thermoacoustic, photoacoustic, ultrasound, and DICOM imaging.

Live Demo:
https://hypoxify-annotation-suite.onrender.com

================================================================================
OVERVIEW
================================================================================

Hypoxify is a physics-informed annotation platform designed for emerging
non-ionizing biomedical imaging modalities. Rather than relying solely on
pixel information, it incorporates microwave scattering, acoustic pressure,
dielectric contrast, and phase information to improve segmentation quality.

Think of it as:
"Segment Anything Model (SAM), but guided by physics instead of only pixels."

================================================================================
FEATURES
================================================================================

• Mobile SAM (CPU inference)
• Multi-modality support
• Physics-guided segmentation
• Active learning
• 3D propagation
• DICOM support
• Project persistence
• MONAI / nnU-Net export
• Client-side de-identification

================================================================================
SUPPORTED MODALITIES
================================================================================

- MITT
- Microwave Imaging
- Photoacoustic Imaging
- Ultrasound
- MRI / CT (DICOM)

================================================================================
INSTALLATION
================================================================================

git clone https://github.com/HeavenlyCloudz/Hypoxify-Annotation-Suite.git

cd Hypoxify-Annotation-Suite

pip install -r requirements.txt

python app.py

================================================================================
PROJECT STRUCTURE
================================================================================

Hypoxify-Annotation-Suite/
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
├── mobile_sam.pt
└── saved_projects/

================================================================================
LICENSE
================================================================================

MIT License

================================================================================
CONTACT
================================================================================

Anie Udofia
Email: anieudofia8@gmail.com

GitHub:
https://github.com/HeavenlyCloudz

⭐ If you find this project useful, please star the repository!
