🧬 Hypoxify Annotation Suite

Clinical-grade, physics-informed medical image annotation platform for microwave, thermoacoustic, photoacoustic, ultrasound, and DICOM imaging.

🌐 Live Demo: https://hypoxify-annotation-suite.onrender.com

Overview

Hypoxify is a physics-informed annotation platform designed specifically for emerging non-ionizing biomedical imaging modalities.

Unlike traditional segmentation tools that rely solely on image appearance, Hypoxify incorporates underlying imaging physics—including microwave scattering, acoustic pressure, dielectric contrast, and phase information—to improve annotation quality in difficult medical images.

Think of it as: Segment Anything Model (SAM), but guided by physics instead of only pixels.

Why Hypoxify?

Biomedical researchers working with experimental imaging modalities face several challenges:

Problem	Impact
Manual annotation	30–48 minutes per image
Generic AI models	Poor performance on noisy MITT/MWI data
No uncertainty estimation	Difficult to trust AI-generated masks
Fragmented workflows	Time-consuming dataset conversion
Limited DICOM support	Poor clinical interoperability
GPU requirements	Many annotation tools require expensive hardware

Hypoxify addresses each of these limitations within a unified workflow.

Features
Feature	Description
🧠 Mobile SAM	Lightweight CPU-based segmentation
📡 Multi-Modality Support	MITT, MWI, Ultrasound, Photoacoustic
🔬 Physics-Guided Conditioning	Imaging physics incorporated into segmentation
📈 Active Learning	Interactive uncertainty refinement
🧪 Synthetic Dataset Generation	Automated augmentation pipeline
📦 3D Mask Propagation	Propagate annotations across image volumes
💾 Project Saving	Resume annotation sessions anytime
🏥 DICOM Support	Clinical imaging with metadata parsing
🔐 Client-side De-identification	HIPAA/PHIPA-friendly workflow
📤 Export Formats	COCO, YOLO, MONAI, nnU-Net, PNG
Supported Modalities
Modality	Supported Data
Microwave-Induced Thermoacoustic Tomography (MITT)	S2P, CSV, MAT, DICOM
Microwave Imaging (MWI)	S2P, CSV, MAT
Photoacoustic Imaging	MAT, H5, DICOM
Ultrasound	RF, DICOM
MRI / CT	DICOM
Novel Contributions
Physics-Guided SAM Conditioning

Instead of conditioning segmentation solely on image coordinates, Hypoxify incorporates:

dielectric contrast
acoustic pressure
absorbed microwave energy
phase information

allowing accurate segmentation of challenging microwave and thermoacoustic images.

Linear-Domain Background Subtraction

Background removal occurs before logarithmic conversion.

This preserves additive signal behavior and improves contrast in high-coupling microwave measurements.

S-Parameter Phase Tokenization

Both

|S₂₁|
∠S₂₁|

are encoded into the segmentation pipeline, allowing microwave phase shifts to contribute to segmentation accuracy.

Active Learning Loop

Researchers can click uncertain regions to:

correct segmentation
identify failure cases
locally refine predictions

without restarting the annotation process.

Mobile SAM Integration

Uses the distilled 40 MB Mobile SAM model instead of the 2.5 GB original.

Benefits:

CPU inference
Free-tier deployment
Low memory footprint
Fast interaction
Annotation Workflow
Upload Data
      │
      ▼
Select Imaging Modality
      │
      ▼
Physics-Guided Mobile SAM
      │
      ▼
Candidate Segmentations
      │
      ▼
Uncertainty Heatmap
      │
      ▼
Interactive Refinement
      │
      ▼
Export Dataset
Supported File Types
Format	Extensions
DICOM	.dcm
Touchstone	.s2p
CSV	.csv
MATLAB	.mat
Images	.png .jpg .bmp .tiff
Projects	.json
Performance
Metric	Result
Physics-guided segmentation	✅
3D mask propagation	✅
Uncertainty estimation	✅
Raw-data reconstruction	93.3%
DICOM parsing	✅
Phase tokenization	✅
Memory usage	<512 MB
Installation
git clone https://github.com/HeavenlyCloudz/Hypoxify-Annotation-Suite.git

cd Hypoxify-Annotation-Suite

pip install -r requirements.txt

python app.py
Dependencies
Gradio
PyTorch
Mobile SAM
OpenCV
NumPy
SciPy
Pillow
pandas
pydicom
scikit-image
Deploy on Render
Build Command:
pip install -r requirements.txt

Start Command:
python app.py
Project Structure
Hypoxify-Annotation-Suite/
│
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
├── mobile_sam.pt
└── saved_projects/
Clinical Features
Client-side DICOM de-identification
Physics-informed segmentation
Active clinician-in-the-loop workflow
MONAI / nnU-Net export
Multi-modality imaging support
Roadmap
Completed
Mobile SAM integration
Multi-modality support
Active learning
DICOM parsing
In Progress
Full volumetric reconstruction
Clinical validation dashboard
Planned
Hypoxify-Net foundation model
Federated learning
OHIF Viewer integration
Contributing
git checkout -b feature/my-feature

git commit -m "Add awesome feature"

git push origin feature/my-feature

Then open a Pull Request.

License

Released under the MIT License.

Acknowledgements
Dr. Elise Fear
Meta AI
Mobile SAM
Gradio
PyTorch
OpenCV
pydicom
Calgary Youth Science Fair
Contact

Anie Udofia

📧 anieudofia8@gmail.com

🐙 https://github.com/HeavenlyCloudz

⭐ Support the Project

If Hypoxify helps your research, consider giving the repository a ⭐ to support future development.
