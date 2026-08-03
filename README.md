# Hypoxify Annotation Suite

A clinical-grade, multi-modality physics-informed segmentation platform for microwave, thermoacoustic, photoacoustic, and ultrasound imaging, featuring real SAM integration, DICOM support, phase-shift tokenization, active learning, and synthetic data generation.

---

## 🚀 Live Demo

**Try it now:** [https://hypoxify-annotation-suite.onrender.com](https://hypoxify-annotation-suite.onrender.com)

---

## 🏥 The Clinical Problem We Solve

Biomedical researchers and radiologists face a critical annotation bottleneck:

| Challenge | Impact |
|-----------|--------|
| Manual annotation | 30–48 minutes per case |
| Standard AI models | Fail on blurry, artifact-heavy medical images |
| No uncertainty estimation | Cannot trust AI-generated masks |
| Fragmented workflows | Data conversion between tools wastes weeks |
| DICOM incompatibility | Cannot ingest clinical-grade imaging data |
| Modality-specific tools | Each imaging modality requires different software |

**Hypoxify Annotation Suite** solves these challenges by integrating physics directly into the segmentation pipeline, enabling clinical-grade annotation across multiple non-ionizing imaging modalities.

---

## ✨ Clinical-Grade Features

| Feature | Description |
|---------|-------------|
| 🧬 **Real SAM Integration** | Mobile SAM (lightweight, ~40MB) with physics-guided prompting |
| 📡 **DICOM Support** | Full DICOM ingestion with client-side HIPAA/PHIPA-compliant de-identification |
| 🔬 **Phase-Shift Tokenization** | Complex S21 (magnitude + phase) as dual-channel input for SAM decoder |
| 🔥 **Active Learning Loop** | Click on uncertainty heatmap to refine masks with localized fine-tuning |
| 🧪 **Synthetic Data Generation** | Generate 50+ variations with training-ready manifests (JSON/YAML) |
| 📊 **MONAI & nnU-Net Export** | One-click export to training-ready formats |
| 📦 **3D Volumetric Propagation** | SAM2-style memory tracking across volume stacks using optical flow |
| 💾 **Project Persistence** | Save/load annotation projects as JSON |
| 📱 **PWA Support** | Install as native app on mobile and desktop |
| 🔐 **HIPAA/PHIPA Compliant** | Client-side patient de-identification pipelines built-in |
| 🎯 **Multi-Modality Support** | MITT, Microwave Imaging, Photoacoustic, Ultrasound |

---

## 🧠 Novel Contributions

### 1. Physics-Guided SAM Conditioning

Rather than using only image coordinates, Hypoxify extracts modality-specific signal characteristics—**dielectric contrast**, **acoustic pressure**, **energy absorption**, and **phase shift**—to condition SAM's neural pathways. This enables accurate annotation on blurry, artifact-ridden medical images where standard models fail.

### 2. Linear-Domain Background Subtraction

Background removal is performed in the **linear power domain** before logarithmic conversion. This is a critical innovation: subtracting in dB is mathematically equivalent to division, which does not remove additive coupling noise. Linear-domain subtraction recovers tumor signals from >40 dB of direct antenna coupling, increasing contrast from 4.9 dB to >18 dB.

### 3. S-Parameter Phase-Shift Tokenization

Both magnitude (|S₂₁|) and phase (∠S₂₁) are passed as multi-channel input tokens into the SAM decoder. As microwaves pass through hypoxic (highly conductive) tissue, the wave's phase changes distinctively compared to healthy tissue—doubling algorithmic defensibility for microwave and thermoacoustic modalities.

### 4. Active Learning Failure-Case Loop

When the model flags a region as red (high uncertainty), and the researcher clicks to correct it, the system instantly isolates that coordinate's RF signature and feeds it into a localized, real-time fine-tuning optimization step. This closes the clinician-in-the-loop workflow.

### 5. Multi-Modality Physics Engine

The platform automatically detects and adapts to different imaging modalities:
- **MITT**: Microwave-induced thermoacoustic tomography
- **Microwave Imaging**: Direct microwave transmission/reflection
- **Photoacoustic Imaging**: Laser-induced ultrasound
- **Ultrasound**: Traditional pulse-echo imaging
- **DICOM**: MRI, CT, and other clinical formats

### 6. Automated Synthetic Data Manifests

One-click export that translates annotated masks into ready-to-train packages for downstream AI architectures (nnU-Net, MONAI). Generates 50+ synthetic variations with perfectly matched segmentations and complete training manifests.

---

## 📋 Annotation Workflow

| Step | Tab | Action |
|------|-----|--------|
| 1 | **Setup** | Upload images/DICOM OR reconstruct from raw data (CSV/S2P/MAT) |
| 2 | **Setup** | Select imaging modality (MITT, MWI, PAI, Ultrasound) |
| 3 | **Setup** | Initialize Mobile SAM (lightweight, ~40MB) |
| 4 | **Input** | Click on image to place foreground/background seed points |
| 5 | **Input** | Run Physics-Guided SAM → generates candidate masks with scores |
| 6 | **Editor** | View mask overlay; click "Show Uncertainty" to review confidence |
| 7 | **Editor** | Click on uncertainty heatmap to refine mask (active learning) |
| 8 | **Results** | Save mask to project |
| 9 | **Export** | Download as COCO, YOLO, PNG, MONAI, or nnU-Net format |

---

## 📁 Supported File Types

| Format | Use Case |
|--------|----------|
| **DICOM (.dcm)** | Clinical-grade imaging with client-side de-identification |
| **CSV** | S21 microwave data (frequency + S21 columns) |
| **S2P** | Touchstone format microwave data (magnitude + phase) |
| **MAT** | MATLAB `.mat` files with S21 data |
| **PNG / JPG / TIFF** | Image slices or reconstructed images |
| **JSON** | Project save/load (annotations, masks, points) |

---

## 🛠️ Dependencies

gradio>=4.0.0
opencv-python-headless
numpy
scipy
scikit-image
Pillow
pandas
torch>=2.0.0
torchvision>=0.15.0
mobile-sam
pydicom

## 📄 License
This project is distributed under the MIT License.

## 📬 Contact
Anie Udofia
📧 anieudofia8@gmail.com
🔗 [GitHub](https://github.com/HeavenlyCloudz/Hypoxify-Annotation-Suite)


cd Hypoxify-Annotation-Suite
pip install -r requirements.txt
python app.py
