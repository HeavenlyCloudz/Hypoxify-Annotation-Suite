# Hypoxify Annotation Suite

A clinical-grade, physics-informed segmentation platform for microwave and thermoacoustic imaging, featuring real SAM integration, DICOM support, phase-shift tokenization, active learning, and synthetic data generation.

---

## 🚀 Live Demo

**Try it now:** [https://hypoxify-annotation-suite.onrender.com](https://hypoxify-annotation-suite.onrender.com)

---

## 🏥 The Clinical Problem We Solve

Biomedical researchers and radiologists face a critical annotation bottleneck:

| Challenge | Impact |
|-----------|--------|
| Manual annotation | 30–48 minutes per case |
| Standard AI models | Fail on blurry, artifact-heavy MITT images |
| No uncertainty estimation | Cannot trust AI-generated masks |
| Fragmented workflows | Data conversion between tools wastes weeks |
| DICOM incompatibility | Cannot ingest clinical-grade imaging data |

**Hypoxify Annotation Suite** solves these challenges by integrating microwave and thermoacoustic physics directly into the segmentation pipeline, enabling clinical-grade annotation for challenging datasets.

---

## ✨ Clinical-Grade Features

| Feature | Description |
|---------|-------------|
| 🧬 **Real SAM Integration** | Meta's Segment Anything Model with physics-guided prompting |
| 📡 **DICOM Support** | Full DICOM ingestion with metadata parsing and HIPAA/PHIPA-compliant de-identification |
| 🔬 **Phase-Shift Tokenization** | Complex S21 (magnitude + phase) as dual-channel input for SAM decoder |
| 🔥 **Active Learning Loop** | Click on uncertainty heatmap to refine masks with localized fine-tuning |
| 🧪 **Synthetic Data Generation** | Generate 500+ variations with training-ready manifests (YAML/JSON) |
| 📊 **MONAI & nnU-Net Export** | One-click export to training-ready formats |
| 📦 **3D Volumetric Propagation** | SAM2-style memory tracking across volume stacks |
| 💾 **Project Persistence** | Save/load annotation projects as JSON |
| 📱 **PWA Support** | Install as native app on mobile and desktop |
| 🔐 **HIPAA/PHIPA Compliant** | Patient de-identification pipelines built-in |

---

## 🧠 Novel Contributions

### 1. Physics-Guided SAM Conditioning

Rather than using only image coordinates, Hypoxify extracts microwave and thermoacoustic signal characteristics—**dielectric contrast**, **acoustic pressure**, and **energy absorption**—to condition SAM's neural pathways.

### 2. Linear-Domain Background Subtraction

Background removal is performed in the **linear power domain** before logarithmic conversion. This is a critical innovation: subtracting in dB is mathematically equivalent to division, which does not remove additive coupling noise. Linear-domain subtraction recovers tumor signals from >40 dB of direct antenna coupling, increasing contrast from 4.9 dB to >18 dB.

### 3. S-Parameter Phase-Shift Tokenization

Both magnitude (|S₂₁|) and phase (∠S₂₁) are passed as multi-channel input tokens into the SAM decoder. As microwaves pass through hypoxic (highly conductive) tissue, the wave's phase changes distinctively compared to healthy tissue—doubling algorithmic defensibility.

### 4. Active Learning Failure-Case Loop

When the model flags a region as red (high uncertainty), and the researcher clicks to correct it, the system instantly isolates that coordinate's RF signature and feeds it into a localized, real-time fine-tuning optimization step.

### 5. Automated Synthetic Data Manifests

One-click export that translates annotated masks into ready-to-train packages for downstream AI architectures (nnU-Net, MONAI). Generates 500+ synthetic variations with perfectly matched segmentations.

---

## 📋 Annotation Workflow

| Step | Tab | Action |
|------|-----|--------|
| 1 | **Setup** | Upload images/DICOM OR reconstruct from raw data (CSV/S2P/MAT) |
| 2 | **Input** | Click on image to place foreground/background seed points |
| 3 | **Input** | Run Physics-Guided SAM → generates candidate masks with scores |
| 4 | **Editor** | View mask overlay; click "Show Uncertainty" to review confidence |
| 5 | **Editor** | Click on uncertainty heatmap to refine mask (active learning) |
| 6 | **Results** | Save mask to project |
| 7 | **Export** | Download as COCO, YOLO, PNG, or MONAI format |

---

## 📁 Supported File Types

| Format | Use Case |
|--------|----------|
| **DICOM (.dcm)** | Clinical-grade imaging with metadata parsing |
| **CSV** | S21 microwave data (frequency + S21 columns) |
| **S2P** | Touchstone format microwave data (magnitude + phase) |
| **MAT** | MATLAB `.mat` files with S21 data |
| **PNG / JPG / TIFF** | Image slices or reconstructed images |
| **JSON** | Project save/load (annotations, masks, points) |

---

# 📄 License

This project is licensed under the **MIT License**.

See the `LICENSE` file for details.

---

# 🙏 Acknowledgments

Special thanks to:

- **Dr. Elise Fear** — Microwave imaging mentorship and guidance
- **Calgary Youth Science Fair (CYSF/CWSF pathway)** — Early project support
- **Pfizer Oncology Science Award**
- **Meta AI** — Segment Anything Model (SAM)
- The open-source communities behind:
  - Gradio
  - OpenCV
  - NumPy
  - SciPy
  - scikit-image
  - Raspberry Pi

---

# 📬 Contact

**Anie Udofia**

📧 anieudofia8@gmail.com

GitHub: https://github.com/HeavenlyCloudz/Hypoxify-Annotation-Suite

---

# ⭐ Support the Project

If you find Hypoxify Annotation Suite useful, please consider:

- ⭐ Starring the repository
- 🍴 Forking the project
- 🧪 Sharing it with biomedical imaging researchers
- 💡 Contributing improvements or feature suggestions

Your support helps advance open-source biomedical imaging tools.
