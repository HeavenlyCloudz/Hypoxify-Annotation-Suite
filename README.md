# Hypoxify Annotation Suite

A **physics-informed biomedical image annotation platform** for microwave and thermoacoustic imaging, featuring candidate mask selection, uncertainty quantification, raw signal reconstruction, and an end-to-end annotation workflow.

---

## 🚀 Live Demo

**Try it here:** https://hypoxify-annotation-suite.onrender.com

> **Note:** The free Render tier may take 30–60 seconds to wake up after inactivity.

---

# 📖 The Problem

Biomedical researchers developing emerging imaging technologies—including **Microwave-Induced Thermoacoustic Tomography (MITT)**, **microwave imaging**, and **thermoacoustic imaging**—often face an annotation bottleneck.

| Challenge | Impact |
|------------|--------|
| Manual annotation | 30–48 minutes per image; impractical for large datasets or 3D volumes |
| Conventional AI segmentation (SAM, Cellpose) | Trained primarily on natural or optical images; struggle with blurry boundaries and scattering artifacts |
| No uncertainty estimation | Difficult to determine when AI predictions are reliable |
| Fragmented workflow | Reconstruction, annotation, and export require multiple disconnected tools |

**Hypoxify Annotation Suite** addresses these limitations by integrating microwave and thermoacoustic physics directly into the segmentation pipeline, enabling efficient annotation for datasets that are challenging for conventional computer vision models.

---

# ✨ Features

| Feature | Description |
|---------|-------------|
| 🧬 **Physics-Guided Segmentation** | Conditions segmentation using dielectric contrast, acoustic pressure, and absorption estimates—not just image pixels |
| 📡 **Raw Data Reconstruction** | Import CSV, S2P, and MATLAB files and reconstruct images with delay-and-sum beamforming |
| 🔬 **Linear-Domain Background Subtraction** | Removes antenna coupling in the linear power domain for substantially improved contrast |
| 🎯 **Multi-Candidate Selection** | Generates three candidate masks for user review and selection |
| 🔥 **Uncertainty Heatmaps** | Pixel-level confidence visualization highlighting uncertain regions |
| 📦 **3D Volumetric Propagation** | Propagates annotations throughout image stacks using optical-flow-inspired tracking |
| 💾 **Project Persistence** | Save and resume annotation sessions using JSON project files |
| 📤 **Multiple Export Formats** | Export annotations as COCO JSON, YOLO TXT, or PNG masks |
| 📱 **Progressive Web App (PWA)** | Installable on desktop and mobile devices |
| 🎨 **No-Code Interface** | Browser-based Gradio application requiring no programming experience |

---

# 🧠 Novel Contributions

## 1. Physics-Guided Prompting

Instead of relying solely on image coordinates, Hypoxify extracts microwave and thermoacoustic signal characteristics—including:

- Dielectric contrast
- Acoustic pressure
- Energy absorption estimates

These physical priors guide segmentation, enabling improved performance on blurry, artifact-heavy MITT datasets where traditional segmentation models often struggle.

---

## 2. Linear-Domain Background Subtraction

Background removal is performed **before logarithmic conversion** in the **linear power domain**.

Unlike subtraction performed in decibels (which is mathematically equivalent to division), linear-domain subtraction removes additive antenna coupling and improves tumor visibility.

Reported improvements include:

- Direct coupling exceeding **40 dB**
- Contrast improvement from approximately **4.9 dB** to **>18 dB**

---

## 3. Multi-Candidate Segmentation

The platform produces **three candidate masks** together with confidence scores, allowing researchers to choose the most appropriate segmentation for each image while retaining a familiar SAM-style workflow.

---

## 4. Uncertainty Quantification

Two complementary uncertainty sources are estimated:

- **Signal uncertainty**
  - Low signal-to-noise ratio
  - Weak thermoacoustic response
  - Poor microwave contrast

- **Model uncertainty**
  - Ambiguous boundaries
  - Geometric uncertainty
  - Low prediction confidence

This supports transparent AI-assisted annotation workflows and aligns with current recommendations for uncertainty reporting in medical AI.

---

## 5. 3D Volumetric Propagation

Using memory-inspired propagation strategies, Hypoxify tracks physical signal characteristics across neighboring slices to generate consistent volumetric annotations with minimal user interaction.

---

# 📋 Annotation Workflow

| Step | Interface | Action |
|------|-----------|--------|
| **1** | Setup | Upload reconstructed images or import raw microwave data (CSV/S2P/MAT) |
| **2** | Input | Place foreground seed points on the image |
| **3** | Results | Review and select one of three candidate masks |
| **4** | Editor | Inspect overlays and visualize uncertainty heatmaps |
| **5** | Export | Download annotations in COCO, YOLO, or PNG formats |
| **Optional** | 3D Propagation | Propagate masks throughout an image volume |
| **Optional** | User Guide | Access built-in documentation |

---

# 📁 Supported File Types

| Format | Purpose |
|---------|---------|
| **CSV** | Microwave S21 measurements |
| **S2P** | Touchstone microwave files |
| **MAT** | MATLAB reconstruction files |
| **PNG / JPG / TIFF** | Image slices and reconstructed images |
| **JSON** | Project save files |

---

# 🛠️ Installation

## Clone the Repository

```bash
git clone https://github.com/HeavenlyCloudz/Hypoxify-Annotation-Suite.git
cd Hypoxify-Annotation-Suite
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Launch

```bash
python app.py
```

---

# 📦 Requirements

```
gradio>=4.0.0
opencv-python-headless
numpy
scipy
scikit-image
Pillow
pandas
```

---

# ☁️ Deploy on Render

1. Fork or clone the repository.
2. Create a new **Web Service** on Render.
3. Connect the GitHub repository.
4. Set the build command:

```bash
pip install -r requirements.txt
```

5. Set the start command:

```bash
python app.py
```

6. Deploy.

---

# 📂 Project Structure

```text
Hypoxify-Annotation-Suite/
│
├── app.py
├── README.md
├── requirements.txt
├── LICENSE
└── saved_projects/
```

---

# 🔬 Validation

| Metric | Goal | Current Status |
|---------|------|----------------|
| Physics-guided segmentation | >85% IoU | Demonstration implementation |
| 3D propagation | >90% IoU | Optical-flow-based propagation |
| Uncertainty calibration | r > 0.8 | Signal + epistemic estimation |
| Raw data reconstruction | Phantom validation | Delay-and-sum beamforming |

> **Current Release:** This version includes a demonstration segmentation backend. Future versions are planned to support integration with Meta AI's Segment Anything Model (SAM).

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
