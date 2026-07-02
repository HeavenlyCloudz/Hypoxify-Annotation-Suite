# 🔬 Hypoxify Annotation Suite

> **Physics-informed segmentation for microwave and thermoacoustic imaging**

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)
![Status](https://img.shields.io/badge/Status-Alpha-orange.svg)

---

# 📖 The Problem We Solve

Biomedical researchers studying emerging imaging modalities—including **Microwave-Induced Thermoacoustic Tomography (MITT)**, **microwave imaging**, and **thermoacoustic imaging**—face a major annotation bottleneck.

Manual annotation typically requires **30–48 minutes per case** because:

- Standard AI models (SAM, UNet) struggle with blurry, artifact-heavy MITT images.
- Existing annotation tools are not designed for microwave or thermoacoustic physics.
- Researchers spend significant time converting data between incompatible software.
- There is no unified workflow from **reconstruction → segmentation → export**.

**Hypoxify Annotation Suite** addresses these challenges by integrating microwave and thermoacoustic physics directly into the segmentation pipeline, enabling efficient annotation for datasets that are difficult for conventional computer vision models.

---

# ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🧬 **Physics-Guided Segmentation** | SAM conditioned on dielectric contrast and acoustic pressure—not just image pixels |
| 📡 **Raw Data Support** | Load CSV, S2P, MAT, DICOM, and HDF5 files directly |
| 🔨 **Signal Processing** | Linear-domain background subtraction, time-gain compensation, denoising |
| 📊 **Multi-Angle Reconstruction** | Rotation averaging for improved signal-to-noise ratio |
| 🎯 **Multiple Imaging Configurations** | Microwave, thermoacoustic, MRI, CT, histology |
| 📤 **Export Anywhere** | COCO, YOLO, MONAI, PNG, DICOM SEG |
| 🎨 **No-Code Interface** | Streamlit dashboard for annotation without programming |
| 🔬 **Python API** | Integrate directly into research pipelines |

---

# 🚀 Quick Start

## Option 1 — Run the Streamlit App

```bash
# Clone repository
git clone https://github.com/HeavenlyCloudz/Hypoxify-Annotation-Suite.git

cd Hypoxify-Annotation-Suite

# Install dependencies
pip install -r requirements.txt

# Launch Streamlit
streamlit run app.py
```

The application will open at:

```
http://localhost:8501
```

---

## Option 2 — Use as a Python Package

```python
from hypoxify_annotate import (
    auto_load,
    delay_and_sum_reconstruction,
    SAMWrapper,
    to_coco_format
)

# Load raw data
frequencies, s21_data = auto_load("patient_scan.csv")

# Reconstruct image
s21_dict = {1: s21_data}
image = delay_and_sum_reconstruction(s21_dict, frequencies)

# Segment
segmenter = SAMWrapper()
segmenter.set_image(image)

mask = segmenter.from_clicks(
    foreground=[(100, 150)]
)

# Export
coco = to_coco_format(
    [mask],
    image_ids=[1],
    image_shapes=[image.shape]
)
```

---

# 📦 Installation

## Prerequisites

- Python 3.8+
- pip

---

## Install from Source

```bash
git clone https://github.com/HeavenlyCloudz/Hypoxify-Annotation-Suite.git

cd Hypoxify-Annotation-Suite

pip install -e .
```

---

## PyPI (Coming Soon)

```bash
pip install hypoxify-annotate
```

---

# 📚 Dependencies

| Package | Purpose |
|---------|----------|
| numpy | Numerical computing |
| scipy | Scientific computing |
| pandas | Data handling |
| pillow | Image I/O |
| streamlit | Web application |
| segment-anything | Meta Segment Anything (optional) |
| opencv-python | Polygon conversion |
| h5py | HDF5 support |
| pydicom | DICOM support |

---

# 🧠 Novel Contributions

## 1. Physics-Guided Prompting

Rather than using only image coordinates, Hypoxify extracts microwave and thermoacoustic signal characteristics—including dielectric contrast, acoustic pressure, and energy absorption estimates—to condition the segmentation process.

---

## 2. Linear-Domain Background Subtraction

Background removal is performed in the **linear power domain** before logarithmic conversion, improving suppression of direct antenna coupling while preserving weaker tissue responses.

---

## 3. Time-Domain Feature Extraction

Frequency-domain S-parameters are transformed into impulse responses, providing additional temporal scattering information that complements image-based segmentation.

---

## 4. Volumetric Propagation *(Planned)*

Future releases aim to leverage **SAM 2** memory mechanisms to propagate annotations across volumetric image stacks.

---

# 🗂️ Project Structure

```text
Hypoxify-Annotation-Suite/
│
├── app.py
├── hypoxify_annotate/
│   ├── io/
│   ├── preprocess/
│   ├── reconstruct/
│   ├── segment/
│   ├── features/
│   ├── export/
│   └── utils/
│
├── requirements.txt
├── setup.py
└── README.md
```

---

# 🖥️ Streamlit Application

## Configuration

Choose:

- Imaging system
- Reconstruction mode
- Physics-guided segmentation
- Export format

Supported systems include:

- 4-Antenna Microwave
- 8-Antenna Microwave
- Thermoacoustic Ring
- Thermoacoustic Linear
- MRI
- CT
- Histology

---

## Annotation Workflow

1. Upload raw data or an image.
2. Add foreground clicks.
3. (Optional) Add background clicks.
4. Generate segmentation.
5. Export annotations.

---

# 📚 Package Modules

| Module | Description |
|---------|-------------|
| **io** | CSV, S2P, MAT, DICOM, HDF5 loaders |
| **preprocess** | Background subtraction, filtering, denoising, TGC |
| **reconstruct** | Delay-and-sum beamforming |
| **segment** | SAM wrapper with physics-guided prompting |
| **features** | Time- and frequency-domain feature extraction |
| **export** | COCO, YOLO, MONAI, PNG |
| **utils** | Validation and antenna configurations |

---

# 🤝 Contributing

Contributions are welcome.

```bash
# Fork repository

git checkout -b feature/MyFeature

git commit -m "Add MyFeature"

git push origin feature/MyFeature
```

Then open a Pull Request.

---

## Development Setup

```bash
git clone https://github.com/HeavenlyCloudz/Hypoxify-Annotation-Suite.git

cd Hypoxify-Annotation-Suite

pip install -e .[dev]

pytest tests/
```

---

# 📄 License

This project is distributed under the **MIT License**.

See the `LICENSE` file for details.

---

# 🙏 Acknowledgments

- **Dr. Elise Fear** — Microwave imaging mentorship
- **Calgary Youth Science Fair (CYSF/CWSF pathway)** — Early project support
- **Pfizer Oncology Science Award**
- **Meta AI** — Segment Anything Model (SAM)

---

# 📬 Contact

**Anie Udofia**

📧 anie.udofia@hypoxify.ai

GitHub:

https://github.com/HeavenlyCloudz/Hypoxify-Annotation-Suite

---

## ⭐ If you find this project useful...

Please consider **starring the repository** and sharing it with other biomedical imaging researchers.
