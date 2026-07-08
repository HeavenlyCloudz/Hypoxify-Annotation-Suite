# Hypoxify Annotation Suite

A physics‑informed segmentation platform for microwave and thermoacoustic imaging, featuring a professional project workflow, candidate selection, and uncertainty quantification.

# The Problem We Solve

Biomedical researchers studying emerging imaging modalities—including **Microwave-Induced Thermoacoustic Tomography (MITT)**, **microwave imaging**, and **thermoacoustic imaging**—face a major annotation bottleneck.

Manual annotation typically requires **30–48 minutes per case** because:

- Standard AI models (SAM, UNet) struggle with blurry, artifact-heavy MITT images.
- Existing annotation tools are not designed for microwave or thermoacoustic physics.
- Researchers spend significant time converting data between incompatible software.
- There is no unified workflow from **reconstruction → segmentation → export**.

**Hypoxify Annotation Suite** addresses these challenges by integrating microwave and thermoacoustic physics directly into the segmentation pipeline, enabling efficient annotation for datasets that are difficult for conventional computer vision models.

---

# Key Features

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


# Novel Contributions

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


## Annotation Workflow

1. Upload raw data or an image.
2. Add foreground clicks.
3. (Optional) Add background clicks.
4. Generate segmentation.
5. Export annotations.

---

# Contributing

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

# License

This project is distributed under the **MIT License**.

See the `LICENSE` file for details.

---

# Acknowledgments

- **Dr. Elise Fear** — Microwave imaging mentorship
- **Calgary Youth Science Fair (CYSF/CWSF pathway)** — Early project support
- **Pfizer Oncology Science Award**
- **Meta AI** — Segment Anything Model (SAM)

---

# Contact

**Anie Udofia**

📧 anieudofia8@gmail.com

GitHub:

https://github.com/HeavenlyCloudz/Hypoxify-Annotation-Suite

---

## ⭐ If you find this project useful...

Please consider **starring the repository** and sharing it with other biomedical imaging researchers.
