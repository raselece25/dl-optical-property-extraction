# 🧠 Deep Learning for Optical Property Extraction

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![NIH Funded](https://img.shields.io/badge/NIH-R01%20Funded-brightgreen?style=flat-square)](https://www.nih.gov/)

> CNN and U-Net based deep learning solvers replacing slow iterative fitting in Spatial Frequency Domain Imaging (SFDI) — achieving real-time optical property maps (μa, μs′) with high spatial resolution and accuracy.

---

## 📖 Overview

Traditional optical property extraction in SFDI relies on pixel-wise iterative fitting against a diffusion or Monte Carlo model — computationally expensive and unsuitable for real-time clinical use.

This project replaces that bottleneck with **deep learning inference**, training convolutional neural networks (CNNs) and U-Net encoder-decoder architectures on synthetic Monte Carlo-generated datasets to predict tissue optical properties directly from raw SFDI measurements.

### Key Advantages over Iterative Fitting

| Metric | Iterative Fitting | Deep Learning (Ours) |
|--------|------------------|----------------------|
| Inference time (512×512) | ~45 s | **< 50 ms** |
| Robustness to noise | Moderate | High |
| Spatial resolution | Pixel-wise | Full-image contextual |

---

## ✨ Key Features

- **U-Net encoder-decoder** for spatially-aware optical property prediction
- **Lightweight CNN solver** for real-time single-pixel inference
- **Monte Carlo synthetic training data** generation pipeline (μa × μs′ grid)
- **Multi-frequency SFDI input** — configurable spatial frequency combinations
- **Noise augmentation** — simulated photon shot noise and detector noise
- **ONNX export** — deployment-ready model export for clinical integration

---

## 🏗️ Architecture

```
Input: AC/DC reflectance maps (N_freq × H × W)
          │
    ┌─────▼──────┐
    │  Encoder   │  ResNet/VGG backbone
    │  (CNN)     │
    └─────┬──────┘
          │ skip connections
    ┌─────▼──────┐
    │  Decoder   │  Upsampling blocks
    │  (U-Net)   │
    └─────┬──────┘
          │
Output: [μa map, μs′ map]  (H × W × 2)
```

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Deep Learning | PyTorch, TensorFlow/Keras |
| Data Pipeline | NumPy, Pandas, HDF5 |
| Image Processing | OpenCV, scikit-image, torchvision |
| Visualization | Matplotlib, TensorBoard, W&B |
| Monte Carlo | Custom Python, MCML |
| Model Export | ONNX, TorchScript |

---

## 🗂️ Repository Structure

```
dl-optical-property-extraction/
├── models/
│   ├── unet.py              # U-Net architecture (PyTorch)
│   ├── cnn_solver.py        # Lightweight CNN solver
│   └── losses.py            # Custom loss functions
├── data/
│   ├── monte_carlo_gen.py   # Synthetic dataset generation
│   ├── dataset.py           # PyTorch Dataset class
│   └── augmentation.py      # Noise & augmentation transforms
├── training/
│   ├── train.py             # Training loop
│   ├── evaluate.py          # Metrics & evaluation
│   └── config.yaml          # Hyperparameter config
├── inference/
│   ├── predict.py           # Real-time inference pipeline
│   └── export_onnx.py       # ONNX model export
└── notebooks/
```

---

## 🚀 Getting Started

```bash
git clone https://github.com/raselece25/dl-optical-property-extraction.git
cd dl-optical-property-extraction
pip install -r requirements.txt
```

```bash
python training/train.py --model unet --epochs 100 --batch_size 32 --lr 1e-4
```

```python
from inference.predict import OPSolver
solver = OPSolver.from_checkpoint('runs/exp01/best_model.pth')
mu_a, mu_sp = solver.predict(ac_dc_maps)
```

---

## 📊 Training Dataset

- **μa range**: 0.001 – 0.5 mm⁻¹ (50 values)
- **μs′ range**: 0.1 – 3.0 mm⁻¹ (50 values)
- **Total samples**: 2,500 (μa × μs′ combinations)
- **Spatial frequencies**: 0, 0.05, 0.10, 0.15, 0.20 mm⁻¹
- **Noise augmentation**: Poisson noise at 5 SNR levels

---

## 📚 Publications

1. **Ahmmed, R.**, Kluiszo, E., Sunar, U. (2025). *Quantitative Fluorescence Imaging of Chemophototherapy Drug Pharmacokinetics Using Laparoscopic SFDI*. **Int. J. Molecular Sciences**, Q1, IF 4.9.

---

## 🏆 Funding

NIH R01-funded — National Cancer Institute, Stony Brook University (2023–2026).

---

## 👤 Author

**Rasel Ahmmed** — PhD Candidate, Biomedical Engineering, Stony Brook University
[LinkedIn](https://www.linkedin.com/in/raahmmed) · [Portfolio](https://raselece25.github.io) · [Email](mailto:rasel.ahmmed@stonybrook.edu)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
