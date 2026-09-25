# Semiconductor Wafer Map Classifier (PyTorch)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-WM--811K-blue.svg)](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

A lightweight, high-performance deep learning classifier built in **PyTorch** to detect and classify spatial defect patterns on semiconductor wafer maps, designed to run smoothly on **Google Colab** with GPU acceleration.

---

## Quick Overview

In semiconductor manufacturing, spatial defect patterns on silicon wafers indicate critical fabrication equipment issues. This project automates wafer defect recognition on the **WM-811K (LSWMD)** dataset using:
- **ResNet Backbones (ResNet-18 / ResNet-50)** adapted for die-level wafer maps.
- **Focal Loss** to handle extreme class imbalance across defect types.
- **Grad-CAM Explainability** to produce heatmaps localizing the exact physical defect regions.

---

## The 9 Wafer Defect Patterns

| Defect Pattern | Description / Fab Implication |
|:---|:---|
| **Center** | Failures concentrated at the center (e.g., spin coating, CMP pressure issues) |
| **Donut** | Circular ring around the center (e.g., thermal gradients, gas flow non-uniformity) |
| **Edge-Loc** | Localized cluster along the wafer boundary (e.g., edge bead removal, handling error) |
| **Edge-Ring** | Concentric defect ring at the outer perimeter (e.g., clamp ring, plasma etching) |
| **Loc (Local)** | Clustered defects on interior dies (e.g., particle contamination) |
| **Random** | Dispersed die failures with no systematic pattern |
| **Scratch** | Linear/curved scratch signatures (e.g., robotic handler or tweezer slip) |
| **Near-full** | Massive die failures covering nearly the entire wafer |
| **None (Normal)** | High-yield golden wafer without systematic defect signatures |

---

## Real-World Fab Realities & Dataset Challenges

While WM-811K is the standard benchmark for wafer defect analysis, **it reflects the messy, noisy reality of high-volume semiconductor manufacturing rather than a clean academic dataset**:

- **Annotation Scarcity**: Over **78%** of the 811,457 wafer maps in the dataset are completely unlabeled, with only ~172,950 possessing defect pattern tags.
- **Subjective & Imperfect Labels**: Annotations stem from legacy heuristic rules and operator judgment, leading to noticeable label noise and inconsistencies across lots.
- **Composite / Mixed Defect Ambiguity**: Real fabrication processes frequently produce multiple overlapping failure mechanisms (e.g., a *Scratch* cutting across an *Edge-Ring* or *Center* defect), yet the dataset forces each wafer into a single class label.
- **Severe Class Imbalance**: High-yield golden wafers (`None`) represent >80% of labeled samples, while critical, yield-killing defect types (`Scratch`, `Donut`, `Near-full`) account for less than 1-2% each.
- **Subtle Defects vs. Sensor Noise**: Low-density defect clusters are frequently mislabeled as `None`, while harmless random die test artifacts are sometimes tagged as `Random`.

### How This Project Addresses These Challenges:
1. **Focal Loss ($\gamma=2.0$)**: Prevents the ubiquitous `None` (golden) wafers from overwhelming the gradient signal, forcing the network to learn subtle boundaries of rare defect classes.
2. **Grad-CAM Visual Verification**: Because ground-truth labels in production cannot always be trusted blindly, Grad-CAM provides an essential sanity check for yield engineers by confirming whether the model is focusing on genuine physical defect signatures or peripheral fab noise.

---

## Running on Google Colab

### 1. Enable GPU
Go to **Runtime** > **Change runtime type** > Select **T4 GPU** (or any available GPU accelerator).

### 2. Download WM-811K Dataset
In a Colab cell, download the dataset via Kaggle API:
```python
# Upload kaggle.json or set API credentials
import os
os.environ['KAGGLE_CONFIG_DIR'] = "/content"
!kaggle datasets download -d qingyi/wm811k-wafer-map
!unzip -q wm811k-wafer-map.zip -d ./data
```

### 3. Pipeline Flow
```
 1. Load LSWMD.pkl (811,457 wafer maps)
 2. Filter labeled defect samples & normalize die coordinates
 3. Augment data (90° rotations & flips) + Apply stratified split
 4. Train ResNet-18 / ResNet-50 with Focal Loss (Handles imbalance)
 5. Evaluate (Macro F1-Score, Confusion Matrix)
 6. Visualize predictions & defect regions via Grad-CAM heatmaps
```

---

## Model & Training Details

- **Input Representation**: Single-channel die failure status maps resized to $64 \times 64$ (or $224 \times 224$).
- **Architecture**: `ResNet-18` (fast training on Colab T4) or `ResNet-50`.
- **Loss Function**: Multi-Class **Focal Loss** ($\gamma=2.0$) to penalize easy negatives and focus gradient updates on rare defect patterns (e.g., *Scratch*, *Donut*).
- **Optimization**: AdamW ($\text{lr} = 10^{-3}$, weight decay $= 10^{-4}$) with Cosine Annealing scheduler.
- **Explainable AI (XAI)**: Grad-CAM generates visual attention heatmaps overlaid on the wafer map to verify the model targets true defect signatures rather than peripheral noise.

---

## License

Distributed under the **Apache License 2.0**. See [LICENSE](LICENSE) for details.
