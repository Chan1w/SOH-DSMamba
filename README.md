# 🚀 DSMamba-net  
## Dual-stream Mamba network with nonuniform feature fusion for lithium-ion battery state-of-health estimation

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)]()
[![Journal of Energy Storage](https://img.shields.io/badge/Published-Journal%20of%20Energy%20Storage-1f8acb.svg)]()
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()

---

## 🔬 Background & Motivation

Accurate State of Health (SOH) estimation is essential for:

- Battery management system (BMS) reliability  
- Electric vehicle safety assurance  
- Lifecycle-aware energy optimization  
- Prognostics and maintenance scheduling  

However, lithium-ion battery degradation data exhibit:

- Nonuniform feature distributions  
- Strong nonlinearity and nonstationarity  
- Long-term temporal dependencies  
- Noise interference in late-life stages  

Conventional RNNs and Transformer-based models:

1. Struggle with efficient long-sequence modeling  
2. Suffer from high computational complexity  
3. Often overlook multi-scale degradation characteristics  
4. Show limited robustness under diverse operating conditions  

To address these challenges, this work proposes a **multi-scale feature learning + dual-stream Mamba architecture** for efficient and robust SOH estimation.

---

## 🧠 Proposed Method: DSMamba net

The proposed framework integrates multi-scale CNN-based feature extraction with bidirectional state-space modeling.

The overall architecture consists of:

- **Multi-Scale Feature Extraction Module (FEM)**
- **Dual-Stream Mamba Module**
- **Task Output Layer**


### 1️⃣ Multi-Scale Feature Extraction Module (FEM)

Battery raw voltage sequences are directly used as model inputs to avoid manual feature engineering.

The FEM:

- Employs multi-scale 1D convolution layers (different kernel sizes)  
- Captures degradation features at multiple temporal resolutions  
- Enhances nonlinear representation capability  
- Improves convergence speed via sparse/dense transformation strategy  

This module effectively extracts degradation-sensitive features from raw voltage signals.


### 2️⃣ Dual-Stream Mamba Architecture

To address long-term dependency modeling limitations of RNNs and Transformers, we adopt a **dual-stream Mamba structure**:

- Forward Mamba block → captures chronological degradation evolution  
- Backward Mamba block → captures reverse temporal dependencies  
- Bidirectional information fusion → enhances global contextual modeling  

Advantages:

- Efficient state-space modeling  
- Reduced computational overhead  
- Improved long-sequence learning capability  
- Better robustness to noise  


### 3️⃣ Task Layer

A lightweight MLP-based task layer:

- Maps high-dimensional features to SOH output  
- Enables end-to-end training  
- Supports efficient deployment in BMS environments  

---

## 💻 Environment

- torch==2.0.0
- numpy==1.22.4
- pandas==1.4.2
- matplotlib==3.6.0
- scikit-learn==1.0.2

Install dependencies:

```
pip install -r requirements.txt
```
---

## 🏃 Training & Testing
```
python main.py
```

Hyperparameters (learning rate, window size, decomposition parameters) can be modified in configuration files or script arguments.

---

## 📌 Citation
If you find this work useful, please consider citing:

```bibtex
@article{shen2025msf_mamba,
  title={A Multi-Scale Feature Extraction and Dual-Stream Mamba Framework for Lithium-Ion Battery SOH Estimation},
  author={Shen, Quanyong and Li, Jian and Nie, Jiahao and Bao, Zhengyi and Wang, Chenhan},
  journal={Journal of Energy Storage},
  year={2025}
}
```
