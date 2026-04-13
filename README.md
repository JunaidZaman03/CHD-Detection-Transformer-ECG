# PACER: Pediatric Automated Cardiac ECG Recognition

**Early Diagnosis of Congenital Heart Disease Using Transformer-Based Deep Learning on Electrocardiogram Signals**

> Bachelor's Thesis — Sichuan University, College of Software Engineering  
> **Author:** Junaid Zaman  
> **Adviser:** Associate Professor Zhang Haixian  
> **Date:** April 2025

---

## Abstract

Congenital heart disease (CHD) remains a leading cause of neonatal morbidity and mortality worldwide. Traditional diagnostic methods such as echocardiography and MRI are expensive, require specialized expertise, and are often inaccessible in resource-limited settings. This thesis presents **PACER**, a transformer-based deep learning model for automated CHD detection from 12-lead pediatric ECG signals.

PACER integrates three complementary input streams — raw ECG waveforms, wavelet-transformed features, and human-concept clinical attributes — through a multi-path architecture combining 1D convolutional residual blocks, a Transformer encoder with temporal attention, and TabNet-based feature selection. The model was trained and evaluated on the Georgia-12Lead-ECG-Challenge-Dataset comprising 10,344 pediatric ECG cases.

### Key Results

| Metric | Score |
|--------|-------|
| Accuracy | **90.93%** |
| AUC | **0.9402** |
| F1 Score | **0.8513** |
| Recall | **0.85** |
| Precision | **0.83** |

The model outperformed expert cardiologists in identifying subtle ECG manifestations of CHD on the Net Reclassification Index (NRI) assessment, demonstrating significant promise for clinical deployment in resource-constrained environments.

---

## Model Architecture

```
                        ┌─────────────────┐
                        │  Raw ECG Input   │
                        │   (5000 × 12)    │
                        └────────┬─────────┘
                                 │
                        ┌────────▼─────────┐
                        │  Pre-processing   │
                        └────────┬─────────┘
                ┌────────────────┼────────────────┐
                │                │                │
    ┌───────────▼──┐   ┌────────▼────────┐  ┌────▼──────────┐
    │   Wavelet    │   │  1D ConvBlock   │  │   Clinical    │
    │  Features    │   │  + ResBlocks    │  │   Features    │
    └──────┬───────┘   └────────┬────────┘  └────┬──────────┘
           │                    │                 │
    ┌──────▼───────┐   ┌────────▼────────┐  ┌────▼──────────┐
    │   TabBlock   │   │  Transformer    │  │   TabBlock    │
    │              │   │  Encoder +      │  │               │
    │              │   │  Temporal       │  │               │
    │              │   │  Attention      │  │               │
    └──────┬───────┘   └────────┬────────┘  └────┬──────────┘
           │                    │                 │
           └────────────┬───────┘─────────────────┘
                        │
                ┌───────▼────────┐
                │  Concatenation │
                │  + TabBlock    │
                │  + Dense       │
                │  + Dropout     │
                └───────┬────────┘
                        │
                ┌───────▼────────┐
                │   Output       │
                │ (CHD / Normal) │
                └────────────────┘
```

### Core Components

- **Multi-path 1D ResBlocks** with kernel sizes 3, 5, and 7 for multi-scale local pattern extraction
- **Transformer Encoder** with positional encoding and multi-head self-attention for long-range temporal dependencies
- **Temporal Attention Mechanism** to focus on diagnostically critical time steps in the ECG signal
- **Discrete Wavelet Transform (DWT)** using the db5 wavelet for time-frequency feature extraction
- **TabNet** for attention-based feature selection on structured clinical and wavelet data
- **SMOTE** for addressing class imbalance in the training set

---

## Project Structure

```
CHD-Detection-Transformer-ECG/
├── configs/
│   └── default.yaml              # Training and preprocessing configuration
├── docs/
│   ├── model_design.md           # Detailed model architecture documentation
│   ├── module_design.md          # Module-level design decisions
│   └── testing_strategy.md       # Testing and evaluation strategy
├── notebooks/
│   └── Code.ipynb                # Original development notebook
├── scripts/
│   ├── preprocess.py             # Data preprocessing pipeline
│   ├── train.py                  # Model training script
│   ├── evaluate.py               # Evaluation and metrics computation
│   ├── perturbation_test.py      # Ablation and perturbation analysis
│   └── visualize.py              # Result visualization generation
├── src/chddecg/
│   ├── data/                     # Data loading and augmentation modules
│   ├── evaluation/               # Metrics and evaluation utilities
│   ├── models/                   # Model architecture definitions
│   ├── training/                 # Training loop and callbacks
│   └── utils/                    # Utility functions
├── tests/                        # Unit tests
├── 2021521460113-Thesis.pdf      # Full thesis document
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Dataset

The model is trained on the **Georgia-12Lead-ECG-Challenge-Dataset** containing:

- **10,344** pediatric ECG recordings (12-lead)
- Associated clinical metadata (age, gender, diagnosis codes)
- Binary classification: CHD-positive vs. Normal

Input data format: `.mat` and `.hea` file pairs (WFDB-compatible).

---

## Quick Start

### Installation

```bash
git clone https://github.com/JunaidZaman03/CHD-Detection-Transformer-ECG.git
cd CHD-Detection-Transformer-ECG
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Run the Pipeline

```bash
# Preprocess ECG data
python scripts/preprocess.py --config configs/default.yaml

# Train the model
python scripts/train.py --config configs/default.yaml

# Evaluate on test set
python scripts/evaluate.py --config configs/default.yaml

# Run ablation studies
python scripts/perturbation_test.py --config configs/default.yaml

# Generate visualizations (ROC, PR curves, confusion matrix)
python scripts/visualize.py --config configs/default.yaml
```

---

## Technical Details

| Parameter | Value |
|-----------|-------|
| ECG input shape | `(5000, 12)` |
| Clinical feature vector | 15 dimensions |
| PCA-reduced feature vector | 100 dimensions |
| Optimizer | Adam with Cosine Annealing LR |
| Loss function | Binary Cross-Entropy |
| Dropout rate | 0.4 |
| L2 weight decay | 0.01 |
| Batch size | 32 |
| Epochs | 100 (with early stopping on F1) |
| Framework | TensorFlow 2.16.1 / Keras 3.5.4 |

---

## Ablation Studies

The thesis includes two ablation cases:

1. **GPU Utilization and Batch Size** — GPU-accelerated training with batch size 32 achieved faster convergence and reduced overfitting compared to CPU-based training.
2. **Effect of SMOTE** — Applying SMOTE improved validation accuracy from 0.73 to 0.92 and F1 score from 0.68 to 0.88, demonstrating the critical importance of addressing class imbalance.

---

## Thesis Document

The full thesis is available in this repository:  
[`2021521460113-Thesis.pdf`](./2021521460113-Thesis.pdf)

---

## Citation

If you use this work, please cite:

```
Junaid Zaman, "Early Diagnosis of Congenital Heart Disease Using Transformer-Based
Deep Learning on Electrocardiogram Signals," Bachelor's Thesis, College of Software
Engineering, Sichuan University, 2025.
```

---

## License

This project is part of an academic thesis submitted to Sichuan University. Please contact the author for permissions regarding reuse or adaptation.

---

## Contact

**Junaid Zaman**  
College of Software Engineering, Sichuan University, Chengdu, China  
GitHub: [@JunaidZaman03](https://github.com/JunaidZaman03)
