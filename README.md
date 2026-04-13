# CHDdECG Production Refactor

This package is a production-oriented refactor of the original `Code.ipynb` workflow. The notebook logic has been split into maintainable Python modules for:

- data loading
- preprocessing
- augmentation
- model construction
- training
- evaluation
- visualization
- perturbation testing

The original notebook is preserved in `notebooks/Code.ipynb` for traceability.

## What was refactored

The notebook contained inline package installs, Kaggle-specific absolute paths, duplicated layer definitions, visualization code mixed into training, and multiple execution-only cells. This refactor separates concerns and makes the codebase easier to run, test, review, and extend.

## Project structure

```text
chddecg_production/
├── configs/
│   └── default.yaml
├── docs/
│   ├── model_design.md
│   ├── module_design.md
│   └── testing_strategy.md
├── notebooks/
│   └── Code.ipynb
├── scripts/
│   ├── evaluate.py
│   ├── perturbation_test.py
│   ├── preprocess.py
│   ├── train.py
│   └── visualize.py
├── src/chddecg/
│   ├── data/
│   ├── evaluation/
│   ├── models/
│   ├── training/
│   └── utils/
└── tests/
```

## Expected input data

The current implementation assumes a Georgia 12-lead ECG style directory that contains matching `.mat` and `.hea` files.

Example:

```text
data/
└── Georgia/
    ├── E00001.mat
    ├── E00001.hea
    ├── E00002.mat
    ├── E00002.hea
    └── ...
```

## Quick start

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Run preprocessing:

```bash
python scripts/preprocess.py --config configs/default.yaml
```

Train the model:

```bash
python scripts/train.py --config configs/default.yaml
```

Evaluate a trained checkpoint:

```bash
python scripts/evaluate.py --config configs/default.yaml
```

Run the perturbation test:

```bash
python scripts/perturbation_test.py --config configs/default.yaml
```

Create result visualizations:

```bash
python scripts/visualize.py --config configs/default.yaml
```

## Outputs

By default, the pipeline writes to:

- `artifacts/processed/` for processed arrays and metadata
- `artifacts/models/` for trained checkpoints and history
- `artifacts/evaluation/` for metrics and plots
- `artifacts/perturbation/` for ablation and perturbation outputs

## Important assumptions carried over from the notebook

- ECG signal input shape: `(5000, 12)`
- Clinical feature vector size: `15`
- Handcrafted / PCA feature vector size for the model: `100`
- Binary target mapping: normal vs abnormal
- TensorFlow / Keras model family kept close to the notebook architecture

## Notes

This refactor is designed to be production-ready in structure and code organization. The exact biomedical validity of labels, diagnosis-code mapping, and advanced handcrafted feature design should still be reviewed by a domain expert before publication or deployment.
