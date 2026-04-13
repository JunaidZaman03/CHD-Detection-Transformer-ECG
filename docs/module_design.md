# Module Design

## Objectives

The original notebook mixed environment setup, feature engineering, model definition, training, testing, and visualization in a single linear execution flow. The refactor separates these responsibilities into stable modules.

## Modules

### `chddecg.data.io`
Responsible for reading `.mat` and `.hea` records, parsing metadata, and discovering record identifiers.

### `chddecg.data.preprocessing`
Contains filtering, baseline correction, signal normalization, handcrafted feature extraction, clinical feature vector construction, PCA projection, and dataset split persistence.

### `chddecg.data.augmentation`
Contains reusable ECG augmentation transforms such as additive noise, baseline wander, amplitude scaling, and batch-level augmentation.

### `chddecg.data.datasets`
Creates TensorFlow datasets from NumPy arrays with the input signatures expected by the model.

### `chddecg.models.*`
Contains the reusable deep learning building blocks refactored from the notebook:
- ResNet-style signal blocks
- Transformer / temporal attention blocks
- TabNet support blocks
- The top-level CHDdECG model factory

### `chddecg.training.*`
Contains metrics, callbacks, model compilation, training orchestration, checkpointing, and history persistence.

### `chddecg.evaluation.*`
Contains prediction utilities, metric calculation, confusion matrix / ROC / PR plotting, dashboard creation, and perturbation tests.

### `scripts/*`
Thin CLI entry points that call into package code. These are intentionally lightweight so that logic stays importable and testable.

## Design choices

- Hard-coded Kaggle paths were removed.
- Notebook-only install cells were removed.
- Repeated class definitions were reduced to single-source modules.
- File output locations are driven by config.
- Functions return typed data structures where possible.
- Side effects are isolated to scripts and explicit save functions.
