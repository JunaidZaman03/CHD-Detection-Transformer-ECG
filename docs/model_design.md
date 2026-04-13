# Model Design

## Overview

The refactored model remains aligned with the notebook's original CHDdECG architecture. It uses three input branches:

1. **ECG signal branch**: raw 12-lead ECG sequence of shape `(5000, 12)`
2. **Clinical branch**: engineered metadata vector of shape `(15,)`
3. **Handcrafted feature branch**: PCA-reduced handcrafted features of shape `(100,)`

## Architecture

### Signal branch
The signal branch starts with a convolutional stem followed by multi-scale residual blocks inspired by 1D ResNet designs. This branch captures local morphology patterns in ECG waveforms.

### Attention branch
A transformer-style encoder and temporal attention layer are used to model broader temporal dependencies when enabled.

### Handcrafted branch
A TabNet-style downsampling layer consumes the handcrafted feature vector and projects it into a compact representation.

### Fusion
The outputs of the signal, clinical, and handcrafted branches are concatenated and passed to dense classification layers.

## Why this is kept

This structure preserves the intent of the notebook:
- combine raw-signal representation learning
- combine engineered signal statistics
- combine metadata and diagnosis-context cues

## Caveats

The notebook architecture is retained for compatibility, but some engineering decisions should still be revisited before real deployment:
- very aggressive float16 usage may not be ideal on CPU inference
- the clinical vector definition is tied to dataset-specific metadata
- the diagnosis-code mapping should be validated clinically
