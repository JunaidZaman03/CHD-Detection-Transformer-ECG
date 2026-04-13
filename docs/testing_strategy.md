# Testing Strategy

## Unit tests

Unit tests cover:
- metadata parsing
- signal padding / trimming
- feature extraction shape contracts
- augmentation output shape preservation
- model factory smoke test

## Integration tests

Integration tests can be added later for:
- end-to-end preprocessing on a miniature fixture dataset
- training one batch on synthetic data
- checkpoint save / load round-trip
- evaluation report generation

## Runtime validation

The code also performs runtime validation:
- NaN / Inf cleanup for signals
- shape assertions for model inputs
- directory creation for all output locations
- skipped TensorFlow-heavy tests when TensorFlow is unavailable

## Recommended CI steps

```bash
pip install -r requirements.txt
pip install -e .
pytest
python -m py_compile $(find src scripts tests -name "*.py")
```
