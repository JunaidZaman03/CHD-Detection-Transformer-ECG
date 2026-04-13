from __future__ import annotations

import pytest

pytest.importorskip("pywt")
pytest.importorskip("imblearn")

import numpy as np
import pandas as pd

from chddecg.data.preprocessing import (
    build_clinical_vector,
    normalize_per_lead,
    pad_or_trim_signal,
)


def test_pad_or_trim_signal_returns_target_length():
    signal = np.random.randn(3000, 12).astype(np.float32)
    output = pad_or_trim_signal(signal, 5000)
    assert output.shape == (5000, 12)


def test_normalize_per_lead_preserves_shape():
    signal = np.random.randn(5000, 12).astype(np.float32)
    output = normalize_per_lead(signal)
    assert output.shape == signal.shape


def test_build_clinical_vector_returns_fixed_dimension():
    row = pd.Series(
        {
            "age": 50,
            "sex": "Male",
            "fs": 500,
            "dx": ["426783006"],
            "heart_rate": 72,
            "bmi": 24.0,
        }
    )
    vector = build_clinical_vector(row, top_dx_codes=["426783006", "999999"], target_dim=15)
    assert vector.shape == (15,)
