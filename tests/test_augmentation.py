from __future__ import annotations

import numpy as np

from chddecg.data.augmentation import augment_batch, augment_signal


def test_augment_signal_preserves_shape():
    signal = np.random.randn(5000, 12).astype(np.float32)
    augmented = augment_signal(signal)
    assert augmented.shape == signal.shape


def test_augment_batch_preserves_batch_shape():
    signals = np.random.randn(8, 5000, 12).astype(np.float32)
    augmented = augment_batch(signals)
    assert augmented.shape == signals.shape
