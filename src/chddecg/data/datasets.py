from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf


def create_tf_dataset(
    signals: np.ndarray,
    clinical: np.ndarray,
    handcrafted: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            {
                "signal_input": tf.convert_to_tensor(signals, dtype=tf.float16),
                "clinical_input": tf.convert_to_tensor(clinical, dtype=tf.float16),
                "wavelet_input": tf.convert_to_tensor(handcrafted, dtype=tf.float16),
            },
            tf.convert_to_tensor(labels, dtype=tf.int32),
        )
    )
    if shuffle:
        dataset = dataset.shuffle(buffer_size=max(1, len(labels)), reshuffle_each_iteration=True)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def load_processed_arrays(processed_dir: str | Path) -> dict[str, np.ndarray]:
    processed_dir = Path(processed_dir)
    return {
        "train_signals": np.load(processed_dir / "train_signals.npy"),
        "train_features": np.load(processed_dir / "train_features.npy"),
        "train_clinical": np.load(processed_dir / "train_clinical.npy"),
        "train_labels": np.load(processed_dir / "train_labels.npy"),
        "val_signals": np.load(processed_dir / "val_signals.npy"),
        "val_features": np.load(processed_dir / "val_features.npy"),
        "val_clinical": np.load(processed_dir / "val_clinical.npy"),
        "val_labels": np.load(processed_dir / "val_labels.npy"),
        "test_signals": np.load(processed_dir / "test_signals.npy"),
        "test_features": np.load(processed_dir / "test_features.npy"),
        "test_clinical": np.load(processed_dir / "test_clinical.npy"),
        "test_labels": np.load(processed_dir / "test_labels.npy"),
    }
