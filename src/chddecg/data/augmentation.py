from __future__ import annotations

import numpy as np


def add_gaussian_noise(signal: np.ndarray, std: float) -> np.ndarray:
    noise = np.random.normal(0.0, std, size=signal.shape).astype(signal.dtype)
    return signal + noise


def scale_amplitude(signal: np.ndarray, factor: float) -> np.ndarray:
    return signal * np.asarray(factor, dtype=signal.dtype)


def add_baseline_wander(signal: np.ndarray, amplitude: float) -> np.ndarray:
    timesteps = signal.shape[0]
    t = np.linspace(0, 2 * np.pi, timesteps, dtype=np.float32)
    wander = amplitude * np.sin(t)
    return signal + wander[:, None]


def augment_signal(
    signal: np.ndarray,
    probability: float = 0.7,
    noise_range: tuple[float, float] = (0.01, 0.03),
    scale_range: tuple[float, float] = (0.85, 1.15),
    baseline_wander_range: tuple[float, float] = (0.02, 0.05),
) -> np.ndarray:
    augmented = signal.copy()

    if np.random.rand() < probability:
        augmented = add_gaussian_noise(augmented, np.random.uniform(*noise_range))
    if np.random.rand() < probability:
        augmented = scale_amplitude(augmented, np.random.uniform(*scale_range))
    if np.random.rand() < probability:
        augmented = add_baseline_wander(augmented, np.random.uniform(*baseline_wander_range))

    return augmented.astype(np.float32)


def augment_batch(signals: np.ndarray, **kwargs) -> np.ndarray:
    return np.stack([augment_signal(signal, **kwargs) for signal in signals], axis=0)
