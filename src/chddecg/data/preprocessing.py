from __future__ import annotations

from pathlib import Path
from typing import Any

import json
import numpy as np
import pandas as pd
import pywt
from imblearn.over_sampling import SMOTE
from scipy import signal
from scipy.ndimage import median_filter
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors

from .augmentation import augment_batch


def bandpass_filter_ecg(
    ecg: np.ndarray,
    fs: float,
    lowcut: float = 0.5,
    highcut: float = 50.0,
    order: int = 3,
) -> np.ndarray:
    nyquist = 0.5 * fs
    low = max(lowcut / nyquist, 1e-6)
    high = min(highcut / nyquist, 0.999)
    b, a = signal.butter(order, [low, high], btype="band")
    return signal.filtfilt(b, a, ecg, axis=0).astype(np.float32)


def remove_baseline_wander(ecg: np.ndarray) -> np.ndarray:
    corrected = np.empty_like(ecg, dtype=np.float32)
    for lead_idx in range(ecg.shape[1]):
        lead = ecg[:, lead_idx]
        baseline = median_filter(lead, size=101)
        baseline = median_filter(baseline, size=301)
        corrected[:, lead_idx] = lead - baseline
    return corrected


def pad_or_trim_signal(ecg: np.ndarray, target_length: int) -> np.ndarray:
    if ecg.shape[0] >= target_length:
        return ecg[:target_length].astype(np.float32)

    output = np.zeros((target_length, ecg.shape[1]), dtype=np.float32)
    output[: ecg.shape[0]] = ecg
    return output


def normalize_per_lead(ecg: np.ndarray) -> np.ndarray:
    mean = np.mean(ecg, axis=0, keepdims=True)
    std = np.std(ecg, axis=0, keepdims=True) + 1e-8
    return ((ecg - mean) / std).astype(np.float32)


def build_clinical_vector(
    row: pd.Series,
    top_dx_codes: list[str],
    target_dim: int = 15,
    age_range: tuple[float, float] = (0.0, 100.0),
    hr_range: tuple[float, float] = (30.0, 200.0),
    bmi_range: tuple[float, float] = (15.0, 45.0),
) -> np.ndarray:
    values: list[float] = []

    age = row.get("age")
    age = float(age) if age is not None and not pd.isna(age) else sum(age_range) / 2.0
    age_norm = (age - age_range[0]) / (age_range[1] - age_range[0])
    values.append(float(np.clip(age_norm, 0.0, 1.0)))

    sex = str(row.get("sex", "Unknown"))
    sex_value = 0.0 if sex == "Male" else 1.0 if sex == "Female" else 0.5
    values.append(sex_value)

    fs = float(row.get("fs", 500.0))
    fs_norm = (fs - 100.0) / (1000.0 - 100.0)
    values.append(float(np.clip(fs_norm, 0.0, 1.0)))

    row_dx = set(row.get("dx", []) if isinstance(row.get("dx"), list) else [])
    values.extend([1.0 if code in row_dx else 0.0 for code in top_dx_codes])

    heart_rate = row.get("heart_rate")
    if heart_rate is None or pd.isna(heart_rate):
        values.append(0.5)
    else:
        hr_norm = (float(heart_rate) - hr_range[0]) / (hr_range[1] - hr_range[0])
        values.append(float(np.clip(hr_norm, 0.0, 1.0)))

    bmi = row.get("bmi")
    if bmi is None or pd.isna(bmi):
        values.append(0.5)
    else:
        bmi_norm = (float(bmi) - bmi_range[0]) / (bmi_range[1] - bmi_range[0])
        values.append(float(np.clip(bmi_norm, 0.0, 1.0)))

    if len(values) < target_dim:
        values.extend([0.0] * (target_dim - len(values)))

    return np.asarray(values[:target_dim], dtype=np.float32)


def extract_wavelet_features(signals: np.ndarray, wavelet: str = "sym4", levels: int = 5) -> np.ndarray:
    features: list[np.ndarray] = []
    for sample in signals:
        per_sample: list[float] = []
        for lead_idx in range(sample.shape[1]):
            coeffs = pywt.wavedec(sample[:, lead_idx], wavelet=wavelet, level=levels)
            for coeff in coeffs:
                per_sample.extend(
                    [
                        float(np.mean(coeff)),
                        float(np.std(coeff)),
                        float(np.max(coeff)),
                        float(np.min(coeff)),
                    ]
                )
        features.append(np.asarray(per_sample, dtype=np.float32))
    return np.asarray(features, dtype=np.float32)


def extract_frequency_features(signals: np.ndarray, fs: float = 500.0) -> np.ndarray:
    features: list[np.ndarray] = []
    for sample in signals:
        per_sample: list[float] = []
        for lead_idx in range(sample.shape[1]):
            freqs, psd = signal.welch(sample[:, lead_idx], fs=fs, nperseg=min(1024, sample.shape[0]))
            per_sample.extend(
                [
                    float(np.mean(psd)),
                    float(np.std(psd)),
                    float(freqs[np.argmax(psd)] if len(freqs) else 0.0),
                    float(np.sum(psd)),
                ]
            )
        features.append(np.asarray(per_sample, dtype=np.float32))
    return np.asarray(features, dtype=np.float32)


def extract_hrv_features(signals: np.ndarray, fs: float = 500.0) -> np.ndarray:
    features: list[np.ndarray] = []
    min_distance = max(1, int(0.25 * fs))

    for sample in signals:
        lead = sample[:, 1] if sample.shape[1] > 1 else sample[:, 0]
        peaks, _ = signal.find_peaks(lead, distance=min_distance)
        if len(peaks) < 2:
            features.append(np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32))
            continue

        rr = np.diff(peaks) / fs
        rmssd = np.sqrt(np.mean(np.square(np.diff(rr)))) if len(rr) > 1 else 0.0
        features.append(
            np.asarray(
                [
                    float(np.mean(rr)),
                    float(np.std(rr)),
                    float(rmssd),
                    float(60.0 / np.mean(rr)) if np.mean(rr) > 0 else 0.0,
                ],
                dtype=np.float32,
            )
        )

    return np.asarray(features, dtype=np.float32)


def extract_morphology_features(signals: np.ndarray) -> np.ndarray:
    features: list[np.ndarray] = []
    for sample in signals:
        per_sample: list[float] = []
        for lead_idx in range(sample.shape[1]):
            lead = sample[:, lead_idx]
            per_sample.extend(
                [
                    float(np.mean(lead)),
                    float(np.std(lead)),
                    float(np.max(lead) - np.min(lead)),
                    float(np.percentile(lead, 95) - np.percentile(lead, 5)),
                ]
            )
        features.append(np.asarray(per_sample, dtype=np.float32))
    return np.asarray(features, dtype=np.float32)


def validate_features(features: np.ndarray) -> np.ndarray:
    output = np.asarray(features, dtype=np.float32)
    invalid_mask = ~np.isfinite(output)
    if invalid_mask.any():
        output[invalid_mask] = 0.0
    return output


def reduce_features(features: np.ndarray, n_components: int = 100) -> tuple[np.ndarray, PCA]:
    n_components = min(n_components, features.shape[0], features.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(features)
    return reduced.astype(np.float32), pca


def align_smote_modalities(
    original_features: np.ndarray,
    original_signals: np.ndarray,
    original_clinical: np.ndarray,
    resampled_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(original_features)
    _, indices = nn.kneighbors(resampled_features)
    indices = indices.flatten()
    return original_signals[indices], original_clinical[indices]


def save_array(path: str | Path, array: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def preprocess_records(
    signals: list[np.ndarray],
    metadata_df: pd.DataFrame,
    processed_dir: str | Path,
    config: dict[str, Any],
) -> dict[str, str]:
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    target_length = int(config["data"]["target_length"])
    sampling_rate = float(config["data"]["sampling_rate"])
    lowcut = float(config["preprocessing"]["lowcut"])
    highcut = float(config["preprocessing"]["highcut"])
    filter_order = int(config["preprocessing"]["filter_order"])
    wavelet = str(config["preprocessing"]["wavelet"])
    wavelet_levels = int(config["preprocessing"]["wavelet_levels"])
    clinical_dim = int(config["data"]["clinical_dim"])

    dx_counts: dict[str, int] = {}
    for dx_list in metadata_df.get("dx", []):
        if isinstance(dx_list, list):
            for code in dx_list:
                dx_counts[code] = dx_counts.get(code, 0) + 1
    top_dx_codes = [code for code, _ in sorted(dx_counts.items(), key=lambda item: item[1], reverse=True)[:10]]

    processed_signals: list[np.ndarray] = []
    clinical_vectors: list[np.ndarray] = []
    labels: list[int] = []

    for signal_array, (_, row) in zip(signals, metadata_df.iterrows()):
        filtered = bandpass_filter_ecg(signal_array, fs=sampling_rate, lowcut=lowcut, highcut=highcut, order=filter_order)
        corrected = remove_baseline_wander(filtered)
        fixed = pad_or_trim_signal(corrected, target_length)
        normalized = normalize_per_lead(fixed)

        processed_signals.append(normalized)
        clinical_vectors.append(build_clinical_vector(row, top_dx_codes, target_dim=clinical_dim))
        labels.append(int(row["label"]))

    signals_np = np.asarray(processed_signals, dtype=np.float32)
    clinical_np = np.asarray(clinical_vectors, dtype=np.float32)
    labels_np = np.asarray(labels, dtype=np.int32)

    if bool(config["preprocessing"]["augment_normal_class"]):
        normal_idx = np.where(labels_np == 0)[0]
        abnormal_idx = np.where(labels_np == 1)[0]
        if len(normal_idx) and len(normal_idx) < len(abnormal_idx):
            augmented = augment_batch(
                signals_np[normal_idx],
                probability=float(config["augmentation"]["probability"]),
                noise_range=(float(config["augmentation"]["noise_std_min"]), float(config["augmentation"]["noise_std_max"])),
                scale_range=(float(config["augmentation"]["scale_min"]), float(config["augmentation"]["scale_max"])),
                baseline_wander_range=(
                    float(config["augmentation"]["baseline_wander_min"]),
                    float(config["augmentation"]["baseline_wander_max"]),
                ),
            )
            signals_np = np.concatenate([signals_np, augmented], axis=0)
            clinical_np = np.concatenate([clinical_np, clinical_np[normal_idx]], axis=0)
            labels_np = np.concatenate([labels_np, np.zeros(len(normal_idx), dtype=np.int32)], axis=0)

    wavelet_features = extract_wavelet_features(signals_np, wavelet=wavelet, levels=wavelet_levels)
    frequency_features = extract_frequency_features(signals_np, fs=sampling_rate)
    hrv_features = extract_hrv_features(signals_np, fs=sampling_rate)
    morphology_features = extract_morphology_features(signals_np)

    handcrafted = np.concatenate(
        [wavelet_features, frequency_features, hrv_features, morphology_features],
        axis=1,
    )
    handcrafted = validate_features(handcrafted)

    pca = None
    if bool(config["preprocessing"]["apply_pca"]):
        handcrafted, pca = reduce_features(handcrafted, n_components=int(config["preprocessing"]["pca_components"]))

    train_val_signals, test_signals, train_val_features, test_features, train_val_clinical, test_clinical, train_val_labels, test_labels = train_test_split(
        signals_np,
        handcrafted,
        clinical_np,
        labels_np,
        test_size=float(config["data"]["test_size"]),
        random_state=int(config["seed"]),
        stratify=labels_np,
    )

    train_signals, val_signals, train_features, val_features, train_clinical, val_clinical, train_labels, val_labels = train_test_split(
        train_val_signals,
        train_val_features,
        train_val_clinical,
        train_val_labels,
        test_size=float(config["data"]["val_size_from_trainval"]),
        random_state=int(config["seed"]),
        stratify=train_val_labels,
    )

    if bool(config["preprocessing"]["use_smote"]):
        smote = SMOTE(random_state=int(config["seed"]))
        resampled_features, resampled_labels = smote.fit_resample(train_features, train_labels)
        resampled_signals, resampled_clinical = align_smote_modalities(train_features, train_signals, train_clinical, resampled_features)
        train_features = resampled_features.astype(np.float32)
        train_labels = resampled_labels.astype(np.int32)
        train_signals = resampled_signals.astype(np.float32)
        train_clinical = resampled_clinical.astype(np.float32)

    save_array(processed_dir / "train_signals.npy", train_signals)
    save_array(processed_dir / "train_features.npy", train_features)
    save_array(processed_dir / "train_clinical.npy", train_clinical)
    save_array(processed_dir / "train_labels.npy", train_labels)

    save_array(processed_dir / "val_signals.npy", val_signals)
    save_array(processed_dir / "val_features.npy", val_features)
    save_array(processed_dir / "val_clinical.npy", val_clinical)
    save_array(processed_dir / "val_labels.npy", val_labels)

    save_array(processed_dir / "test_signals.npy", test_signals)
    save_array(processed_dir / "test_features.npy", test_features)
    save_array(processed_dir / "test_clinical.npy", test_clinical)
    save_array(processed_dir / "test_labels.npy", test_labels)

    save_json(
        processed_dir / "metadata.json",
        {
            "top_dx_codes": top_dx_codes,
            "train_size": int(len(train_labels)),
            "val_size": int(len(val_labels)),
            "test_size": int(len(test_labels)),
        },
    )

    if pca is not None:
        save_array(processed_dir / "pca_components.npy", pca.components_.astype(np.float32))
        save_array(processed_dir / "pca_explained_variance_ratio.npy", pca.explained_variance_ratio_.astype(np.float32))

    if len(labels_np) >= 5:
        cv_dir = processed_dir / "cv_splits"
        cv_dir.mkdir(parents=True, exist_ok=True)
        splitter = StratifiedKFold(n_splits=min(5, np.bincount(labels_np).min()), shuffle=True, random_state=int(config["seed"]))
        for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(handcrafted, labels_np), start=1):
            save_array(cv_dir / f"fold_{fold_idx}_train_idx.npy", train_idx)
            save_array(cv_dir / f"fold_{fold_idx}_val_idx.npy", val_idx)

    return {
        "train_signals": str(processed_dir / "train_signals.npy"),
        "train_features": str(processed_dir / "train_features.npy"),
        "train_clinical": str(processed_dir / "train_clinical.npy"),
        "train_labels": str(processed_dir / "train_labels.npy"),
        "val_signals": str(processed_dir / "val_signals.npy"),
        "val_features": str(processed_dir / "val_features.npy"),
        "val_clinical": str(processed_dir / "val_clinical.npy"),
        "val_labels": str(processed_dir / "val_labels.npy"),
        "test_signals": str(processed_dir / "test_signals.npy"),
        "test_features": str(processed_dir / "test_features.npy"),
        "test_clinical": str(processed_dir / "test_clinical.npy"),
        "test_labels": str(processed_dir / "test_labels.npy"),
    }
