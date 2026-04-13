from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import scipy.io


@dataclass
class RecordMetadata:
    file_id: str
    dx: list[str]
    age: float | None
    sex: str
    fs: float
    height: float | None = None
    weight: float | None = None
    heart_rate: float | None = None
    bmi: float | None = None
    label: int = 1

    def to_dict(self) -> dict:
        return asdict(self)


NORMAL_CODES = {
    "426783006",
    "59118001",
    "164889003",
    "427393009",
    "426177001",
    "74390002",
    "164934002",
    "17338001",
}
ABNORMAL_CODES = {
    "270492004",
    "713427006",
    "39732003",
    "445211001",
    "233917008",
    "251146004",
    "698252002",
}


def discover_record_ids(base_path: str | Path) -> list[str]:
    base_path = Path(base_path)
    return sorted(path.stem for path in base_path.glob("*.mat"))


def _parse_numeric(value: str) -> float | None:
    value = value.strip().split()[0]
    if value.lower() == "nan":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_header(hea_path: str | Path) -> RecordMetadata:
    hea_path = Path(hea_path)
    lines = hea_path.read_text(encoding="utf-8").splitlines()

    file_id = hea_path.stem
    fs = 500.0
    dx: list[str] = []
    age = None
    sex = "Unknown"
    height = None
    weight = None
    heart_rate = None

    if lines:
        header = lines[0].split()
        if len(header) >= 3:
            try:
                fs = float(header[2])
            except ValueError:
                fs = 500.0

    for line in lines:
        if line.startswith("#Dx:"):
            dx = [token.strip() for token in line.split(":", maxsplit=1)[1].split(",") if token.strip()]
        elif line.startswith("#Age:"):
            age = _parse_numeric(line.split(":", maxsplit=1)[1])
        elif line.startswith("#Sex:"):
            sex = line.split(":", maxsplit=1)[1].strip()
        elif line.startswith("#Sampling frequency:"):
            parsed = _parse_numeric(line.split(":", maxsplit=1)[1])
            fs = parsed if parsed is not None else fs
        elif line.startswith("#Height:"):
            height = _parse_numeric(line.split(":", maxsplit=1)[1])
        elif line.startswith("#Weight:"):
            weight = _parse_numeric(line.split(":", maxsplit=1)[1])
        elif line.startswith("#Heart rate:"):
            heart_rate = _parse_numeric(line.split(":", maxsplit=1)[1])

    bmi = None
    if height and weight and height > 0:
        bmi = weight / ((height / 100.0) ** 2)

    if any(code in ABNORMAL_CODES for code in dx):
        label = 1
    elif any(code in NORMAL_CODES for code in dx):
        label = 0
    else:
        label = 1

    return RecordMetadata(
        file_id=file_id,
        dx=dx,
        age=age,
        sex=sex,
        fs=fs,
        height=height,
        weight=weight,
        heart_rate=heart_rate,
        bmi=bmi,
        label=label,
    )


def load_mat_signal(mat_path: str | Path, gain: float | None = None) -> np.ndarray:
    mat_path = Path(mat_path)
    data = scipy.io.loadmat(mat_path)
    if "val" not in data:
        raise KeyError(f"'val' key not found in {mat_path}")

    signal = np.asarray(data["val"], dtype=np.float32).T
    if signal.ndim != 2 or signal.shape[1] != 12:
        raise ValueError(f"Expected ECG shape (n, 12), got {signal.shape}")

    invalid_mask = ~np.isfinite(signal)
    if invalid_mask.any():
        signal[invalid_mask] = 0.0

    if gain and gain > 0:
        signal = signal / gain

    return signal


def load_dataset(record_ids: Iterable[str], base_path: str | Path) -> tuple[list[np.ndarray], pd.DataFrame]:
    base_path = Path(base_path)
    signals: list[np.ndarray] = []
    metadata: list[dict] = []

    for record_id in record_ids:
        mat_path = base_path / f"{record_id}.mat"
        hea_path = base_path / f"{record_id}.hea"
        if not mat_path.exists() or not hea_path.exists():
            continue

        signal = load_mat_signal(mat_path)
        meta = parse_header(hea_path)

        signals.append(signal)
        metadata.append(meta.to_dict())

    return signals, pd.DataFrame(metadata)
