from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_directories(config: dict[str, Any]) -> None:
    for key, value in config.get("paths", {}).items():
        if key.endswith("_dir"):
            Path(value).mkdir(parents=True, exist_ok=True)
