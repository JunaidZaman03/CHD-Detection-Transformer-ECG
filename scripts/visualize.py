from __future__ import annotations

import argparse
from pathlib import Path

from chddecg.config import ensure_directories, load_config
from chddecg.evaluation.visualization import create_dashboard
from chddecg.utils.logging import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    configure_logging()
    config = load_config(args.config)
    ensure_directories(config)

    results_path = Path(config["paths"]["evaluation_dir"]) / "test_results.json"
    output_path = Path(config["paths"]["evaluation_dir"]) / "dashboard.png"
    dashboard = create_dashboard(results_path, output_path)
    print({"dashboard": dashboard})


if __name__ == "__main__":
    main()
