from __future__ import annotations

import argparse

from chddecg.config import ensure_directories, load_config
from chddecg.training.train import train_model
from chddecg.utils.logging import configure_logging
from chddecg.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    configure_logging()
    config = load_config(args.config)
    ensure_directories(config)
    set_seed(int(config["seed"]))

    outputs = train_model(config)
    print(outputs)


if __name__ == "__main__":
    main()
