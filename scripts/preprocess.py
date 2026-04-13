from __future__ import annotations

import argparse

from chddecg.config import ensure_directories, load_config
from chddecg.data.io import discover_record_ids, load_dataset
from chddecg.data.preprocessing import preprocess_records
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

    record_ids = discover_record_ids(config["paths"]["data_dir"])
    signals, metadata_df = load_dataset(record_ids, config["paths"]["data_dir"])
    outputs = preprocess_records(signals, metadata_df, config["paths"]["processed_dir"], config)
    print(outputs)


if __name__ == "__main__":
    main()
