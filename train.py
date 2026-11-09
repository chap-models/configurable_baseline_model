"""Train: collect the seasonal shift ratios of the training data."""

from __future__ import annotations

import argparse
import logging

import joblib
import pandas as pd

from config import load_config
from seasonal_shift_persistence import SeasonalShiftPersistence

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
logger = logging.getLogger("configurable_baseline_model")


def train(train_data_path: str, model_path: str, config_path: str | None = None) -> None:
    logger.info("User options: %s", load_config(config_path).model_dump())
    df = pd.read_csv(train_data_path)
    joblib.dump(SeasonalShiftPersistence.train(df), model_path)
    logger.info("Saved seasonal shifts for %d locations to %s", df["location"].nunique(), model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the configurable baseline model.")
    parser.add_argument("train_data")
    parser.add_argument("model")
    parser.add_argument("model_config", nargs="?", default=None)
    args = parser.parse_args()
    train(args.train_data, args.model, args.model_config)
