"""Predict: point forecast by the chosen method, then log-space bias and noise."""

from __future__ import annotations

import argparse
import logging

import joblib
import numpy as np
import pandas as pd

from config import load_config
from model import point_forecasts, sample

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
logger = logging.getLogger("configurable_baseline_model")


def predict(
    model_path: str,
    historic_data_path: str,
    future_data_path: str,
    out_file_path: str,
    config_path: str | None = None,
) -> None:
    config = load_config(config_path)
    logger.info("User options: %s", config.model_dump())
    profiles = joblib.load(model_path)
    historic = pd.read_csv(historic_data_path)
    future = pd.read_csv(future_data_path)
    rng = np.random.default_rng(config.seed)

    outputs = []
    for location, future_group in future.groupby("location"):
        future_group = future_group.sort_values("time_period")
        historic_group = historic[historic["location"] == location].sort_values("time_period")
        log_point = point_forecasts(config.method, profiles, location, historic_group, future_group)
        samples = sample(log_point, config, rng)
        out = future_group[["time_period", "location"]].reset_index(drop=True)
        sample_columns = pd.DataFrame(samples, columns=[f"sample_{i}" for i in range(config.n_samples)])
        outputs.append(pd.concat([out, sample_columns], axis=1))

    pd.concat(outputs).to_csv(out_file_path, index=False)
    logger.info("Saved predictions to %s", out_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict with the configurable baseline model.")
    parser.add_argument("model")
    parser.add_argument("historic_data")
    parser.add_argument("future_data")
    parser.add_argument("out_file")
    parser.add_argument("model_config", nargs="?", default=None)
    args = parser.parse_args()
    predict(args.model, args.historic_data, args.future_data, args.out_file, args.model_config)
