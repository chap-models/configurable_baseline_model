"""Forecasting logic, all in log(1 + cases) space.

Training stores, per location, the mean of log(1 + cases) for each period of year
(month 1-12 or week 1-52). Prediction picks a point forecast with the chosen method,
adds `log_bias`, draws Gaussian noise with sd `log_noise_sd`, and maps back with
exp(x) - 1, clipped at zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import ModelConfig


def period_of_year(time_period: pd.Series) -> pd.Series:
    """Month (1-12) for 'YYYY-MM', week (1-52) for 'YYYY-Www'. Week 53 counts as week 52."""
    text = time_period.astype(str)
    if text.str.contains("W").all():
        return text.str.split("W").str[-1].astype(int).clip(upper=52)
    return text.str[5:7].astype(int)


def seasonal_profiles(df: pd.DataFrame) -> dict:
    """Mean log(1 + cases) by period of year, per location and pooled over locations."""
    df = df.assign(season=period_of_year(df["time_period"]), log_cases=np.log1p(df["disease_cases"]))
    df = df[np.isfinite(df["log_cases"])]
    by_location = {
        location: group.groupby("season")["log_cases"].mean().to_dict() for location, group in df.groupby("location")
    }
    pooled = df.groupby("season")["log_cases"].mean().to_dict()
    return {"by_location": by_location, "pooled": pooled, "overall": float(df["log_cases"].mean())}


def _profile_value(profiles: dict, location: str, season: int) -> float:
    """The location's mean for the season, else the pooled mean for the season, else the overall mean."""
    for profile in (profiles["by_location"].get(location, {}), profiles["pooled"]):
        if season in profile:
            return profile[season]
    return profiles["overall"]


def point_forecasts(
    method: str, profiles: dict, location: str, historic: pd.DataFrame, future: pd.DataFrame
) -> np.ndarray:
    """Point forecasts in log(1 + cases) space for the future periods of one location."""
    target_seasons = period_of_year(future["time_period"]).to_numpy()
    seasonal = np.array([_profile_value(profiles, location, s) for s in target_seasons])
    if method == "baseline":
        return seasonal
    observed = historic[np.isfinite(historic["disease_cases"])]
    if observed.empty:
        raise ValueError(f"No observed disease_cases for location {location!r}; '{method}' needs a last value")
    last = observed.iloc[-1]
    last_log = float(np.log1p(last["disease_cases"]))
    if method == "persistence":
        return np.full(len(future), last_log)
    last_season = int(period_of_year(pd.Series([last["time_period"]])).iloc[0])
    return last_log + seasonal - _profile_value(profiles, location, last_season)


def sample(log_point: np.ndarray, config: ModelConfig, rng: np.random.Generator) -> np.ndarray:
    """Samples (periods x n_samples) on the case scale, with log-space bias and noise."""
    noise = rng.normal(0.0, config.log_noise_sd, size=(len(log_point), config.n_samples))
    return np.maximum(np.expm1(log_point[:, None] + config.log_bias + noise), 0.0)
