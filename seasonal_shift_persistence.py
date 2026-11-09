"""Seasonal shift persistence, copied from chap-core's built-in baseline.

Source: chap_core/models/builtin/seasonal_shift_persistence.py and empirical.py in
https://github.com/dhis2-chap/chap-core. Kept free of chap-core imports so the model
does not depend on it. Keep it in sync by hand if the built-in model changes.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

N_SAMPLES = 100
# 99 quantile levels symmetric around 0.5, plus 0.5 once more. The two middle samples
# are then both the median, so the median of the samples is exactly the median of
# the values whichever median definition a metric uses.
QUANTILE_LEVELS = np.sort(np.append((np.arange(N_SAMPLES - 1) + 0.5) / (N_SAMPLES - 1), 0.5))


def quantile_samples(values: np.ndarray) -> np.ndarray:
    """``N_SAMPLES`` deterministic samples spread over the values, with their median in the middle."""
    return np.asarray(np.quantile(values, QUANTILE_LEVELS))


def periods_per_year(time_period: pd.Series) -> int:
    """52 for weekly periods ('YYYY-MM-DD/YYYY-MM-DD', as CHAP writes them, or 'YYYY-Www'), else 12 (monthly)."""
    text = time_period.astype(str)
    return 52 if (text.str.contains("/") | text.str.contains("W")).all() else 12


def period_of_year(time_period: pd.Series) -> np.ndarray:
    """The month (1-12) or ISO week of the start date (1-52) of each period. Week 53 counts as week 52."""
    text = time_period.astype(str)
    if periods_per_year(time_period) == 12:
        return text.str[5:7].astype(int).to_numpy()
    if text.str.contains("/").all():
        weeks = pd.to_datetime(text.str.split("/").str[0]).dt.isocalendar().week
    else:
        weeks = text.str.split("W").str[-1]
    return np.minimum(weeks.astype(int).to_numpy(), 52)


@dataclass(frozen=True)
class SeasonalShifts:
    """Ratios ``(y_t + 1) / (y_{t-h} + 1)`` by horizon ``h`` and the period of year of ``t``."""

    by_horizon: dict[int, dict[int, np.ndarray]]

    @classmethod
    def from_cases(cls, cases: np.ndarray, seasons: np.ndarray, max_horizon: int) -> SeasonalShifts:
        by_horizon = {}
        for horizon in range(1, min(max_horizon, len(cases) - 1) + 1):
            ratios = (cases[horizon:] + 1) / (cases[:-horizon] + 1)
            target_seasons = seasons[horizon:]
            finite = np.isfinite(ratios)
            by_horizon[horizon] = {
                int(season): ratios[finite & (target_seasons == season)] for season in np.unique(target_seasons[finite])
            }
        return cls(by_horizon)

    @classmethod
    def pooled(cls, shifts: Iterable[SeasonalShifts]) -> SeasonalShifts:
        """The ratios of all the given shifts together."""
        collected: dict[int, dict[int, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
        for location_shifts in shifts:
            for horizon, by_season in location_shifts.by_horizon.items():
                for season, ratios in by_season.items():
                    collected[horizon][season].append(ratios)
        return cls(
            {
                horizon: {season: np.concatenate(ratios) for season, ratios in by_season.items()}
                for horizon, by_season in collected.items()
            }
        )

    def ratios(self, season: int, horizon: int) -> np.ndarray | None:
        """The ratios ending in the period of year, else in the nearest ones earlier and later in the year.

        Without ratios both earlier and later in the year, the ratios of every period of year
        are used. None when there are no ratios for the horizon.
        """
        by_season = self.by_horizon.get(horizon, {})
        if not by_season:
            return None
        if season in by_season:
            return by_season[season]
        earlier = [other for other in by_season if other < season]
        later = [other for other in by_season if other > season]
        if earlier and later:
            return np.concatenate([by_season[max(earlier)], by_season[min(later)]])
        return np.concatenate(list(by_season.values()))


@dataclass(frozen=True)
class SeasonalShiftPersistence:
    """Forecasts the last observed value, scaled by how cases changed in earlier years.

    For a target ``h`` periods after the last observed value ``y``, the samples are
    ``(y + 1) * r - 1`` for the ratios ``r = (y_t + 1) / (y_{t-h} + 1)`` in the training
    data of the same location, over every ``t`` with the target's period of year. The
    ratios are quantile samples, so the median is ``y`` scaled by the median ratio.
    Missing ratios fall back to the nearest periods of year, then to those of all
    locations, then to the longest shorter horizon that has them. Without any ratio the
    last value is forecast unchanged.
    """

    shifts: dict[str, SeasonalShifts]
    pooled_shifts: SeasonalShifts

    @classmethod
    def train(cls, df: pd.DataFrame) -> SeasonalShiftPersistence:
        max_horizon = periods_per_year(df["time_period"])
        shifts = {}
        for location, group in df.groupby("location"):
            group = group.sort_values("time_period")
            shifts[location] = SeasonalShifts.from_cases(
                group["disease_cases"].to_numpy(dtype=float), period_of_year(group["time_period"]), max_horizon
            )
        return cls(shifts, SeasonalShifts.pooled(shifts.values()))

    def ratios(self, location: str, season: int, horizon: int) -> np.ndarray:
        """The ratios of the location, else of all locations, at the horizon or else the longest shorter one.

        A single ratio gives no spread, so the first candidate with more than one is used, else
        the first single ratio, else 1.
        """
        candidates = [s for s in (self.shifts.get(location), self.pooled_shifts) if s is not None]
        single = None
        for shorter_horizon in range(horizon, 0, -1):
            for shifts in candidates:
                ratios = shifts.ratios(season, shorter_horizon)
                if ratios is not None and len(ratios) > 1:
                    return ratios
                if single is None:
                    single = ratios
        return np.ones(1) if single is None else single

    def predict_location(self, location: str, historic: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
        """Samples (periods x N_SAMPLES) for one location; both frames sorted by time_period."""
        cases = historic["disease_cases"].to_numpy(dtype=float)
        observed = np.flatnonzero(np.isfinite(cases))
        if len(observed) == 0:
            raise ValueError(f"No observed disease_cases for location {location!r} in the historic data")
        last_value = cases[observed[-1]]
        periods_since_last = len(cases) - observed[-1]
        rows = [
            np.maximum((last_value + 1) * quantile_samples(self.ratios(location, season, horizon)) - 1, 0)
            for horizon, season in enumerate(period_of_year(future["time_period"]), start=periods_since_last)
        ]
        return np.array(rows)
