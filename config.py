"""User options for the configurable baseline model.

`ModelConfig` is the single source of truth: it validates the values CHAP passes
in, and `dump_user_options.py` generates the `user_options` block of `MLproject`
from it.
"""

from __future__ import annotations

from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """User-tunable options. Regenerate MLproject with `uv run python dump_user_options.py` after editing."""

    method: Literal["seasonal_shift", "persistence", "baseline"] = Field(
        default="seasonal_shift",
        description=(
            "Point forecast in log space. 'persistence': the last observed value. "
            "'baseline': the location's mean for the period of year. "
            "'seasonal_shift': the last observed value shifted by the difference between the "
            "location's means for the target and the last observed period of year."
        ),
    )
    log_bias: float = Field(
        default=0.0,
        description="Added to every forecast in log(1 + cases) space. 0.1 scales forecasts by about 1.1.",
    )
    log_noise_sd: float = Field(
        default=0.0,
        ge=0.0,
        description="Standard deviation of the Gaussian noise added to each sample in log(1 + cases) space.",
    )
    n_samples: int = Field(default=100, ge=1, le=1000, description="Number of samples per forecast.")
    seed: int = Field(default=0, description="Seed for the noise, for reproducible forecasts.")


def load_config(config_path: str | None) -> ModelConfig:
    """Read `user_option_values` from the run configuration CHAP writes; defaults when run without CHAP."""
    if config_path is None:
        return ModelConfig()
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}
    return ModelConfig(**(raw.get("user_option_values") or {}))
