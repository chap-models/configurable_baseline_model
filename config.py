"""User options for the configurable baseline model.

`ModelConfig` is the single source of truth: it validates the values CHAP passes
in, and `dump_user_options.py` generates the `user_options` block of `MLproject`
from it.
"""

from __future__ import annotations

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """User-tunable options. Regenerate MLproject with `uv run python dump_user_options.py` after editing."""

    spread_factor: float = Field(
        default=1.0,
        ge=0.0,
        description=(
            "Multiplies the distance of each sample from the median in log(1 + cases) space. "
            "1 keeps the seasonal shift persistence samples, 0 gives a point forecast at the median, "
            "below 1 narrows and above 1 widens the forecast."
        ),
    )
    log_bias: float = Field(
        default=0.0,
        description="Added to every sample in log(1 + cases) space. 0.1 scales forecasts by about 1.1.",
    )


def load_config(config_path: str | None) -> ModelConfig:
    """Read `user_option_values` from the run configuration CHAP writes; defaults when run without CHAP."""
    if config_path is None:
        return ModelConfig()
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}
    return ModelConfig(**(raw.get("user_option_values") or {}))
