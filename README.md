# configurable_baseline_model

CHAP's seasonal shift persistence baseline, with a configurable spread and bias in
log space. With the default options the samples are identical to CHAP's built-in
`seasonal_shift_persistence` model. Use the options to check how evaluation metrics
react to a forecast that is too high, too low, too narrow or too wide by a known amount.

## Method

`seasonal_shift_persistence.py` is a copy of the built-in model from chap-core
(`chap_core/models/builtin/seasonal_shift_persistence.py` and `empirical.py`),
without a chap-core dependency. For a target `h` periods after the last observed
value `y`, its 100 samples are quantiles of `(y + 1) * r - 1`, where `r` are the
ratios `(y_t + 1) / (y_{t-h} + 1)` in the training data for the same location and
the target's month or week of year.

Each forecast's samples `s` are then adjusted in `log(1 + cases)` space:

```
z = log(1 + s)
z' = median(z) + spread_factor * (z - median(z)) + log_bias
samples = max(exp(z') - 1, 0)
```

## Options

| option          | default | meaning                                                                  |
|-----------------|---------|--------------------------------------------------------------------------|
| `spread_factor` | `1.0`   | 1 keeps the samples, 0 gives a point forecast at the median, 2 doubles the log-space spread |
| `log_bias`      | `0.0`   | added in log space; `0.1` scales forecasts by about 1.1                  |

The options are defined in `config.py` (`ModelConfig`). After changing them,
regenerate the `user_options` block in `MLproject` with
`uv run python dump_user_options.py`.

## Running

Without CHAP, with the default options:

```
uv run python isolated_run.py
```

With CHAP:

```
cat > config.yaml <<YAML
user_option_values:
  spread_factor: 2.0
  log_bias: 0.3
YAML
chap eval . data.csv eval.nc --model-configuration-yaml config.yaml
```

## Seeding into CHAP

`chap_seed.yaml` seeds the model with six configurations: the unchanged baseline,
a bias of +0.3 and -0.3, and spread factors 0, 0.5 and 2. Copy it into chap-core's
`config/configured_models/`.
