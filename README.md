# configurable_baseline_model

A CHAP model template with three simple forecasting methods and a configurable,
known bias and noise in log space. Use it as a reference baseline, or to check
how evaluation metrics react to a forecast that is too high, too low, or too
narrow or wide by a known amount.

## Method

All computation is in `log(1 + cases)` space. Training stores, per location, the
mean of `log(1 + cases)` for each period of year (month 1-12 or week 1-52, with
week 53 counted as 52). A missing period of year falls back to the mean across
locations, then to the overall mean.

For each future period the point forecast `m` is:

| `method`         | point forecast `m`                                               |
|------------------|------------------------------------------------------------------|
| `persistence`    | `log(1 + y_last)`, the last observed value                       |
| `baseline`       | the location's mean for the target's period of year             |
| `seasonal_shift` | `log(1 + y_last)` + mean(target period) - mean(last observed period) |

Samples are `max(exp(m + log_bias + e) - 1, 0)` with `e ~ N(0, log_noise_sd)`.

## Options

| option         | default          | meaning                                                   |
|----------------|------------------|-----------------------------------------------------------|
| `method`       | `seasonal_shift` | `seasonal_shift`, `persistence` or `baseline`             |
| `log_bias`     | `0.0`            | added in log space; `0.1` scales forecasts by about 1.1   |
| `log_noise_sd` | `0.0`            | sd of the Gaussian noise per sample in log space           |
| `n_samples`    | `100`            | samples per forecast                                      |
| `seed`         | `0`              | seed for the noise                                        |

With `log_noise_sd: 0` every sample is the same, which gives a point forecast.

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
  method: persistence
  log_bias: 0.2
  log_noise_sd: 0.5
YAML
chap eval . data.csv eval.nc --model-configuration-yaml config.yaml
```

## Seeding into CHAP

`chap_seed.yaml` seeds the model with eight configurations: the three methods
with `log_noise_sd: 0.3`, and `seasonal_shift` with a bias of +0.3 and -0.3 and
with noise of 0, 0.1 and 1.0. Copy it into chap-core's `config/configured_models/`.
