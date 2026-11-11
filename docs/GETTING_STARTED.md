# Getting started

This page covers how to give Simons your data and read its forecasts, how to add covariates, and
how to fine-tune a released model. Two notebooks run the same steps end to end:
[examples/demo.ipynb](../examples/demo.ipynb) for forecasting and
[examples/finetune.ipynb](../examples/finetune.ipynb) for fine-tuning.

## Install

You need Python 3.10+ and PyTorch 2.5+. A GPU is optional for inference.

```shell
pip install "git+https://github.com/Abel-ai-lab/Simons"
```

Load a model by its Hugging Face name. This page uses Simons-25M, the smallest, throughout;
[Models](MODEL.md) compares the four sizes.

```python
import torch
from simons import Simons

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Simons.from_pretrained("abel-lab/Simons-25M", device=device)
```

## Input format

`model.forecast(history, horizon)` takes a tensor or array of past values and returns quantile
forecasts.

| Input `history` | Meaning | Output |
|---|---|---|
| `[batch, time]` | Independent series | `[batch, horizon, 9]` |
| `[batch, variates, time]` | Multivariate series | `[batch, variates, horizon, 9]` |

Each multivariate row holds up to 32 variates.

- **Time order.** The last value of `history` is the most recent one. The forecast starts at the
  next step.
- **Missing values.** Use `NaN`. To batch series of different lengths, left-pad the shorter ones
  with `NaN`, or use `forecast_series` below.
- **Context.** The model reads the last 16,384 steps (Simons-25M and 60M) or 32,768 steps
  (Simons-178M and 297M). Longer histories are cut to that length.
- **Frequency.** The model reads values without frequency labels or timestamps.
- **Horizon.** Any length. The model forecasts 512 steps per pass and continues block by block for
  longer horizons.
- **Output.** The last axis holds the 9 quantile levels 0.1, 0.2, ..., 0.9 (`simons.QUANTILES`).
  Level 0.5 is the median; levels 0.1 and 0.9 bound an 80% prediction interval.

```python
history = torch.randn(8, 2000).cumsum(-1)         # [batch, time]
quantiles = model.forecast(history, horizon=96)   # [batch, 96, 9]
median = quantiles[..., 4]

panel = torch.randn(4, 3, 1000).cumsum(-1)        # [batch, variates, time]
quantiles = model.forecast(panel, horizon=48)     # [batch, variates, 48, 9]
```

### Series of different lengths

`forecast_series` takes a list of arrays of any lengths, `[time]` or `[variates, time]`, and
returns one numpy array per series. It batches the series for you, and splits a series with more
than 32 variates into groups of 32.

```python
import numpy as np
from simons import forecast_series

series = [np.random.randn(500).cumsum(), np.random.randn(3, 800).cumsum(-1)]
outputs = forecast_series(model, series, horizon=24)  # [24, 9] and [3, 24, 9]
```

## Covariates

A covariate is a series that helps forecast the target. Simons reads **known-future** covariates:
values you know over the forecast horizon when you make the forecast, such as calendar features,
weather forecasts, planned prices and promotions.

- **Length.** Each covariate covers the history and the horizon: `time + horizon` values, aligned
  with the target step by step.
- **Known at forecast time.** Give the value that was available when forecasting each step: a
  weather forecast, not the weather that was observed later.
- **Missing values.** Use `NaN` where a value is unknown.
- **Sharing.** The variates of one series share its covariates.
- **Limits.** The released models read up to 16 covariates per series and ignore past-only
  covariates. A covariate with fewer than two distinct values is dropped.

With tensors, pass numeric covariates as `[batch, covariates, time + horizon]`:

```python
horizon = 48
history = torch.randn(4, 3000).cumsum(-1)
covariates = torch.randn(4, 2, 3000 + horizon)
quantiles = model.forecast(history, horizon=horizon, covariates=covariates)
```

With `forecast_series`, pass one entry per series: an array, a list of arrays, a dict of named
columns or a pandas DataFrame. Columns accept numeric or categorical values.

```python
history = np.random.randn(500).cumsum()
temperature = np.random.randn(500 + horizon)
holiday = np.array(["no"] * 540 + ["yes"] * 8)
outputs = forecast_series(
    model,
    [history],
    horizon=horizon,
    future_covariates=[{"temperature": temperature, "holiday": holiday}],
)
```

## Options

- **`views=False`** turns off the transformed views of the history that a forecast combines by
  default. Forecasts are faster and slightly less accurate.
- **`batch_size`** in `forecast_series` sets how many series go into one call. Lower it when
  memory runs out; use 16 on a CPU.
- **GluonTS.** After `pip install gluonts`,
  `simons.predictor.SimonsPredictor(model, prediction_length)` yields `QuantileForecast` objects
  from a GluonTS dataset.

## Fine-tune

Fine-tuning trains a released model further on your own series. Hold out a test period and forecast
it with the released model first.

Fine-tuning needs a clone of the repository:

```shell
git clone https://github.com/Abel-ai-lab/Simons
cd Simons
pip install -e .
```

The configs target GPUs with 80 GB of memory: one for Simons-25M and 60M, two for Simons-178M and
297M. The notebook runs a small fine-tune on a CPU.

### Data format

Write your series to `train-*.parquet` files in a folder, one row per series. Keep the test period
out of these files.

| Column | Training corpus | Covariate corpus (optional) |
|---|---|---|
| `item_id` | Series id (string) | Same |
| `source` | Dataset name (string) | Same |
| `freq` | pandas frequency string, such as `D` or `h` | Same |
| `target` | Floats, `[variate][time]` | Floats, `[time]` |
| `covariates` | Not used | Floats, `[covariate][time]`, aligned with `target` |

The covariate corpus teaches the model the effect of your known-future covariates. Its rows follow
the rules of [Covariates](#covariates), without the horizon: store, at every step, the value that
was known when forecasting that step.

### Configure

Start from the stage-4 config of the size you fine-tune and override these training keys. Save
them as `finetune.yaml`:

```yaml
warm_start: models/Simons-25M  # the released model folder
data: data/finetune/train
covariate_data: data/finetune/covariates  # null without a covariate corpus
steps: 2000
checkpoint_every: 500
loader_workers: 0
long_weight: 0  # the long covariate pass needs covariate series of 16,384+ steps
shape_mix: 0    # no precomputed shape-cluster labels
# Start at step 0 with the full training context
context_schedule: null
synthetic_only_stages: 0
warm_start_stage: 0
warm_start_step: null
far_from_stage: 0
stitch_from_stage: 0
# Learning rate: a quarter of the peak, re-warmed over 100 steps, then a linear decay to 0
lr_schedule: linear
warm_lr_scale: 0.25
rewarm_from: 0.1
rewarm_steps: 100
```

Only the model's weights carry over; the optimizer starts fresh. Adjust `steps` and
`warm_lr_scale` for your data.

Merge the overrides into the stage-4 config and download the released model:

```python
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download

config = yaml.safe_load(Path("configs/simons-25m-stage4.yaml").read_text())
config["training"].update(yaml.safe_load(Path("finetune.yaml").read_text()))
Path("finetune-25m.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
snapshot_download("abel-lab/Simons-25M", local_dir="models/Simons-25M",
                  allow_patterns=["config.json", "model.safetensors", "LICENSE", "NOTICE.md"])
```

For another size, replace `25m` and `25M` in the config name and the paths.

### Train and export

```shell
python -m simons.training train --config finetune-25m.yaml --output runs/finetune-25m
python -m simons.training export --checkpoint runs/finetune-25m/checkpoint.pt \
    --output models/finetune-25m
```

For Simons-178M or 297M, train on two GPUs:

```shell
torchrun --standalone --nproc-per-node 2 -m simons.training train \
    --config finetune-178m.yaml --output runs/finetune-178m
```

- **Progress.** Losses are in `training.jsonl` and the run's status in `run.json`.
- **Resume.** Add `--resume runs/finetune-25m/checkpoint.pt` and use a new `--output` folder.
- **Memory.** Raise `checkpoint_blocks` (up to the model's layer count) or `micro_batches` to use
  less GPU memory at the cost of speed.

The exported folder loads like a released model:

```python
tuned = Simons.from_pretrained("models/finetune-25m", device=device)
```

Forecast your test period with `tuned` and with `model`, the released Simons-25M loaded above, and
compare their errors. The
[notebook](../examples/finetune.ipynb) does this with the weighted quantile loss.

When you share a fine-tuned model, include the released model's `LICENSE` and `NOTICE.md`.
