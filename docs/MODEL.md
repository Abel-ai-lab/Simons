# Models

Simons comes in four sizes, which share one architecture and differ in depth, width, and context
length. Model cards:
[Simons-25M](https://huggingface.co/abel-lab/Simons-25M),
[Simons-60M](https://huggingface.co/abel-lab/Simons-60M),
[Simons-178M](https://huggingface.co/abel-lab/Simons-178M) and
[Simons-297M](https://huggingface.co/abel-lab/Simons-297M).

| Model | Parameters | Layers / width / heads | Context | Horizon per pass |
|---|---|---|---|---|
| Simons-25M | 25.1M | 12 / 384 / 6 | 16,384 | 512 |
| Simons-60M | 60.3M | 8 / 704 / 11 | 16,384 | 512 |
| Simons-178M | 177.7M | 12 / 1024 / 16 | 32,768 | 512 |
| Simons-297M | 297.2M | 13 / 1280 / 20 | 32,768 | 512 |

Parameter counts include the covariate modules. Context and horizon are in time steps.
For longer horizons, the model forecasts successive blocks.

## Scores

Lower CRPS and MASE are better. Higher SQL skill is better.

| Model | GIFT-Eval CRPS | GIFT-Eval MASE | fev-bench SQL skill | TIME MASE |
|---|---|---|---|---|
| Simons-25M | 0.4683 | 0.6860 | 47.7% | 0.6473 |
| Simons-60M | 0.4654 | 0.6812 | 47.6% | 0.6427 |
| Simons-178M | 0.4609 | 0.6754 | 48.6% | 0.6393 |
| Simons-297M | 0.4574 | 0.6731 | 48.4% | 0.6371 |

- **[GIFT-Eval](https://github.com/SalesforceAIResearch/gift-eval):** 97 tasks, targets only,
  relative to Seasonal Naive (geometric mean over tasks).
- **[fev-bench](https://github.com/autogluon/fev):** 100 tasks (42 with dynamic covariates, 30 of
  them known-future), SQL skill score relative to Seasonal Naive, with no task declared as trained
  on.
- **[TIME](https://github.com/zqiao11/TIME):** 98 tasks, relative to Seasonal Naive (geometric mean
  over tasks). All four models are on the
  [leaderboard](https://huggingface.co/spaces/Real-TSF/TIME-leaderboard).

The code of the three benchmark runs is in [`benchmarks/`](../benchmarks):

| File | Benchmark | How to run |
|---|---|---|
| [`gift_eval.ipynb`](../benchmarks/gift_eval.ipynb) | GIFT-Eval | The notebook; its first cell gives the setup and the pinned gift-eval commit |
| [`fev_bench.py`](../benchmarks/fev_bench.py) | fev-bench | A `fev.ForecastingModel`; pass it to fev's evaluation |
| [`time.py`](../benchmarks/time.py) | TIME | Copy into a TIME checkout as `experiments/simons_model.py` |

## CPU memory and time

Peak memory in GiB at 2 CPU threads:

| Request | Simons-25M | Simons-60M | Simons-178M | Simons-297M |
|---|---|---|---|---|
| 1 series, 2,688 steps | 0.44 | 0.68 | 1.56 | 2.45 |
| 1 series, 2,688 steps, 16 covariates | 0.97 | 1.30 | 2.00 | 2.66 |
| 1 series, 32,768 steps | 0.69 | 0.82 | 1.56 | 2.45 |
| 64 series, 2,688 steps | 0.67 | 0.96 | 1.63 | 2.45 |

Seconds per forecast at 2 CPU threads (4 threads in parentheses):

| Request | Simons-25M | Simons-60M | Simons-178M | Simons-297M |
|---|---|---|---|---|
| 1 series, 2,688 steps | 0.1 (0.1) | 0.1 (0.1) | 0.4 (0.3) | 0.7 (0.5) |
| 1 series, 2,688 steps, 16 covariates | 1.2 (0.9) | 2.4 (1.7) | 6.9 (4.7) | 11.2 (7.7) |
| 1 series, 32,768 steps | 0.1 (0.1) | 0.3 (0.2) | 1.2 (0.8) | 1.9 (1.3) |
| 64 series, 2,688 steps | 4.4 (3.0) | 8.8 (6.1) | 26.2 (18.1) | 43.1 (29.5) |

- **Setup:** an AMD Ryzen 9 9950X3D2 with PyTorch 2.15 (CPU build), horizon 96, one process per
  measurement. Set the thread count with `torch.set_num_threads` or `OMP_NUM_THREADS`.
- **Memory** is the peak resident memory of the whole process, loading included. **Seconds** are
  for a forecast after the first one.
- **Requests:** the 2,688-step series are 28 days of 15-minute data and the 32,768-step series is
  hourly, both with daily and weekly cycles; Simons-25M and 60M read its last 16,384 steps. The
  covariates are 12 numeric and 4 calendar columns. The 64 series go through `forecast_series`
  with `batch_size=16`.
- **Long histories:** period detection builds a lookup table on the first forecast of a history
  longer than 7,680 steps with a cycle thousands of steps long, such as a weekly cycle in 5-minute
  data or a yearly cycle in hourly data, and of a history longer than a few thousand steps with
  covariates. It keeps the table for later forecasts: 1 GiB for Simons-25M and 60M, 4 GiB for
  Simons-178M and 297M. A 32,768-step history with such a cycle peaks at 1.7, 1.9, 5.4 and 5.8 GiB
  on the four sizes.

## Forecast options

By default, forecasts combine transformed views of the history. Pass `views=False` to turn off
these views. See [Getting started](GETTING_STARTED.md) for the input format,
[covariates](GETTING_STARTED.md#covariates) and [fine-tuning](GETTING_STARTED.md#fine-tune), the
README for the [training quickstart](../README.md#train), and [Training](TRAINING.md) for the
recipe of the released models.
