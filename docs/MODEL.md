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

## Forecast options

By default, forecasts combine transformed views of the history. Pass `views=False` to turn off
these views. See [Getting started](GETTING_STARTED.md) for the input format,
[covariates](GETTING_STARTED.md#covariates) and [fine-tuning](GETTING_STARTED.md#fine-tune), the
README for the [training quickstart](../README.md#train), and [Training](TRAINING.md) for the
recipe of the released models.
