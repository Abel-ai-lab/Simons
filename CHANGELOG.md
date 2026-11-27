# Changelog

All notable changes to Simons are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). The model weights have not changed since 0.1.0.

## [Unreleased]

## [0.1.3] - 2026-11-26

### Changed

- Simons is on PyPI as `simons-forecasting`: `pip install simons-forecasting`. The import name stays
  `simons`. The package installs only the inference dependencies (PyTorch, NumPy, safetensors and
  huggingface_hub); training and fine-tuning need the `train` extra, `pip install -e ".[train]"` in a
  clone.
- On a CPU, period detection builds no lookup table for a context of 16,384 or 32,768 steps when
  few candidate periods are needed, as with covariates on a history of a few thousand steps. The
  table takes 1 GiB for Simons-25M and 60M and 4 GiB for Simons-178M and 297M. At 2 CPU threads,
  Simons-297M with 16 covariates on a 2,688-step history peaks at 2.7 GiB instead of 6.6 GiB, and
  without covariates on a 32,768-step history with daily and weekly cycles at 2.4 GiB instead of
  5.8 GiB. Forecasts are unchanged. ([#24](https://github.com/Abel-ai-lab/Simons/pull/24))

### Documentation

- `docs/MODEL.md` gives the CPU memory and time of each size ([#24](https://github.com/Abel-ai-lab/Simons/pull/24)).

## [0.1.2] - 2026-11-16

### Changed

- With covariates, a series with at most one nonzero observed value no longer gets the
  square-root test-time views. Once normalized, such a history and its square root look the same
  to the model. When a covariate rose over the horizon, squaring the square-root forecast back
  squared the forecast's relative growth, with upper quantiles up to about 10⁸ on some fev-bench
  series. Forecasts change only for such series; forecasts without covariates are unchanged. The
  scores reported for 0.1.1 used the old views.
  ([#15](https://github.com/Abel-ai-lab/Simons/pull/15))

### Documentation and tests

- `docs/DATA.md` gives each source's share of the stored observations instead of one run's
  planned training-token shares ([#13](https://github.com/Abel-ai-lab/Simons/pull/13)). The README's
  benchmark place table is updated ([#12](https://github.com/Abel-ai-lab/Simons/pull/12)).
- The zero-forecast test is seeded and no longer compares forecasts across batch sizes, which
  failed on some machines ([#10](https://github.com/Abel-ai-lab/Simons/pull/10)).

## [0.1.1] - 2026-11-11

### Changed

- Covariate forecasts use less memory. Rows of a short series are kept from their own start
  instead of the full context window; for Simons-297M with 256 short items and 16 covariates,
  peak GPU memory drops from 17.0 GB to 7.2 GB. Forecasts are unchanged on the CPU
  ([#7](https://github.com/Abel-ai-lab/Simons/pull/7)).
- A variate with no observed value in the context is forecast as zero in every quantile, instead
  of a finite value of about ±670 from scaling a series without data
  ([#8](https://github.com/Abel-ai-lab/Simons/pull/8)).

### Fixed

- Forecast pass sizes count the padded tokens of covariate rows, so one pass stays within the
  memory budget ([#7](https://github.com/Abel-ai-lab/Simons/pull/7)).
- A caller's `series_ids` can no longer collide with internal ids. Before, ids of -1000 or lower
  could ([#7](https://github.com/Abel-ai-lab/Simons/pull/7)).

### Documentation

- `docs/TRAINING.md` states what the full public data needs and how long Simons-25M takes to
  train ([#3](https://github.com/Abel-ai-lab/Simons/pull/3)). The README links the release post,
  and `docs/MODEL.md` links the TIME leaderboard
  ([#5](https://github.com/Abel-ai-lab/Simons/pull/5),
  [#6](https://github.com/Abel-ai-lab/Simons/pull/6)).

## [0.1.0] - 2026-11-11

Initial release: inference and training code, the training recipe, and the weights of
Simons-25M, Simons-60M, Simons-178M and Simons-297M.

[Unreleased]: https://github.com/Abel-ai-lab/Simons/compare/v0.1.3...HEAD
[0.1.3]: https://github.com/Abel-ai-lab/Simons/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/Abel-ai-lab/Simons/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/Abel-ai-lab/Simons/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Abel-ai-lab/Simons/releases/tag/v0.1.0
