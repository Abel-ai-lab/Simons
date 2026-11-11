"""GluonTS predictor for Simons; requires GluonTS."""

import numpy as np
from gluonts.model.forecast import QuantileForecast

from .forecast import QUANTILES, forecast_series


class SimonsPredictor:
    """Yield GluonTS forecasts for entries with target [T] or [V,T] and start (pandas Period).
    Forecast arrays [10,H] or [10,H,V] hold quantiles 0.1,...,0.9 and their mean.
    With covariate weights and `use_covariates`, the entries' `feat_dynamic_real` [C,T+H] and
    `past_feat_dynamic_real` [C,T] are passed; `**options` go to `forecast_series`."""

    def __init__(self, model, prediction_length: int, batch_size: int = 1024,
                 use_covariates: bool = True, **options):
        self.model = model
        self.prediction_length = prediction_length
        self.batch_size = batch_size
        self.use_covariates = use_covariates and bool(model.config.covariates)
        self.options = options

    def predict(self, dataset, **kwargs):
        """Yield forecasts lazily in groups of `batch_size` entries."""
        keys = [str(q) for q in QUANTILES] + ["mean"]
        for entries in batcher(dataset, self.batch_size):
            targets = [np.asarray(entry["target"], dtype=np.float32) for entry in entries]
            covariates = {}
            if self.use_covariates and any("feat_dynamic_real" in e or
                                           "past_feat_dynamic_real" in e for e in entries):
                covariates = {
                    "future_covariates": [e.get("feat_dynamic_real") for e in entries],
                    "past_covariates": [e.get("past_feat_dynamic_real") for e in entries],
                }
            forecasts = forecast_series(self.model, targets, self.prediction_length,
                                        self.batch_size, **covariates, **self.options)
            for entry, target, prediction in zip(entries, targets, forecasts):
                array = prediction.T if target.ndim == 1 else prediction.transpose(2, 1, 0)
                yield QuantileForecast(
                    forecast_arrays=np.concatenate([array, array.mean(axis=0, keepdims=True)]),
                    start_date=entry["start"] + target.shape[-1],
                    forecast_keys=keys,
                    item_id=entry.get("item_id"),
                )


def batcher(iterable, size: int):
    """Consecutive lists of at most `size` items, reading the iterable lazily."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch
