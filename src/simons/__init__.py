"""Simons: small, open time series foundation models for probabilistic forecasting."""

from .config import SimonsConfig
from .covariates import encode_categorical, prepare_covariates
from .forecast import QUANTILES, forecast, forecast_items, forecast_series
from .model import Simons

__all__ = ["QUANTILES", "Simons", "SimonsConfig", "encode_categorical", "forecast",
           "forecast_items", "forecast_series", "prepare_covariates"]
__version__ = "0.1.1"
