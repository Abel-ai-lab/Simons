
import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch

from simons import Simons, SimonsConfig, forecast_series, prepare_covariates
from simons.covariates import encode_categorical, usable_covariates

FIXTURE = Path(__file__).parent / "fixtures" / "covariates"


@pytest.fixture(scope="module")
def tiny():
    return Simons.from_pretrained(FIXTURE)


@pytest.fixture(scope="module")
def cases():
    return dict(np.load(FIXTURE / "cases.npz"))


def test_matches_training_code(tiny, cases):
    history = torch.from_numpy(cases["cov_history"])
    q = tiny.forecast(history, 16, covariates=list(cases["cov_covariates"]))
    np.testing.assert_allclose(q.numpy(), cases["cov_expected"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(tiny.forecast(history, 16).numpy(), cases["plain_expected"],
                               rtol=1e-5, atol=1e-5)


def test_matches_training_code_beyond_one_pass(tiny, cases):
    history = torch.from_numpy(cases["cov_history"])
    q = tiny.forecast(history, 80, covariates=torch.from_numpy(cases["long_covariates"]))
    np.testing.assert_allclose(q.numpy(), cases["long_expected"], rtol=1e-5, atol=1e-5)


def test_covariates_change_the_forecast(cases):
    assert not np.allclose(cases["cov_expected"], cases["plain_expected"])


def test_unusable_covariates_give_the_plain_forecast(tiny, cases):
    history = torch.from_numpy(cases["cov_history"])
    plain = tiny.forecast(history, 16)
    flat = torch.ones(4, 2, history.shape[-1] + 16)
    assert torch.equal(tiny.forecast(history, 16, covariates=flat), plain)
    assert torch.equal(tiny.forecast(history, 16, covariates=[None] * 4), plain)


def test_no_covariate_weights_is_univariate():
    config = SimonsConfig(context_length=1024, prediction_length=64, width=32, layers=2, heads=2,
                          near_context=512, far_pool=2, far_min_history=256, stats_window=512,
                          detector_mid=256)
    torch.manual_seed(0)
    model = Simons(config).eval()
    assert not any(k.startswith("covariates.") for k in model.state_dict())
    with pytest.raises(ValueError):
        model.forecast(torch.randn(1, 100), 8, covariates=torch.randn(1, 1, 108))


def test_series_api_with_categorical_columns(tiny):
    rng = np.random.default_rng(0)
    days = np.array(["mon", "tue", "wed", "thu", "fri", "sat", "sun"] * 60, dtype=object)
    series, future, past = [], [], []
    for n in (200, 350):
        y = 10 + rng.normal(size=n).cumsum() * 0.2 + 3 * (np.arange(n) % 7 >= 5)
        series.append(y.astype(np.float32))
        future.append({"day": days[: n + 12], "promo": rng.integers(0, 2, n + 12)})
        past.append({"lagged": np.r_[np.nan, y[:-1]]})
    out = forecast_series(tiny, series, 12, future_covariates=future, past_covariates=past)
    assert [o.shape for o in out] == [(12, 9), (12, 9)]
    assert all(np.isfinite(o).all() and (np.diff(o, axis=-1) >= 0).all() for o in out)


def test_prepare_covariates():
    covs = prepare_covariates(5, 2, future={"a": [1, 2, 3, 4, 5, 6, 7], "flat": [1] * 7},
                              past=[np.array([0, 1, 0, 1, 0.5])])
    assert covs.shape == (2, 7)
    assert np.isnan(covs[1, 5:]).all()
    assert list(encode_categorical(np.array(["b", "a", "b"], dtype=object))) == [1.0, 0.0, 1.0]
    with pytest.raises(ValueError):
        prepare_covariates(5, 2, future=[[1, 2, 3]])


def test_config_round_trip(tiny, tmp_path):
    tiny.save_pretrained(tmp_path)
    loaded = Simons.from_pretrained(tmp_path)
    assert loaded.config == tiny.config
    with pytest.raises(ValueError):
        dataclasses.replace(tiny.config, read_after=2)


def test_categorical_codes_shared_or_separate():
    column = np.array(["A", "B", "C", "A", "B", "C"], dtype=object)
    shared = prepare_covariates(4, 2, future=[column])
    separate = prepare_covariates(4, 2, future=[column], shared_categories=False)
    assert list(shared[0]) == [0, 1, 2, 0, 1, 2]
    assert list(separate[0]) == [0, 1, 2, 0, 0, 1]


def test_missing_categories_stay_missing():
    codes = encode_categorical(np.array(["x", None, "y", float("nan")], dtype=object))
    assert codes[0] == 0 and codes[2] == 1 and np.isnan(codes[1]) and np.isnan(codes[3])


def test_numbers_with_pandas_na_and_given_categories():
    pd = pytest.importorskip("pandas")
    codes = encode_categorical(np.array(["1", pd.NA, "10", "2"], dtype=object))
    np.testing.assert_array_equal(codes, [0, np.nan, 1, 2])
    codes = encode_categorical(np.array([1, "x", 2], dtype=object), categories=["1", "2", "x"])
    assert list(codes) == [0, 2, 1]


def test_dataframe_and_series_inputs():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"a": np.arange(7.0), "b": ["u", "v"] * 3 + ["u"]})
    covs = prepare_covariates(5, 2, future=frame)
    assert covs.shape == (2, 7)
    assert prepare_covariates(5, 2, future=frame["a"]).shape == (1, 7)


def test_unread_covariates_are_dropped_without_changing_the_forecast(tiny, cases):
    history = torch.from_numpy(cases["cov_history"])
    covs = torch.from_numpy(cases["cov_covariates"])
    past_only = covs[:, 3:4]
    assert usable_covariates(history, past_only, 16) is None
    plain = tiny.forecast(history, 16)
    assert torch.equal(tiny.forecast(history, 16, covariates=past_only), plain)
    many = torch.cat([covs[:, :3]] * 7, 1)
    kept = usable_covariates(history, many, 16)
    assert torch.isfinite(kept).any(-1).sum(-1).max() == 16


def test_row_crop_and_shared_rows_are_exact(tiny, cases, monkeypatch):
    import simons.model as model_module

    history = torch.from_numpy(cases["cov_history"]).clone()
    covs = torch.from_numpy(cases["cov_covariates"]).clone()
    length = history.shape[-1]
    for row, start in enumerate((length - 700, length - 300, length - 90, 0)):
        history[row, :, :start] = float("nan")
    covs[1, 1, : length - 100] = float("nan")
    covs[2, 2, : length - 90] = float("nan")
    history[3] = float("nan")
    crops, shared = [], []
    row_crops, shared_rows = model_module.row_crops, model_module.shared_rows
    monkeypatch.setattr(model_module, "row_crops",
                        lambda *a: crops.append(row_crops(*a)) or crops[-1])
    monkeypatch.setattr(model_module, "shared_rows",
                        lambda *a: shared.append(shared_rows(*a)) or shared[-1])
    fast = tiny.forecast(history, 16, covariates=covs)
    assert any(len(c.unique()) > 1 for c in crops)
    assert any(compute is not None for compute, _ in shared)
    monkeypatch.setattr(model_module, "row_crops", lambda start, *_: torch.zeros(
        start.shape[0], dtype=torch.long, device=start.device))
    monkeypatch.setattr(model_module, "shared_rows", lambda *_: (None, None))
    full = tiny.forecast(history, 16, covariates=covs)
    torch.testing.assert_close(fast, full, rtol=1e-5, atol=1e-5, equal_nan=True)
