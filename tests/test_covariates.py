
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from simons import Simons, SimonsConfig, forecast_series, prepare_covariates
from simons.covariates import (
    ROLE_FIT,
    ROLE_KNOWN,
    ROLE_PAST,
    ROLE_TARGET,
    covariate_rows,
    encode_categorical,
    read_mask,
    usable_covariates,
    xreg_fit,
)

FIXTURE = Path(__file__).parent / "fixtures" / "covariates"


@pytest.fixture(scope="module")
def tiny():
    return Simons.from_pretrained(FIXTURE)


@pytest.fixture(scope="module")
def cases():
    return dict(np.load(FIXTURE / "cases.npz"))


@pytest.fixture(scope="module")
def past(tiny):
    """The fixture checkpoint with `past_rows` on."""
    model = Simons(dataclasses.replace(tiny.config, past_rows=True)).eval()
    model.load_state_dict(tiny.state_dict())
    return model


@pytest.fixture
def roles(monkeypatch):
    """The roles [B,V'] of every `covariate_rows` call the model makes."""
    import simons.model as model_module

    seen, rows = [], model_module.covariate_rows

    def recorded(*args, **kwargs):
        grown = rows(*args, **kwargs)
        seen.append(grown[2])
        return grown

    monkeypatch.setattr(model_module, "covariate_rows", recorded)
    return seen


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


def test_past_rows_read_history_only_covariates(past, cases, roles, monkeypatch):
    history = torch.from_numpy(cases["cov_history"])
    past_only = torch.from_numpy(cases["cov_covariates"])[:, 3:4]
    plain = past.forecast(history, 16)
    monkeypatch.setattr(sys.modules["simons.forecast"], "xreg_fit", None)  # XReg must not run
    q = past.forecast(history, 16, covariates=past_only)
    assert not torch.allclose(q, plain, rtol=1e-3, atol=1e-3)
    assert {int(r) for used in roles for r in used.unique()} == {ROLE_TARGET, ROLE_PAST}


def test_targets_read_past_rows():
    related = torch.ones(1, 1, 2, 2, dtype=torch.bool)
    mask, update = read_mask(torch.tensor([[ROLE_TARGET, ROLE_PAST]]), related, 1)
    assert mask[0, 0].tolist() == [[True, True], [False, True]] and update[0, :, 0].tolist() == [
        True, False]


def test_past_or_known_is_decided_at_the_forecast_origin(tiny, past):
    torch.manual_seed(0)
    values = F.pad(torch.randn(1, 1, 1024), (0, 64), value=float("nan"))
    values[..., 1000:] = float("nan")  # the target stops 24 steps before the origin
    covs = torch.randn(1, 2, 1088)
    covs[0, 0, 1024:] = float("nan")  # no value from the origin on
    covs[0, 1, 1025:] = float("nan")  # one value at the origin
    ids = torch.zeros(1, 1, dtype=torch.long)
    rows, _, roles = covariate_rows(past.config, values, ids, covs)
    assert roles[0, :3].tolist() == [ROLE_TARGET, ROLE_PAST, ROLE_KNOWN]
    torch.testing.assert_close(rows[0, 1], covs[0, 0], rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(rows[0, 2, :-32], covs[0, 1, 32:], rtol=0, atol=0, equal_nan=True)
    roles = covariate_rows(tiny.config, values, ids, covs)[2]  # off: after the target's last step
    assert roles[0, :3].tolist() == [ROLE_TARGET, ROLE_KNOWN, ROLE_KNOWN]


def test_fits_ignore_past_covariates(past):
    torch.manual_seed(0)
    known, other = torch.randn(2, 1, 1, 664)
    other[..., 600:] = float("nan")
    values = 2 * known + other + 0.1 * torch.randn(1, 1, 664)
    values[..., 600:] = float("nan")
    both = torch.cat([other, known], 1)
    ids = torch.zeros(1, 1, dtype=torch.long)
    rows, _, roles = covariate_rows(past.config, values, ids, both)
    assert roles[0].tolist() == [ROLE_TARGET, ROLE_PAST, ROLE_KNOWN, ROLE_FIT]
    alone = covariate_rows(past.config, values, ids, known)[0]
    torch.testing.assert_close(rows[0, -1], alone[0, -1], equal_nan=True)
    future = torch.arange(664) >= 600
    fitted = xreg_fit(values, both, future)
    assert fitted.any()
    torch.testing.assert_close(fitted, xreg_fit(values, known, future))
    assert not xreg_fit(values, other, future).any()


def test_a_known_column_that_runs_out_is_a_past_row_in_later_passes(past, cases, roles):
    history = torch.from_numpy(cases["cov_history"])[3:]
    covs = torch.from_numpy(cases["long_covariates"])[3:, :1].clone()
    covs[..., -70:] = float("nan")  # ten future values: none from the second pass's origin on
    q = past.forecast(history, 80, covariates=covs, views=False)
    assert torch.isfinite(q).all()
    assert [int(used[0, 1]) for used in roles] == [ROLE_KNOWN, ROLE_PAST]


def test_backtest_does_not_read_a_past_covariate_after_its_origin(past, cases, monkeypatch):
    module = sys.modules["simons.forecast"]
    calls, variants = [], module.covariate_variants

    def recorded(model, history, horizon, ids, covariates, rows, *rest):
        out = variants(model, history, horizon, ids, covariates, rows, *rest)
        calls.append((covariates[1:], out[rows >= 1]))  # the backtest's covariates and forecasts
        return out

    monkeypatch.setattr(module, "covariate_variants", recorded)
    history = torch.from_numpy(cases["cov_history"])[1:2]  # 250 observed steps: guarded
    covs = torch.from_numpy(cases["cov_covariates"])[1:2, [0, 3]]  # a known-future and a past one
    poisoned = covs.clone()
    poisoned[0, 1, -32:-16] += 100.0  # the past covariate over the backtest's held-out block
    q = [past.forecast(history, 16, covariates=c) for c in (covs, poisoned)]
    (inputs, held), (other_inputs, other_held) = calls
    assert len(held) == 2 and not torch.allclose(q[0], q[1], rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(inputs, other_inputs, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(held, other_held, rtol=1e-5, atol=1e-5)


def test_mixed_batch_matches_each_item_alone(past, cases):
    history = torch.from_numpy(cases["cov_history"])  # 80, 250, 400 and 900 observed steps
    covs = torch.from_numpy(cases["cov_covariates"])  # three known-future columns, then a past one
    items = [covs[0, 3:], covs[1, 2:], covs[2, :2], None]  # past, both, known, none
    together = past.forecast(history, 16, covariates=items)
    for i, item in enumerate(items):
        alone = past.forecast(history[i : i + 1], 16, covariates=[item])
        torch.testing.assert_close(together[i : i + 1], alone, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("items", [4, 3])  # with far tokens; short histories, their prefix skipped
def test_crops_and_skips_are_exact_with_past_rows(past, cases, roles, monkeypatch, items):
    import simons.model as model_module

    history = torch.from_numpy(cases["cov_history"])[:items].clone()
    length = history.shape[-1]
    covs = torch.cat([torch.from_numpy(cases["long_covariates"])[:, :1],
                      F.pad(torch.from_numpy(cases["cov_covariates"])[:, 3:, :length], (0, 80),
                            value=float("nan"))], 1)[:items]  # a known-future and a past column
    history[1, :, : length - 120] = float("nan")  # a target starting after its past covariate
    covs[2, 0] = float("nan")
    covs[2, 1, : length - 100] = float("nan")  # a past covariate starting after its target
    crops, skips = [], []
    row_crops, skip = model_module.row_crops, Simons._skip
    monkeypatch.setattr(model_module, "row_crops",
                        lambda *a: crops.append(row_crops(*a)) or crops[-1])
    monkeypatch.setattr(Simons, "_skip", lambda *a: skips.append(skip(*a)) or skips[-1])
    fast = past.forecast(history, 80, covariates=covs)
    assert any(len(c.unique()) > 1 for c in crops)
    assert any((used == ROLE_PAST).any() for used in roles)
    assert items == 4 or all(skips)
    monkeypatch.setattr(model_module, "row_crops", lambda start, *_: torch.zeros(
        start.shape[0], dtype=torch.long, device=start.device))
    monkeypatch.setattr(model_module, "shared_rows", lambda *_: (None, None))
    monkeypatch.setattr(Simons, "_skip", lambda *_: 0)
    full = past.forecast(history, 80, covariates=covs)
    torch.testing.assert_close(fast, full, rtol=1e-5, atol=1e-5)


def test_from_pretrained_can_switch_past_rows_on():
    assert not Simons.from_pretrained(FIXTURE).config.past_rows
    assert Simons.from_pretrained(FIXTURE, past_rows=True).config.past_rows
