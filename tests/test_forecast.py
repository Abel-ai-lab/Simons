import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from simons import Simons, SimonsConfig, forecast_items, forecast_series, periodicity
from simons.forecast import pass_size

TINY = SimonsConfig(context_length=1024, prediction_length=64, patch_length=32, width=64,
                        layers=2, heads=4, near_context=512, far_pool=2, far_min_history=256,
                        stats_window=512, detector_mid=256)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return Simons(TINY).eval()


def test_shapes_and_order(model):
    history = torch.randn(3, 300).cumsum(-1)
    for horizon in (10, 64, 150):
        q = model.forecast(history, horizon)
        assert q.shape == (3, horizon, 9)
        assert torch.isfinite(q).all()
        assert (q.diff(dim=-1) >= 0).all()


def test_multivariate_and_missing(model):
    history = torch.randn(2, 4, 2000)
    history[0, 1, :1500] = float("nan")
    history[1, 2, ::7] = float("nan")
    q = model.forecast(history, 70, series_ids=torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]]))
    assert q.shape == (2, 4, 70, 9) and torch.isfinite(q).all()


def test_post_processing(model):
    counts = torch.poisson(torch.full((1, 400), 3.0))
    q = model.forecast(counts, 20)
    assert (q >= 0).all() and torch.equal(q, q.round())
    assert (model.forecast(-counts - 1, 20) <= 0).all()


def test_series_api_matches_batched(model):
    series = [np.random.default_rng(i).normal(size=n).cumsum().astype(np.float32)
              for i, n in enumerate((50, 700, 1300))]
    series.append(np.random.default_rng(9).normal(size=(3, 400)).astype(np.float32))
    out = forecast_series(model, series, 30)
    assert [o.shape for o in out] == [(30, 9)] * 3 + [(3, 30, 9)]
    for o, s in zip(out, series):
        np.testing.assert_allclose(o, model.forecast(torch.from_numpy(s[None]), 30)[0].numpy(),
                                   atol=1e-4)


def test_save_and_load(model, tmp_path):
    model.save_pretrained(tmp_path)
    loaded = Simons.from_pretrained(tmp_path)
    assert loaded.config == model.config
    history = torch.randn(2, 500)
    assert torch.equal(loaded.forecast(history, 40), model.forecast(history, 40))


@pytest.mark.parametrize("clip", [True, False])
def test_sign_clip_setting(clip):
    config = dataclasses.replace(TINY, sign_clip=clip, views=False, integer_snap=False)
    torch.manual_seed(0)
    model = Simons(config).eval()
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.fill_(-10.0)
    ramp = torch.arange(1.0, 401.0)[None]
    q = model.forecast(ramp, 20)
    if clip:
        assert (q >= 0).all()
    else:
        assert (q < 0).any()


def test_covariates_that_would_be_misread_raise():
    model = Simons.from_pretrained(Path(__file__).parent / "fixtures" / "covariates")
    history, covariates = torch.randn(1, 2, 300), torch.randn(1, 1, 324)
    with pytest.raises(ValueError):
        forecast_items(model, [(history[0].numpy(), covariates[0, :, :300].numpy())], 24)
    with pytest.raises(ValueError):
        model.forecast(history, 24, series_ids=[[0, 1]], covariates=covariates)


def test_negative_series_ids_do_not_change_the_forecast():
    config = dataclasses.replace(TINY, covariates=True, read_after=0, views=False,
                                 integer_snap=False)
    torch.manual_seed(0)
    model = Simons(config).eval()
    torch.manual_seed(44)
    history, covariates = torch.randn(1, 100), torch.randn(1, 1, 133)
    ids = [torch.tensor([[i]]) for i in (0, -1000, -1001, -2**63)]
    ids.append(torch.tensor([[0]], dtype=torch.uint8))
    q = [model.forecast(history, 33, series_ids=i, covariates=covariates) for i in ids]
    assert all(torch.equal(q[0], other) for other in q[1:])


def test_pass_size_charges_the_rows_padded_up_to_a_fit_row():
    window = torch.full((3, 1, 1024), float("nan"))
    window[..., -100:] = 1.0
    covariates = torch.ones(3, 4, 1024 + 64)
    covariates[..., :-164] = float("nan")
    whole = 8 + 16 + 2  # far, near and future tokens of the whole window
    assert pass_size(TINY, window, covariates, 64, 6, False)[0] == 6 * (5 + 2)
    spent = 5 * (5 + 2) + whole  # five short rows and the fit row
    assert pass_size(TINY, window, covariates, 64, 6, True)[0] == spent + (6 * whole - spent) // 4
    window[1] = float("nan")  # an item without history: its covariate rows run the whole window
    assert pass_size(TINY, window, covariates, 64, 6, True)[0] == 6 * whole


def test_far_min_history_at_or_beyond_the_context_keeps_no_far_tokens(model):
    history = torch.randn(2, 1024).cumsum(-1)
    q = []
    for steps in (1024, 1500):
        other = Simons(dataclasses.replace(TINY, far_min_history=steps)).eval()
        other.load_state_dict(model.state_dict())
        q.append(other.forecast(history, 20))
    assert torch.equal(*q)


def test_period_tables_match_the_direct_formula():
    length = 1200
    candidates, cos, sin = periodicity._refine_table(length, torch.device("cpu"))
    assert torch.equal(candidates, torch.arange(2.0, length // 2 + 1))
    steps = torch.arange(length, dtype=torch.float64)
    angle = 2 * math.pi * torch.remainder(steps[:, None] / candidates.double()[None], 1.0)
    assert cos.dtype == sin.dtype == torch.float32
    assert torch.equal(cos, angle.cos().float())
    assert torch.equal(sin, angle.sin().float())
