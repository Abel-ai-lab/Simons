
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from simons.config import SimonsConfig
from simons.covariates import ROLE_BINARY, ROLE_KNOWN
from simons.model import Simons
from simons.synth import SyntheticMixer, augment, mech
from simons.synth.covariates import families
from simons.synth.multi import MultiPool
from simons.training import network, objective, schedule
from simons.training.covariates import roles, stream, windows

TINY = dict(context_length=1536, prediction_length=512, width=32, layers=3, heads=2,
            near_context=512, far_min_history=768, stats_window=1024, detector_mid=768)
COVARIATES = dict(covariates=True, binary_rows=True, fit_row=True, read_after=1)


@pytest.fixture(autouse=True)
def small_banks():
    mech.configure(64, 1024, 400, 1)
    augment.configure("", 0.0, None)
    yield


def tiny_model(seed=0, **extra):
    torch.manual_seed(seed)
    return Simons(SimonsConfig(**{**TINY, **COVARIATES, **extra}))


@pytest.mark.parametrize("family", families.FAMILIES)
def test_effect_families_deterministic(family):
    base = np.random.default_rng(0).standard_normal(4000).cumsum()
    windows_ = [np.random.default_rng(i).standard_normal(4000).cumsum() for i in range(4)]

    def run(seed):
        counter = iter(range(100))
        return families.make_task(family, base[-2000:], np.random.default_rng(seed), 1744, 256,
                                  8, lambda n: windows_[next(counter) % 4][-n:])

    y, c = run(1)
    y2, c2 = run(1)
    assert y.shape == (2000,) and c.ndim == 2 and c.shape[1] == 2000 and 1 <= len(c) <= 8
    assert np.array_equal(y, y2, equal_nan=True) and np.array_equal(c, c2, equal_nan=True)
    assert np.isfinite(y[:1744]).any()


def test_family_mix_parsing():
    names, p = families.parse_mix({"generic": 0.2, "energy": 0.6, "lagged": 0.2})
    assert names == ("generic", "energy", "lagged") and math.isclose(p.sum(), 1.0)
    with pytest.raises(ValueError):
        families.parse_mix({"weather": 1.0})


def role_mixer(seed, spec):
    gen = torch.Generator().manual_seed(seed)
    multi = MultiPool(0.5, 16, ("identity", "mix", "lag", "coint", "func", "event"), 64,
                      seed=seed)
    mixer = SyntheticMixer(0.5, gen, ("kernel", "sarima"), 0.3, 0.4, 2, multi=multi)
    mixer.track_first = 32
    hook = roles.Hook(spec, seed, 0, "cpu", 10, 1152)
    multi.roles = hook.planner
    return mixer, hook


def test_role_groups_and_effect_rows():
    spec = {"boundaries": [0.5], "p_role": [0.5, 1.0], "p_eff": [0.0, 0.25],
            "effect_workers": 0, "effect_mix": dict.fromkeys(families.FAMILIES, 0.2)}
    mixer, hook = role_mixer(3, spec)
    for step in (0, 7):
        hook.planner.step = step
        sequence = torch.randn(8, 32, 1152)
        ids = torch.arange(256).view(8, 32)
        sequence, ids = mixer(sequence, ids)
        before = sequence.clone()
        first = (np.zeros((8, 32), np.int64), np.zeros((8, 32), np.int64))
        out, _, _, batch = hook.prepare(step, mixer, sequence, ids, lambda first=first: first)
        covariate = batch.role_ids > 0
        assert torch.equal(covariate, torch.isin(batch.role_ids,
                                                 torch.tensor([ROLE_KNOWN, ROLE_BINARY])))
        assert torch.isnan(out[covariate][:, -32:]).all()
        target = batch.target(torch.zeros(8, 32, 36, 32))
        assert torch.isnan(target[covariate]).all() and not torch.isnan(target[~covariate]).any()
        if step == 0:
            assert torch.equal(torch.nan_to_num(out[~covariate]),
                               torch.nan_to_num(before[~covariate]))
    stats = hook.summary()
    assert stats["batch"]["role_slots"] > 0 and stats["plan"]["role_groups"] > 0
    assert stats["batch"]["effect_rows"] == 2 and sum(stats["batch"]["families"].values()) > 0


def test_effect_rows_carry_the_augmentation(monkeypatch):
    monkeypatch.setattr(augment, "CONFIG", None)
    augment.configure("mix,season,filter,lattice", 1.0, None)
    rows = roles.EffectRows(roles.settings({"effect_mix": {"generic": 1.0}}), 3, 0, 1024)
    first = next(iter(rows))["values"]
    augment.CONFIG = None
    assert torch.equal(torch.nan_to_num(next(iter(rows))["values"]), torch.nan_to_num(first))


def test_zero_init_read_and_roles_are_exact_no_ops():
    from simons.training.trainer import init_read

    spec = {"boundaries": [], "p_role": [1.0], "p_eff": [0.0]}
    mixer, hook = role_mixer(5, spec)
    sequence, ids = mixer(torch.randn(6, 32, 2048).cumsum(-1), torch.arange(192).view(6, 32))
    first = (np.zeros((6, 32), np.int64), mixer.first_rows().reshape(3, 32))

    def slots():
        host = ids.numpy().copy()
        start = np.zeros((6, 32), np.int64)
        start[3:] = first[1]
        host[3:] = mixer.host_ids(3, 32)
        return host, start
    sequence, ids, slots, batch = hook.prepare(0, mixer, sequence, ids, slots)
    assert int((batch.role_ids > 0).sum()) > 0
    out = []
    for read_after in (1, -1):
        model = tiny_model(read_after=read_after)
        if read_after >= 0:
            init_read(model.covariates.read, model.variate_block)
        torch.manual_seed(1)
        loss, _ = objective.packed_loss(model, sequence, slots, 0.4, 8, row_tokens=288,
                                        roles=batch)
        loss.backward()
        out.append((loss.detach(), {n: p.grad.clone() for n, p in model.named_parameters()
                                    if p.grad is not None and "covariates" not in n}))
    assert torch.equal(out[0][0], out[1][0])
    assert all(torch.allclose(out[0][1][n], out[1][1][n], atol=1e-7) for n in out[1][1])


def test_covariate_forward_without_covariates_is_the_plain_model():
    model = tiny_model()
    model.eval()
    values = torch.randn(3, 1, 640).cumsum(-1)
    values[..., -512:] = float("nan")
    ids = torch.zeros(3, 1, dtype=torch.long)
    q, loc, _ = network.covariate_forward(model, values, ids,
                                          torch.full((3, 2, 640), float("nan")))
    tokens, rotary, started, loc2, _, far, _ = network.front(model, values)
    plain = network.trunk(model, tokens, rotary, started, ids[:, :, None].expand(3, 1, 20), far)
    keep = torch.isfinite(q)
    assert torch.equal(q[keep], plain[keep]) and torch.equal(loc, loc2)


def corpus_folder(path, seed=0):
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(seed)
    rows = []
    for i in range(12):
        variates = 1 if i % 3 else 3
        length = int(rng.integers(600, 3000))
        rows.append({"item_id": f"s{i}@0", "source": f"src{i % 3}", "freq": "H",
                     "domain": "d", "subdomain": f"x{i % 2}",
                     "target": rng.standard_normal((variates, length)).cumsum(-1).tolist()})
    path.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path / "train-00000.parquet")
    return path


def covariate_folder(path, seed=1):
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(seed)
    rows = []
    for i in range(10):
        length = int(rng.integers(3000, 20000))
        t = np.arange(length)
        driver = np.sin(2 * np.pi * t / 24) + 0.1 * rng.standard_normal(length)
        rows.append({"item_id": f"c{i}", "source": f"cov{i % 4}", "freq": "h",
                     "target": (5 * driver + rng.standard_normal(length)).tolist(),
                     "covariates": [driver.tolist(), (t % 168 < 48).astype(float).tolist()],
                     "covariate_names": ["forecast", "weekend"]})
    path.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path / "train-00000.parquet")
    return path


def test_covariate_corpus_draws(tmp_path):
    folder = covariate_folder(tmp_path / "cov")
    corpus = windows.CovariateCorpus(str(folder))
    assert corpus.sources == ["cov0", "cov1", "cov2", "cov3"] and len(corpus.recs) == 10
    y, c, h = corpus.train_sample(np.random.default_rng(0), 512)
    assert y.shape == (512 + h,) and c.shape == (2, 512 + h)
    gated = windows.CovariateCorpus(str(folder), weights={"cov1": 0.0},
                                    exclude={"cov0": {"c0"}}, min_length=5000)
    assert "cov1" not in gated.sources and gated.excluded == 1
    assert all(len(r["target"]) >= 5000 for r in gated.recs)
    with pytest.raises(ValueError):
        windows.CovariateCorpus(str(folder), weights={"nope": 1.0})
    with pytest.raises(ValueError, match="no usable rows"):
        stream.real_corpus({**stream.DEFAULTS, "real": str(tmp_path / "missing" / "covariates")})


def test_corpora_in_dataset_subfolders(tmp_path):
    import shutil

    from simons.training import data

    flat_uni, flat_cov = corpus_folder(tmp_path / "flat"), covariate_folder(tmp_path / "flatcov")
    root = tmp_path / "release"
    for kind, flat in (("univariate", flat_uni), ("covariates", flat_cov)):
        for name in ("a", "b"):
            (root / kind / name).mkdir(parents=True)
            shutil.copy(flat / "train-00000.parquet", root / kind / name / "train-00000.parquet")
    nested = data.load(str(root / "univariate"))
    assert len(nested) == 2 * len(data.load(str(flat_uni)))
    corpus = windows.CovariateCorpus(str(root / "covariates"))
    assert len(corpus.recs) == 2 * len(windows.CovariateCorpus(str(flat_cov)).recs)
    assert data.resolve(str(root), "univariate") == str(root)
    assert data.resolve(root, "univariate") == str(root)


def test_base_series_span_the_datasets(tmp_path):
    import shutil

    import pyarrow as pa
    import pyarrow.parquet as pq

    flat = corpus_folder(tmp_path / "flat")
    shutil.copy(flat / "train-00000.parquet", flat / "train-00001.parquet")
    assert windows.base_files(str(flat)) == [str(flat / "train-00000.parquet")]
    whole = windows.BaseSeries(windows.base_files(str(flat)))
    assert len(whole.uni) == 20 and len(whole.multi) == 4

    root = tmp_path / "release" / "univariate"
    rng = np.random.default_rng(0)
    for name, variates in (("a", 1), ("b", 3)):
        rows = [{"item_id": f"{name}{i}@0", "source": name, "freq": "H",
                 "target": rng.standard_normal((variates, 1000)).tolist()} for i in range(40)]
        (root / name).mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(rows), root / name / "train-00000.parquet")
        shutil.copy(root / name / "train-00000.parquet", root / name / "train-00001.parquet")
    files = windows.base_files(str(root))
    assert [Path(f).parent.name for f in files] == ["a", "b"]
    assert all(Path(f).name == "train-00000.parquet" for f in files)
    base = windows.BaseSeries(files, values=20_000)
    assert len(base.uni) == 10 + 4 * 3 and len(base.multi) == 4
    one = windows.BaseSeries(files, values=2)
    assert len(one.uni) == 1 + 3 and len(one.multi) == 1


def test_unreadable_corpus_name(tmp_path, monkeypatch):
    import huggingface_hub

    from simons.training import data

    def missing(*args, **kwargs):
        raise ValueError("no such dataset")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", missing)
    for name in ("data/trian", Path("data/trian")):
        with pytest.raises(FileNotFoundError, match="not a local folder"):
            data.load(name)
    with pytest.raises(FileNotFoundError, match="not a local folder"):
        windows.base_files("data/trian")


def test_covariate_batches_and_loss(tmp_path):
    base = windows.BaseSeries(windows.base_files(corpus_folder(tmp_path / "corpus")))
    real = windows.CovariateCorpus(str(covariate_folder(tmp_path / "cov")))
    spec = {**stream.DEFAULTS, "hists": [256, 512], "hist_weights": [0.5, 0.5]}
    kinds = ["synthetic", "sibling", "real"]
    batch = stream.make_batch(spec, np.random.default_rng(0), base, real, kinds,
                              np.array([0.63, 0.25, 0.12]), np.array([0.5, 0.5]))
    assert batch["values"].shape[-1] == batch["hist"] + 512
    assert batch["covs"].shape[0] == batch["values"].shape[0] >= 4
    model = tiny_model()
    model.train()
    loss, weight = stream.batch_loss(model, batch, 0.1, torch.device("cpu"))
    loss.backward()
    assert math.isfinite(float(loss.detach())) and float(weight) > 0
    assert model.covariates.role.grad is not None
    assert stream.tokens_per_window(4096) == 144 and stream.tokens_per_window(2048) == 80


def test_soup_and_wise(tmp_path):
    from simons.training.optim import NorMuonAdamW
    from simons.training.soup import merge, write
    from simons.training.trainer import save_checkpoint

    paths = []
    for i in range(2):
        model = tiny_model(seed=i)
        path = tmp_path / f"m{i}.pt"
        save_checkpoint(path, model, NorMuonAdamW(model), {"export_context_length": 1024}, 5,
                        1.0)
        paths.append(path)
    a, b = (torch.load(p, weights_only=True)["model"] for p in paths)
    soup = merge(paths)["model"]
    wise = merge(paths, wise_alpha=0.25)["model"]
    for key in a:
        assert torch.equal(soup[key], ((a[key].double() + b[key].double()) / 2).to(a[key].dtype))
        assert torch.equal(wise[key], (0.75 * a[key].double() + 0.25 * b[key].double())
                           .to(a[key].dtype))
    one = merge(paths[:1])["model"]
    assert all(torch.equal(one[k], a[k]) for k in a)
    out = write(paths, tmp_path / "soup" / "checkpoint.pt")
    assert "optimizer" not in torch.load(out, weights_only=True)


def test_warm_start_of_a_covariate_model(tmp_path):
    from simons.training.optim import NorMuonAdamW
    from simons.training.trainer import load_weights, save_checkpoint

    torch.manual_seed(0)
    plain = Simons(SimonsConfig(**TINY))
    path = tmp_path / "plain.pt"
    save_checkpoint(path, plain, NorMuonAdamW(plain), {}, 3, 0.0)
    model = tiny_model(seed=1)
    info = load_weights(model, path)
    assert info["read"] == "from the variate block"
    read, variate = model.covariates.read, model.variate_block
    assert torch.equal(read.qkv.weight, variate.qkv.weight)
    assert not read.out.weight.any() and not read.mlp[2].weight.any()
    assert not model.covariates.role.any()
    for key, value in plain.state_dict().items():
        assert torch.equal(value, model.state_dict()[key])
    torch.manual_seed(7)
    a = Simons(SimonsConfig(**TINY))
    after_a = torch.rand(3)
    torch.manual_seed(7)
    b = Simons(SimonsConfig(**TINY, **COVARIATES))
    after_b = torch.rand(3)
    assert torch.equal(after_a, after_b)
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())


def test_schedule_extensions():
    s = {"steps": 100, "lr_schedule": "wsd", "warmup_fraction": 0.05, "decay_fraction": 0.2,
         "decay_shape": "sqrt", "final_lr_fraction": 0.1, "rewarm_from": 0.1, "rewarm_steps": 10,
         "warm_lr_scale": 2.0}
    assert schedule.multiplier(70, s) == 1.0
    assert math.isclose(schedule.multiplier(90, s), 0.1 + 0.9 * (1 - math.sqrt(0.5)))
    value = schedule.multiplier(85, s, 85, True)
    assert math.isclose(value, 2.0 * (0.1 + 0.9 * 0.1) * (0.1 + 0.9 * (1 - math.sqrt(0.25))))


def write_config(tmp_path, name, model, training):
    import yaml

    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump({"model": model, "training": training}))
    return path


def test_tiny_covariate_chain(tmp_path, monkeypatch):
    from simons.covariates import prepare_covariates
    from simons.training.soup import write
    from simons.training.trainer import export, train

    monkeypatch.setitem(stream.DEFAULTS, "hists", [256, 512])
    monkeypatch.setitem(stream.DEFAULTS, "hist_weights", [0.5, 0.5])
    monkeypatch.setitem(stream.LONG, "hists", [512, 1024])
    corpus = corpus_folder(tmp_path / "corpus")
    covariate_data = covariate_folder(tmp_path / "covariates")
    common = {"seed": 1, "device": "cpu", "checkpoint_every": 100, "log_every": 2,
              "synthetic_families": "kernel,sarima,sde", "synthetic_rows": 2,
              "synthetic_context": 1024, "synthetic_pool_steps": 2, "loader_workers": 0,
              "mech_bank": 64, "mech_length": 1024}
    plain_model = dict(TINY)
    cov_model = {**TINY, **COVARIATES, "fit_row": False}
    final_model = {**TINY, **COVARIATES, "xreg": True, "xreg_guard": True,
                   "xreg_guard_margin": 0.9, "rows_only_history": 256}
    stage1 = train(write_config(tmp_path, "stage1", plain_model, {
        **common, "stage": "stage1", "steps": 4, "lr_schedule": "warmup_decay",
        "warmup_steps": 2, "decay_steps": 2}), tmp_path / "stage1")
    curriculum = train(write_config(tmp_path, "stage2", cov_model, {
        **common, "stage": "stage1", "steps": 6, "lr_schedule": "warmup_decay",
        "warmup_steps": 1, "decay_steps": 2, "decay_shape": "sqrt", "final_lr_fraction": 0.1,
        "warm_start": str(stage1), "rewarm_from": 0.1, "rewarm_steps": 2,
        "covariate_read_init": "variate", "role_phases": "0.3,0.6",
        "role_share": "0.5,0.7,0.9", "effect_share": "0.5,0.5,1.0",
        "effect_mix": "0.2,0.2,0.2,0.2,0.2", "effect_workers": 0}), tmp_path / "stage2")
    run = json.loads((tmp_path / "stage2" / "run.json").read_text())
    assert run["covariates"]["roles"]["batch"]["effect_rows"] > 0
    real = {**common, "stage": "stage2", "steps": 10, "data": str(corpus), "batch_size": 16,
            "lr_schedule": "wsd", "context_schedule": "1024:0.3,768:0.3,1536:0.4",
            "warm_start_stage": 1, "stitch_from_stage": 2, "far_from_stage": 2,
            "role_share": "0.4", "covariate_weight": 0.025,
            "covariate_data": str(covariate_data), "covariate_tokens": 512,
            "covariate_workers": 0}
    model2 = {**TINY, **COVARIATES, "xreg": True, "xreg_guard": True, "xreg_guard_margin": 0.9}
    post = train(write_config(tmp_path, "stage3", model2, {**real, "warm_start": str(curriculum)}),
                 tmp_path / "stage3")
    members = []
    for name, start, scale in (("stage4", 8, 2.0), ("stage4-long", 6, 1.0)):
        members.append(train(write_config(tmp_path, name, final_model, {
            **real, "seed": 7, "warm_start": str(post), "warm_start_step": start,
            "warm_lr_scale": scale, "rewarm_from": 0.1, "rewarm_steps": 1,
            "long_weight": 0.025, "long_workers": 0, "long_min_length": 4000,
            "long_tokens": 256}), tmp_path / name))
        log = [json.loads(line) for line in (tmp_path / name / "training.jsonl").open()]
        records = [r for r in log if "loss" in r]
        assert records and all(math.isfinite(r["covariate_loss"]) and
                               math.isfinite(r["long_loss"]) for r in records)
        assert json.loads((tmp_path / name / "run.json").read_text())["steps_this_run"] == \
            10 - start
    soup = write(members, tmp_path / "soup" / "checkpoint.pt")
    folder = export(soup, tmp_path / "model", context_length=1024)
    model = Simons.from_pretrained(folder)
    assert model.config.covariates and model.config.rows_only_history == 256
    history = np.sin(np.arange(700) * 2 * np.pi / 24) + np.arange(700) / 100
    future = np.sin(np.arange(700 + 48) * 2 * np.pi / 24)
    covariates = torch.from_numpy(prepare_covariates(700, 48, [future]))[None]
    forecast = model.forecast(torch.from_numpy(history).float()[None], 48, covariates=covariates)
    plain = model.forecast(torch.from_numpy(history).float()[None], 48)
    assert forecast.shape == (1, 48, 9) and torch.isfinite(forecast).all()
    assert not torch.equal(forecast, plain)


def test_micro_batches_same_loss_and_gradients():
    spec = {"boundaries": [], "p_role": [1.0], "p_eff": [0.0]}
    mixer, hook = role_mixer(9, spec)
    sequence, ids = mixer(torch.randn(6, 32, 2048).cumsum(-1), torch.arange(192).view(6, 32))
    first = mixer.first_rows().reshape(3, 32)

    def slots():
        host = ids.numpy().copy()
        start = np.zeros((6, 32), np.int64)
        start[3:] = first
        host[3:] = mixer.host_ids(3, 32)
        return host, start
    sequence, ids, slots, batch = hook.prepare(0, mixer, sequence, ids, slots)
    out = []
    for chunks in (1, 3):
        model = tiny_model(seed=2)
        with torch.no_grad():
            for p in model.covariates.parameters():
                p.normal_(0, 0.1)
        torch.manual_seed(1)
        loss, _ = objective.packed_loss(model, sequence, slots, 0.4, 8, row_tokens=288,
                                        roles=batch, micro_batches=chunks)
        if chunks == 1:
            loss.backward()
        out.append((float(loss.detach()), {n: p.grad.clone() for n, p in model.named_parameters()
                                  if p.grad is not None}))
    assert math.isclose(out[0][0], out[1][0], rel_tol=1e-6)
    assert out[0][1].keys() == out[1][1].keys()
    for name, grad in out[0][1].items():
        assert torch.allclose(grad, out[1][1][name], rtol=1e-4, atol=1e-7), name
    assert objective.chunk_bounds(10, 4) == [(0, 3), (3, 6), (6, 8), (8, 10)]


def test_model_init_leaves_the_gpu_generator_alone(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: calls.append(seed))
    monkeypatch.setattr(torch.cuda, "manual_seed", lambda seed: calls.append(seed))
    Simons(SimonsConfig(**TINY, **COVARIATES))
    assert calls == []


@pytest.mark.parametrize("covariates", [[[0.0] * 90], [[0.0] * 100, [0.0] * 90]])
def test_covariate_corpus_rejects_misaligned_covariates(tmp_path, covariates):
    import pyarrow as pa
    import pyarrow.parquet as pq

    row = {"item_id": "x", "source": "s", "freq": "h", "target": [1.0] * 100,
           "covariates": covariates}
    tmp_path.mkdir(exist_ok=True)
    pq.write_table(pa.Table.from_pylist([row]), tmp_path / "train-00000.parquet")
    with pytest.raises(ValueError, match="aligned"):
        windows.CovariateCorpus(str(tmp_path))


def test_example_data_keeps_every_segment():
    import importlib.util
    from pathlib import Path

    import pandas as pd

    path = Path(__file__).parents[1] / "scripts" / "prepare_example_data.py"
    spec = importlib.util.spec_from_file_location("prepare_example_data", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    length = 2 * script.SEGMENT + 500
    values = np.arange(length, dtype=np.float32)
    record = {"id": "a", "timestamp": pd.date_range("2020-01-01", periods=length, freq="h"),
              "x": values, "y": values * 2}
    covariate_rows = []
    segments = script.segments_of(record, ["x", "y"], "cfg", "d", "s", covariate_rows)
    assert [r["item_id"] for r in segments] == ["cfg/a@0", f"cfg/a@{script.SEGMENT}",
                                                f"cfg/a@{2 * script.SEGMENT}"]
    for row, start in zip(segments, (0, script.SEGMENT, 2 * script.SEGMENT)):
        assert row["target"][0][0] == start and row["target"][1][0] == 2 * start
    assert len(covariate_rows) == 3
    assert all(len(r["covariates"][0]) == len(r["target"]) for r in covariate_rows)


def test_stage1_leg_warm_start_step(tmp_path):
    from simons.training.trainer import train

    common = {"seed": 1, "device": "cpu", "checkpoint_every": 100, "log_every": 1,
              "synthetic_families": "kernel,sarima", "synthetic_rows": 2,
              "synthetic_context": 1024, "synthetic_pool_steps": 2, "mech_bank": 64,
              "mech_length": 1024, "lr_schedule": "wsd", "decay_fraction": 0.5}
    leg1 = train(write_config(tmp_path, "leg1", TINY, {**common, "stage": "stage1",
                                                       "steps": 10}), tmp_path / "leg1",
                 max_steps=4)
    train(write_config(tmp_path, "leg2", TINY, {
        **common, "stage": "stage1", "steps": 12, "warm_start": str(leg1),
        "warm_start_step": 4, "rewarm_from": 0.15, "rewarm_steps": 2}), tmp_path / "leg2")
    log = [json.loads(line) for line in (tmp_path / "leg2" / "training.jsonl").open()]
    steps = [r["step"] for r in log if "loss" in r]
    assert steps[0] == 5 and steps[-1] == 12
    first = next(r for r in log if "loss" in r)
    assert math.isclose(first["lr_multiplier"], 0.15 + 0.85 * 0.5)

