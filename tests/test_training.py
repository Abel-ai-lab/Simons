
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from simons.config import SimonsConfig
from simons.model import Simons
from simons.synth import SyntheticMixer, augment, base, mech, observe
from simons.synth.multi import MultiPool
from simons.training import data, mixture, network, objective, packing, schedule
from simons.training.config import FAMILIES

TINY = dict(context_length=1536, prediction_length=128, width=32, layers=1, heads=2,
            near_context=512, far_min_history=768, stats_window=1024, detector_mid=768)


@pytest.fixture(autouse=True)
def small_banks():
    mech.configure(64, 1024, 400, 1)
    augment.configure("", 0.0, None)
    yield
    augment.configure("", 0.0, None)


def draw(family, seed, n=8, length=600):
    mech.configure(64, 1024, 400, 1)
    gen = torch.Generator().manual_seed(seed)
    return base.draw(n, length, gen, "cpu", families=(family,))


@pytest.mark.parametrize("family", FAMILIES)
def test_family_shapes_finite_deterministic(family):
    x = draw(family, 1)
    assert x.shape == (8, 600) and x.dtype == torch.float32
    assert torch.isfinite(x).all()
    assert torch.equal(x, draw(family, 1))
    assert not torch.equal(x, draw(family, 2))


def test_post_and_layers():
    augment.configure("mix,season,filter,lattice", 1.0, "0.35,0.25,0.2,0.2")
    view = observe.make_config("duty", 1.0, 0.5)
    gen = torch.Generator().manual_seed(3)
    x = base.generate(16, 700, gen, "cpu", families=("kernel", "sarima"), noise=0.3)
    assert x.shape == (16, 700)
    assert torch.isfinite(x[:, -32:]).all()
    saved, observe.CONFIG = observe.CONFIG, view
    try:
        y = base.generate(16, 700, gen, "cpu", families=("kernel",))
    finally:
        observe.CONFIG = saved
    assert y.shape == (16, 700)


def test_augment_weight_zero_is_off(monkeypatch):
    monkeypatch.setitem(augment._MAKERS, "lattice", lambda gen, x: torch.zeros_like(x))
    augment.configure("mix,lattice", 1.0, "1,0")
    x = base.draw(256, 64, torch.Generator().manual_seed(0), "cpu", ("kernel",))
    assert (x.abs().amax(-1) > 0).all()


def test_mixer_rows_and_ids():
    gen = torch.Generator().manual_seed(0)
    mixer = SyntheticMixer(0.5, gen, ("kernel", "sde"), 0.3, 0.4, 2,
                           multi=MultiPool(0.5, 8, ("identity", "mix", "lag"), 64, seed=1))
    mixer.track_first = 32
    sequence = torch.randn(4, 32, 640)
    ids = torch.arange(128).view(4, 32)
    out, out_ids = mixer(sequence, ids)
    assert torch.equal(out[:2], sequence[:2]) and torch.equal(out_ids[:2], ids[:2])
    assert (out_ids[2:] >= base.SYNTHETIC_ID).all()
    assert mixer.first_rows().shape == (64,)
    assert mixer.host_ids(2, 32).shape == (2, 32)


def corpus(n_sources=4):
    from datasets import Dataset

    rng = np.random.default_rng(0)
    rows = {k: [] for k in ("item_id", "source", "freq", "domain", "subdomain", "target")}
    for s in range(n_sources):
        for b in range(3):
            offset = 0
            for _ in range(2):
                length = int(rng.integers(300, 2000))
                x = rng.standard_normal((1 + s % 3, length)).cumsum(-1).astype("float32")
                rows["item_id"].append(f"s{s}b{b}@{offset}")
                rows["source"].append(f"src{s}")
                rows["freq"].append("H")
                rows["domain"].append("d" + str(s % 2))
                rows["subdomain"].append("x")
                rows["target"].append(x.tolist())
                offset += length
    return Dataset.from_dict(rows)


def test_windows_deterministic_and_shaped():
    windows = data.Windows(corpus(), 1023, 1, 50, 7, downsample=0.4, stretch=0.25,
                           stitch_index=True)
    windows.stitch = True
    h, t = windows[5]
    h2, t2 = windows[5]
    assert torch.equal(torch.nan_to_num(h), torch.nan_to_num(h2)) and torch.equal(t, t2)
    assert h.shape[1] == 1023 and t.shape[1] == 1 and h.shape[0] == t.shape[0] <= 32
    assert sum(p >= 0 for prev in windows.prev for p in prev) > 0


def test_corpus_files_one_level_deep(tmp_path):
    for name in ("train-0.parquet", "a/train-0.parquet"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(b"")
    assert data.corpus_files(tmp_path) == [str(tmp_path / "a/train-0.parquet"),
                                          str(tmp_path / "train-0.parquet")]
    (tmp_path / "a/old").mkdir()
    (tmp_path / "a/old/train-0.parquet").write_bytes(b"")
    with pytest.raises(ValueError, match="more than one folder below"):
        data.corpus_files(tmp_path)


def test_mixture_budgets():
    train = corpus()
    windows = data.Windows(train, 1023, 1, 50, 7)
    meta = mixture.source_meta(train)
    capacity = sum(float((np.minimum(s[:, 0], 32) * s[:, 1]).sum()) for s in windows.shapes)
    mix = mixture.Mixture(windows.names, windows.shapes, meta, 2.0, capacity)
    assert math.isclose(mix.share.sum(), 1.0)
    assert (mix.epochs <= 2.0 + 1e-9).all()
    p, t = mix.weights(1024, 0, 0.4, 8, 512, 1024, 0.25, 0.5)
    assert math.isclose(p.sum(), 1.0) and (t > 0).all()

    def make(b):
        return mixture.Mixture(windows.names, windows.shapes, meta, 6.0, b)

    budget, per_step = mixture.plan_budget(make, [(10, (1024, 0, 0.4, 8, 512, 1024, 0.25, 0.5))],
                                           16, [0.4])
    assert budget > 0 and len(per_step) == 1


def test_schedules():
    s = {"steps": 100, "lr_schedule": "warmup_decay", "warmup_steps": 10, "decay_steps": 20,
         "decay_shape": "sqrt", "final_lr_fraction": 0.1}
    values = [schedule.multiplier(t, s) for t in range(100)]
    assert values[0] == 0.1 and values[9] == 1.0 and values[50] == 1.0
    assert math.isclose(values[-1], 0.1)
    s = {"steps": 100, "lr_schedule": "wsd", "warmup_fraction": 0.05, "decay_fraction": 0.5,
         "final_lr_fraction": 0.0, "rewarm_from": 0.15, "rewarm_steps": 10}
    assert math.isclose(schedule.multiplier(40, s, 40, True), 0.15 + 0.85 * 0.1)
    stages = schedule.parse_schedule("512:0.3,1024:0.7", 100, 1024, 32)
    assert stages == [(512, 0, 30), (1024, 30, 100)]
    with pytest.raises(ValueError, match="no step"):
        schedule.parse_schedule("512:0.3,1024:0.7", 1, 1024, 32)
    share = schedule.share_function(stages, 1, 10, 0.4)
    assert share(29) == 1.0 and math.isclose(share(39), 0.4) and share(80) == 0.4
    keep = schedule.ramp_rows(10, 0.7, 0.4)
    assert len(keep) == 3 + 4


def test_packing_plan_covers_spans():
    ids = np.array([[0, 0, 1, -1], [2, 3, 3, 3]])
    first = np.array([[2, 3, 0, 8], [5, 1, 1, 7]])
    source, segment, _ = packing.plan(ids.reshape(-1), first.reshape(-1), 8, width=4)
    spans = {0: 2 * 6, 1: 8, 2: 3, 3: 3 * 7}
    for window, tokens in spans.items():
        assert (segment == window).sum() == tokens
    assert len(np.unique(source[source >= 0])) == sum(spans.values())


def test_packed_loss_is_exact():
    torch.manual_seed(0)
    model = Simons(SimonsConfig(**TINY))
    rng = np.random.default_rng(1)
    shapes = [(2, 300), (1, 1664), (3, 900), (1, 70)]
    seq = np.full((1, 32, 1664), np.nan, np.float32)
    ids = np.full((1, 32), -1)
    slot = 0
    for w, (v, n) in enumerate(shapes):
        seq[0, slot:slot + v, -n:] = rng.standard_normal((v, n)).cumsum(-1)
        ids[0, slot:slot + v] = w
        slot += v
    seq = torch.from_numpy(seq)

    def loss(window_ids):
        keep = torch.from_numpy(np.isin(ids, window_ids))[..., None]
        part = torch.where(keep, seq, torch.nan)
        first = data.first_patches(part, 32)
        slot_ids = np.where(keep[..., 0].numpy(), ids, -1)
        return objective.packed_loss(model, part, lambda: (slot_ids.copy(), first.numpy().copy()),
                                     0.0, 1)

    with torch.no_grad():
        everything, count = loss([0, 1, 2, 3])
        parts = [loss([w]) for w in range(len(shapes))]
    weighted = sum(float(value) * int(n) for value, n in parts) / sum(int(n) for _, n in parts)
    assert int(count) == sum(int(n) for _, n in parts)
    assert math.isclose(float(everything), weighted, rel_tol=1e-4)


def test_adamw_bias_correction_per_tensor():
    from simons.training.optim import NorMuonAdamW

    model = Simons(SimonsConfig(**TINY))
    optimizer = NorMuonAdamW(model, adam_lr=1.0)
    adam = optimizer.param_groups[1]["params"]
    for step in range(4):
        for p in model.parameters():
            p.grad = None if p is model.far_type and step < 3 else torch.ones_like(p)
        before = [p.detach().clone() for p in adam]
        optimizer.step()
    for p, old in zip(adam, before):
        assert torch.allclose(old - p.detach(), torch.ones_like(old))


def test_checkpoint_blocks_same_loss_and_gradients():
    torch.manual_seed(0)
    model = Simons(SimonsConfig(**{**TINY, "layers": 3}))
    seq = torch.randn(2, 32, 1664).cumsum(-1)
    seq[:, 3:] = torch.nan
    ids = np.where(np.arange(32) < 3, 0, -1)[None].repeat(2, 0)
    ids[1, :3] = 1
    first = data.first_patches(seq, 32).numpy()

    def run(blocks):
        model.zero_grad(set_to_none=True)
        loss, _ = objective.packed_loss(model, seq, lambda: (ids.copy(), first.copy()), 0.0, 1,
                                        checkpoint_blocks=blocks)
        loss.backward()
        return loss.detach(), [p.grad.clone() for p in model.parameters() if p.grad is not None]

    plain, grads = run(0)
    for blocks in (2, 3):
        loss, other = run(blocks)
        assert torch.equal(loss, plain)
        assert len(other) == len(grads)
        assert all(torch.allclose(a, b, rtol=1e-5, atol=1e-7) for a, b in zip(grads, other))


def tiny_config(tmp_path, stage, **extra):
    import yaml

    training = {"stage": stage, "steps": 20, "seed": 1, "device": "cpu", "checkpoint_every": 10,
                "log_every": 5, "synthetic_families": "kernel,sarima,sde", "synthetic_rows": 2,
                "synthetic_context": 1024, "synthetic_pool_steps": 2, "loader_workers": 1,
                "mech_bank": 64, "mech_length": 1024, **extra}
    path = tmp_path / f"{stage}.yaml"
    path.write_text(yaml.safe_dump({"model": TINY, "training": training}))
    return path


def test_tiny_training_runs(tmp_path):
    from simons.training.trainer import export, train

    first = train(tiny_config(tmp_path, "stage1", lr_schedule="warmup_decay", warmup_steps=2,
                              decay_steps=5, synthetic_observe="duty"), tmp_path / "stage1")
    stage2 = tiny_config(tmp_path, "stage2", data="toy", validation_data="toy", batch_size=32,
                       lr_schedule="wsd", context_schedule="1024:0.3,768:0.3,1536:0.4",
                       synthetic_only_stages=1, synthetic_ramp_steps=3, warm_start=str(first),
                       warm_start_stage=1, stitch_from_stage=2, far_from_stage=2)
    checkpoint = train(stage2, tmp_path / "stage2")
    log = [json.loads(line) for line in (tmp_path / "stage2" / "training.jsonl").open()]
    losses = [r["loss"] for r in log if "loss" in r]
    assert losses and all(math.isfinite(v) for v in losses)
    assert any("validation_loss" in r for r in log)
    assert json.loads((tmp_path / "stage2" / "run.json").read_text())["steps_this_run"] == 14
    folder = export(checkpoint, tmp_path / "model", context_length=1024)
    model = Simons.from_pretrained(folder)
    forecast = model.forecast(torch.randn(2, 400).cumsum(-1), 150)
    assert forecast.shape == (2, 150, 9) and torch.isfinite(forecast).all()


def test_front_layouts():
    model = Simons(SimonsConfig(**TINY))
    values = torch.randn(1, 2, 1664)
    full = network.front(model, values, True)
    dropped = network.front(model, values, False)
    short = network.front(model, values[..., -640:], True)
    assert full[6] == 4 and full[0].shape[2] == 4 + 20
    assert dropped[6] == 0 and dropped[0].shape[2] == 20
    assert short[5] is None and short[0].shape[2] == 20


def test_mixup_alpha_below_one_is_rejected(tmp_path):
    from simons.training import config

    with pytest.raises(ValueError, match="augment_mixup_alpha"):
        config.load(tiny_config(tmp_path, "stage1", augment_mixup_alpha=0.3))


CONFIGS = sorted((Path(__file__).parents[1] / "configs").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=[p.stem for p in CONFIGS])
def test_shipped_configs_validate(path):
    from simons.training import config

    model, settings = config.load(path)
    assert settings["stage"] in ("stage1", "stage2")
    if settings["stage"] == "stage2":
        stages = schedule.parse_schedule(settings["context_schedule"], settings["steps"],
                                         model.context_length, model.patch_length)
        assert stages[settings["warm_start_stage"]][1] > 0


def test_shipped_config_set():
    names = {p.stem for p in CONFIGS}
    runs = ("stage1", "stage2", "stage3", "stage4")
    expected = {f"simons-{size}-{run}" for size in ("25m", "60m", "178m", "297m") for run in runs}
    expected |= {f"simons-{size}-stage4-long" for size in ("25m", "60m", "178m")}
    expected |= {"simons-297m-stage1-ext"}
    assert names == expected


def test_predictor_quantile_forecasts():
    pytest.importorskip("gluonts")
    import pandas as pd

    from simons.predictor import SimonsPredictor

    model = Simons(SimonsConfig(**TINY)).eval()
    start = pd.Period("2024-01-01 00:00", freq="h")
    rng = np.random.default_rng(0)
    dataset = [{"target": rng.standard_normal(300).cumsum().astype("float32"), "start": start,
                "item_id": "uni"},
               {"target": rng.standard_normal((3, 200)).cumsum(-1).astype("float32"),
                "start": start, "item_id": "multi"}]
    uni, multi = SimonsPredictor(model, prediction_length=48).predict(dataset)
    assert uni.forecast_array.shape == (10, 48) and multi.forecast_array.shape == (10, 48, 3)
    assert uni.forecast_keys == ([str(q) for q in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]
                                 + ["mean"])
    for f in (uni, multi):
        q = f.forecast_array[:9]
        assert np.isfinite(f.forecast_array).all() and (np.diff(q, axis=0) >= -1e-5).all()
        assert np.allclose(f.forecast_array[9], q.mean(axis=0), atol=1e-5)
    assert uni.start_date == start + 300 and multi.start_date == start + 200
    assert uni.item_id == "uni" and uni.quantile(0.5).shape == (48,)
    assert multi.quantile(0.5).shape == (48, 3)


def test_predictor_streams_batches():
    pytest.importorskip("gluonts")
    import pandas as pd

    from simons.predictor import SimonsPredictor

    model = Simons(SimonsConfig(**TINY)).eval()
    start = pd.Period("2024-01-01 00:00", freq="h")
    read = []

    def entries():
        for i in range(5):
            read.append(i)
            yield {"target": np.arange(200, dtype="float32") + i, "start": start}

    stream = SimonsPredictor(model, prediction_length=16, batch_size=2).predict(entries())
    next(stream)
    assert read == [0, 1]
    assert len(list(stream)) == 4 and read == [0, 1, 2, 3, 4]


def test_export_cli(tmp_path):
    import os
    import subprocess
    import sys

    from simons.training.optim import NorMuonAdamW
    from simons.training.trainer import save_checkpoint

    torch.manual_seed(0)
    model = Simons(SimonsConfig(**TINY))
    checkpoint = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint, model, NorMuonAdamW(model), {"export_context_length": 1024}, 0, 0.0)
    src = Path(__file__).parents[1] / "src"
    result = subprocess.run(
        [sys.executable, "-m", "simons.training", "export", "--checkpoint", str(checkpoint),
         "--output", str(tmp_path / "model")],
        capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(src)})
    assert result.returncode == 0, result.stderr
    exported = Simons.from_pretrained(tmp_path / "model")
    assert exported.config.context_length == 1024
    for key, value in model.state_dict().items():
        assert torch.equal(value, exported.state_dict()[key])


def toy_windows():
    return data.Windows(data.load("toy", "train"), 1663, 1, 100, 3)


def test_cluster_build_save_load(tmp_path):
    from simons.training import clusters

    torch.manual_seed(0)
    model = Simons(SimonsConfig(**TINY))
    windows = toy_windows()
    one, info = clusters.build_labels(model, windows, 4, 64, 256, 0, 1, "cpu")
    again, _ = clusters.build_labels(model, windows, 4, 64, 256, 0, 1, "cpu")
    two, _ = clusters.build_labels(model, windows, 4, 64, 256, 0, 2, "cpu")
    n = sum(len(s) for s in windows.sources)
    assert one.shape == (n,) and info["rows"] == n and np.array_equal(one, again)
    assert one.min() >= 0 and one.max() < 4 and two.min() >= 0 and two.max() < 4
    path = tmp_path / "clusters.npz"
    clusters.save(path, one, info, windows)
    labels, k = clusters.load(path, windows)
    assert k == 4 and np.array_equal(labels, one)
    other = data.Windows(corpus(), 1663, 1, 100, 3)
    with pytest.raises(ValueError, match="different corpus rows"):
        clusters.load(path, other)


def test_stage_table_and_window_hook():
    from simons.training.clusters import StageTable

    windows = data.Windows(corpus(), 1023, 1, 200, 7)
    n = sum(len(s) for s in windows.sources)
    labels = (np.arange(n) % 5).astype(np.int32)
    table = StageTable(labels, windows, None, 5, 2.0, 0.5)
    assert math.isclose(table.p_cum[-1], 1.0)
    rng = np.random.default_rng(0)
    for _ in range(50):
        s, pos = table.draw(rng)
        assert 0 <= s < len(windows.sources) and 0 <= pos < len(windows.sources[s])
    plain = [windows[i] for i in range(40)]
    windows.shape = StageTable(labels, windows, None, 5, 2.0, 0.0)
    same = [windows[i] for i in range(40)]
    assert all(torch.equal(torch.nan_to_num(a[0]), torch.nan_to_num(b[0])) for a, b in
               zip(plain, same))
    windows.shape = StageTable(labels, windows, None, 5, 2.0, 1.0)
    mixed = [windows[i] for i in range(40)]
    assert any(not torch.equal(torch.nan_to_num(a[0]), torch.nan_to_num(b[0])) for a, b in
               zip(plain, mixed))
    assert all(torch.equal(torch.nan_to_num(a[0]), torch.nan_to_num(b[0])) for a, b in
               zip(mixed, [windows[i] for i in range(40)]))


def test_tiny_post_with_clusters(tmp_path):
    import subprocess
    import sys

    from simons.training.trainer import train

    first = train(tiny_config(tmp_path, "stage1", lr_schedule="warmup_decay", warmup_steps=2,
                              decay_steps=5, steps=4), tmp_path / "stage1")
    stage2 = tiny_config(tmp_path, "stage2", data="toy", batch_size=32, steps=10,
                       lr_schedule="wsd", context_schedule="1024:0.3,768:0.3,1536:0.4",
                       warm_start=str(first), warm_start_stage=1, stitch_from_stage=2,
                       far_from_stage=2, shape_mix=0.5, shape_k=4, shape_anchors=64,
                       shape_context=256, shape_shards=2,
                       shape_clusters=str(tmp_path / "clusters.npz"))
    src = Path(__file__).parents[1] / "src"
    result = subprocess.run(
        [sys.executable, "-m", "simons.training", "clusters", "--config", str(stage2),
         "--embedder", str(first), "--output", str(tmp_path / "clusters.npz")],
        capture_output=True, text=True, env={**__import__("os").environ, "PYTHONPATH": str(src)})
    assert result.returncode == 0, result.stderr
    train(stage2, tmp_path / "stage2")
    log = [json.loads(line) for line in (tmp_path / "stage2" / "training.jsonl").open()]
    assert [r for r in log if "loss" in r] and all(math.isfinite(r["loss"]) for r in log
                                                   if "loss" in r)
