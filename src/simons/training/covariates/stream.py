"""Compute weighted auxiliary losses on synthetic, sibling and real covariate windows."""

import time
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from simons.model import QUANTILES
from simons.synth.covariates import effects
from simons.training.covariates import windows as W

PATCH = 32
FUTURE = 512
DEFAULTS = {
    "base": None,
    "real": None,
    "weight": 0.025,
    "history_weight": 0.1,
    "mix": {"synthetic": 0.63, "sibling": 0.25, "real": 0.12},
    "hists": [128, 256, 512, 1024, 2048, 4096],
    "hist_weights": [0.1, 0.15, 0.2, 0.2, 0.2, 0.15],
    "tokens": 4096,
    "max_batch": 128,
    "max_cov": 8,
    "horizons": [1, 2, 3, 4, 6, 8, 12, 16],
    "drop_all": 0.15,
    "drop_each": 0.15,
    "drop_fit": 0.2,
    "flip": 0.3,
    "sqrt": 0.2,
    "workers": 3,
    "source_weights": None,
    "exclude": None,
    "min_length": 0,
    "min_observed": 0.0,
    "future": FUTURE,
    "fovea_tokens": False,
    "fovea_near": 4096,
    "fovea_pool": 8,
    # Separate the long pass's random stream from the main pass.
    "stream_tag": 3232,
}
LONG = {"mix": {"real": 1.0}, "hists": [2048, 4096], "hist_weights": [0.5, 0.5],
        "fovea_tokens": True, "drop_all": 0.05, "stream_tag": 4242}


def long_spec(main: dict, overrides: dict) -> dict:
    """Apply long-pass overrides to the main spec, dropping per-source weights."""
    spec = {k: v for k, v in main.items() if k != "source_weights"}
    spec.update(LONG)
    spec.update(overrides)
    return spec


def tokens_per_window(hist: int, near: int = 4096, pool: int = 8, patch: int = PATCH,
                      future: int = FUTURE) -> int:
    """Count near, pooled-far and future tokens for one window."""
    near_steps = min(hist, near) if near else hist
    far = (hist - near_steps) // (patch * pool) if near else 0
    return near_steps // patch + far + future // patch


def real_corpus(s):
    """Load the pass's covariate corpus, or return None when disabled; reject empty corpora."""
    if not s["mix"].get("real", 0) > 0 or not s["real"]:
        return None
    exclude = W.load_items(s["exclude"]) if s["exclude"] else None
    corpus = W.CovariateCorpus(s["real"], exclude=exclude, min_length=int(s["min_length"]),
                               weights=s["source_weights"])
    if not corpus.recs:
        raise ValueError(f"covariate corpus {s['real']}: no usable rows (train-*.parquet rows "
                         f"with covariates, at least {s['min_length']} steps)")
    return corpus


class Batches(IterableDataset):
    """An endless stream of covariate batches (one numpy generator per loader worker)."""

    def __init__(self, spec, seed: int, first_step: int):
        self.spec, self.seed, self.first_step = spec, seed, first_step

    def __iter__(self):
        info = get_worker_info()
        worker = info.id if info is not None else 0
        rng = np.random.default_rng([self.seed, int(self.spec["stream_tag"]), worker,
                                     self.first_step])
        s = self.spec
        base = W.BaseSeries(W.base_files(s["base"])) if s["base"] else None
        real = real_corpus(s)
        kinds = [k for k in s["mix"] if s["mix"][k] > 0 and (k != "real" or real is not None)
                 and (k != "sibling" or (base is not None and base.multi))
                 and (k != "synthetic" or base is not None)]
        if not kinds:
            raise ValueError(f"the covariate pass (mix {s['mix']}) has no windows: no base "
                             f"series and no covariate-corpus series of at least "
                             f"{s['min_length']} steps")
        dropped = [k for k in s["mix"] if s["mix"][k] > 0 and k not in kinds]
        if dropped and worker == 0:
            warnings.warn(f"covariate pass: no {' or '.join(dropped)} windows (sibling windows "
                          f"need base rows with several channels of equal length, real windows "
                          f"a covariate corpus); its mix is renormalized over {kinds}",
                          stacklevel=1)
        p = np.array([s["mix"][k] for k in kinds], float)
        p /= p.sum()
        hw = np.array(s["hist_weights"], float)
        hw /= hw.sum()
        while True:
            yield make_batch(s, rng, base, real, kinds, p, hw)


def make_batch(s, rng, base, real, kinds, p, hw):
    hist = int(rng.choice(s["hists"], p=hw))
    future_steps = int(s["future"])
    per_window = (hist + future_steps) // PATCH
    if s["fovea_tokens"]:
        per_window = tokens_per_window(hist, s["fovea_near"], s["fovea_pool"],
                                       future=future_steps)
    count = int(np.clip(round(s["tokens"] / per_window), 4, s["max_batch"]))
    samples, kind_count, misses = [], dict.fromkeys(kinds, 0), 0
    while len(samples) < count:
        if misses == 10_000:
            raise ValueError(f"the covariate pass (kinds {kinds}) found no usable window in "
                             f"10,000 draws at history {hist}: its series are too short or too "
                             f"sparsely observed (min_length {s['min_length']}, min_observed "
                             f"{s['min_observed']})")
        misses += 1
        kind = kinds[rng.choice(len(kinds), p=p)]
        if kind == "real":
            got = real.train_sample(rng, hist, float(s["min_observed"]))
            if got is None:
                continue
            y, c, H = got
        else:
            H = int(rng.choice(s["horizons"]) * PATCH)
            maker = W.sibling if kind == "sibling" else effects.synthetic
            y, c = maker(base, rng, hist, H)
            if np.isfinite(y[-H:]).sum() < H // 4 or np.isfinite(y[:hist]).sum() < 32:
                continue
        misses = 0
        c = c[: s["max_cov"]]
        if rng.random() < s["drop_all"]:
            c = c[:0]
        elif len(c):
            c = c[rng.random(len(c)) >= s["drop_each"]]
        y = np.asarray(y, "float64")
        fin = np.isfinite(y)
        if s["sqrt"] and fin.any() and (y[fin] >= 0).all() and rng.random() < s["sqrt"]:
            y = np.sqrt(np.clip(y, 0, None))
        if rng.random() < s["flip"]:
            y = -y
        samples.append((y.astype("float32"), np.asarray(c, "float32"), H))
        kind_count[kind] += 1
    L = hist + future_steps
    B = len(samples)
    C = max(1, max(len(c) for _, c, _ in samples))
    values = np.full((B, 1, L), np.nan, "float32")
    covs = np.full((B, C, L), np.nan, "float32")
    future = np.zeros((B, L), bool)
    for b, (y, c, H) in enumerate(samples):
        lo = hist - (len(y) - H)
        values[b, 0, lo:lo + len(y)] = y
        if len(c):
            covs[b, : len(c), lo:lo + len(y)] = c
        future[b, hist:hist + H] = True
    fit_keep = rng.random(B) >= s["drop_fit"]
    return {"values": torch.from_numpy(values), "future": torch.from_numpy(future),
            "covs": torch.from_numpy(covs), "fit_keep": torch.from_numpy(fit_keep),
            "kinds": kind_count, "hist": hist}


def batch_loss(model, batch, history_weight: float, device):
    """(weighted mean pinball loss, summed target weight) of one covariate batch."""
    from simons.training import network

    values = batch["values"].to(device, non_blocking=True)
    future = batch["future"].to(device, non_blocking=True)
    covs = batch["covs"].to(device, non_blocking=True)
    fit_keep = batch["fit_keep"].to(device, non_blocking=True)
    inputs = values.masked_fill(future[:, None], float("nan"))
    ids = torch.zeros(values.shape[:2], dtype=torch.long, device=device)
    q, loc, scale = network.covariate_forward(model, inputs, ids, covs, fit_keep)
    q = q[:, :, :-1].float()
    target = values.unflatten(-1, (-1, PATCH))[:, :, 1:].double()
    target = torch.asinh((target - loc[..., :-1, None]) / scale[..., :-1, None]).float()
    seen = torch.isfinite(inputs).unflatten(-1, (-1, PATCH)).any(-1).cummax(-1).values
    target = target.masked_fill(~seen[..., :-1, None], float("nan"))
    fut = future.unflatten(-1, (-1, PATCH))[:, 1:].any(-1)[:, None, :, None]
    weight = torch.where(fut, 1.0, float(history_weight))
    observed = torch.isfinite(target)
    levels = q.new_tensor(QUANTILES)
    err = torch.where(observed, target, 0.0).unsqueeze(-1) - torch.nan_to_num(q)
    loss = torch.maximum(levels * err, (levels - 1) * err).mean(-1)
    w = weight * observed
    return (loss * w).sum() / w.sum().clamp_min(1), w.sum().detach()


class CovariateStream:
    """One covariate pass of a run (per rank): its loader and its loss."""

    def __init__(self, spec, seed: int, first_step: int, device):
        unknown = set(spec) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"covariate pass: unknown fields {sorted(unknown)}")
        self.spec = {**DEFAULTS, **spec}
        self.weight = float(self.spec["weight"])
        self.device = device
        workers = int(self.spec["workers"])
        # A private generator keeps loader creation independent of the global random stream.
        private = torch.Generator()
        tag = int(self.spec["stream_tag"])
        private.manual_seed(int(seed) * 1009 + 3304 + int(first_step)
                            + (0 if tag == 3232 else 7 * tag))
        self.loader = DataLoader(Batches(self.spec, seed, first_step), batch_size=None,
                                 generator=private, num_workers=workers,
                                 pin_memory=device.type == "cuda",
                                 persistent_workers=workers > 0,
                                 prefetch_factor=4 if workers > 0 else None)
        self.batches = iter(self.loader)
        self.last_weight_sum = None
        self.stats = {"batches": 0, "windows": 0, "wait_seconds": 0.0, "kinds": {}}

    def loss(self, model):
        """The pass's loss on its next batch (its summed target weight in `last_weight_sum`)."""
        begin = time.perf_counter()
        batch = next(self.batches)
        self.stats["wait_seconds"] += time.perf_counter() - begin
        self.stats["batches"] += 1
        self.stats["windows"] += len(batch["values"])
        for k, n in batch["kinds"].items():
            self.stats["kinds"][k] = self.stats["kinds"].get(k, 0) + int(n)
        loss, self.last_weight_sum = batch_loss(model, batch, self.spec["history_weight"],
                                                self.device)
        return loss

    def summary(self):
        return {**self.stats, "wait_seconds": round(self.stats["wait_seconds"], 2)}

    def close(self):
        self.batches = None
        self.loader = None

