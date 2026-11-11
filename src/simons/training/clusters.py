"""Sample real windows by shape clusters within per-source token budgets."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from simons.model import time_blocks
from simons.training import network
from simons.training.mixture import waterfill

LABELS_FORMAT = "simons-clusters-1"


def features(model, values: torch.Tensor) -> torch.Tensor:
    """Return mean final-norm features [B,D] over observed patches of histories [B,H].
    H must be a patch-length multiple within near context; leading NaN marks padding."""
    c = model.config
    patch = c.patch_length
    batch, history = values.shape
    full = torch.nn.functional.pad(values, (0, network.future_steps(c)), value=float("nan"))
    tokens, rotary, started, _, _, _, _ = network.front(model, full[:, None])
    p = history // patch
    tokens, started = tokens[:, :, :p], started[:, :, :p]
    cos, sin = (t[:, :, :p] for t in rotary)
    x = (model.embed(tokens) + model.embed_mlp(tokens)).reshape(batch, p, -1)
    x = time_blocks(model.time_blocks, x, started.reshape(batch, p), (cos, sin), None)
    vb, width = model.variate_block, x.shape[-1]
    x = x + vb.out(vb.qkv(vb.attention_norm(x))[..., 2 * width:])
    x = x + vb.mlp(vb.mlp_norm(x))
    h = model.norm(x).float()
    seen = tokens[:, 0, :, patch:].amax(-1) > 0
    weight = seen.float()
    return (h * weight[..., None]).sum(1) / weight.sum(1, keepdim=True).clamp_min(1)


class Rows:
    """Flat row index over a `data.Windows` sampler's sources: (source, position) per row."""

    def __init__(self, windows):
        self.sizes = np.array([len(s) for s in windows.sources], dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)])
        self.source = np.repeat(np.arange(len(self.sizes)), self.sizes)
        shapes = np.concatenate(windows.shapes)
        self.channels, self.length = shapes[:, 0], shapes[:, 1]
        self.windows = windows

    def __len__(self):
        return int(self.offsets[-1])

    def array(self, flat: int):
        s = self.source[flat]
        return self.windows.sources[s][flat - self.offsets[s]]


def embed_rows(model, rows: Rows, flat: np.ndarray, variate: np.ndarray, context: int, device,
               tokens_per_batch: int = 65536, max_rows: int = 8192) -> torch.Tensor:
    """[len(flat), D] fp32 features (on `device`) of the given rows, batched by patch count."""
    patch = model.config.patch_length
    hist = np.minimum(rows.length[flat], context)
    patches = -(-hist // patch)
    out = torch.empty(len(flat), model.config.width, device=device)
    order = np.argsort(patches, kind="stable")
    sorted_patches = patches[order]
    i = 0
    while i < len(order):
        p = int(patches[order[i]])
        size = max(1, min(max_rows, tokens_per_batch // p))
        same = int(np.searchsorted(sorted_patches, p, side="right"))
        j = min(same, i + size)
        chunk = order[i:j]
        buf = np.full((len(chunk), p * patch), np.nan, dtype=np.float32)
        for r, k in enumerate(chunk):
            a = rows.array(flat[k])
            h = int(hist[k])
            buf[r, p * patch - h:] = a[variate[k], a.shape[1] - h:]
        values = torch.from_numpy(buf).to(device, non_blocking=True)
        out[torch.from_numpy(chunk).to(device)] = features(model, values)
        i = j
    return out


def kmeans(x: torch.Tensor, k: int, iters: int, seed: int):
    """Spherical k-means of unit rows x [N, D]: (centroids [k, D], cluster sizes)."""
    gen = torch.Generator().manual_seed(seed)
    n = x.shape[0]
    centroids = x[torch.randperm(n, generator=gen)[:k].to(x.device)].clone()
    for _ in range(iters):
        labels = assign(x, centroids)
        sums = torch.zeros_like(centroids).index_add_(0, labels, x)
        counts = torch.bincount(labels, minlength=k)
        empty = counts == 0
        if empty.any():
            pick = torch.randint(n, (int(empty.sum()),), generator=gen).to(x.device)
            sums[empty] = x[pick]
        centroids = torch.nn.functional.normalize(sums, dim=-1)
    return centroids, torch.bincount(assign(x, centroids), minlength=k)


def assign(x: torch.Tensor, centroids: torch.Tensor, chunk: int = 65536) -> torch.Tensor:
    return torch.cat([(x[i:i + chunk] @ centroids.T).argmax(-1)
                      for i in range(0, x.shape[0], chunk)])


@torch.no_grad()
def build_labels(model, windows, k: int, anchors: int, context: int, seed: int, shards: int,
                 device):
    """Cluster label (int32) of every row of `windows`, and build facts."""
    rows = Rows(windows)
    n = len(rows)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 8017]))
    variate = (rng.random(n) * rows.channels).astype(np.int64)
    per = math.ceil(int(anchors) / len(rows.sizes))
    anchor_rows = np.concatenate([rows.offsets[s] + rng.integers(rows.sizes[s], size=per)
                                  for s in range(len(rows.sizes))])
    if len(anchor_rows) < k:
        raise ValueError(f"shape_anchors ({anchors}) gives {len(anchor_rows)} anchor rows: it "
                         f"must be at least shape_k ({k})")
    was_training = model.training
    model.eval()
    with torch.autocast("cuda", torch.bfloat16, enabled=torch.device(device).type == "cuda"):
        x = torch.cat([embed_rows(model, rows, anchor_rows[r::shards],
                                  variate[anchor_rows[r::shards]], context, device)
                       for r in range(shards)])
        mean = x.mean(0)
        x = torch.nn.functional.normalize(x - mean, dim=-1)
        centroids, sizes = kmeans(x, k, 40, int(seed))
        hist_p = -(-np.minimum(rows.length, context) // model.config.patch_length)
        order = np.argsort(hist_p, kind="stable")
        labels = np.empty(n, dtype=np.int32)
        step = 1 << 20
        for r in range(shards):
            shard = order[r::shards]
            out = np.empty(len(shard), dtype=np.int32)
            for i in range(0, len(shard), step):
                part = shard[i:i + step]
                f = embed_rows(model, rows, part, variate[part], context, device)
                f = torch.nn.functional.normalize(f - mean, dim=-1)
                out[i:i + len(part)] = assign(f, centroids).int().cpu().numpy()
            labels[shard] = out
    model.train(was_training)
    info = {"rows": n, "k": k, "anchors": len(anchor_rows), "shards": shards,
            "anchor_cluster_sizes_q": np.quantile(sizes.cpu().numpy(), [0, .1, .5, .9, 1]).tolist(),
            "row_clusters_nonempty": int((np.bincount(labels, minlength=k) > 0).sum())}
    return labels, info


def fingerprint(windows) -> str:
    """Hash of the rows the labels index: source names and (variates, steps) of every row."""
    digest = hashlib.sha256()
    for name, shapes in zip(windows.names, windows.shapes):
        digest.update(name.encode() + b"\0")
        digest.update(np.ascontiguousarray(shapes, dtype=np.int64).tobytes())
    return digest.hexdigest()


def save(path, labels, info, windows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"format": LABELS_FORMAT, "fingerprint": fingerprint(windows), **info}
    with open(path, "wb") as file:  # Opening the path prevents np.savez from appending .npz.
        np.savez(file, labels=labels, meta=json.dumps(meta))


def load(path, windows):
    """(labels, k) of a clusters file, checked against the corpus rows of `windows`."""
    with np.load(path) as data:
        labels, meta = data["labels"], json.loads(str(data["meta"]))
    if meta.get("format") != LABELS_FORMAT:
        raise ValueError(f"{path} is not a Simons clusters file")
    if meta["fingerprint"] != fingerprint(windows) or len(labels) != meta["rows"]:
        raise ValueError(f"{path} was built on different corpus rows; rebuild it for this corpus")
    return labels, int(meta["k"])


class StageTable:
    """The cluster stream of one curriculum phase (picklable, for loader workers)."""

    def __init__(self, labels: np.ndarray, windows, source_weights, k: int, cap: float,
                 mix: float):
        rows = Rows(windows)
        sizes, source = rows.sizes, rows.source
        w = np.asarray(source_weights if source_weights is not None
                       else np.full(len(sizes), 1.0 / len(sizes)), dtype=np.float64)
        r = w[source] / sizes[source]
        mass = np.bincount(labels, weights=r, minlength=k)
        mass = mass / mass.sum()
        p = waterfill(1.0, cap * mass)
        p = np.where(mass > 0, p, 0.0)
        p = p / p.sum()
        order = np.argsort(labels, kind="stable")
        self.order = order.astype(np.int64)
        self.row_cum = np.cumsum(r[order])
        counts = np.bincount(labels, minlength=k)
        self.end = np.cumsum(counts)
        self.start = self.end - counts
        self.p_cum = np.cumsum(p)
        self.p_cum[-1] = 1.0
        self.offsets, self.source = rows.offsets, source.astype(np.int32)
        self.mix = float(mix)

    def draw(self, rng) -> tuple[int, int]:
        """(source index, position) of one cluster-stream row."""
        c = int(np.searchsorted(self.p_cum, rng.random(), side="right"))
        c = min(c, len(self.p_cum) - 1)
        lo, hi = int(self.start[c]), int(self.end[c])
        base = self.row_cum[lo - 1] if lo > 0 else 0.0
        u = base + rng.random() * (self.row_cum[hi - 1] - base)
        j = int(np.searchsorted(self.row_cum, u, side="right"))
        flat = int(self.order[min(max(j, lo), hi - 1)])
        s = int(self.source[flat])
        return s, flat - int(self.offsets[s])
