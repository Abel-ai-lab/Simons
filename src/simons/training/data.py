"""Load training corpora and sample packed time-series windows.
Parquet rows hold item_id, source, freq, target [variate,time] and optional domain/subdomain.
NaN marks missing values; `<base>@<offset>` IDs identify contiguous segments."""

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from simons.model import MAX_VARIATES

TOY = "toy"


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def resolve(name, part: str) -> str:
    """Resolve a local corpus folder or download a Hub dataset's `part` folder to its cache.
    `part` is "univariate" or "covariates"; local paths expand ~."""
    name = os.path.expanduser(str(name))
    if name == TOY or Path(name).exists() or name.count("/") != 1 or name.startswith((".", "/")):
        return name
    from huggingface_hub import snapshot_download

    try:
        root = snapshot_download(name, repo_type="dataset", allow_patterns=[f"{part}/**"])
    except (OSError, ValueError) as error:
        reason = (str(error).splitlines() or [""])[0]
        raise FileNotFoundError(
            f"{name!r} is not a local folder, and no Hugging Face dataset of that name could be "
            f"read ({type(error).__name__}: {reason})") from error
    return str(Path(root) / part)


def corpus_files(folder, split: str = "train") -> list[str]:
    """Find `<split>-*.parquet` files in a folder or its dataset subfolders."""
    folder, pattern = Path(folder), f"{split}-*.parquet"
    deeper = next(folder.glob(f"*/*/**/{pattern}"), None)
    if deeper is not None:
        raise ValueError(f"{deeper} is more than one folder below the corpus folder {folder}: "
                         "put each dataset's files in <corpus>/<dataset>/")
    return sorted(str(p) for p in [*folder.glob(pattern), *folder.glob(f"*/{pattern}")])


def load(name: str, split: str = "train"):
    """The toy corpus, or a corpus of train-*.parquet / validation-*.parquet files: a local
    folder (files in it or in its subfolders) or a Hugging Face dataset repo (`resolve`)."""
    from datasets import load_dataset

    if name == TOY:
        return toy(split)
    folder = resolve(name, "univariate")
    if not Path(folder).is_dir():
        raise FileNotFoundError(f"no folder {folder!r} (data must be {TOY!r}, a local parquet "
                                "folder or a Hugging Face dataset repo id)")
    files = corpus_files(folder, split)
    if not files:
        raise FileNotFoundError(f"no {split}-*.parquet files in {folder}")
    return load_dataset("parquet", data_files=files, split="train")


def column(dataset, name: str) -> list:
    """A dataset column as a Python list, read through Arrow (much faster than dataset[name])."""
    return dataset.with_format("arrow")[:].column(name).to_pylist()


def arrays(dataset) -> list[np.ndarray]:
    """Float32 [variate, time] arrays of the `target` column (memory-mapped views may be
    read-only)."""
    import pyarrow as pa

    out = []
    for chunk in dataset.with_format("arrow")[:].column("target").chunks:
        variates, points = chunk.offsets.to_numpy(), chunk.values.offsets.to_numpy()
        values = chunk.values.values
        if values.type != pa.float32():
            values = values.cast(pa.float32())
        values = values.to_numpy(zero_copy_only=False)
        lengths = np.diff(points)
        counts = np.diff(variates)
        if len(counts) and (np.any(counts == 0) or np.any(
                lengths[variates[0]:variates[-1]] != np.repeat(lengths[variates[:-1]], counts))):
            raise ValueError("Each target must be a nonempty [variate][time] list")
        for row in range(len(chunk)):
            first, last = variates[row], variates[row + 1]
            out.append(values[points[first] : points[last]].reshape(last - first, -1))
    return out


def toy(split: str, seed: int = 7, series: int = 24, length: int = 512):
    from datasets import Dataset as HFDataset

    rng = np.random.default_rng(seed)
    time = np.arange(length)
    rows = {"item_id": [], "source": [], "freq": [], "target": []}
    for index in range(series):
        period = rng.choice([12, 24, 48])
        values = (
            rng.uniform(0.5, 3) * np.sin(2 * np.pi * time / period + rng.random())
            + rng.normal(0, 0.1, length)
            + 0.002 * time
        ).astype("float32")
        values[rng.random(length) < 0.02] = np.nan
        if ("validation" if index % 5 == 0 else "train") == split:
            rows["item_id"].append(f"{TOY}/{index}")
            rows["source"].append(f"{TOY}-{index % 3}")
            rows["freq"].append("H")
            rows["target"].append([values.tolist()])
    if not rows["item_id"]:
        raise ValueError(f"The toy data has no {split} split")
    return HFDataset.from_dict(rows)


class Windows(Dataset):
    """Draw deterministic source, series, variate and window samples by index.
    Return NaN-padded, right-aligned `context` history steps and `horizon` target steps."""

    def __init__(
        self, dataset, context: int, horizon: int, count: int, seed: int,
        downsample=0.0, downsample_max=8, downsample_min_points=512,
        stretch=0.0, stretch_up=0.5, stitch_index=False,
    ):
        self.context, self.horizon, self.count, self.seed = context, horizon, count, seed
        self.downsample, self.downsample_max = downsample, downsample_max
        self.downsample_min_points = downsample_min_points
        self.end_min = 0
        self.stretch, self.stretch_up = stretch, stretch_up
        self.stitch, self.prev = False, None
        self.weights = None
        self.shape = None
        sources, rows = {}, {}
        for row, (source, values) in enumerate(zip(column(dataset, "source"), arrays(dataset),
                                                   strict=True)):
            if values.shape[1] > horizon:
                sources.setdefault(source, []).append(values)
                rows.setdefault(source, []).append(row)
        self.sources = list(sources.values())
        if stitch_index:
            from simons.training.stitching import positions

            ids = column(dataset, "item_id")
            self.prev = [positions([ids[r] for r in rows[name]], [a.shape for a in members])
                         for name, members in sources.items()]
        self.names = list(sources.keys())
        self.shapes = [np.array([a.shape for a in s], dtype=np.int64) for s in self.sources]
        if not self.sources or min(context, horizon, count) <= 0:
            raise ValueError("No series longer than the horizon, or an invalid window/count")

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(index)]))
        srng = (np.random.default_rng(np.random.SeedSequence([self.seed, int(index), 7907]))
                if self.stretch > 0 else None)
        crng = (np.random.default_rng(np.random.SeedSequence([self.seed, int(index), 4441]))
                if self.shape is not None else None)
        for _ in range(100):
            if self.weights is None:
                source_index = rng.integers(len(self.sources))
            else:
                source_index = rng.choice(len(self.sources), p=self.weights)
            source = self.sources[source_index]
            position = rng.integers(len(source))
            if crng is not None and crng.random() < self.shape.mix:
                # Preserve the main stream's draws when substituting a cluster-stream row.
                source_index, position = self.shape.draw(crng)
                source = self.sources[source_index]
            group = source[position]
            prev = self.prev[source_index] if self.stitch else None
            channels = np.sort(
                rng.choice(group.shape[0], min(group.shape[0], MAX_VARIATES), replace=False)
            )
            if self.downsample > 0 and rng.random() < self.downsample:
                window = downsampled_window(
                    group, channels, self.context + self.horizon, rng, self.downsample_max,
                    self.downsample_min_points,
                )
                length = window.shape[1]
            elif self.stretch > 0 and srng.random() < self.stretch:
                from simons.training.warp import stretch_window

                window = stretch_window(
                    group, channels, self.context + self.horizon, srng, self.stretch_up,
                    self.downsample_min_points,
                )
                length = window.shape[1]
            elif self.end_min and group.shape[1] > self.end_min:
                end = self.end_min + rng.integers(group.shape[1] - self.end_min + 1)
                start = max(0, end - (self.context + self.horizon))
                if prev is not None and start == 0 and prev[position] >= 0:
                    from simons.training.stitching import extend

                    window = extend(source, prev, position, channels, end,
                                    self.context + self.horizon)
                else:
                    window = np.array(group[channels, start:end], copy=True)
                length = window.shape[1]
            else:
                length = min(group.shape[1], self.context + self.horizon)
                start = rng.integers(group.shape[1] - length + 1)
                if prev is not None and start == 0 and prev[position] >= 0:
                    from simons.training.stitching import extend

                    window = extend(source, prev, position, channels, length,
                                    self.context + self.horizon)
                    length = window.shape[1]
                else:
                    window = np.array(group[channels, start : start + length], copy=True)
            if length <= self.horizon:
                continue
            cut = length - self.horizon
            if np.isfinite(window[:, :cut]).any() and np.isfinite(window[:, cut:]).any():
                history = np.full((len(channels), self.context), np.nan, dtype="float32")
                history[:, -cut:] = window[:, :cut]
                return torch.from_numpy(history), torch.from_numpy(window[:, cut:])
        raise ValueError("Could not find an observed window after 100 attempts")


def downsampled_window(group, channels, want, rng, k_max, min_points):
    """Return up to `want` steps after NaN-aware mean pooling by k ~ LogU[2,k_max].
    Retain at least `min_points` steps when available; empty blocks remain NaN."""
    cap = min(k_max, group.shape[1] // max(1, min_points))
    k = 1
    if cap >= 2:
        k = int(np.floor(np.exp(rng.uniform(np.log(2), np.log(cap + 1)))))
        k = min(max(k, 2), cap)
    raw = min(group.shape[1] // k * k, want * k)
    start = rng.integers(group.shape[1] - raw + 1)
    block = np.asarray(group[channels, start : start + raw], dtype="float32")
    block = block.reshape(len(channels), raw // k, k)
    finite = np.isfinite(block)
    counts = finite.sum(-1)
    sums = np.where(finite, block, 0.0).sum(-1, dtype="float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        pooled = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    return pooled.astype("float32")


def pack_windows(samples):
    """Pack aligned (history, target) samples into rows of up to 32 variate slots; returns
    history [R,W,C], target [R,W,H] and series ids [R,W] (the sample index, -1 = empty)."""
    if not samples:
        raise ValueError("Cannot pack an empty batch")
    rows, row, used = [], [], 0
    context, horizon = samples[0][0].shape[-1], samples[0][1].shape[-1]
    for index, (history, target) in enumerate(samples):
        variates = history.shape[0]
        if (
            history.shape != (variates, context)
            or target.shape != (variates, horizon)
            or not 0 < variates <= MAX_VARIATES
        ):
            raise ValueError("Samples must be aligned [V,C]/[V,H] with 1 <= V <= 32")
        if used + variates > MAX_VARIATES:
            rows.append(row)
            row, used = [], 0
        row.append((index, history, target))
        used += variates
    rows.append(row)
    width = max(sum(h.shape[0] for _, h, _ in row) for row in rows)
    history = torch.full((len(rows), width, context), float("nan"))
    target = torch.full((len(rows), width, horizon), float("nan"))
    ids = torch.full((len(rows), width), -1, dtype=torch.long)
    for batch, row in enumerate(rows):
        offset = 0
        for index, h, t in row:
            stop = offset + h.shape[0]
            history[batch, offset:stop] = h
            target[batch, offset:stop] = t
            ids[batch, offset:stop] = index
            offset = stop
    return history, target, ids


def first_patches(sequence: torch.Tensor, patch: int) -> torch.Tensor:
    """First patch with an observed value per slot of [..., T] (T // patch when none)."""
    seen = torch.isfinite(sequence).unflatten(-1, (-1, patch)).any(-1)
    patches = seen.shape[-1]
    position = torch.arange(patches, device=sequence.device)
    return torch.where(seen, position, patches).amin(-1)


def collate(patch: int):
    """`pack_windows` returning (sequence [R,W,T], ids [R,W], first observed patch [R,W])."""

    def collate_fn(samples):
        history, target, ids = pack_windows(samples)
        sequence = torch.cat([history, target], dim=-1)
        return sequence, ids, first_patches(sequence, patch)

    return collate_fn


class StepIndices:
    """Window indices of one rank's loader: for step s in [start, end), `size` indices from
    s * stride + offset (stride = global batch; one device: offset 0, size = stride)."""

    def __init__(self, start: int, end: int, stride: int, offset: int, size: int):
        self.start, self.end, self.stride, self.offset, self.size = start, end, stride, offset, size

    def __len__(self):
        return max(0, self.end - self.start) * self.size

    def __getitem__(self, i):
        if i < 0:
            i += len(self)
        if not 0 <= i < len(self):
            raise IndexError(i)
        step, j = divmod(i, self.size)
        return (self.start + step) * self.stride + self.offset + j
