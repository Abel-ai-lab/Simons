"""Sample synthetic effects, sibling pairs and real known-future covariate windows.
Corpora use train-*.parquet rows with item_id, source, freq, target [T], covariates [C,T].
Covariates must be known at forecast time; NaN marks unknown values."""

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from simons.synth.covariates.effects import MAX_COV
from simons.training.data import corpus_files, resolve

PATCH = 32


def horizon_for(freq: str, rng) -> int:
    """A frequency-typical horizon (multiple of PATCH): long for sub-hourly, short for low."""
    f = (freq or "").upper()
    if f.endswith(("MIN", "T", "S")):
        patches = [1, 2, 3, 4, 6, 8]
    elif f.endswith("H"):
        patches = [1, 2, 3, 4, 6]
    else:
        patches = [1, 1, 2]
    return int(rng.choice(patches) * PATCH)


BASE_VALUES = 32_000_000


class BaseSeries:
    """Load univariate channels and equal-length multivariate rows for sibling sampling."""

    def __init__(self, paths, min_len: int = 384, values: int = BASE_VALUES):
        paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
        self.uni, self.multi = [], []
        if len(paths) == 1:
            table = pq.read_table(paths[0], columns=["target", "freq"]).to_pydict()
            for chans, freq in zip(table["target"], table["freq"]):
                self._add(chans, freq, min_len)
            return
        share = values // len(paths)
        for path in paths:
            read = 0
            with pq.ParquetFile(path) as file:
                for batch in file.iter_batches(batch_size=16, columns=["target", "freq"]):
                    rows = batch.to_pydict()
                    for chans, freq in zip(rows["target"], rows["freq"]):
                        self._add(chans, freq, min_len)
                        read += sum(len(c) for c in chans)
                        if read >= share:
                            break
                    if read >= share:
                        break

    def _add(self, chans, freq, min_len: int):
        arrs = [np.asarray(c, dtype="float32") for c in chans]
        arrs = [a for a in arrs if np.isfinite(a).sum() >= min_len]
        for a in arrs:
            self.uni.append((a, freq))
        if len(arrs) >= 2 and len({len(a) for a in arrs}) == 1:
            self.multi.append((np.stack(arrs), freq))


def sibling(base: BaseSeries, rng, hist: int, horizon: int):
    L = hist + horizon
    arr, _ = base.multi[rng.integers(len(base.multi))]
    j = int(rng.integers(len(arr)))
    if arr.shape[1] >= L:
        s = rng.integers(0, arr.shape[1] - L + 1)
        w = arr[:, s:s + L]
    else:
        w = np.concatenate([np.full((len(arr), L - arr.shape[1]), np.nan, "float32"), arr], 1)
    others = [i for i in range(len(arr)) if i != j]
    rng.shuffle(others)
    return w[j].copy(), w[others[:MAX_COV]].copy()


def base_files(folder) -> list[str]:
    """Find the first training file in a corpus, or one per dataset subfolder."""
    folder = Path(resolve(folder, "univariate"))
    files = corpus_files(folder)
    if not files:
        raise FileNotFoundError(f"no train-*.parquet files in {folder}")
    flat = [f for f in files if Path(f).parent == folder]
    if flat:
        return flat[:1]
    first = {}
    for f in files:
        first.setdefault(Path(f).parent, f)
    return list(first.values())


def load_items(path) -> dict:
    """{source: set of item ids} from a JSON file {"items": {source: [item ids]}} (or the
    mapping itself)."""
    import json

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data = data.get("items", data)
    return {str(s): set(map(str, v)) for s, v in data.items()}


def source_weights(spec, sources):
    """Sampling probabilities over `sources` (in that order) from {source: weight} (sources not
    named: 1), or None when `spec` is None. Unknown names are an error."""
    if spec is None:
        return None
    if isinstance(spec, (str, Path)):
        import json

        spec = json.loads(Path(spec).read_text(encoding="utf-8"))
    weights = {str(k): float(v) for k, v in dict(spec).items()}
    unknown = sorted(set(weights) - set(sources))
    if unknown:
        raise ValueError(f"covariate_source_weights: sources not in the covariate corpus: "
                         f"{unknown}")
    w = np.array([weights.get(s, 1.0) for s in sources], float)
    if (w < 0).any() or w.sum() <= 0:
        raise ValueError("covariate_source_weights: weights must be non-negative with a positive "
                         "sum")
    return w / w.sum()


def _list_arrays(column):
    import pyarrow as pa

    if not len(column):
        return []
    values = pa.concat_arrays(column.chunks)
    offsets = values.offsets.to_numpy()
    flat = values.values.to_numpy(zero_copy_only=False).astype("float32")
    nulls = values.is_null().to_numpy(zero_copy_only=False)
    return [None if nulls[i] else flat[offsets[i]:offsets[i + 1]] for i in range(len(values))]


RAGGED = "ragged"


def _matrix_arrays(column):
    """Decode Arrow rows to float32 [C,L], None for empty/null, or RAGGED for unequal lengths."""
    import pyarrow as pa

    if not len(column):
        return []
    rows = pa.concat_arrays(column.chunks)
    row_offsets = rows.offsets.to_numpy()
    inner = rows.values
    offsets = inner.offsets.to_numpy()
    flat = inner.values.to_numpy(zero_copy_only=False)
    nulls = rows.is_null().to_numpy(zero_copy_only=False)
    out = []
    for i in range(len(rows)):
        a, b = row_offsets[i], row_offsets[i + 1]
        if nulls[i] or b <= a:
            out.append(None)
            continue
        lengths = offsets[a + 1:b + 1] - offsets[a:b]
        if len(set(lengths.tolist())) != 1:
            out.append(RAGGED)
            continue
        out.append(flat[offsets[a]:offsets[b]].astype("float32").reshape(b - a, int(lengths[0])))
    return out


class CovariateCorpus:
    """Load aligned targets and known-future covariates from train-*.parquet rows."""

    def __init__(self, folder: str, exclude=None, min_length: int = 0, weights=None):
        self.recs = []
        self.excluded = 0
        for path in corpus_files(resolve(folder, "covariates")):
            table = pq.read_table(path, columns=["item_id", "source", "freq", "target",
                                                 "covariates"])
            names = table.column("source").to_pylist()
            items = table.column("item_id").to_pylist()
            freqs = table.column("freq").to_pylist()
            targets = _list_arrays(table.column("target"))
            covariates = _matrix_arrays(table.column("covariates"))
            for i in range(len(names)):
                if exclude and str(items[i]) in exclude.get(names[i], ()):
                    self.excluded += 1
                    continue
                if targets[i] is None or (min_length and len(targets[i]) < min_length):
                    continue
                if covariates[i] is None:
                    continue
                if covariates[i] is RAGGED or covariates[i].shape[1] != len(targets[i]):
                    steps = ("unequal lengths" if covariates[i] is RAGGED
                             else f"{covariates[i].shape[1]} steps")
                    raise ValueError(
                        f"{path}: series {items[i]!r} of {names[i]!r} has covariates of {steps} "
                        f"for a target of {len(targets[i])}: every covariate must be aligned "
                        "with the target step by step")
                self.recs.append({"source": names[i], "freq": freqs[i], "target": targets[i],
                                  "covs": covariates[i]})
        self.by_source = {}
        for i, r in enumerate(self.recs):
            self.by_source.setdefault(r["source"], []).append(i)
        self.sources = sorted(self.by_source)
        self.p = source_weights(weights, self.sources)
        if self.p is not None and (self.p == 0).any():
            keep = self.p > 0
            self.sources = [s for s, k in zip(self.sources, keep) if k]
            self.p = self.p[keep] / self.p[keep].sum()

    def train_sample(self, rng, hist: int, min_observed: float = 0.0):
        """Return (target [hist+H], covariates [C,hist+H], H), or None after 20 failed draws.
        `min_observed` requires that fraction of history steps to be observed."""
        for _ in range(20):
            if self.p is None:
                src = self.sources[rng.integers(len(self.sources))]
            else:
                src = self.sources[rng.choice(len(self.sources), p=self.p)]
            r = self.recs[self.by_source[src][rng.integers(len(self.by_source[src]))]]
            horizon = horizon_for(r["freq"], rng)
            n = len(r["target"])
            if n < horizon + 64:
                continue
            low = horizon + 64
            if min_observed > 0:
                low = max(low, horizon + int(np.ceil(min_observed * hist)))
                if low > n:
                    continue
            end = int(rng.integers(low, n + 1))
            start = end - horizon - hist
            y = r["target"][max(start, 0):end]
            c = r["covs"][:MAX_COV, max(start, 0):end]
            pad = (hist + horizon) - y.shape[-1]
            if pad:
                y = np.concatenate([np.full(pad, np.nan, "float32"), y])
                c = np.concatenate([np.full((len(c), pad), np.nan, "float32"), c], 1)
            if (np.isfinite(y[-horizon:]).sum() < horizon // 4
                    or np.isfinite(y[:-horizon]).sum() < 32):
                continue
            if min_observed > 0 and np.isfinite(y[:-horizon]).sum() < min_observed * hist:
                continue
            return y, c, horizon
        return None
