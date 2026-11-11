"""Token-budgeted real-data mixture with hierarchical UniMax repetition caps."""

import numpy as np

from simons.training.data import MAX_VARIATES, column


def waterfill(budget: float, caps) -> np.ndarray:
    """Divide a budget equally under per-item caps, redistributing surplus.
    If the budget exceeds all caps, allocate in proportion to caps."""
    caps = np.asarray(caps, dtype=float)
    total = caps.sum()
    if total <= 0:
        return np.zeros_like(caps)
    if budget >= total:
        return caps * (budget / total)
    out = np.empty_like(caps)
    remaining = budget
    order = np.argsort(caps, kind="stable")
    for i, index in enumerate(order):
        take = min(remaining / (len(caps) - i), caps[index])
        out[index] = take
        remaining -= take
    return out


def draw_tokens(shapes: np.ndarray, length: int, end_min: int = 0, downsample: float = 0.0,
                downsample_max: int = 8, min_points: int = 512, trained: int = 0,
                stretch: float = 0.0, stretch_up: float = 0.5) -> float:
    """Return expected observed values per draw for series shapes [n,2] (variates,steps).
    `trained` > 0 counts only the last `trained` window steps."""
    keep = trained or length
    variates = np.minimum(shapes[:, 0], MAX_VARIATES).astype(float)
    steps = shapes[:, 1].astype(float)
    natural = np.minimum(steps, length)
    if end_min:
        a, b = float(end_min), steps
        long = steps > end_min
        mean_end = np.where(
            length >= b, (a + b) / 2,
            np.where(length <= a, length,
                     ((length ** 2 - a ** 2) / 2 + length * (b - length)) / np.maximum(b - a, 1)),
        )
        natural = np.where(long, mean_end, natural)
    natural = np.minimum(natural, keep)
    if stretch > 0:
        from simons.training.warp import expected_steps

        stretched = np.minimum(expected_steps(steps, length, stretch_up, min_points), keep)
        natural = (1 - stretch) * natural + stretch * stretched
    if downsample > 0:
        cap = np.minimum(downsample_max, steps // max(1, min_points))
        pooled = np.minimum(steps, length)
        for c in np.unique(cap[cap >= 2]).astype(int):
            ks = np.arange(2, c + 1)
            p = (np.log(ks + 1) - np.log(ks)) / (np.log(c + 1) - np.log(2))
            rows = cap == c
            per_k = np.minimum(steps[rows, None] // ks[None, :], length)
            pooled[rows] = np.minimum(per_k, keep) @ p
        pooled = np.minimum(pooled, keep)
        natural = (1 - downsample) * natural + downsample * pooled
    return float((variates * natural).mean())


_PER_DRAW = {}


class Mixture:
    """Per-source token budgets for a `Windows` sampler's sources (its `names`, `shapes`)."""

    def __init__(self, names, shapes, meta, cap_epochs=4.0, budget_tokens=6.4e9):
        self.names, self.shapes = list(names), shapes
        self.capacity = np.array([float((np.minimum(s[:, 0], MAX_VARIATES) * s[:, 1]).sum())
                                  for s in shapes])
        self.meta = meta
        self.groups = [self.group_of(name) for name in self.names]
        keys = sorted(set(self.groups))
        index = {k: i for i, k in enumerate(keys)}
        member = np.array([index[g] for g in self.groups])
        caps = cap_epochs * self.capacity
        group_caps = np.bincount(member, weights=caps, minlength=len(keys))
        group_budget = waterfill(budget_tokens, group_caps)
        self.budget = np.zeros(len(self.names))
        for g in range(len(keys)):
            rows = np.flatnonzero(member == g)
            self.budget[rows] = waterfill(group_budget[g], caps[rows])
        self.share = self.budget / self.budget.sum()
        self.epochs = self.budget / np.maximum(self.capacity, 1)

    def group_of(self, name):
        m = self.meta.get(name, {})
        return f"{m.get('domain', '?')}/{m.get('subdomain', '?')}"

    def weights(self, length, end_min=0, downsample=0.0, downsample_max=8, min_points=512,
                trained=0, stretch=0.0, stretch_up=0.5):
        """Return per-source draw probabilities and expected tokens per draw for a phase."""
        spec = (length, end_min, downsample, downsample_max, min_points, trained, stretch,
                stretch_up)
        cached = _PER_DRAW.get(id(self.shapes))
        if cached is None or cached[0] is not self.shapes:
            cached = (self.shapes, {})
            _PER_DRAW[id(self.shapes)] = cached
        per_draw = cached[1].get(spec)
        if per_draw is None:
            per_draw = np.array([draw_tokens(s, length, end_min, downsample, downsample_max,
                                             min_points, trained, stretch, stretch_up)
                                 for s in self.shapes])
            cached[1][spec] = per_draw
        per_draw = per_draw.copy()
        w = self.share / np.maximum(per_draw, 1.0)
        return w / w.sum(), per_draw


def plan_budget(make, stages, batch_size, synthetic_shares, rounds=4, start=6.4e9):
    """Return (expected real-token budget, real tokens per step of each phase).
    `make(B)` builds a Mixture; `stages` holds (steps, draw_tokens arguments) pairs."""
    budget, per_step = float(start), []
    for _ in range(rounds):
        mixture = make(budget)
        total, per_step = 0.0, []
        for (steps, spec), share in zip(stages, synthetic_shares):
            p, t = mixture.weights(*spec)
            tokens = batch_size * (1 - share) * float(p @ t)
            per_step.append(tokens)
            total += steps * tokens
        budget = total
    return budget, per_step


def source_meta(dataset) -> dict:
    """{source: {domain, subdomain}} from the dataset columns (those it has), from each source's
    first row."""
    columns = set(dataset.column_names)
    names = column(dataset, "source")
    cols = {c: column(dataset, c) for c in ("domain", "subdomain") if c in columns}
    meta = {}
    for i, name in enumerate(names):
        if name not in meta:
            meta[name] = {c: cols[c][i] for c in cols}
    return meta
