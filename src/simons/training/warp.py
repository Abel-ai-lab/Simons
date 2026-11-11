"""Resample and power-warp real training windows using private random streams."""

import math

import numpy as np
import torch

UP_RANGE = (1.15, 3.0)
DOWN_RANGE = (1.15, 6.0)


def _log_uniform(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def stretch_window(group, channels, want, rng, up_share=0.5, min_points=512):
    """A window [V, n] float32 of `group[channels]` at a new sampling rate, up or down."""
    length = group.shape[1]
    down_hi = min(DOWN_RANGE[1], length / max(1, min_points))
    if rng.random() < up_share or down_hi < DOWN_RANGE[0] * 1.01:
        factor, c = _log_uniform(rng, *UP_RANGE), float(rng.random())
        return upsample(group, channels, want, factor, c, rng)
    return aggregate(group, channels, want, _log_uniform(rng, DOWN_RANGE[0], down_hi), rng)


def upsample(group, channels, want, r, c, rng):
    length = group.shape[1]
    raw = int(min(length, math.ceil((want - 1) / r) + 1))
    start = int(rng.integers(length - raw + 1))
    x = np.asarray(group[channels, start : start + raw], dtype="float64")
    out = int(min(want, math.floor((raw - 1) * r + 1e-9) + 1))
    t = (raw - 1) - (out - 1 - np.arange(out)) / r
    t = np.clip(t, 0.0, raw - 1)
    i0 = np.minimum(np.floor(t + 1e-9).astype(np.int64), raw - 1)
    u = np.clip(t - i0, 0.0, 1.0)
    on_knot = u < 1e-9
    i1 = np.minimum(i0 + 1, raw - 1)
    a, b = x[:, i0], x[:, i1]
    with np.errstate(invalid="ignore"):
        y = np.where(on_knot[None], a, (1 - u)[None] * a + u[None] * b)
    if c > 0 and out > 1:
        bridge = _bridge(i0, u, on_knot, len(channels), rng)  # [V, out]
        d = np.diff(x, axis=1)
        sigma = np.zeros(len(channels))
        for v in range(len(channels)):
            dv = d[v][np.isfinite(d[v])]
            if dv.size >= 4:
                sigma[v] = 1.4826 * np.median(np.abs(dv - np.median(dv)))
        with np.errstate(invalid="ignore"):
            fill = c * sigma[:, None] * bridge
            fill = np.where((a == b) | on_knot[None], 0.0, fill)
            y = y + fill
            nonneg = ~(np.nanmin(np.where(np.isfinite(x), x, np.inf), axis=1) < 0)
            y = np.where(nonneg[:, None] & np.isfinite(y), np.maximum(y, 0.0), y)
    return y.astype("float32")


def _bridge(i0, u, on_knot, variates, rng):
    """Sample Brownian bridges between native knots at output positions u."""
    out = len(u)
    prev_u = np.concatenate([[0.0], u[:-1]])
    new_seg = np.concatenate([[True], i0[1:] != i0[:-1]])
    du = np.where(new_seg, u, u - prev_u).clip(min=0.0)
    z = rng.standard_normal((variates, out))
    inc = np.sqrt(du)[None] * z
    cs = np.cumsum(inc, axis=1)
    seg_start = np.maximum.accumulate(np.where(new_seg, np.arange(out), 0))
    base = np.where(seg_start > 0, cs[:, np.maximum(seg_start - 1, 0)], 0.0)
    w = cs - base
    # Completing W(1) needs the unsampled increment with variance 1 - u_last.
    last = np.concatenate([i0[1:] != i0[:-1], [True]])
    end_idx = np.flatnonzero(last)
    seg_id = np.cumsum(new_seg) - 1
    w1_last = w[:, end_idx] + np.sqrt(np.clip(1 - u[end_idx], 0, None))[None] * \
        rng.standard_normal((variates, len(end_idx)))
    w1 = w1_last[:, seg_id]
    bridge = w - u[None] * w1
    return np.where(on_knot[None], 0.0, bridge)


def aggregate(group, channels, want, k, rng):
    length = group.shape[1]
    n = int(min(want, math.floor(length / k)))
    span = n * k
    s = float(rng.uniform(0.0, max(0.0, length - span)))
    lo, hi = math.floor(s), int(min(length, math.ceil(s + span)))
    x = np.asarray(group[channels, lo:hi], dtype="float64")
    finite = np.isfinite(x)
    vals = np.where(finite, x, 0.0)
    cv = np.concatenate([np.zeros((len(channels), 1)), np.cumsum(vals, 1)], 1)
    cn = np.concatenate([np.zeros((len(channels), 1)), np.cumsum(finite, 1)], 1)
    edges = np.clip(s + k * np.arange(n + 1) - lo, 0.0, hi - lo)
    e0 = np.minimum(np.floor(edges).astype(np.int64), hi - lo - 1)
    frac = edges - e0

    def at(cum, per):
        return cum[:, e0] + frac[None] * per[:, e0]

    sv, sn = at(cv, vals), at(cn, finite.astype("float64"))
    num, cnt = np.diff(sv, axis=1), np.diff(sn, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        y = np.where(cnt > 0.25, num / np.maximum(cnt, 1e-9), np.nan)
    return y.astype("float32")


def expected_steps(steps, length, up_share=0.5, min_points=512):
    """Expected window steps of a stretched draw for series of `steps` (array), up to `length`."""
    steps = np.asarray(steps, dtype=float)
    lo, hi = np.log(UP_RANGE[0]), np.log(UP_RANGE[1])
    rs = np.exp(np.linspace(lo, hi, 9))
    up = np.minimum(steps[:, None] * rs[None], length).mean(1)
    down_hi = np.minimum(DOWN_RANGE[1], steps / max(1, min_points))
    can_down = down_hi >= DOWN_RANGE[0] * 1.01
    grid = np.linspace(0, 1, 9)
    ks = np.exp(np.log(DOWN_RANGE[0]) + grid[None] * (np.log(np.maximum(down_hi, DOWN_RANGE[0]))
                                                      - np.log(DOWN_RANGE[0]))[:, None])
    down = np.minimum(np.floor(steps[:, None] / ks), length).mean(1)
    p_up = np.where(can_down, up_share, 1.0)
    return p_up * up + (1 - p_up) * np.where(can_down, down, up)


def power_warp(sequence, series_ids, real_rows, gen, rate, lo=1.25, hi=2.5, convex=0.65):
    """Scale-preserving power warp of non-negative real variates (dense, no host syncs)."""
    if rate <= 0 or real_rows <= 0:
        return sequence
    _, width, length = sequence.shape
    device = sequence.device
    x = sequence[:real_rows].reshape(real_rows * width, length).float()
    ids = series_ids[:real_rows].reshape(-1)
    buckets = int(series_ids.shape[0] * width) + 1
    index = torch.where(ids >= 0, ids, buckets - 1).clamp(max=buckets - 1)
    draws = torch.rand((buckets, 3), generator=gen, device=device)[index]
    finite = torch.isfinite(x)
    count = finite.sum(-1)
    nonneg = ~(finite & (x < 0)).any(-1)
    s = (torch.where(finite, x.abs(), 0.0).sum(-1) / count.clamp_min(1)).clamp_min(1e-6)
    logp = math.log(lo) + draws[:, 1] * (math.log(hi) - math.log(lo))
    p = torch.exp(torch.where(draws[:, 2] < convex, logp, -logp))
    warped = s[:, None] * torch.pow((x / s[:, None]).clamp_min(0.0), p[:, None])
    use = (draws[:, 0] < rate) & (ids >= 0) & (count > 0) & nonneg
    x = torch.where(use[:, None] & finite, warped, x)
    sequence = sequence.clone()
    sequence[:real_rows] = x.view(real_rows, width, length).to(sequence.dtype)
    return sequence
