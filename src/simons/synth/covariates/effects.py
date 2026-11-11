"""Add generated covariate effects to corpus-series windows."""

import numpy as np

MAX_COV = 16
NONE_RELEVANT = 0.35
RELEVANT = 0.5


def smooth(x, k):
    """Trailing mean of x over k steps (the first value held before the start)."""
    if k <= 1:
        return x
    c = np.cumsum(np.pad(x, (k, 0), mode="edge"))
    return (c[k:] - c[:-k]) / k


def log_uniform(rng, lo, hi):
    """Log-uniform draw in [lo, hi]."""
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def base_window(base, rng, L):
    """(window [L] float64, freq) of a random channel of `base.uni`: a random window, or the whole
    channel left-padded with NaN when it is shorter than L."""
    a, freq = base.uni[rng.integers(len(base.uni))]
    if len(a) >= L:
        s = rng.integers(0, len(a) - L + 1)
        y0 = a[s:s + L].astype("float64")
    else:
        y0 = np.concatenate([np.full(L - len(a), np.nan), a]).astype("float64")
    return y0, freq


def covariate_window(base, rng, L):
    """A window [L] float64 of a random channel of `base.uni` as a covariate: left-padded with NaN
    when short, then every NaN filled with the window's mean (0 when it has none)."""
    b, _ = base.uni[rng.integers(len(base.uni))]
    s0 = rng.integers(0, max(1, len(b) - L + 1))
    x = b[s0:s0 + L].astype("float64")
    if len(x) < L:
        x = np.concatenate([np.full(L - len(x), np.nan), x])
    return np.where(np.isfinite(x), x, np.nanmean(x) if np.isfinite(x).any() else 0.0)


def gen_covariate(kind, L, rng):
    t = np.arange(L, dtype="float64")
    if kind == "event":
        if rng.random() < 0.5:
            p = rng.choice([7, 12, 24, 52, 168, 30])
            phase = rng.integers(p)
            width = rng.integers(1, max(2, p // 4))
            x = (((t + phase) % p) < width).astype(float)
        else:
            starts = rng.random(L) < rng.uniform(0.005, 0.08)
            length = rng.integers(1, 15)
            x = np.convolve(starts.astype(float), np.ones(length), mode="full")[:L].clip(0, 1)
        return x
    if kind == "smooth":
        walk = np.cumsum(rng.normal(size=L)) * rng.uniform(0, 0.3)
        p = rng.uniform(8, 400)
        x = walk + np.sin(2 * np.pi * t / p + rng.uniform(0, 6.3)) * rng.uniform(0.5, 2)
        return smooth(x + rng.normal(size=L) * rng.uniform(0, 0.5), int(rng.integers(1, 8)))
    if kind == "step":
        changes = rng.random(L) < rng.uniform(0.002, 0.03)
        levels = rng.normal(size=changes.sum() + 1)
        return levels[np.cumsum(changes)]
    if kind == "calendar":
        p = rng.choice([7, 24, 12, 52, 168, 96, 48])
        if rng.random() < 0.5:
            return np.sin(2 * np.pi * t / p + rng.uniform(0, 6.3))
        return ((t + rng.integers(p)) % p) / p
    raise ValueError(kind)


def effect(x, rng):
    shape = rng.choice(["linear", "relu", "tanh", "square"], p=[0.5, 0.2, 0.2, 0.1])
    z = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)
    if shape == "relu":
        z = np.maximum(z - rng.normal(0, 0.5), 0)
    elif shape == "tanh":
        z = np.tanh(z * rng.uniform(0.5, 2))
    elif shape == "square":
        z = z ** 2
    lag = int(rng.choice([0, 0, 0, 0, 1, 2, 3]))
    if lag:
        z = np.concatenate([np.full(lag, z[0]), z[:-lag]])
    if rng.random() < 0.3:
        k = rng.uniform(0.3, 0.9)
        from scipy.signal import lfilter

        z = lfilter([1.0], [1.0, -k], z) * (1 - k)
    return z / (np.std(z) + 1e-9)


def synthetic(base, rng, hist: int, horizon: int):
    """Return float32 (target [hist+horizon], covariates [C,hist+horizon]).
    Draw a window from `base.uni`, a list of (values,freq) pairs."""
    L = hist + horizon
    y0, _ = base_window(base, rng, L)
    hist_part = y0[:hist]
    mu, sd = np.nanmean(hist_part), np.nanstd(hist_part)
    if not np.isfinite(sd) or sd == 0:
        sd = 1.0
    if not np.isfinite(mu):
        mu = 0.0
    z0 = (y0 - mu) / sd
    if rng.random() < 0.25:
        z0 = z0 * rng.uniform(0, 0.3)
    n = int(rng.integers(1, 7))
    kinds = rng.choice(["event", "smooth", "step", "calendar", "forecast", "real"], size=n,
                       p=[0.25, 0.2, 0.12, 0.1, 0.13, 0.2])
    none_relevant = rng.random() < NONE_RELEVANT
    covs, total = [], np.zeros(L)
    multiplicative = rng.random() < 0.2
    pending_forecasts = 0
    for kind in kinds:
        if kind == "forecast":
            pending_forecasts += 1
            continue
        if kind == "real":
            x = covariate_window(base, rng, L)
        else:
            x = gen_covariate(kind, L, rng)
        if not none_relevant and rng.random() < RELEVANT:
            total += (effect(x, rng) * rng.choice([-1, 1])
                      * np.exp(rng.uniform(np.log(0.05), np.log(2.5))))
        covs.append(x)
    if multiplicative:
        y = z0 + (np.abs(np.nan_to_num(z0)) + 1) * 0.5 * total
    else:
        y = z0 + total
    y = y + rng.normal(size=L) * rng.uniform(0, 0.3)
    for _ in range(pending_forecasts):
        err = (smooth(rng.normal(size=L), int(rng.integers(1, 12))) * rng.uniform(0.05, 1.5)
               * np.nanstd(y))
        f = y + err
        if rng.random() < 0.3:
            f = f * rng.uniform(0.7, 1.3) + rng.normal() * np.nanstd(y)
        covs.append(f)
    if not covs:
        covs.append(gen_covariate("calendar", L, rng))
    rng.shuffle(covs)
    covs = np.stack(covs)
    for i in range(len(covs)):
        if rng.random() < 0.15:
            covs[i, hist:] = np.nan
        if rng.random() < 0.1:
            covs[i, rng.random(L) < rng.uniform(0, 0.1)] = np.nan
    scale, offset = 10 ** rng.uniform(-2, 4), rng.normal() * 10 ** rng.uniform(-1, 4)
    y = y * scale + offset
    y[~np.isfinite(y0)] = np.nan
    if rng.random() < 0.3:
        covs = covs * 10 ** rng.uniform(-2, 3, size=(len(covs), 1))
    return y.astype("float32"), covs.astype("float32")
