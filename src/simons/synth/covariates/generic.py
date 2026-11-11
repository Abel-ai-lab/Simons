"""Generate covariates and their effects on a base window."""

import numpy as np
from scipy.signal import lfilter

from simons.synth.covariates.effects import log_uniform

KINDS = ("forecast", "weather", "promo", "closure", "price", "schedule", "setpoint", "decoy")
KIND_P = (0.15, 0.17, 0.18, 0.10, 0.08, 0.10, 0.10, 0.12)
NONE_RELEVANT = 0.2
INTERACT = 0.2
MAX_COV = 8
PERIODS = (7, 12, 24, 48, 96, 144, 168, 288, 52, 30)


def _std(x):
    s = float(np.std(x))
    return s if s > 1e-9 else 1.0


def _ou(rng, L, tau):
    phi = np.exp(-1.0 / max(tau, 1.0))
    x = lfilter([1.0], [1.0, -phi], rng.normal(size=L + 200))[200:]
    return (x - x.mean()) / _std(x)


def _smooth_noise(rng, L, lo, hi):
    return _ou(rng, L, log_uniform(rng, lo, hi))


def _period(base, rng, hist):
    """Use a valid history FFT peak with probability 0.7; otherwise draw from PERIODS."""
    h = np.nan_to_num(base[:hist])
    if hist >= 64 and rng.random() < 0.7:
        spec = np.abs(np.fft.rfft(h - h.mean())) ** 2
        spec[:2] = 0
        k = int(np.argmax(spec))
        if k > 0:
            p = round(hist / k)
            if 3 <= p <= hist // 3:
                return p
    return int(rng.choice(PERIODS))


def _runs(rng, L, duty, mean_len):
    """Draw binary runs with on-share `duty` and geometric run lengths of mean `mean_len`."""
    mean_len = max(1.0, mean_len)
    n = max(1, rng.poisson(duty * L / mean_len))
    x = np.zeros(L)
    starts = rng.integers(0, L, size=n)
    lengths = rng.geometric(1.0 / mean_len, size=n)
    for s, m in zip(starts, lengths):
        x[s:s + m] = 1.0
    return x


def _edges(x):
    d = np.diff(np.concatenate([[0.0], x]))
    return (d > 0).astype(float), (d < 0).astype(float)


def _decay(impulse, k):
    return lfilter([1.0], [1.0, -k], impulse)


def _phase_gate(rng, L, P):
    t = np.arange(L)
    q = rng.uniform(0.5, 3.0)
    return (0.5 * (1 + np.cos(2 * np.pi * (t - rng.uniform(0, P)) / P))) ** q



def _forecast_driver(rng, L, P):
    d = _smooth_noise(rng, L, 3, 150)
    if rng.random() < 0.5:
        d = d + rng.uniform(0.3, 1.5) * np.sin(2 * np.pi * np.arange(L) / P + rng.uniform(0, 6.3))
        d = (d - d.mean()) / _std(d)
    e = rng.choice([-1, 1]) * rng.uniform(0.15, 0.8) * d
    return d, e


def _issue(rng, truth, hist):
    """Issue a forecast with small history error and horizon error growing with lead."""
    L = len(truth)
    sd = _std(truth)
    lead = np.clip((np.arange(L) - hist + 1) / max(1, L - hist), 0, None)
    err = rng.uniform(0.02, 0.4) * _smooth_noise(rng, L, 1, 20)
    err = err + rng.uniform(0.0, 1.0) * lead * _smooth_noise(rng, L, 5, 100)
    x = truth + err * sd
    if rng.random() < 0.3:
        x = x * rng.uniform(0.8, 1.2) + rng.normal() * 0.3 * sd
    return x


def _weather(rng, L, P):
    t = np.arange(L)
    d = _ou(rng, L, log_uniform(rng, 20, 500))
    if rng.random() < 0.6:
        d = d + rng.uniform(0, 1.2) * np.sin(2 * np.pi * t / P + rng.uniform(0, 6.3))
    d = (d - d.mean()) / _std(d)
    theta = float(np.quantile(d, rng.uniform(0.2, 0.8)))
    shape = rng.choice(["hinge_up", "hinge_down", "v", "tanh", "linear"],
                       p=[0.25, 0.2, 0.25, 0.15, 0.15])
    if shape == "hinge_up":
        r = np.maximum(d - theta, 0)
    elif shape == "hinge_down":
        r = np.maximum(theta - d, 0)
    elif shape == "v":
        r = rng.uniform(0.3, 1.0) * np.maximum(d - theta, 0) + rng.uniform(0.3, 1.0) * np.maximum(
            theta - d, 0)
    elif shape == "tanh":
        r = np.tanh(rng.uniform(0.5, 3) * (d - theta))
    else:
        r = d
    if rng.random() < 0.4:
        k = rng.uniform(0.5, 0.98)
        r = _decay(r, k) * (1 - k)
    if rng.random() < 0.3:
        r = r * _phase_gate(rng, L, P)
    e = rng.choice([-1, 1]) * rng.uniform(0.1, 0.8) * (r - r.mean()) / _std(r)
    x = d * rng.uniform(2, 15) + rng.uniform(-10, 30)
    return x, e


def _promo(rng, L):
    x = _runs(rng, L, rng.uniform(0.01, 0.15), log_uniform(rng, 1, 30))
    lift = np.log1p(log_uniform(rng, 0.1, 1.5))
    if rng.random() < 0.3:
        k = rng.uniform(0.8, 0.99)
        on = x * _decay(_edges(x)[0], k)
        e = lift * np.where(x > 0, np.clip(on, 0.3, 1.0), 0.0)
    else:
        e = lift * x
    starts, ends = _edges(x)
    if rng.random() < 0.5:
        k = rng.uniform(0.5, 0.95)
        e = e + lift * rng.uniform(0.2, 0.7) * _decay(ends, k) * (1 - x)
    if rng.random() < 0.4:
        k = rng.uniform(0.6, 0.97)
        e = e - lift * rng.uniform(0.1, 0.5) * _decay(ends, k) * (1 - x)
    if rng.random() < 0.3:
        r = int(rng.integers(2, 32))
        ramp = np.zeros(len(x))
        for s in np.flatnonzero(starts):
            lo = max(0, s - r)
            ramp[lo:s] = np.maximum(ramp[lo:s], np.linspace(0, 1, s - lo + 1)[1:])
        e = e + lift * rng.uniform(0.1, 0.5) * ramp * (1 - x)
    if rng.random() < 0.2:
        depth = rng.uniform(0.2, 1.0, size=L)
        depth = np.round(depth * 4) / 4
        run_id = np.cumsum(starts)
        level = depth[np.minimum(run_id.astype(int), L - 1)]
        x = x * level
        e = e * np.where(x > 0, level, 1.0)
    return x, e


def _closure(rng, L, P):
    return _runs(rng, L, rng.uniform(0.005, 0.1), log_uniform(rng, 1, max(2, P)))


def _price(rng, L):
    changes = rng.random(L) < rng.uniform(0.002, 0.02)
    walk = np.cumsum(np.where(changes, rng.normal(0, 0.1, size=L), 0))
    disc = _runs(rng, L, rng.uniform(0.03, 0.2), rng.uniform(3, 20))
    starts, _ = _edges(disc)
    depth = rng.uniform(0.1, 0.4, size=L)[np.minimum(np.cumsum(starts).astype(int), L - 1)]
    logp = walk - disc * depth
    eps = rng.uniform(0.5, 3.0)
    e = -eps * (logp - logp.mean())
    e = np.clip(e, -2.5, 2.5)
    x = np.exp(logp) * log_uniform(rng, 1, 100)
    return x, e


def _schedule(rng, L, P):
    P = max(2, int(P))
    segs = int(rng.integers(2, 7))
    cuts = np.sort(rng.choice(np.arange(1, P), size=min(segs - 1, P - 1), replace=False))
    vals = rng.random(len(cuts) + 1)
    vals[rng.random(len(vals)) < 0.3] = 0.0
    phase = (np.arange(L) + int(rng.integers(P))) % P
    pattern = vals[np.searchsorted(cuts, phase, side="right")]
    K = int(rng.integers(1, 13))
    level = np.full(L, float(K))
    if rng.random() < 0.3:
        at = int(rng.integers(L // 4, L))
        level[at:] = max(1, K + int(rng.integers(-K // 2 - 1, K // 2 + 2)))
    c = np.round(level * pattern)
    if rng.random() < 0.3:
        c = c * (1 - _runs(rng, L, rng.uniform(0.01, 0.05), rng.uniform(1, 10)))
    gamma = rng.uniform(0.5, 1.2)
    delta = 0.1 * max(1.0, float(c.mean()))
    e = gamma * (np.log(c + delta) - np.log(c.mean() + delta))
    return c, e


def _setpoint(rng, L):
    changes = rng.random(L) < log_uniform(rng, 0.002, 0.03)
    levels = rng.normal(size=int(changes.sum()) + 1)
    x = levels[np.cumsum(changes)]
    if rng.random() < 0.3:
        w = int(rng.integers(2, 64))
        x = np.convolve(np.pad(x, (w - 1, 0), mode="edge"), np.ones(w) / w, mode="valid")
    k1 = rng.uniform(0.5, 0.995)
    r = _decay(x - x[0], k1) * (1 - k1) + x[0]
    if rng.random() < 0.4:
        k2 = rng.uniform(0.5, 0.98)
        r = _decay(r - r[0], k2) * (1 - k2) + r[0]
    e = rng.choice([-1, 1]) * rng.uniform(0.3, 1.0) * (r - r.mean()) / _std(r)
    return x * log_uniform(rng, 0.1, 100) + rng.normal() * 20, e


def _decoy(rng, L, P):
    kind = rng.choice(["smooth", "runs", "price", "schedule", "periodic", "weather"])
    if kind == "smooth":
        return _smooth_noise(rng, L, 3, 300) * log_uniform(rng, 0.1, 100) + rng.normal() * 10
    if kind == "runs":
        return _runs(rng, L, rng.uniform(0.01, 0.2), log_uniform(rng, 1, 30))
    if kind == "price":
        return _price(rng, L)[0]
    if kind == "schedule":
        return _schedule(rng, L, P)[0]
    if kind == "weather":
        return _weather(rng, L, P)[0]
    q = int(rng.choice(PERIODS))
    return np.sin(2 * np.pi * np.arange(L) / q + rng.uniform(0, 6.3))


def make_task(base, rng, hist: int, horizon: int, k: int):
    """(target [L] float32, covariates [k, L] float32) from one base window [L]."""
    L = hist + horizon
    base = np.asarray(base[:L], dtype="float64")
    pad = ~np.isfinite(base)
    b = np.where(pad, 0.0, base)
    obs = b[:hist][~pad[:hist]] if (~pad[:hist]).any() else b
    b = (b - obs.mean()) / _std(obs)
    b = np.clip(b, -8, 8)
    P = _period(np.where(pad, np.nan, b), rng, hist)
    none_relevant = rng.random() < NONE_RELEVANT
    kinds = rng.choice(KINDS, size=k, p=KIND_P)
    covs, effects, switches, target_fc = [], [], [], []
    for kind in kinds:
        relevant = not none_relevant and kind != "decoy"
        if not relevant:
            covs.append(_decoy(rng, L, P))
            effects.append(None)
            continue
        if kind == "forecast":
            if rng.random() < 0.4:
                target_fc.append(len(covs))
                covs.append(None)
                effects.append(None)
                continue
            d, e = _forecast_driver(rng, L, P)
            covs.append(_issue(rng, d, hist) * log_uniform(rng, 0.1, 100))
            effects.append(e)
        elif kind == "weather":
            x, e = _weather(rng, L, P)
            covs.append(x)
            effects.append(e)
        elif kind == "promo":
            x, e = _promo(rng, L)
            covs.append(x)
            effects.append(e)
        elif kind == "closure":
            x = _closure(rng, L, P)
            covs.append(x)
            effects.append(None)
            switches.append(x)
        elif kind == "setpoint":
            x, e = _setpoint(rng, L)
            covs.append(x)
            effects.append(e)
        elif kind == "price":
            x, e = _price(rng, L)
            covs.append(x)
            effects.append(e)
        else:
            x, e = _schedule(rng, L, P)
            covs.append(x)
            effects.append(e)
    live = [i for i, e in enumerate(effects) if e is not None]
    for i in live:
        others = [j for j in range(len(covs)) if j != i and covs[j] is not None]
        if others and rng.random() < INTERACT:
            z = covs[int(rng.choice(others))]
            z = np.clip((z - z.mean()) / _std(z), -2, 2)
            effects[i] = effects[i] * np.clip(1 + rng.uniform(0.2, 0.6) * z, 0, None)
    total = np.zeros(L)
    for e in effects:
        if e is not None:
            total = total + e
    log_mode = rng.random() < 0.6
    if log_mode:
        if rng.random() < 0.5:
            level = np.exp(rng.uniform(0.1, 0.6) * b)
        else:
            level = b - b.min() + rng.uniform(0.02, 0.5) * (b.max() - b.min() + 1e-6)
        y = level * np.exp(np.clip(total, -6, 6))
    else:
        if rng.random() < 0.25:
            b = b * rng.uniform(0, 0.3)
        y = b + rng.uniform(0.5, 3.0) * total
    for x in switches:
        if rng.random() < 0.5:
            floor = rng.uniform(0, 0.1) * float(np.mean(np.abs(y)))
            alt = np.full(L, floor) if log_mode else np.full(L, float(y.min()))
        else:
            w = max(2, int(P))
            kern = np.ones(w) / w
            smooth = np.convolve(np.pad(y, (w - 1, 0), mode="edge"), kern, mode="valid")
            lam = rng.uniform(0.3, 1.3)
            alt = smooth * lam if log_mode else smooth + (lam - 1) * _std(y)
        y = np.where(x > 0, alt, y)
    sd = _std(y)
    if log_mode:
        y = y * np.exp(rng.normal(size=L) * rng.uniform(0.01, 0.2))
        if rng.random() < 0.2:
            scale = log_uniform(rng, 0.5, 50) / max(1e-9, float(np.mean(y)))
            y = rng.poisson(np.clip(y * scale, 0, 1e6)).astype("float64")
    else:
        y = y + rng.normal(size=L) * rng.uniform(0.01, 0.3) * sd
    for i in target_fc:
        w = int(rng.integers(1, 9))
        smooth = np.convolve(np.pad(y, (w - 1, 0), mode="edge"), np.ones(w) / w, mode="valid")
        covs[i] = _issue(rng, smooth, hist)
    units = log_uniform(rng, 1e-2, 1e4)
    y = y * units
    if not log_mode:
        y = y + rng.normal() * units * log_uniform(rng, 0.1, 30)
    for i in target_fc:
        covs[i] = covs[i] * units
    c = np.stack(covs)
    order = rng.permutation(len(c))
    c = c[order]
    for i in range(len(c)):
        if rng.random() < 0.05:
            c[i, rng.random(L) < rng.uniform(0, 0.1)] = np.nan
    y[pad] = np.nan
    return y.astype("float32"), c.astype("float32")

