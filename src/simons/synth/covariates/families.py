"""Mix generic, classical, energy, retail and lagged covariate-effect families."""

import math
import warnings

import numpy as np

from simons.synth.covariates.effects import smooth

FAMILIES = ("generic", "classical", "energy", "retail", "lagged")
RETAIL_HORIZONS = (1, 1, 2, 2, 3, 4)
ENERGY_CFG = {"kinds": {"load": 0.35, "pv": 0.2, "wind": 0.1, "price": 0.15, "tariff": 0.1}}
PERIOD_FREQ = {24: "H", 168: "H", 48: "30T", 96: "15T", 144: "10T", 288: "5T", 7: "D",
               30: "D", 365: "D", 52: "W", 12: "M"}
FREQ_FALLBACK = (("H", 0.35), ("D", 0.25), ("15T", 0.12), ("30T", 0.08), ("W", 0.12),
                 ("M", 0.08))


def parse_mix(mix):
    """(names, probabilities) of an effect_mix dict {family: weight}."""
    unknown = set(mix) - set(FAMILIES)
    if unknown:
        raise ValueError(f"effect_mix: unknown families {sorted(unknown)}")
    names = tuple(k for k in FAMILIES if float(mix.get(k, 0)) > 0)
    if not names:
        raise ValueError("effect_mix: no family has a positive weight")
    p = np.array([float(mix[k]) for k in names])
    return names, p / p.sum()


def horizon(name, rng, horizons):
    """The family's horizon in patches."""
    if name == "retail":
        return int(rng.choice(RETAIL_HORIZONS))
    return int(rng.choice(horizons))


def freq_label(b, rng, hist):
    """Infer a frequency from the base period within 10%, or draw from FREQ_FALLBACK."""
    from simons.synth.covariates.generic import _period

    p = _period(b, rng, hist)
    best = min(PERIOD_FREQ, key=lambda q: abs(q - p))
    if abs(best - p) <= 0.1 * best:
        return PERIOD_FREQ[best]
    names, w = zip(*FREQ_FALLBACK)
    return names[int(rng.choice(len(names), p=np.array(w) / sum(w)))]


class SynthUni:
    """Expose synthetic windows as (values,freq) pairs in `.uni`."""

    def __init__(self, windows):
        self.uni = windows


def _classical(b, rng, hist, H, more):
    from simons.synth.covariates import classical

    L = hist + H
    freq = freq_label(b, rng, hist)
    pool = SynthUni([(b.astype("float32"), freq), (more(L).astype("float32"), freq)])
    return classical.sample(pool, rng, hist, H)


def _energy(b, rng, hist, H):
    from simons.synth.covariates import energy

    y, c = energy.sample(None, rng, hist, H, ENERGY_CFG)
    nan0 = ~np.isfinite(b)
    y = y.copy()
    y[nan0] = np.nan
    return y, c


def _runs(L, duty, mean_run, rng):
    on = np.zeros(L, bool)
    rid = np.full(L, -1)
    mean_gap = max(1.0, mean_run * (1 - duty) / max(duty, 1e-3))
    t, r = int(rng.integers(0, int(mean_gap) + 1)), 0
    while t < L:
        n = int(rng.geometric(1.0 / mean_run))
        on[t:t + n] = True
        rid[t:t + n] = r
        r += 1
        t += n + int(rng.geometric(1.0 / mean_gap))
    return on, rid, r


def _retail_decoy(kind, n, rng):
    t = np.arange(n)
    if kind == "flag":
        return (rng.random(n) < rng.uniform(0.02, 0.4)).astype(float)
    if kind == "fraction":
        x = rng.uniform(0.02, 0.15) + np.cumsum(rng.normal(0, 0.003, n))
        hold = int(rng.integers(1, 8))
        return np.clip(x[(t // hold) * hold], 0, 1)
    if kind == "int":
        hi = int(rng.integers(1, 6))
        changes = rng.random(n) < rng.uniform(0.01, 0.2)
        lv = rng.integers(0, hi + 1, changes.sum() + 1)
        return lv[np.cumsum(changes)].astype(float)
    p0 = np.exp(rng.uniform(np.log(0.5), np.log(20)))
    on, _, _ = _runs(n, rng.uniform(0.05, 0.5), rng.uniform(1.3, 3), rng)
    return np.round(p0 * (1 - on * rng.uniform(0.05, 0.4)), 2)


def _retail(b, rng, hist, H):
    """Apply multiplicative SCAN*PRO-style lifts, capped at 4, to a smoothed observed baseline."""
    L = hist + H
    fin = np.isfinite(b)
    first = int(np.argmax(fin)) if fin.any() else L
    n = L - first
    a = np.where(fin, b, 0.0)[first:].astype("float64")
    a = smooth(a, int(rng.integers(4, 13)))
    z = (a - a.mean()) / (a.std() + 1e-9)
    t = np.arange(n)
    daily = rng.random() < 0.5
    loglam = np.clip(rng.uniform(0.05, 0.5) * z, -3, 3)
    if daily:
        dow = rng.normal(0, rng.uniform(0.05, 0.35), 7)
        loglam = loglam + dow[(t + int(rng.integers(7))) % 7]
    else:
        loglam = loglam + rng.uniform(0, 0.4) * np.sin(2 * np.pi * t / 52 + rng.uniform(0, 6.3))
    loglam = loglam + rng.uniform(np.log(0.3), np.log(3000))
    eff = 0.0 if rng.random() < 0.25 else 1.0
    duty, mean_run = rng.uniform(0.05, 0.6), rng.uniform(1.3, 3.0)
    on, rid, nruns = _runs(n, duty, mean_run, rng)
    reg = np.exp(rng.uniform(np.log(0.1), np.log(20)))
    steps = rng.random(n) < 0.004
    regular = reg * np.cumprod(np.where(steps, 1 + rng.choice([-1, 1], n)
                                        * rng.uniform(0.03, 0.15, n), 1))
    cut_run = rng.random(max(nruns, 1)) < rng.uniform(0.6, 1.0)
    delta_run = rng.uniform(0.05, 0.5, max(nruns, 1)) * cut_run
    delta = np.where(on, delta_run[np.maximum(rid, 0)], 0.0)
    price = regular * (1 - delta)
    if rng.random() < 0.5:
        price = np.round(price, 2)
    if rng.random() < 0.3:
        price = price * np.exp(rng.normal(0, 0.01, n))
    beta = 0.0 if rng.random() < 0.15 else -np.clip(np.exp(rng.normal(np.log(2.5), 0.6)), 0.2, 6)
    run_lift = beta * np.log(np.maximum(1 - delta, 1e-3))
    flags = []
    for _ in range(int(rng.integers(1, 4))):
        pick = rng.random(max(nruns, 1)) < rng.uniform(0.6, 0.9)
        f = (on & pick[np.maximum(rid, 0)]) | (rng.random(n) < rng.uniform(0, 0.03))
        run_lift = run_lift + rng.normal(0.3, 0.3) * f
        flags.append(f.astype(float))
    sigma_ev = rng.uniform(0, 0.6)
    het = np.exp(rng.normal(0, sigma_ev, max(nruns, 1)))[np.maximum(rid, 0)]
    het = np.where(rid >= 0, het, np.exp(rng.normal(0, sigma_ev, n)))
    lift = run_lift * het
    if rng.random() < 0.4:
        d, m = rng.uniform(0.05, 0.3), int(rng.integers(1, 4))
        ends = np.flatnonzero(on[:-1] & ~on[1:])
        for j in range(1, m + 1):
            idx = ends + j
            idx = idx[idx < n]
            idx = idx[~on[idx]]
            src = idx - j
            lift[idx] -= d * np.maximum(lift[src], 0) * (0.6 ** (j - 1))
    covs = [price, *flags]
    if rng.random() < 0.25:
        hon, _, _ = _runs(n, rng.uniform(0.05, 0.4), rng.uniform(1.3, 3), rng)
        lift = lift + rng.normal(0, 0.2) * hon
        covs.append(hon.astype(float))
    hol = np.zeros(n)
    if rng.random() < 0.6:
        per = 365 if daily else 52
        days = rng.choice(per, int(rng.integers(1, 6 if daily else 4)), replace=False)
        hol = np.isin((t + int(rng.integers(per))) % per, days).astype(float)
        lift = lift + rng.normal(0, 0.5) * hol
        covs.append(hol)
    closed = np.zeros(n, bool)
    if daily and rng.random() < 0.5:
        closed = rng.random(n) < rng.uniform(0, 0.03)
        if rng.random() < 0.5:
            closed |= ((t + int(rng.integers(7))) % 7) == 0
        if hol.any() and rng.random() < 0.5:
            closed |= hol > 0
        covs.append((~closed).astype(float) if rng.random() < 0.6 else closed.astype(float))
    lift = np.clip(lift, -4.0, 4.0) * eff
    mu = np.exp(loglam + lift)
    mu = np.where(closed & (eff > 0), 0.0, mu)
    if rng.random() < 0.7:
        r = np.exp(rng.uniform(np.log(0.5), np.log(200)))
        y = rng.poisson(np.minimum(rng.gamma(r, np.maximum(mu, 1e-12) / r), 1e8)).astype(float)
    else:
        y = mu * np.exp(rng.normal(0, rng.uniform(0.05, 0.5), n))
    if rng.random() < 0.1:
        yc = np.maximum(np.round(np.maximum(y, 0)), 0).astype(np.int64)
        covs.append(rng.binomial(yc, rng.uniform(0.15, 0.7)).astype(float))
    for _ in range(int(rng.integers(0, 4))):
        covs.append(_retail_decoy(rng.choice(["flag", "fraction", "int", "price"]), n, rng))
    covs = np.stack([covs[i] for i in rng.permutation(len(covs))])
    if first:
        y = np.concatenate([np.full(first, np.nan), y])
        covs = np.concatenate([np.full((len(covs), first), np.nan), covs], 1)
    return y.astype("float32"), covs.astype("float32")


def _shift(x, lag):
    """Return x(t - lag) with held edges; positive lag delays, negative lag leads."""
    if lag == 0:
        return x
    if lag > 0:
        lag = min(lag, len(x) - 1)
        return np.concatenate([np.full(lag, x[0]), x[:-lag]])
    lag = min(-lag, len(x) - 1)
    return np.concatenate([x[lag:], np.full(lag, x[-1])])


def _lagged_cov(rng, L, more):
    kind = rng.choice(["base", "step", "bell", "event"], p=[0.35, 0.2, 0.2, 0.25])
    t = np.arange(L, dtype="float64")
    if kind == "base":
        x = more(L)
        fill = np.nanmean(x) if np.isfinite(x).any() else 0.0
        return np.where(np.isfinite(x), x, fill)
    if kind == "step":
        changes = rng.random(L) < 10 ** rng.uniform(-3.3, -1.5)
        return rng.normal(size=int(changes.sum()) + 1)[np.cumsum(changes)]
    if kind == "bell":
        n = max(1, rng.poisson(L * 10 ** rng.uniform(-3.3, -2)))
        x = np.zeros(L)
        for c, w, a in zip(rng.integers(0, L, n), np.exp(rng.uniform(np.log(2), np.log(64), n)),
                           rng.uniform(0.3, 1.5, n)):
            lo, hi = max(0, int(c - 4 * w)), min(L, int(c + 4 * w) + 1)
            x[lo:hi] += a * np.exp(-0.5 * ((t[lo:hi] - c) / w) ** 2)
        return x
    starts = rng.random(L) < 10 ** rng.uniform(-3, -1.3)
    width = int(rng.integers(1, 12))
    return np.convolve(starts.astype(float), np.ones(width), mode="full")[:L].clip(0, 1)


def _lagged(b, rng, hist, H, kmax, more):
    L = hist + H
    k = int(min(kmax, 8, rng.geometric(0.25)))
    nan0 = ~np.isfinite(b)
    h = b[:hist][np.isfinite(b[:hist])]
    mu, sd = (float(h.mean()), float(h.std())) if len(h) else (0.0, 1.0)
    z = np.where(nan0, 0.0, (b - mu) / (sd if sd > 1e-9 else 1.0))
    z = z * rng.uniform(0.0, 1.0)
    impact = np.zeros(L)
    covs = []
    for _ in range(k):
        x = _lagged_cov(rng, L, more)
        covs.append(x)
        if rng.random() < 0.25:
            continue
        xs = (x - x.mean()) / (x.std() + 1e-9)
        lag = 0 if rng.random() < 0.3 else int(min(512, math.exp(rng.uniform(0, math.log(512)))))
        g = _shift(xs, lag)
        if rng.random() < 0.15:
            g = np.where(g > np.quantile(g, rng.uniform(0.5, 0.95)), g, 0.0)
        elif rng.random() < 0.2:
            g = np.diff(g, prepend=g[0])
            g = g / (g.std() + 1e-9)
        elif rng.random() < 0.3:
            g = np.tanh(g * rng.uniform(0.5, 2.0))
        if rng.random() < 0.1:
            g = g + rng.uniform(0.2, 0.8) * _shift(g, -int(rng.integers(1, 33)))
        impact += rng.normal() * g
    y = z + impact * rng.uniform(0.3, 2.0) + rng.normal(size=L) * rng.uniform(0.0, 0.3)
    if rng.random() < 0.3:
        y = np.exp(np.clip(0.3 * y, -6, 6))
    y = y * 10 ** rng.uniform(-1, 3) + (rng.normal() * 10 ** rng.uniform(-1, 2)
                                         if rng.random() < 0.5 else 0.0)
    y[nan0] = np.nan
    covs = np.stack(covs) * 10 ** rng.uniform(-1, 2, size=(len(covs), 1))
    return y.astype("float32"), covs.astype("float32")


def make_task(name, b, rng, hist, H, kmax, more):
    """(y [hist+H], covs [k <= kmax, hist+H]) of family `name` on synthetic base b."""
    from simons.synth.covariates import generic

    if name == "generic":
        return generic.make_task(b, rng, hist, H, kmax)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        if name == "classical":
            y, c = _classical(b, rng, hist, H, more)
        elif name == "energy":
            y, c = _energy(b, rng, hist, H)
        elif name == "retail":
            y, c = _retail(b, rng, hist, H)
        elif name == "lagged":
            y, c = _lagged(b, rng, hist, H, kmax, more)
        else:
            raise ValueError(name)
    y = np.asarray(y, np.float32)
    c = np.asarray(c, np.float32)[:kmax]
    y[~np.isfinite(y)] = np.nan
    c[~np.isfinite(c)] = np.nan
    return y, c
