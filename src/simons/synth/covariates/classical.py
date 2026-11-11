"""Sample classical covariate effects on corpus or synthetic base channels."""

from typing import ClassVar

import numpy as np
from scipy.signal import lfilter

from simons.synth.covariates import effects as D
from simons.synth.covariates.effects import log_uniform as _lu

WEIGHTS = {"F0": 0.04, "F1": 0.14, "F2": 0.16, "F3": 0.10, "F4": 0.18, "F5": 0.08, "F6": 0.12,
           "F7": 0.10, "F8": 0.08}
DEFAULTS = {"positive": 0.5, "count": 0.15, "null": 0.25, "weak_base": 0.25, "past_only": 0.0,
            "gaps": 0.1, "units": 0.3, "decoys": 3, "fc_addon": 0.35}



def _z(x):
    m, s = np.nanmean(x), np.nanstd(x)
    return (x - m) / (s if np.isfinite(s) and s > 1e-9 else 1.0)


def _unit(g):
    s = np.nanstd(g)
    return g / s if np.isfinite(s) and s > 1e-9 else g * 0.0


def _ar1(L, rng, phi, sd=1.0, heavy=False):
    e = rng.standard_t(3, L) / np.sqrt(3) if heavy else rng.normal(size=L)
    x = lfilter([1.0], [1.0, -phi], e)
    return x * sd * np.sqrt(max(1 - phi * phi, 1e-6))


def _lowpass(L, rng, period):
    """Filter white noise through a two-pole low-pass with the given cutoff period."""
    a = np.exp(-2 * np.pi / max(period, 2.0))
    return _unit(lfilter([1.0], [1.0, -2 * a, a * a], rng.normal(size=L)))


def _delay(x, d):
    if d <= 0:
        return x
    d = min(d, len(x) - 1)
    return np.concatenate([np.full(d, x[0]), x[:-d]])


def _filt(b, a, x):
    """Start lfilter at steady state for x[0] to avoid a start-up transient."""
    gain = np.sum(b) / np.sum(a)
    return lfilter(b, a, x - x[0]) + x[0] * gain


def _holds(L, rng, mean_hold):
    """Return segment IDs [L] for piecewise-constant signals with geometric hold times."""
    n = int(L / max(mean_hold, 1.0) * 3) + 4
    ends = np.cumsum(rng.geometric(1.0 / max(mean_hold, 1.0), size=n))
    return np.minimum(np.searchsorted(ends, np.arange(L), side="right"), n - 1)


class Cal:
    """Daily (pd), weekly (pw) and annual (py) periods in steps; p1 is the shortest cycle.
    pd/pw are None for daily/weekly or coarser data."""

    PD: ClassVar[dict[str, int | None]] = {"H": 24, "30T": 48, "15T": 96, "10T": 144, "5T": 288,
                                           "D": None, "W": None, "M": None}

    def __init__(self, freq, L, rng):
        self.pd = self.PD[freq]
        if self.pd:
            self.pw, self.py = 7 * self.pd, 365 * self.pd
        elif freq == "D":
            self.pw, self.py = 7, 365
        elif freq == "W":
            self.pw, self.py = None, 52
        else:
            self.pw, self.py = None, 12
        self.p1 = self.pd or self.pw or self.py
        self.t = np.arange(L, dtype="float64")
        self.t0 = int(rng.integers(0, 10 ** 6))
        self.phase = ((self.t + self.t0) % self.p1) / self.p1
        if self.pw:
            day = ((self.t + self.t0) // (self.pd or 1)) % 7
            self.weekend = (day >= 5).astype(float)
        else:
            self.weekend = None

    def profile(self, rng, harmonics=3):
        """Draw a unit-std smooth periodic profile over the shortest calendar cycle."""
        g = np.zeros_like(self.t)
        for k in range(1, harmonics + 1):
            g += rng.normal() / k * np.sin(2 * np.pi * k * self.phase + rng.uniform(0, 6.3))
        return _unit(g)


def _issued(truth, hist, rng, rel_sd=None, lead=None):
    """Issue forecasts with autocorrelated error, optionally switching vintages at the origin."""
    L = len(truth)
    H = L - hist
    sd = np.nanstd(truth)
    sd = sd if np.isfinite(sd) and sd > 0 else 1.0
    rel = _lu(rng, 0.03, 1.0) if rel_sd is None else rel_sd
    phi = rng.uniform(0.0, 0.97)
    heavy = rng.random() < 0.3
    e = _ar1(L, rng, phi, rel * sd, heavy)
    if lead is None:
        lead = rng.random() < 0.5
    if lead and H > 0:
        fresh = _ar1(H, rng, phi, rel * sd, heavy)
        k = rng.uniform(0.5, 4.0)
        e[hist:] = fresh * np.sqrt(1 + k * np.arange(1, H + 1) / H)
    f = np.nan_to_num(truth, nan=np.nanmean(truth) if np.isfinite(truth).any() else 0.0) + e
    if rng.random() < 0.3:
        f = f * rng.uniform(0.8, 1.2) + rng.normal() * 0.3 * sd
    return f


def _driver(L, rng, cal):
    k = rng.random()
    if k < 0.4:
        x = _lowpass(L, rng, _lu(rng, 4, 800))
    elif k < 0.6:
        x = _z(D.gen_covariate("smooth", L, rng))
    elif k < 0.8:
        x = _z(_lowpass(L, rng, _lu(rng, 20, 2000)) + 0.7 * cal.profile(rng))
    else:
        x = _z(D.gen_covariate("step", L, rng))
    return x




def f0_legacy(L, hist, rng, cal, lev):
    n = int(rng.integers(1, 7))
    kinds = rng.choice(["event", "smooth", "step", "calendar", "forecast"], size=n,
                       p=[0.3, 0.25, 0.15, 0.13, 0.17])
    g, covs, nf = np.zeros(L), [], 0
    for kind in kinds:
        if kind == "forecast":
            nf += 1
            continue
        x = D.gen_covariate(kind, L, rng)
        if rng.random() < 0.6:
            g += D.effect(x, rng) * rng.choice([-1, 1]) * _lu(rng, 0.05, 2.5)
        covs.append(x)
    return {"g": g, "covs": covs, "fc_target": nf}


def _input(L, rng, cal):
    k = rng.choice(["setpoint", "prbs", "steps", "band", "schedule"])
    if k == "setpoint":
        seg = _holds(L, rng, _lu(rng, 20, 2000))
        return rng.normal(size=seg.max() + 1)[seg]
    if k == "prbs":
        seg = _holds(L, rng, _lu(rng, 2, 200))
        return (rng.random(seg.max() + 1) < 0.5).astype(float)[seg]
    if k == "steps":
        return D.gen_covariate("step", L, rng)
    if k == "band":
        return _lowpass(L, rng, _lu(rng, 4, 500))
    slots = int(rng.choice([2, 3, 4, 6]))
    lv = rng.normal(size=slots)
    x = lv[np.minimum((cal.phase * slots).astype(int), slots - 1)]
    if rng.random() < 0.5:
        x = x + D.gen_covariate("step", L, rng) * 0.5
    return x


def _tf_coeffs(rng):
    if rng.random() < 0.55:
        tau = _lu(rng, 1, 300)
        a1 = np.exp(-1 / tau)
        a = np.array([1.0, -a1])
        b = np.array([1.0 - a1])
        if rng.random() < 0.3:
            z = rng.uniform(-0.5, 0.8)
            b = (1.0 - a1) * np.array([1.0, -z]) / (1.0 - z)
    elif rng.random() < 0.6:
        P = _lu(rng, 8, 400)
        r = np.exp(-1 / (P * rng.uniform(0.15, 2.0)))
        a = np.array([1.0, -2 * r * np.cos(2 * np.pi / P), r * r])
        b = np.array([a.sum()])
    else:
        r1, r2 = np.exp(-1 / _lu(rng, 1, 200)), np.exp(-1 / _lu(rng, 1, 200))
        a = np.array([1.0, -(r1 + r2), r1 * r2])
        b = np.array([a.sum()])
    return b, a


def _tf(x, rng, dmax=64):
    b, a = _tf_coeffs(rng)
    d = round(np.exp(rng.uniform(0, np.log(dmax + 1)))) - 1
    return _delay(_filt(b, a, x), d)


def f1_transfer(L, hist, rng, cal, lev):
    covs, g = [], np.zeros(L)
    for _ in range(int(rng.choice([1, 1, 2, 3]))):
        x = _input(L, rng, cal)
        covs.append(x)
        if rng.random() < 0.3:
            v = _z(_tf(x, rng, 16))
            s = rng.uniform(0.5, 2.0)
            w = s * np.tanh(v / s) + rng.uniform(-0.3, 0.3) * v * v
            y = _tf(w, rng)
        else:
            y = _tf(x, rng)
        g += _unit(y) * rng.choice([-1, 1]) * rng.uniform(0.3, 1.0)
    if rng.random() < 0.3:
        g += _ar1(L, rng, rng.uniform(0.9, 0.999), np.nanstd(g) * rng.uniform(0.1, 0.6))
    return {"g": g, "covs": covs}


def _events(L, hist, rng, cal):
    H = L - hist
    if cal.pw and rng.random() < 0.3:
        day = ((cal.t + cal.t0) // (cal.pd or 1)).astype(int)
        start_day = ((day % 7 == rng.integers(7))
                     & (rng.random(day.max() + 1)[day] < rng.uniform(0.3, 1)))
        start = start_day & (np.r_[True, day[1:] != day[:-1]])
        dur = cal.pd or 1
        return start.astype(float), dur
    rate = _lu(rng, 2, 60) / L
    start = rng.random(L) < rate
    if H > 4 and rng.random() < 0.7:
        start[hist + rng.integers(0, H - 1)] = True
    dur = int(max(1, round(_lu(rng, 1, 3 * (cal.pd or 1) * (2 if rng.random() < 0.5 else 1)))))
    return start.astype(float), dur


def f2_intervention(L, hist, rng, cal, lev):
    start, dur = _events(L, hist, rng, cal)
    ind = np.convolve(start, np.ones(dur), mode="full")[:L].clip(0, 1)
    n = int(start.sum())
    ids = np.cumsum(start).astype(int)
    lift = np.exp(rng.normal(0, rng.uniform(0, 0.7), size=n + 1))
    covs = []
    depth = None
    if rng.random() < 0.35:
        depth = rng.uniform(0.1, 1.0, size=n + 1)
        x = ind * depth[ids]
        lift = lift * depth / depth.mean()
    else:
        x = ind
    covs.append(x)
    resp = rng.choice(["abrupt_temp", "pulse_decay", "step_perm", "gradual_perm"],
                      p=[0.45, 0.25, 0.15, 0.15])
    if resp == "abrupt_temp":
        g = ind * lift[ids]
    elif resp == "pulse_decay":
        dlt = rng.uniform(0.3, 0.97)
        g = lfilter([1.0], [1.0, -dlt], start * lift[ids])
    elif resp == "step_perm":
        g = np.cumsum(start * rng.normal(0, 1, L)) if n else np.zeros(L)
        covs[0] = np.cumsum(start)
    else:
        dlt = rng.uniform(0.8, 0.995)
        stepx = np.minimum(np.cumsum(start), 1.0) if rng.random() < 0.5 else np.cumsum(start)
        g = _filt([1 - dlt], [1.0, -dlt], stepx)
        covs[0] = stepx
    sign = 1.0 if rng.random() < 0.8 else -1.0
    g = g * sign
    if resp in ("abrupt_temp", "pulse_decay"):
        if rng.random() < 0.4:
            k = int(max(1, round(dur * rng.uniform(0.5, 3))))
            end = np.r_[0.0, np.maximum(ind[:-1] - ind[1:], 0)]
            dip = np.convolve(end, np.ones(k), mode="full")[:L] * (1 - ind)
            g = g - sign * dip * rng.uniform(0.1, 0.5) * np.mean(lift)
        if rng.random() < 0.3:
            k = int(max(1, round(_lu(rng, 1, 2 * (cal.pd or 3)))))
            lead = np.convolve(start[::-1], np.ones(k), mode="full")[:L][::-1]
            lead = np.r_[lead[1:], 0.0] * (1 - ind)
            g = g + sign * lead * rng.uniform(0.1, 0.6) * np.mean(lift)
    zero = None
    if rng.random() < 0.15 and resp == "abrupt_temp":
        zero = ind > 0
        g = g * 0
    if rng.random() < 0.3:
        s2, d2 = _events(L, hist, rng, cal)
        ind2 = np.convolve(s2, np.ones(d2), mode="full")[:L].clip(0, 1)
        covs.append(ind2)
        g = g + _unit(ind2) * np.nanstd(g if np.nanstd(g) > 0 else ind2) * rng.normal(0, 0.8)
    return {"g": g, "covs": covs, "zero": zero}


def f3_dynregr(L, hist, rng, cal, lev):
    covs, g = [], np.zeros(L)
    for _ in range(int(rng.integers(1, 4))):
        x = _driver(L, rng, cal)
        covs.append(x)
        b0 = rng.normal()
        if rng.random() < 0.5:
            drift = rng.uniform(0.3, 2.0) * max(abs(b0), 0.3)
            beta = b0 + np.cumsum(rng.normal(size=L)) * drift / np.sqrt(L)
        else:
            seg = _holds(L, rng, _lu(rng, 50, 5000))
            levels = b0 + rng.normal(0, 1, size=seg.max() + 1)
            beta = levels[seg % 2] if rng.random() < 0.5 else levels[seg]
        g += beta * x
    g = _unit(g)
    if rng.random() < 0.7:
        r1, r2 = rng.uniform(-0.5, 0.98), rng.uniform(-0.5, 0.9)
        eta = lfilter([1.0, rng.uniform(-0.8, 0.8)], [1.0, -(r1 + r2), r1 * r2], rng.normal(size=L))
        g = g + _unit(eta) * rng.uniform(0.1, 0.7)
    return {"g": g, "covs": covs}


def _temperature(L, rng, cal):
    T = _ar1(L, rng, rng.uniform(0.95, 0.999) if cal.pd else rng.uniform(0.6, 0.95), 1.0)
    if cal.pd:
        T = T + rng.uniform(0.3, 1.5) * np.sin(2 * np.pi * cal.phase + rng.uniform(0, 6.3))
    T = T + rng.uniform(0.5, 3.0) * np.sin(2 * np.pi * (cal.t + cal.t0) / cal.py
                                           + rng.uniform(0, 6.3))
    return T


def f4_weather(L, hist, rng, cal, lev):
    kind = rng.choice(["load", "solar", "wind"], p=[0.55, 0.25, 0.2])
    covs = []
    if kind == "load":
        T = _temperature(L, rng, cal)
        Tf = _issued(T, hist, rng, rel_sd=_lu(rng, 0.03, 0.6), lead=rng.random() < 0.25)
        a = rng.uniform(0.02, 0.6) if rng.random() < 0.7 else 1.0
        Ts = _filt([a], [1.0, -(1 - a)], T)
        mid = np.quantile(T, rng.uniform(0.3, 0.7))
        band = rng.uniform(0.0, 1.0) * np.std(T)
        hdd, cdd = np.maximum(0, mid - band - Ts), np.maximum(0, Ts - mid - band)
        sh, sc = rng.uniform(0, 1), rng.uniform(0, 1)
        if cal.pd and rng.random() < 0.7:
            sh = sh * (1 + 0.6 * cal.profile(rng, 2))
            sc = sc * (1 + 0.6 * cal.profile(rng, 2))
        g = sh * hdd + sc * cdd + (rng.uniform(0, 0.2) * Ts ** 2 if rng.random() < 0.3 else 0)
        covs.append(Tf)
        if rng.random() < 0.3:
            covs.append(Tf * rng.uniform(0.5, 1) + _ar1(L, rng, 0.9, 0.5))
        if rng.random() < 0.3:
            covs.append(cal.weekend if cal.weekend is not None else cal.phase)
        return {"g": g, "covs": covs}
    if kind == "solar":
        lat = np.deg2rad(rng.uniform(-10, 60))
        doy = 2 * np.pi * ((cal.t + cal.t0) % cal.py) / cal.py
        decl = np.deg2rad(23.44) * np.sin(doy - 1.39)
        if cal.pd:
            ha = 2 * np.pi * (cal.phase - 0.5)
            elev = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(ha)
            clear = np.maximum(0.0, elev) ** rng.uniform(1.0, 1.5)
        else:
            clear = np.clip(0.6 + 0.4 * np.sin(doy - 1.39) * np.sign(lat + 1e-3), 0.05, None)
        cloud = 1 / (1 + np.exp(-(
            _ar1(L, rng, rng.uniform(0.9, 0.995) if cal.pd else rng.uniform(0.3, 0.8), 1.5)
            + rng.normal())))
        c, k = rng.uniform(0.5, 0.95), rng.uniform(1.0, 3.5)
        out = clear * (1 - c * cloud ** k)
        if rng.random() < 0.4:
            out = np.minimum(out, np.quantile(out, rng.uniform(0.85, 0.98)))
        cf = np.clip(_issued(cloud, hist, rng, rel_sd=_lu(rng, 0.05, 0.8),
                             lead=rng.random() < 0.25), 0, 1)
        r = rng.random()
        if r < 0.4:
            covs.append(cf)
        elif r < 0.8:
            covs.append(clear * (1 - c * cf ** k))
        else:
            covs += [cf, clear]
        return {"own": out / (np.mean(out) + 1e-9), "covs": covs, "noise_floor": True}
    v = np.exp(_ar1(L, rng, rng.uniform(0.95, 0.998) if cal.pd else rng.uniform(0.3, 0.8),
                    rng.uniform(0.3, 0.6)))
    v = v * rng.uniform(4, 10)
    vci, vr, vco = rng.uniform(2.5, 4), rng.uniform(10, 14), rng.uniform(20, 30)
    p = np.clip((v ** 3 - vci ** 3) / (vr ** 3 - vci ** 3), 0, 1)
    p[v > vco] = 0.0
    vf = np.maximum(0, _issued(v, hist, rng, rel_sd=_lu(rng, 0.05, 0.6), lead=rng.random() < 0.25))
    covs.append(vf)
    if rng.random() < 0.3:
        covs.append(vf * rng.uniform(1.1, 1.4) + _ar1(L, rng, 0.9, 0.5))
    return {"own": p / (np.mean(p) + 1e-9) + 1e-3, "covs": covs, "noise_floor": True}


def f5_marketing(L, hist, rng, cal, lev):
    kind = rng.choice(["bass", "adstock", "price"], p=[0.25, 0.4, 0.35])
    covs = []
    seg = _holds(L, rng, _lu(rng, 20, 1000))
    price = np.exp(np.cumsum(rng.normal(0, 0.05, size=seg.max() + 1)))[seg]
    if rng.random() < 0.6:
        start, dur = _events(L, hist, rng, cal)
        cut = np.convolve(start, np.ones(dur), mode="full")[:L].clip(0, 1)
        price = price * (1 - cut * rng.uniform(0.1, 0.4))
    flights = _holds(L, rng, _lu(rng, 10, 500))
    on = rng.random(flights.max() + 1) < rng.uniform(0.2, 0.7)
    spend = on[flights] * np.exp(rng.normal(0, 0.5, size=flights.max() + 1))[flights]
    lam = rng.uniform(0.3, 0.97)
    ad = lfilter([1 - lam], [1.0, -lam], spend)
    kap, h = (rng.uniform(0.7, 3.0),
              np.quantile(ad[ad > 0], rng.uniform(0.2, 0.8)) if (ad > 0).any() else 1.0)
    hill = ad ** kap / (ad ** kap + h ** kap + 1e-12)
    if kind == "bass":
        q = rng.uniform(1.0, 6.0) / (_lu(rng, 0.3, 1.5) * L)
        p = q * _lu(rng, 0.005, 0.3)
        xm = (price / price[:hist].mean()) ** (-rng.uniform(0, 3)) * (1 + rng.uniform(0, 2) * hill)
        effort = np.cumsum(xm)
        e = np.exp(-(p + q) * effort)
        F = (1 - e) / (1 + (q / p) * e)
        f = np.r_[F[0], np.diff(F)]
        covs += [price, spend] if rng.random() < 0.6 else [spend]
        own = f / (np.mean(f) + 1e-12) + 0.02
        return {"own": own, "covs": covs, "mult_level": True}
    if kind == "adstock":
        g = hill * rng.uniform(0.5, 1.5)
        covs.append(spend)
        if rng.random() < 0.4:
            g = g - rng.uniform(0.5, 3) * np.log(price)
            covs.append(price)
        return {"g": g, "covs": covs, "log_effect": True}
    g = -rng.uniform(0.5, 4.0) * np.log(price)
    covs.append(price)
    if rng.random() < 0.3:
        seg2 = _holds(L, rng, _lu(rng, 20, 1000))
        cp = np.exp(np.cumsum(rng.normal(0, 0.05, size=seg2.max() + 1)))[seg2]
        g = g + rng.uniform(0.2, 1.5) * np.log(cp)
        covs.append(cp)
    return {"g": g, "covs": covs, "log_effect": True}


def f6_capacity(L, hist, rng, cal, lev):
    slots = int(rng.choice([3, 4, 6, 8]))
    prof = np.exp(rng.normal(0, 0.5, size=slots))
    s = prof[np.minimum((cal.phase * slots).astype(int), slots - 1)]
    if cal.weekend is not None and rng.random() < 0.7:
        s = s * (1 - rng.uniform(0.1, 0.6) * cal.weekend)
    if rng.random() < 0.5:
        seg = _holds(L, rng, _lu(rng, 200, 5000))
        s = s * np.exp(rng.normal(0, 0.2, size=seg.max() + 1))[seg]
    covs = []
    disr = np.zeros(L)
    if rng.random() < 0.6:
        start, dur = _events(L, hist, rng, cal)
        disr = np.convolve(start, np.ones(max(dur, 2)), mode="full")[:L].clip(0, 1)
    cut = rng.uniform(0.0, 0.8)
    service = s * (1 - cut * disr)
    lev = np.nan_to_num(lev, nan=np.nanmean(lev))
    rho = rng.uniform(0.2, 1.2)
    demand = lev * (service / service.mean()) ** rho
    cap = rng.uniform(0.6, 1.6) * service / service.mean() * demand.mean()
    if rng.random() < 0.4:
        S = np.cumsum(demand - cap)
        B = S - np.minimum(0.0, np.minimum.accumulate(S))
        served = demand + np.r_[0.0, B[:-1]] - B
    else:
        served = np.minimum(demand, cap)
    covs.append(service if rng.random() < 0.7 else s)
    if disr.any() and rng.random() < 0.7:
        covs.append(disr)
    if rng.random() < 0.3:
        covs.append(cap)
    served = np.maximum(served, 0)
    return {"own": served / (np.mean(served) + 1e-12), "covs": covs}


def f7_interaction(L, hist, rng, cal, lev):
    kind = rng.choice(["lstr", "hill", "product", "daytype", "phase"])
    x = _driver(L, rng, cal)
    covs = [x]
    if kind == "lstr":
        if rng.random() < 0.5:
            sv = x
        else:
            sv = _driver(L, rng, cal)
            covs.append(sv)
        G = 1 / (1 + np.exp(-_lu(rng, 1, 20) * (sv - rng.uniform(-1, 1))))
        a1, a2 = rng.normal(), rng.normal()
        g = a1 * x + (a2 - a1) * x * G
    elif kind == "hill":
        xp = x - x.min() + 1e-3
        h, k = np.quantile(xp, rng.uniform(0.2, 0.8)), rng.uniform(1, 4)
        g = xp ** k / (xp ** k + h ** k)
    elif kind == "product":
        x2 = _driver(L, rng, cal) if rng.random() < 0.6 else (D.gen_covariate("event", L, rng))
        covs.append(x2)
        g = _z(x) * _z(x2) + rng.normal(0, 0.5) * x + rng.normal(0, 0.5) * _z(x2)
    elif kind == "daytype":
        if cal.weekend is not None:
            dt = cal.weekend
        else:
            dt = (cal.phase < rng.uniform(0.2, 0.5)).astype(float)
        b1, b2 = rng.normal(), rng.normal()
        g = x * (b1 + (b2 - b1) * dt)
        if rng.random() < 0.6:
            covs.append(dt)
    else:
        prof = cal.profile(rng, int(rng.integers(1, 4)))
        g = x * (rng.normal() + rng.uniform(0.5, 1.5) * prof)
        if rng.random() < 0.3:
            covs.append(np.sin(2 * np.pi * cal.phase))
    return {"g": g, "covs": covs}


def f8_issued(L, hist, rng, cal, lev):
    if rng.random() < 0.5:
        g = np.zeros(L)
        covs = []
        if rng.random() < 0.5:
            x = _driver(L, rng, cal)
            covs.append(x)
            g = x * rng.uniform(0.2, 1)
        return {"g": g, "covs": covs, "fc_target": int(rng.choice([1, 1, 2]))}
    u = _driver(L, rng, cal)
    g = u if rng.random() < 0.6 else np.tanh(u * rng.uniform(0.5, 2)) + rng.uniform(0, 0.4) * u * u
    covs = [_issued(u, hist, rng)]
    if rng.random() < 0.3:
        covs.append(_issued(u, hist, rng))
    return {"g": g, "covs": covs}


FAMILIES = {"F0": f0_legacy, "F1": f1_transfer, "F2": f2_intervention, "F3": f3_dynregr,
            "F4": f4_weather, "F5": f5_marketing, "F6": f6_capacity, "F7": f7_interaction,
            "F8": f8_issued}



def _level(y0, z0, hist, rng):
    h = y0[:hist]
    fin = np.isfinite(h)
    if fin.any() and np.nanmin(h) > 0 and rng.random() < 0.7:
        lev = y0 / np.nanmean(h)
        if np.nanstd(z0) < 0.5 * np.nanstd((y0 - np.nanmean(h)) / (np.nanstd(h) + 1e-9)):
            lev = 1 + (lev - 1) * 0.3
    elif rng.random() < 0.5:
        lev = np.exp(np.clip(rng.uniform(0.05, 0.6) * z0, -12, 12))
    else:
        lev = z0 - np.nanmin(z0[:hist]) + rng.uniform(0.3, 4.0)
    lev = np.maximum(lev, 0.01)
    m = np.nanmean(lev[:hist]) if np.isfinite(lev[:hist]).any() else 1.0
    return lev / (m if np.isfinite(m) and m > 0 else 1.0)


def sample(base, rng, hist: int, horizon: int):
    """Return float32 (target [hist+horizon], covariates [C,hist+horizon]) using `rng`.
    `base.uni` holds (values,freq) pairs; NaN marks target padding and unknown covariates."""
    for _ in range(4):
        try:
            y, c = _sample(base, rng, hist, horizon)
        except (ValueError, FloatingPointError, ZeroDivisionError, IndexError):
            continue
        if not np.isinf(y).any() and not np.isinf(c).any() and np.isfinite(y).any():
            return y, c
    return D.synthetic(base, rng, hist, horizon)


def _sample(base, rng, hist, horizon):
    names = list(WEIGHTS)
    p = np.array(list(WEIGHTS.values()), float)
    L = hist + horizon
    y0, freq = D.base_window(base, rng, L)
    nan0 = ~np.isfinite(y0)
    mu, sd = np.nanmean(y0[:hist]), np.nanstd(y0[:hist])
    if not np.isfinite(sd) or sd == 0:
        sd = 1.0
    if not np.isfinite(mu):
        mu = 0.0
    z0 = (y0 - mu) / sd
    if rng.random() < DEFAULTS["weak_base"]:
        z0 = z0 * rng.uniform(0, 0.3)
    cal = Cal(freq, L, rng)
    positive = rng.random() < DEFAULTS["positive"]
    lev = _level(y0, z0, hist, rng)
    fam = names[rng.choice(len(names), p=p / p.sum())]
    out = FAMILIES[fam](L, hist, rng, cal, lev)
    covs = [np.asarray(c, dtype="float64") for c in out["covs"]]
    null = rng.random() < DEFAULTS["null"]
    zfill = np.nan_to_num(z0, nan=0.0)
    lfill = np.nan_to_num(lev, nan=1.0)
    if "own" in out and not null:
        own = np.asarray(out["own"], dtype="float64")
        if out.get("mult_level") or rng.random() < 0.5:
            own = own * lfill ** rng.uniform(0.0, 1.0)
        elif out.get("noise_floor"):
            own = own + rng.uniform(0, 0.3) * lfill * rng.random()
        if positive:
            y = own
        else:
            y = _z(own) * rng.uniform(0.5, 1.5) + zfill * rng.uniform(0, 0.5)
    else:
        g = np.zeros(L) if null else _unit(np.nan_to_num(np.asarray(out["g"], dtype="float64")))
        if positive:
            if out.get("log_effect") or rng.random() < 0.65:
                s = _lu(rng, 0.05, 1.2)
                y = lfill * np.exp(np.clip(s * g, -6, 4))
            else:
                y = np.maximum(0.0, lfill + _lu(rng, 0.1, 2.0) * g)
        else:
            y = zfill + _lu(rng, 0.1, 3.0) * g
    if positive and out.get("zero") is not None and not null:
        y = np.where(out["zero"], 0.0, y)
    if positive:
        if rng.random() < DEFAULTS["count"]:
            m = _lu(rng, 0.5, 50.0)
            lam = np.maximum(y, 0) * m / (np.mean(y) + 1e-12)
            if rng.random() < 0.5:
                k = _lu(rng, 1.0, 50.0)
                lam = rng.gamma(k, 1.0 / k, size=L) * lam
            lam = np.nan_to_num(lam, nan=0.0, posinf=1e7, neginf=0.0).clip(0.0, 1e7)
            y = rng.poisson(lam).astype("float64")
        else:
            y = y * np.exp(rng.normal(size=L) * rng.uniform(0, 0.2))
    else:
        y = y + rng.normal(size=L) * rng.uniform(0, 0.3)
    for _ in range(int(out.get("fc_target", 0))):
        f = _issued(y, hist, rng)
        covs.append(np.maximum(f, 0) if positive else f)
    if rng.random() < DEFAULTS["fc_addon"]:
        for _ in range(int(rng.choice([1, 1, 2]))):
            f = _issued(y, hist, rng, lead=rng.random() < 0.25)
            covs.append(np.maximum(f, 0) if positive else f)
    for _ in range(int(rng.integers(0, DEFAULTS["decoys"] + 1))):
        if rng.random() < 0.4:
            x = D.covariate_window(base, rng, L)
        else:
            x = D.gen_covariate(rng.choice(["event", "smooth", "step", "calendar"]), L, rng)
        covs.append(x)
    if not covs:
        covs.append(D.gen_covariate("calendar", L, rng))
    covs = covs[: D.MAX_COV]
    order = rng.permutation(len(covs))
    covs = np.stack([covs[i] for i in order])
    covs[~np.isfinite(covs)] = 0.0
    for i in range(len(covs)):
        if rng.random() < DEFAULTS["past_only"]:  # Preserve the draw even at probability zero.
            covs[i, hist:] = np.nan
        if rng.random() < DEFAULTS["gaps"]:
            covs[i, rng.random(L) < rng.uniform(0, 0.1)] = np.nan
    if positive:
        y = y * 10 ** rng.uniform(-1, 4)
    else:
        y = y * 10 ** rng.uniform(-2, 4) + rng.normal() * 10 ** rng.uniform(-1, 4)
    y[nan0] = np.nan
    if rng.random() < DEFAULTS["units"]:
        covs = covs * 10 ** rng.uniform(-2, 3, size=(len(covs), 1))
    return y.astype("float32"), covs.astype("float32")
