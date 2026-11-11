"""Energy targets driven by latent weather, with issued-forecast covariates."""

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import lfilter
from scipy.special import expit, ndtr

DEFAULT_CFG = {
    "kinds": {"load": 0.35, "pv": 0.2, "wind": 0.2, "price": 0.15, "tariff": 0.1},
    "spd": {24: 0.45, 48: 0.25, 96: 0.25, 288: 0.05},
}


def _ar1(rng, shape, phi, axis=-1):
    """Draw stationary unit-variance AR(1) paths along `axis` with scalar coefficient phi."""
    e = rng.standard_normal(shape)
    phi = float(phi)
    s = np.sqrt(max(1 - phi * phi, 1e-6))
    idx = [slice(None)] * e.ndim
    idx[axis] = 0
    e[tuple(idx)] /= s
    return lfilter([s], [1.0, -phi], e, axis=axis)


def _phi(phi_hour, spd):
    return float(phi_hour) ** (24.0 / spd)


def _ewma(x, alpha):
    y, _ = lfilter([alpha], [1.0, -(1 - alpha)], x, zi=[(1 - alpha) * x[0]])
    return y


def _smooth(x, size):
    size = int(size)
    return x if size <= 1 else uniform_filter1d(x, size, mode="nearest")


def _u(rng, a, b):
    return float(rng.uniform(a, b))


class Calendar:
    def __init__(self, rng, L, spd):
        self.L, self.spd = L, spd
        self.t = np.arange(L)
        off = int(rng.integers(spd))
        tt = self.t + off
        self.day = tt // spd
        self.hour = (tt % spd) * (24.0 / spd)
        doy0 = int(rng.integers(365))
        self.doy = (doy0 + self.day) % 365
        self.wday = (int(rng.integers(7)) + self.day) % 7
        hol = rng.choice(365, size=int(rng.integers(0, 13)), replace=False)
        self.holiday = np.isin(self.doy, hol)
        self.dtype = np.where((self.wday == 6) | self.holiday, 2, np.where(self.wday == 5, 1, 0))
        self.ndays = int(self.day[-1]) + 1
        self.off = off


def _clear_sky(cal, lat):
    decl = np.deg2rad(23.45) * np.sin(2 * np.pi * (284 + cal.doy) / 365.0)
    omega = np.deg2rad(15.0 * (cal.hour + 0.5 * 24.0 / cal.spd - 12.0))
    la = np.deg2rad(lat)
    se = np.sin(la) * np.sin(decl) + np.cos(la) * np.cos(decl) * np.cos(omega)
    se = np.clip(se, 0, None)
    cs = np.where(se > 0.01, 1098.0 * se * np.exp(-0.057 / np.maximum(se, 0.01)), 0.0)
    return cs, se


def _weibull(z, k, c):
    u = np.clip(ndtr(z), 1e-9, 1 - 1e-9)
    return c * (-np.log1p(-u)) ** (1.0 / k)


class Weather:
    """Latent weather at one site (+ helpers to draw correlated sites)."""

    def __init__(self, rng, cal):
        L, spd = cal.L, cal.spd
        self.cal = cal
        lat = _u(rng, 25, 60) if rng.random() < 0.78 else _u(rng, -45, -10)
        peak = 200 if lat > 0 else 20
        seas = np.cos(2 * np.pi * (cal.doy + cal.hour / 24.0 - peak) / 365.0)
        self.c_mean = rng.normal(0.4, 1.0)
        self.c_gain = _u(rng, 1.5, 3.0)
        self.c_phi = _u(rng, 0.85, 0.97)
        zc = _ar1(rng, L, _phi(self.c_phi, spd)) + 0.4 * seas * rng.choice([-1, 1])
        self.zc = zc
        fast = _u(rng, 0.0, 0.8)
        self.zc_fast = fast * _ar1(rng, L, _phi(_u(rng, 0.3, 0.8), spd))
        self.cloud = expit(self.c_gain * (zc + self.zc_fast + self.c_mean))
        self.cs, self.sin_elev = _clear_sky(cal, lat)
        self.kc = 1 - 0.75 * self.cloud ** 3.4
        self.ghi = self.cs * self.kc
        Tm, Aa, Ad = _u(rng, -5, 25), _u(rng, 3, 13), _u(rng, 2, 8)
        hpk = _u(rng, 13, 16)
        diur = np.cos(2 * np.pi * (cal.hour - hpk) / 24.0) * (0.55 + 0.45 * (1 - self.cloud))
        syn = _ar1(rng, L, _phi(_u(rng, 0.95, 0.99), spd)) * _u(rng, 2, 6)
        self.T = Tm + Aa * seas + Ad * diur + syn
        spread = (np.exp(_ar1(rng, L, _phi(0.99, spd)) * 0.4) * _u(rng, 2, 9)
                  * (0.6 + 0.4 * (1 - self.cloud)))
        self.Td = self.T - spread
        self.w_k, self.w_c = _u(rng, 1.6, 2.6), _u(rng, 5, 10)
        ds, hw = _u(rng, 0, 0.3), _u(rng, 0, 24)
        self.w_phi = _u(rng, 0.85, 0.97)
        zw = _ar1(rng, L, _phi(self.w_phi, spd))
        self.zw_diur = ds * np.cos(2 * np.pi * (cal.hour - hw) / 24.0) * 2
        self.zw = (zw + self.zw_diur) / np.sqrt(1 + 2 * ds * ds)
        self.ws = _weibull(self.zw, self.w_k, self.w_c)
        theta0 = _u(rng, 0, 2 * np.pi)
        self.theta = (theta0 + np.cumsum(rng.standard_normal(L)) * _u(rng, 0.02, 0.12)
                      * np.sqrt(24.0 / spd))

    def cloud_sites(self, rng, n, rho):
        """n cloud fields sharing this site's latent with correlation rho."""
        zi = _ar1(rng, (n, self.cal.L), _phi(self.c_phi, self.cal.spd), axis=1)
        zf = _ar1(rng, (n, self.cal.L), _phi(0.6, self.cal.spd), axis=1) * np.std(self.zc_fast)
        z = rho * self.zc[None] + np.sqrt(1 - rho * rho) * zi + zf
        return expit(self.c_gain * (z + self.c_mean))


class IssuePlan:
    """Issue times and leads shared by all forecasts of one NWP/TSO product."""

    def __init__(self, rng, cal, hist):
        spd = cal.spd
        self.cal, self.hist = cal, hist
        L = cal.L
        o = round(_u(rng, 6, 12) * spd / 24.0)
        dd = 1 if rng.random() < 0.8 else 2
        day = cal.day
        issue_hist = (day - dd) * spd + o - cal.off
        self.mode = "latest" if rng.random() < 0.6 else "archive"
        t = cal.t
        origin_day = cal.day[hist] if hist < L else cal.day[-1]
        last_issue = (origin_day * spd + o - cal.off)
        if last_issue > hist:
            last_issue -= spd
        issue = issue_hist.copy()
        block = day
        if self.mode == "latest":
            issue[hist:] = last_issue
            block = np.where(t >= hist, block.max() + 1, block)
        self.lead = (t - issue).astype(float)
        self.lead_h = self.lead * 24.0 / spd
        self.block = block
        self.nblocks = int(block.max()) + 1
        self.timing = rng.random() < 0.5
        self.delta = np.round(rng.normal(0, _u(rng, 0.5, 3), self.nblocks) * spd / 24.0).astype(int)
        self.smooth = max(1, round(_u(rng, 1, 6) * spd / 24.0))
        self.cut = None
        if rng.random() < 0.1 and L - hist > 2:
            self.cut = int(rng.integers(hist + 1, L))
        self.actuals_hist = rng.random() < 0.15


def issue(x, plan, rng, rel_sd, diurnal=True, timing=False):
    """An issued forecast of truth x under `plan`; rel_sd: day-ahead error sd / std(x)."""
    cal = plan.cal
    L, spd = cal.L, cal.spd
    s = float(np.nanstd(x)) + 1e-9
    xs = _smooth(x, plan.smooth if rng.random() < 0.8 else 1)
    if timing and plan.timing:
        idx = np.clip(cal.t - plan.delta[plan.block], 0, L - 1)
        xs = xs[idx]
    clim = _smooth(x, 7 * spd)
    if diurnal:
        hb = (cal.t + cal.off) % spd
        prof = np.bincount(hb, weights=x - clim, minlength=spd) / np.maximum(
            np.bincount(hb, minlength=spd), 1)
        clim = clim + prof[hb]
    rho0, rho1 = _u(rng, 0.8, 1.0), _u(rng, 0.0, 0.12)
    rho = np.clip(rho0 - rho1 * plan.lead_h / 24.0, 0.25, 1.0)
    f = clim + rho * (xs - clim)
    if diurnal and rng.random() < 0.6:
        f = f - _u(rng, 0, 0.3) * prof[hb]
    f = f + rng.normal(0, 0.15) * s
    phi = _phi(_u(rng, 0.8, 0.98), spd)
    e = _ar1(rng, L, phi)
    starts = np.r_[True, plan.block[1:] != plan.block[:-1]]
    if starts.sum() > 1:
        e2 = rng.standard_normal(plan.nblocks)
        e = np.sqrt(0.5) * e + np.sqrt(0.5) * e2[plan.block]
    if rng.random() < 0.3:
        g = rng.chisquare(3, plan.nblocks) / 3.0
        e = e / np.sqrt(g[plan.block]) * np.sqrt(1 / 3.0)
    kappa = _u(rng, 0, 1.5)
    sd = rel_sd * np.sqrt(1 + kappa * np.clip(plan.lead_h - 24, 0, None) / 24.0)
    f = f + sd * s * e
    if rng.random() < 0.1:
        k = int(rng.integers(1, L))
        f[k:] = f[k:] * _u(rng, 0.9, 1.1) + rng.normal(0, 0.1) * s
    if plan.actuals_hist:
        f[: plan.hist] = x[: plan.hist]
    if plan.cut is not None:
        f[plan.cut:] = np.nan
    return f


class WeatherForecast:
    """Issued forecasts of a Weather's variables under one plan (physically consistent)."""

    def __init__(self, W, plan, rng):
        q = _u(rng, 0.5, 1.6)
        self.T = issue(W.T, plan, rng, 0.25 * q)
        self.Td = issue(W.Td, plan, rng, 0.3 * q)
        zc = issue(W.zc, plan, rng, 0.35 * q, diurnal=False, timing=True)
        self.cloud = expit(W.c_gain * (zc + W.c_mean))
        self.ghi = W.cs * (1 - 0.75 * self.cloud ** 3.4)
        zw = issue(W.zw, plan, rng, 0.5 * q, diurnal=False, timing=True)
        self.ws = _weibull(zw, W.w_k, W.w_c)
        self.theta = (W.theta + rng.normal(0, 0.3)
                      + _ar1(rng, W.cal.L, _phi(0.95, W.cal.spd)) * 0.2 * q)
        for v in ("T", "Td", "cloud", "ghi", "ws", "theta"):
            a = getattr(self, v)
            if plan.cut is not None:
                a[plan.cut:] = np.nan


def _daily_profile(rng, spd):
    K = int(rng.integers(2, 5))
    h = np.arange(spd) * 2 * np.pi / spd
    p = np.zeros(spd)
    for k in range(1, K + 1):
        p += rng.normal(0, 1.0 / k) * np.cos(k * h + _u(rng, 0, 2 * np.pi))
    p += _u(rng, 0.5, 1.5) * -np.cos(h - _u(rng, -0.6, 0.6))
    return (p - p.mean()) / (p.std() + 1e-9)


class LoadModel:
    def __init__(self, rng, cal):
        spd = cal.spd
        self.cal = cal
        prof = np.stack([_daily_profile(rng, spd) for _ in range(3)])
        prof[1] = 0.6 * prof[0] + 0.4 * prof[1]
        prof[2] = 0.5 * prof[0] + 0.5 * prof[2]
        amp = _u(rng, 0.1, 0.35)
        lev = np.array([1.0, _u(rng, 0.8, 0.97), _u(rng, 0.7, 0.93)])
        hb = (cal.t + cal.off) % spd
        shape = lev[cal.dtype] * (1 + amp * prof[cal.dtype, hb])
        self.occ = np.tanh(prof[0, hb])
        trend = 1 + rng.normal(0, 0.05) * cal.t / cal.L
        self.base = shape * trend * (1 + _u(rng, 0, 0.08) * rng.choice([-1, 1]) * np.cos(
            2 * np.pi * (cal.doy - 15) / 365.0))
        self.alpha = 1 - 0.5 ** (1.0 / (np.exp(_u(rng, np.log(1), np.log(24))) * spd / 24.0))
        self.lo = _u(rng, 12, 18)
        self.hi = self.lo + _u(rng, 0, 6)
        heat_cool = rng.random()
        self.bh = _u(rng, 0.01, 0.06) * (heat_cool > 0.15)
        self.bc = _u(rng, 0.01, 0.1) * (heat_cool < 0.85)
        self.pc = _u(rng, 1.0, 1.5)
        self.hmod = _u(rng, 0, 0.6)
        self.hum = _u(rng, 0, 0.5) if rng.random() < 0.5 else 0.0
        self.light = _u(rng, 0, 0.05) if rng.random() < 0.5 else 0.0
        self.season_mod = _u(rng, 0, 0.3)

    def __call__(self, T, Td, ghi, cs):
        Ts = _ewma(np.nan_to_num(T, nan=np.nanmean(T)), self.alpha)
        hdd = np.maximum(self.lo - Ts, 0)
        Te = Ts + self.hum * np.maximum(np.nan_to_num(Td, nan=0.0) - 12, 0) * 0.5
        cdd = np.maximum(Te - self.hi, 0) ** self.pc
        hm = 1 + self.hmod * self.occ
        resp = hm * (self.bh * hdd + self.bc * cdd)
        if self.light:
            dark = 1 - np.clip(np.nan_to_num(ghi) / np.maximum(cs.max(), 1), 0, 1)
            resp = resp + self.light * dark * (self.occ > -0.3)
        return self.base * (1 + resp)


class PVModel:
    def __init__(self, rng, W):
        self.W = W
        self.gamma = -_u(rng, 0.003, 0.0045)
        self.kT = _u(rng, 20, 35) / 1000.0
        self.dcac = _u(rng, 1.1, 1.4)
        self.tilt = _u(rng, 0, 0.4)
        self.eta = _u(rng, 0.9, 0.97)
        self.nsites = 1 if rng.random() < 0.6 else int(rng.integers(2, 9))
        self.rho = _u(rng, 0.5, 0.95)
        cal = W.cal
        self.curtail = np.ones(cal.L)
        if rng.random() < 0.15:
            days = rng.random(cal.ndays) < _u(rng, 0.05, 0.4)
            mid = (cal.hour > _u(rng, 9, 11)) & (cal.hour < _u(rng, 13, 16))
            self.curtail = np.where(days[cal.day] & mid, _u(rng, 0.3, 0.8), 1.0)
        self.soil = 1 - (_u(rng, 0, 0.08) * (cal.t % int(rng.integers(cal.spd * 5, cal.spd * 40)))
                         / (cal.spd * 40))

    def power(self, cloud, T):
        W = self.W
        k = 1 - 0.75 * np.clip(cloud, 0, 1) ** 3.4
        poa = W.cs * k * (1 + self.tilt * (1 - W.sin_elev)) * (W.sin_elev > 0.02)
        Tc = np.nan_to_num(T, nan=np.nanmean(T)) + self.kT * poa
        pdc = poa / 1000.0 * (1 + self.gamma * (Tc - 25))
        pac = np.minimum(pdc * self.dcac * self.eta, 1.0)
        if pac.ndim == 2:
            pac = pac.mean(0)
        return np.clip(pac, 0, None) * self.soil

    def truth_clouds(self, rng):
        if self.nsites == 1:
            return self.W.cloud
        return self.W.cloud_sites(rng, self.nsites, self.rho)

    def __call__(self, cloud, T, truth=True):
        p = self.power(cloud, T)
        return p * self.curtail if truth else p


_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(7)
_GH_W = _GH_W / _GH_W.sum()


class WindModel:
    def __init__(self, rng, W):
        self.W = W
        self.ci, self.rated = _u(rng, 2.5, 4), _u(rng, 10, 14)
        self.co = _u(rng, 20, 25)
        self.sig0, self.sig1 = _u(rng, 0.4, 0.8), _u(rng, 0.1, 0.25)
        self.wake = _u(rng, 0, 0.2)
        self.theta0 = _u(rng, 0, np.pi)
        self.shear = _u(rng, 0.8, 1.25)
        cal = W.cal
        changes = rng.random(cal.L) < _u(rng, 0.0005, 0.005)
        levels = np.clip(1 - np.abs(rng.normal(0, 0.06, changes.sum() + 1)), 0.6, 1)
        self.avail = levels[np.cumsum(changes)]
        self.cap = np.ones(cal.L)
        if rng.random() < 0.1:
            blocks = rng.random(cal.ndays) < 0.1
            self.cap = np.where(blocks[cal.day], _u(rng, 0.3, 0.8), 1.0)

    def curve(self, v):
        mid, s = (self.ci + self.rated) / 2, (self.rated - self.ci) / 7.0
        p = expit((v - mid) / s)
        p = np.where(v < self.ci, 0.0, p)
        return np.clip((p - expit((self.ci - mid) / s))
                       / (expit((self.rated - mid) / s) - expit((self.ci - mid) / s)), 0, 1)

    def __call__(self, ws, theta, truth=True):
        v = np.nan_to_num(ws, nan=np.nanmean(ws)) * self.shear
        sig = self.sig0 + self.sig1 * v
        vv = v[:, None] + sig[:, None] * _GH_X[None]
        p = (self.curve(np.clip(vv, 0, None)) * _GH_W).sum(1)
        ev = np.where(v > self.co, 1, np.where(v < self.co - 3, -1, 0))
        idx = np.where(ev != 0, np.arange(len(v)), 0)
        np.maximum.accumulate(idx, out=idx)
        state = ev[idx] == 1
        p = np.where(state, 0.0, p)
        p = p * (1 - self.wake * np.cos(np.nan_to_num(theta) - self.theta0) ** 2)
        if truth:
            p = p * self.avail * self.cap
        return p


def _cfg(cfg):
    out = {**DEFAULT_CFG, **(cfg or {})}
    out["kinds"] = {**DEFAULT_CFG["kinds"], **((cfg or {}).get("kinds") or {})}
    return out


def _tso_forecast(rng, truth_fn_value, y_true, plan):
    """Evaluate target response on forecast drivers with smooth error shared across issues."""
    s = float(np.nanstd(y_true)) + 1e-9
    e = _ar1(rng, len(y_true), _phi(_u(rng, 0.8, 0.98), plan.cal.spd))
    f = truth_fn_value + _u(rng, 0.02, 0.25) * s * e + rng.normal(0, 0.05) * s
    if plan.actuals_hist:
        f[: plan.hist] = y_true[: plan.hist]
    if plan.cut is not None:
        f[plan.cut:] = np.nan
    return f


def sample(base, rng, hist: int, horizon: int, cfg=None):
    """Return float32 (target [hist+horizon], covariates [C,hist+horizon]); NaN marks gaps.
    `cfg` overrides DEFAULT_CFG and merges `kinds` by key; `base` is unused."""
    c = _cfg(cfg)
    L = hist + horizon
    kinds = list(c["kinds"])
    kp = np.array([c["kinds"][k] for k in kinds], float)
    kind = kinds[rng.choice(len(kinds), p=kp / kp.sum())]
    spds = list(c["spd"])
    sp = np.array([c["spd"][k] for k in spds], float)
    spd = int(spds[rng.choice(len(spds), p=sp / sp.sum())])
    cal = Calendar(rng, L, spd)
    W = Weather(rng, cal)
    plan = IssuePlan(rng, cal, hist)
    Wf = WeatherForecast(W, plan, rng)
    covs = []
    noise_sd = _u(rng, 0.005, 0.03)
    tso = rng.random() < 0.3
    want = int(rng.choice([1, 2, 3], p=[0.35, 0.4, 0.25]))

    def obs_noise(n):
        return _ar1(rng, n, _phi(_u(rng, 0.3, 0.9), spd))

    temp_f = rng.random() < 0.15
    def T_out(a):
        return a * 1.8 + 32 if temp_f else a

    if kind == "load":
        lm = LoadModel(rng, cal)
        net = rng.random() < 0.3
        y = lm(W.T, W.Td, W.ghi, W.cs)
        yf = lm(Wf.T, Wf.Td, Wf.ghi, W.cs)
        if net:
            pv = PVModel(rng, W)
            share = _u(rng, 0.05, 0.4)
            y = y - share * pv(pv.truth_clouds(rng), W.T)
            yf = yf - share * pv(Wf.cloud, Wf.T, truth=False)
        y = y * (1 + noise_sd * obs_noise(L))
        opts = [T_out(Wf.T), T_out(Wf.Td), Wf.ghi, 100 * Wf.cloud]
        order = [0, *rng.permutation([1, 2, 3])]
        if net:  # Embedded PV makes irradiance relevant, so prefer it second.
            order = [0, 2, 1, 3] if rng.random() < 0.7 else order
        for i in order[:want]:
            covs.append(opts[i])
        scale = 10 ** _u(rng, 0.5, 4)
    elif kind == "pv":
        pm = PVModel(rng, W)
        y = pm(pm.truth_clouds(rng), W.T)
        yf = pm(Wf.cloud, Wf.T, truth=False)
        y = np.clip(y + noise_sd * 0.3 * obs_noise(L) * (W.cs > 0), 0, None)
        opts = [Wf.ghi, 100 * Wf.cloud, T_out(Wf.T)]
        order = [0, 1, 2] if rng.random() < 0.7 else [1, 0, 2]
        for i in order[:want]:
            covs.append(opts[i])
        scale = 10 ** _u(rng, 0.5, 4)
    elif kind == "wind":
        wm = WindModel(rng, W)
        y = wm(W.ws, W.theta)
        yf = wm(Wf.ws, Wf.theta, truth=False)
        y = np.clip(y + noise_sd * obs_noise(L), 0, None)
        u, v = -Wf.ws * np.sin(Wf.theta), -Wf.ws * np.cos(Wf.theta)
        if rng.random() < 0.5:
            opts = [Wf.ws, u, v]
        else:
            opts = [Wf.ws, np.degrees(np.mod(Wf.theta, 2 * np.pi)), T_out(Wf.T)]
        for i in range(want):
            covs.append(opts[i])
        scale = 10 ** _u(rng, 0.5, 4)
    elif kind == "price":
        lm = LoadModel(rng, cal)
        pm, wm = PVModel(rng, W), WindModel(rng, W)
        ws_share, pv_share = _u(rng, 0, 0.6), _u(rng, 0, 0.4)
        load, load_f = lm(W.T, W.Td, W.ghi, W.cs), lm(Wf.T, Wf.Td, Wf.ghi, W.cs)
        wind, wind_f = wm(W.ws, W.theta), wm(Wf.ws, Wf.theta, truth=False)
        pv, pv_f = pm(pm.truth_clouds(rng), W.T), pm(Wf.cloud, Wf.T, truth=False)
        R = load - ws_share * wind - pv_share * pv
        Rf = load_f - ws_share * wind_f - pv_share * pv_f
        Rmax = float(np.quantile(R, 0.99)) + 1e-6
        Rmin = float(np.quantile(R, 0.01))
        kappa = _u(rng, 1, 4)
        a0, b = _u(rng, 0, 60), _u(rng, 10, 80)
        fuel = np.exp(np.cumsum(rng.normal(0, _u(rng, 0, 0.03), cal.ndays))[cal.day])

        def stack(Rx):
            z = (Rx - Rmin) / max(Rmax - Rmin, 1e-6)
            return fuel * (a0 + b * np.expm1(kappa * z) / np.expm1(kappa))

        y = stack(R)
        yf = stack(Rf)
        capq = float(np.quantile(R, _u(rng, 0.95, 0.995)))
        over = np.clip((R - capq) / max(Rmax - capq, 1e-6), 0, None)
        spikes = (R > capq) & (rng.random(L) < _u(rng, 0.05, 0.5))
        y = y + spikes * (0.3 + over) * np.exp(rng.normal(np.log(_u(rng, 0.5, 5) * b), 0.5, L))
        if rng.random() < 0.4:
            lowq = float(np.quantile(R, _u(rng, 0.01, 0.1)))
            floor = -_u(rng, 0, 60)
            y = np.where(R < lowq, np.minimum(y, floor + (y - floor) * _u(rng, 0, 0.3)), y)
        y = y + _u(rng, 0.01, 0.08) * np.nanstd(y) * obs_noise(L)
        opts = [load_f, ws_share * wind_f + pv_share * pv_f, Rf, ws_share * wind_f,
                pv_share * pv_f]
        choice = [[0], [0, 1], [0, 3, 4], [2], [0, 1, 2]][
            int(rng.choice(5, p=[0.3, 0.3, 0.2, 0.1, 0.1]))]
        for i in choice:
            scl = 10 ** _u(rng, 2, 4)
            covs.append(opts[i] * scl)
        scale = 1.0
    else:
        lm = LoadModel(rng, cal)
        bl = lm(W.T, W.Td, W.ghi, W.cs)
        blf = lm(Wf.T, Wf.Td, Wf.ghi, W.cs)
        hours = cal.hour
        ev_day = rng.random(cal.ndays) < _u(rng, 0.1, 0.5)
        h0, dur = _u(rng, 16, 19), _u(rng, 2, 5)
        high = ev_day[cal.day] & (hours >= h0) & (hours < h0 + dur)
        low_day = rng.random(cal.ndays) < _u(rng, 0.0, 0.4)
        l0 = _u(rng, 0, 6)
        low = low_day[cal.day] & (hours >= l0) & (hours < l0 + _u(rng, 2, 6)) & ~high
        eh, el = -_u(rng, 0.05, 0.4), _u(rng, 0.0, 0.3)
        r = eh * high + el * low
        k = _u(rng, 0.5, 0.95) ** (24.0 / spd)
        deferred = -eh * high
        reb = _u(rng, 0, 1) * lfilter([1 - k], [1, -k], deferred) * ~high
        y = bl * (1 + r + reb)
        yf = blf * (1 + r + reb)
        y = y * (1 + _u(rng, 0.01, 0.05) * obs_noise(L))
        covs += [high.astype(float), low.astype(float)]
        if rng.random() < 0.4:
            covs.append(T_out(Wf.T))
        scale = 10 ** _u(rng, 0.5, 4)
    if tso:
        tf = _tso_forecast(rng, yf, y, plan)
        covs.append(tf * scale if rng.random() < 0.85 else tf)
    if rng.random() < 0.5:
        for _ in range(int(rng.integers(1, 3))):
            if rng.random() < 0.5:
                W2 = Weather(rng, cal)
                covs.append(issue(W2.T, plan, rng, 0.3))
            else:
                p = rng.choice([spd, 7 * spd, spd // 2])
                covs.append(np.sin(2 * np.pi * cal.t / p + _u(rng, 0, 6.3)))
    covs = np.stack(covs).astype("float64")
    y = y * scale
    for i in range(len(covs)):
        r = rng.random()
        if r < 0.1:
            covs[i, hist:] = np.nan
        elif r < 0.35 and hist > 64:
            covs[i, : int(rng.integers(0, hist - 32))] = np.nan
        if rng.random() < 0.1:
            covs[i, rng.random(L) < _u(rng, 0, 0.1)] = np.nan
    if rng.random() < 0.05:
        y[rng.random(L) < _u(rng, 0, 0.05)] = np.nan
    covs = covs[rng.permutation(len(covs))]
    return y.astype("float32"), covs.astype("float32")

