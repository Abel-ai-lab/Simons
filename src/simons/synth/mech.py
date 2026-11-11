"""Mechanistic priors: polynomial flows, epidemics, rainfall-runoff, queues and thermal systems."""

import math

import torch

from simons.synth.base import _logu, _randn, _standardise, _u

BANK = {"chaos": 16384, "epi": 8192, "hydro": 8192, "thermal": 8192}
BANK_LENGTH = 9216
REFRESH = 400
PART = 1
SPARE = 0.35
_STATE = {}


def configure(bank=None, length=None, refresh=None, part=None):
    """Update trajectory-bank settings and clear shared caches; None retains a setting.
    `length`: steps per path; `refresh`: calls between refreshes (0: never); fraction: 1/part."""
    if bank is not None:
        scale = int(bank) / 16384
        for k in BANK:
            BANK[k] = max(64, round({"chaos": 16384}.get(k, 8192) * scale))
    global BANK_LENGTH, REFRESH, PART
    if part is not None:
        PART = max(1, int(part))
    if length is not None:
        BANK_LENGTH = int(length)
    if refresh is not None:
        REFRESH = int(refresh)
    _STATE.clear()


def _ar1(gen, n, length, device, phi, sd):
    """Draw AR(1) paths [n,L] from zero with per-row phi and asymptotic sd [n,1]."""
    from simons.synth.nonstat import _ou_scan

    z = _randn(gen, (n, length), device).double()
    phi = phi.double()
    a = phi.expand(n, length).contiguous()
    drive = z * torch.sqrt(1 - phi**2)
    y = _ou_scan(a, drive, z[:, 0] * 0)
    return (y * sd.double()).float()


# Founder tuples: (name, period, amplitude, init center, init scale, equations).
X, Y, Z, U, V = 0, 1, 2, 3, 4


def _sprott():
    return [
        ("sprott_a", 6.0, 1.5, (0, 5, 0), 0.5,
         [[(1, (Y,))],
          [(-1, (X,)), (1, (Y, Z))],
          [(1, ()), (-1, (Y, Y))]]),
        ("sprott_b", 6.0, 1.5, (0.4, 0.3, 0.1), 0.5,
         [[(1, (Y, Z))],
          [(1, (X,)), (-1, (Y,))],
          [(1, ()), (-1, (X, Y))]]),
        ("sprott_c", 6.0, 1.5, (0.4, 0.3, 0.1), 0.5,
         [[(1, (Y, Z))],
          [(1, (X,)), (-1, (Y,))],
          [(1, ()), (-1, (X, X))]]),
        ("sprott_e", 6.0, 1.5, (0.5, 0.5, 0.5), 0.3,
         [[(1, (Y, Z))],
          [(1, (X, X)), (-1, (Y,))],
          [(1, ()), (-4, (X,))]]),
        ("sprott_g", 6.0, 1.5, (0.5, 0.5, 0.5), 0.3,
         [[(0.4, (X,)), (1, (Z,))],
          [(1, (X, Z)), (-1, (Y,))],
          [(-1, (X,)), (1, (Y,))]]),
        ("sprott_h", 6.0, 1.5, (0.5, 0.5, 0.5), 0.3,
         [[(-1, (Y,)), (1, (Z, Z))],
          [(1, (X,)), (0.5, (Y,))],
          [(1, (X,)), (-1, (Z,))]]),
        ("sprott_j", 6.0, 1.5, (0.5, 0.5, 0.5), 0.3,
         [[(2, (Z,))],
          [(-2, (Y,)), (1, (Z,))],
          [(-1, (X,)), (1, (Y,)), (1, (Y, Y))]]),
        ("sprott_k", 6.0, 1.5, (0.5, 0.5, 0.5), 0.3,
         [[(1, (X, Y)), (-1, (Z,))],
          [(1, (X,)), (-1, (Y,))],
          [(1, (X,)), (0.3, (Z,))]]),
        ("sprott_n", 6.0, 1.5, (0.5, 0.5, 0.5), 0.3,
         [[(-2, (Y,))],
          [(1, (X,)), (1, (Z, Z))],
          [(1, ()), (1, (Y,)), (-2, (Z,))]]),
        ("sprott_s", 6.0, 1.5, (0.5, 0.5, 0.5), 0.3,
         [[(-1, (X,)), (-4, (Y,))],
          [(1, (X,)), (1, (Z, Z))],
          [(1, ()), (1, (X,))]]),
    ]


FOUNDERS = [
    ("lorenz", 0.75, 8.0, (0, 1, 25), 5.0,
     [[(-10, (X,)), (10, (Y,))],
      [(28, (X,)), (-1, (X, Z)), (-1, (Y,))],
      [(1, (X, Y)), (-8 / 3, (Z,))]]),
    ("roessler", 6.1, 5.0, (0, -5, 0.5), 2.0,
     [[(-1, (Y,)), (-1, (Z,))],
      [(1, (X,)), (0.2, (Y,))],
      [(0.2, ()), (1, (X, Z)), (-5.7, (Z,))]]),
    ("chen", 0.45, 8.0, (-3, 2, 20), 4.0,
     [[(-35, (X,)), (35, (Y,))],
      [(-7, (X,)), (-1, (X, Z)), (28, (Y,))],
      [(1, (X, Y)), (-3, (Z,))]]),
    ("lu", 0.6, 8.0, (1, 1, 15), 4.0,
     [[(-36, (X,)), (36, (Y,))],
      [(-1, (X, Z)), (20, (Y,))],
      [(1, (X, Y)), (-3, (Z,))]]),
    *_sprott(),
    ("halvorsen", 2.5, 4.0, (-1, 0, 0), 1.0,
     [[(-1.89, (X,)), (-4, (Y,)), (-4, (Z,)), (-1, (Y, Y))],
      [(-1.89, (Y,)), (-4, (Z,)), (-4, (X,)), (-1, (Z, Z))],
      [(-1.89, (Z,)), (-4, (X,)), (-4, (Y,)), (-1, (X, X))]]),
    ("shimizu", 8.0, 1.0, (0.1, 0, 0), 0.3,
     [[(1, (Y,))],
      [(1, (X,)), (-0.75, (Y,)), (-1, (X, Z))],
      [(-0.45, (Z,)), (1, (X, X))]]),
    ("arneodo", 5.0, 2.0, (0.2, 0.2, 0.2), 0.3,
     [[(1, (Y,))],
      [(1, (Z,))],
      [(5.5, (X,)), (-3.5, (Y,)), (-1, (Z,)), (-1, (X, X, X))]]),
    ("genesio", 6.0, 0.5, (0.1, 0.1, 0.1), 0.1,
     [[(1, (Y,))],
      [(1, (Z,))],
      [(-1, (X,)), (-1.1, (Y,)), (-0.44, (Z,)), (1, (X, X))]]),
    ("burke_shaw", 1.0, 1.0, (0.6, 0, 0), 0.3,
     [[(-10, (X,)), (-10, (Y,))],
      [(-1, (Y,)), (-10, (X, Z))],
      [(10, (X, Y)), (4.272, ())]]),
    ("hindmarsh_rose", 10.0, 1.0, (-1, -5, 3), 0.3,
     [[(1, (Y,)), (-1, (X, X, X)), (3, (X, X)), (-1, (Z,)), (3.25, ())],
      [(1, ()), (-5, (X, X)), (-1, (Y,))],
      [(0.02, (X,)), (0.032, ()), (-0.005, (Z,))]]),
    ("lotka_volterra", 40.0, 0.3, (0.3, 0.3, 0.3), 0.1,
     [[(0.001, ()), (1, (X,)), (-1, (X, X)), (-0.8, (X, Y)), (-1.3, (X, Z))],
      [(0.001, ()), (1, (Y,)), (-1.3, (X, Y)), (-1, (Y, Y)), (-0.8, (Y, Z))],
      [(0.001, ()), (1, (Z,)), (-0.8, (X, Z)), (-1.3, (Y, Z)), (-1, (Z, Z))]]),
    ("moore_spiegel", 3.0, 1.0, (0.1, 0, 0), 0.3,
     [[(1, (Y,))],
      [(1, (Z,))],
      [(-1, (Z,)), (14, (Y,)), (-20, (X, X, Y)), (-6, (X,))]]),
    ("rucklidge", 4.0, 2.0, (1, 0, 4.5), 1.0,
     [[(-2, (X,)), (6.7, (Y,)), (-1, (Y, Z))],
      [(1, (X,))],
      [(-1, (Z,)), (1, (Y, Y))]]),
    ("aizawa", 1.8, 0.7, (0.1, 0, 0), 0.1,
     [[(1, (X, Z)), (-0.7, (X,)), (-3.5, (Y,))],
      [(3.5, (X,)), (1, (Y, Z)), (-0.7, (Y,))],
      [(0.6, ()), (0.95, (Z,)), (-1 / 3, (Z, Z, Z)), (-1, (X, X)), (-1, (Y, Y)), (-0.25, (X, X, Z)),
       (-0.25, (Y, Y, Z))]]),
    ("dadras", 2.0, 3.0, (1, 1, 0), 1.0,
     [[(1, (Y,)), (-3, (X,)), (2.7, (Y, Z))],
      [(1.7, (Y,)), (-1, (X, Z)), (1, (Z,))],
      [(2, (X, Y)), (-9, (Z,))]]),
]
# Forced tuples: (name, forcing frequency, amplitude, init, scale, [eq_x, eq_y]).
FORCED = [
    ("duffing", 1.2, 1.0, (1, 0, 0), 0.5,
     [[(1, (Y,))],
      [(-0.3, (Y,)), (1, (X,)), (-1, (X, X, X)), (0.5, (U,))]]),
    ("vanderpol", 1.1, 2.0, (1, 0, 0), 0.5,
     [[(1, (Y,))],
      [(2, (Y,)), (-2, (X, X, Y)), (-1, (X,)), (1.0, (U,))]]),
    ("fitzhugh", 0.2, 1.5, (0, 0, 0), 0.5,
     [[(1, (X,)), (-1 / 3, (X, X, X)), (-1, (Y,)), (0.5, ()), (0.3, (U,))],
      [(0.056, ()), (0.08, (X,)), (-0.064, (Y,))]]),
    ("brusselator", 0.81, 1.0, (1, 1, 0), 0.3,
     [[(0.4, ()), (-2.2, (X,)), (1, (X, X, Y)), (0.05, (U,))],
      [(1.2, (X,)), (-1, (X, X, Y))]]),
]
DIM, FEATURES = 6, 1 + 6 + 36 + 216


def _feature(mono):
    m = sorted(mono)
    if not m:
        return 0
    if len(m) == 1:
        return 1 + m[0]
    if len(m) == 2:
        return 7 + 6 * m[0] + m[1]
    return 43 + 36 * m[0] + 6 * m[1] + m[2]


def _template(eqs, shift):
    c = torch.zeros(FEATURES, DIM)
    for row, terms in enumerate(eqs):
        for coef, mono in terms:
            c[_feature(tuple(i + shift for i in mono)), row + shift] += coef
    return c


_TEMPLATES = {}


def _templates(device):
    key = str(device)
    if key not in _TEMPLATES:
        a = [_template(f[5], 0) for f in FOUNDERS] + [_template(f[5], 0) for f in FORCED]
        b = [_template(f[5], 3) for f in FOUNDERS]
        for i, f in enumerate(FORCED):
            a[len(FOUNDERS) + i][_feature((V,)), U] = f[1]
            a[len(FOUNDERS) + i][_feature((U,)), V] = -f[1]
        meta = torch.tensor(
            [[2 * math.pi / f[1] if j >= len(FOUNDERS) else f[1], f[2], *f[3], f[4]]
             for j, f in enumerate(FOUNDERS + FORCED)], dtype=torch.float32)
        _TEMPLATES[key] = (torch.stack(a).to(device), torch.stack(b).to(device), meta.to(device))
    return _TEMPLATES[key]


TERMS = 32


def _rhs(x, feat, eq, val, ones):
    """Evaluate sparse polynomial terms: sum val_k * monomial(feat_k) into equation eq_k."""
    n = x.shape[0]
    x2 = (x[:, :, None] * x[:, None, :]).reshape(n, 36)
    x3 = (x2[:, :, None] * x[:, None, :]).reshape(n, 216)
    f = torch.cat([ones, x, x2, x3], 1)
    return torch.zeros_like(x).scatter_add_(1, eq, f.gather(1, feat) * val)


def _chaos_bank(gen, n, length, device, channels=3, burn=768, sub=3):
    ta, tb, meta = _templates(device)
    nf, nforced = len(FOUNDERS), len(FORCED)
    founder = torch.randint(0, nf + nforced, (n,), generator=gen, device=device)
    forced = founder >= nf
    response = torch.randint(0, nf, (n,), generator=gen, device=device)
    skew = (_u(gen, (n,), device=device) < 0.4) & ~forced
    period, amp = meta[founder, 0], meta[founder, 1]
    coef = ta[founder]
    rate = meta[response, 0] / period * _logu(gen, (n,), 0.5, 2.0, device)
    coef = coef + torch.where(skew[:, None, None], tb[response] * rate[:, None, None], 0.0)
    kappa = _u(gen, (n,), 0.05, 0.6, device) * meta[response, 1] / meta[response, 0] * rate / amp
    coef[:, 1 + X, 3] += torch.where(skew, kappa, 0.0)
    sigma = _u(gen, (n, 1, 1), 0.0, 0.2, device)
    mutate = torch.exp(sigma * _randn(gen, coef.shape, device))
    keep = torch.zeros(DIM, device=device, dtype=torch.bool)
    keep[3:5] = True
    keep = forced[:, None, None] & keep[None, None, :]
    coef = torch.where(keep, coef, coef * mutate)
    x = torch.zeros(n, DIM, device=device)
    x[:, :3] = meta[founder, 2:5] + meta[founder, 5:6] * _randn(gen, (n, 3), device)
    x[:, 3:] = meta[response, 2:5] + meta[response, 5:6] * _randn(gen, (n, 3), device)
    phase = _u(gen, (n,), 0.0, 2 * math.pi, device)
    x[:, 3] = torch.where(forced, torch.cos(phase), x[:, 3])
    x[:, 4] = torch.where(forced, -torch.sin(phase), x[:, 4])
    x[:, 5] = torch.where(forced, 0.0, x[:, 5])
    block = torch.where(skew & (_u(gen, (n,), device=device) < 0.6), 3, 0)
    active = torch.where(forced, 2, 3)
    w = torch.zeros(n, DIM, channels, device=device)
    rows = torch.arange(n, device=device)
    for c in range(channels):
        if c == 1:
            g = (_randn(gen, (n, 3), device)
                 * (torch.arange(3, device=device)[None] < active[:, None]))
            for k in range(3):
                w[rows, block + k, c] = g[:, k]
        else:
            k = (_u(gen, (n,), device=device) * active).long().clamp(max=2)
            other = torch.where(skew & (_u(gen, (n,), device=device) < 0.5), 3 - block, block)
            w[rows, (other if c == 2 else block) + k, c] = 1.0
    flat = coef.reshape(n, FEATURES * DIM)
    top = flat.abs().topk(TERMS, dim=1).indices
    val, feat, eq = flat.gather(1, top), top // DIM, top % DIM
    del coef, flat, mutate
    ppp = _logu(gen, (n,), 14.0, 200.0, device)
    dt = (period / ppp / sub)[:, None]
    dt = torch.minimum(dt, (period / 42)[:, None])
    half, sixth = dt / 2, dt / 6
    ones = torch.ones(n, 1, device=device)
    out = torch.empty(n, length, channels, device=device, dtype=torch.float16)
    for t in range(burn + length):
        for _ in range(sub):
            k1 = _rhs(x, feat, eq, val, ones)
            k2 = _rhs(torch.addcmul(x, half, k1), feat, eq, val, ones)
            k3 = _rhs(torch.addcmul(x, half, k2), feat, eq, val, ones)
            k4 = _rhs(torch.addcmul(x, dt, k3), feat, eq, val, ones)
            x = torch.addcmul(x, sixth, k1 + 2 * (k2 + k3) + k4)
            x = x.clamp(-1e5, 1e5)
        if t >= burn:
            out[:, t - burn] = torch.bmm(x[:, None], w)[:, 0].clamp(-6e4, 6e4).half()
    return out


def _flow(x, p, z):
    """Approximate Binomial(x,p) with normal draws clamped to [0,x]."""
    m = x * p
    return torch.minimum(
        torch.addcmul(m, torch.sqrt((m * (1 - p)).clamp_min(0)), z).clamp_min(0), x)


def _epi_bank(gen, n, length, device, burn=512):
    total = burn + length
    t = torch.arange(total, device=device, dtype=torch.float32)[None]
    weekly = _u(gen, (n, 1), device=device) < 0.35
    dt = torch.where(weekly, 7.0, 1.0)
    year = 365.25 / dt
    pop = _logu(gen, (n, 1), 1e4, 1e7, device)
    gamma = 1 / _u(gen, (n, 1), 2.0, 10.0, device)
    sir = _u(gen, (n, 1), device=device) < 0.3
    sigma = torch.where(sir, 50.0, 1 / _u(gen, (n, 1), 1.0, 6.0, device))
    r0 = _u(gen, (n, 1), 1.1, 4.0, device)
    waning = torch.where(_u(gen, (n, 1), device=device) < 0.7,
                         1 / _logu(gen, (n, 1), 60.0, 2000.0, device), 0.0)
    births = torch.where((_u(gen, (n, 1), device=device) < 0.4) | (waning == 0),
                         1 / _logu(gen, (n, 1), 5 * 365.0, 70 * 365.0, device), 0.0)
    season = torch.ones(n, total, device=device)
    for k, (hi, p) in enumerate(((0.45, 1.0), (0.15, 0.5), (0.1, 0.3))):
        a = _u(gen, (n, 1), 0.0, hi, device) * (_u(gen, (n, 1), device=device) < p)
        ph = _u(gen, (n, 1), 0.0, 2 * math.pi, device)
        season = season + a * torch.cos(2 * math.pi * (k + 1) * t / year + ph)
    logm = torch.zeros(n, total, device=device)
    count = torch.randint(0, 5, (n, 1), generator=gen, device=device)
    for e in range(4):
        start = _u(gen, (n, 1), 0.0, float(total), device)
        dur = _logu(gen, (n, 1), 10.0, 600.0, device) / dt
        m = torch.log(_logu(gen, (n, 1), 0.35, 1.6, device))
        logm = logm + m * ((t >= start) & (t < start + dur) & (count > e))
    od = _u(gen, (n, 1), 0.0, 0.4, device)
    noise = torch.exp(od * _randn(gen, (n, total), device) - od**2 / 2)
    beta = (r0 * gamma) * season.clamp_min(0.05) * torch.exp(logm) * noise * dt / pop
    imports = torch.poisson((_logu(gen, (n, 1), 0.01, 3.0, device) * dt)
                            .expand(n, total).contiguous(), generator=gen)
    z = _randn(gen, (n, total, 4), device)
    p_const = 1 - torch.exp(-torch.cat([sigma, gamma, waning], 1) * dt)
    p = torch.cat([torch.zeros(n, 1, device=device), p_const], 1)
    p_birth = 1 - torch.exp(-births * dt)
    s0 = _u(gen, (n, 1), 0.2, 1.0, device)
    i0 = _logu(gen, (n, 1), 1e-5, 1e-2, device)
    x = torch.cat([s0 * (1 - i0), i0 / 2, i0 / 2, (1 - s0) * (1 - i0)], 1) * pop
    inc = torch.empty(n, length, device=device)
    prev = torch.empty(n, length, device=device)
    for step in range(total):
        p[:, 0] = 1 - torch.exp(-beta[:, step] * x[:, 2])
        f = _flow(x, p, z[:, step])
        x = x - f + f.roll(1, 1)
        x = x * (1 - p_birth)
        x[:, 0] += (p_birth * pop)[:, 0]
        x[:, 1] += imports[:, step]
        if step >= burn:
            inc[:, step - burn] = f[:, 1]
            prev[:, step - burn] = x[:, 2]
    from simons.synth.nonstat import _ou_scan

    tt = t[:, :length]
    r_a, r_b = _logu(gen, (n, 1), 0.02, 0.8, device), _logu(gen, (n, 1), 0.02, 0.8, device)
    centre, width = (_u(gen, (n, 1), 0.0, float(length), device),
                     _logu(gen, (n, 1), 5.0, 500.0, device))
    report = r_a + (r_b - r_a) * torch.sigmoid((tt - centre) / width)
    q = _u(gen, (n, 1), math.exp(-0.25), 0.97, device)
    inc = torch.where(torch.isfinite(inc), inc, 0).clamp(0, 1e9)
    delayed = _ou_scan(q.double().expand(n, length).contiguous(),
                       ((1 - q).double() * inc.double()), inc[:, 0].double() * 0).float()
    delayed = torch.where(_u(gen, (n, 1), device=device) < 0.6, delayed, inc)
    wd = torch.exp(_u(gen, (n, 1), 0.0, 0.4, device) * _randn(gen, (n, 7), device))
    wd = torch.where(weekly, 1.0, wd)
    weekday = wd.gather(1, (tt.long() % 7).expand(n, length))
    obs_sd = _u(gen, (n, 1), 0.0, 0.5, device)
    mean = report * delayed * weekday * torch.exp(obs_sd * _randn(gen, (n, length), device)
                                                  - obs_sd**2 / 2)
    cases = torch.poisson(torch.nan_to_num(mean, 0.0, 0.0, 0.0).clamp(0, 1e8), generator=gen)
    occupancy = prev * _logu(gen, (n, 1), 0.001, 0.3, device) * torch.exp(
        0.05 * _randn(gen, (n, length), device))
    return torch.stack([cases, occupancy], -1)


def _hydro_bank(gen, n, length, device, burn=512):
    from simons.synth.heavy import _gamma

    total = burn + length
    t = torch.arange(total, device=device, dtype=torch.float32)[None]
    hourly = _u(gen, (n, 1), device=device) < 0.4
    per_day = torch.where(hourly, 24.0, 1.0)
    year = 365.25 * per_day
    ph = _u(gen, (n, 1), 0.0, 2 * math.pi, device)
    seas = torch.cos(2 * math.pi * t / year + ph)
    p01 = torch.where(hourly, _logu(gen, (n, 1), 0.003, 0.04, device),
                      _logu(gen, (n, 1), 0.03, 0.4, device))
    p11 = torch.where(hourly, _u(gen, (n, 1), 0.6, 0.95, device), _u(gen, (n, 1), 0.2, 0.8, device))
    p01_t = (p01 * (1 + _u(gen, (n, 1), 0.0, 0.8, device) * seas)).clamp(0, 1)
    shape = _u(gen, (n, 1), 0.5, 1.3, device)
    amount = _gamma(shape.expand(n, total).contiguous(), gen) / shape
    amount = amount * torch.where(hourly, _u(gen, (n, 1), 0.5, 3.0, device),
                                  _u(gen, (n, 1), 3.0, 15.0, device))
    amount = amount * (1 + _u(gen, (n, 1), 0.0, 0.5, device) * seas)
    occ = _u(gen, (n, total), device=device)
    snowy = _u(gen, (n, 1), device=device) < 0.35
    temp = (_u(gen, (n, 1), -5.0, 12.0, device) + _u(gen, (n, 1), 5.0, 15.0, device) * seas
            + torch.where(hourly, _u(gen, (n, 1), 0.0, 6.0, device), 0.0)
            * torch.cos(2 * math.pi * t / 24 + _u(gen, (n, 1), 0.0, 6.3, device))
            + _ar1(gen, n, total, device, torch.where(hourly, 0.99, 0.8),
                   torch.full((n, 1), 3.0, device=device)))
    snow_frac = torch.where(snowy, torch.sigmoid(-temp), 0.0)
    melt_rate = _u(gen, (n, 1), 1.0, 5.0, device) / per_day * torch.relu(temp) * snowy
    smax = _logu(gen, (n, 1), 20.0, 300.0, device)
    k = _logu(gen, (n, 1), 2e-3, 0.15, device) / per_day
    a = _u(gen, (n, 1), 1.0, 3.0, device)
    pet = _u(gen, (n, 1), 0.5, 5.0, device) / per_day * (1 + 0.8 * -seas).clamp_min(0)
    pet = pet * torch.where(hourly, (math.pi * torch.sin(2 * math.pi * t / 24)).clamp_min(0), 1.0)
    alpha = torch.where(hourly, _logu(gen, (n, 1), 0.01, 0.3, device),
                        _logu(gen, (n, 1), 0.1, 0.9, device))
    fg = _u(gen, (n, 1), 0.1, 0.7, device)
    g = _logu(gen, (n, 1), 2e-4, 2e-2, device) / per_day * torch.where(hourly, 4.0, 1.0)
    soil = smax[:, 0] * _u(gen, (n,), 0.2, 0.8, device)
    snow, route = torch.zeros(n, device=device), torch.zeros(n, device=device)
    ground = torch.zeros(n, device=device)
    wet = torch.zeros(n, device=device, dtype=torch.bool)
    smax, k, a, alpha, fg, g = (v[:, 0] for v in (smax, k, a, alpha, fg, g))
    flow = torch.empty(n, length, device=device)
    moist = torch.empty(n, length, device=device)
    for step in range(total):
        wet = torch.where(wet, occ[:, step] < p11[:, 0], occ[:, step] < p01_t[:, step])
        rain = amount[:, step] * wet
        sf = snow_frac[:, step]
        melt = torch.minimum(snow, melt_rate[:, step])
        snow = snow + rain * sf - melt
        soil = soil + rain * (1 - sf) + melt
        soil = (soil - pet[:, step] * (soil / smax).clamp(max=1)).clamp_min(0)
        over = torch.relu(soil - smax)
        soil = soil - over
        q = torch.minimum(k * smax * (soil / smax) ** a, soil)
        soil = soil - q
        ground = ground + fg * q - g * ground
        route = route + alpha * ((1 - fg) * q + over - route)
        if step >= burn:
            flow[:, step - burn] = route + g * ground
            moist[:, step - burn] = soil + snow
    return torch.stack([flow + 1e-3 * smax[:, None], moist], -1)


def _thermal_bank(gen, n, length, device, burn=512):
    total = burn + length
    t = torch.arange(total, device=device, dtype=torch.float32)[None]
    fine = _u(gen, (n, 1), device=device) < 0.35
    day = torch.where(fine, 96.0, 24.0)
    year = 365.25 * day
    ambient = (_u(gen, (n, 1), -5.0, 25.0, device)
               + _u(gen, (n, 1), 0.0, 12.0, device)
               * torch.cos(2 * math.pi * t / year + _u(gen, (n, 1), 0.0, 6.3, device))
               + _u(gen, (n, 1), 0.0, 6.0, device)
               * torch.cos(2 * math.pi * t / day + _u(gen, (n, 1), 0.0, 6.3, device))
               + _ar1(gen, n, total, device, torch.where(fine, 0.995, 0.98),
                      _u(gen, (n, 1), 0.5, 3.0, device)))
    hour = 2 * math.pi * t / day
    profile = (1 + _u(gen, (n, 1), 0.0, 0.6, device)
               * torch.cos(hour + _u(gen, (n, 1), 0.0, 6.3, device))
               + _u(gen, (n, 1), 0.0, 0.3, device)
               * torch.cos(2 * hour + _u(gen, (n, 1), 0.0, 6.3, device)))
    weekend = ((t / day).floor() % 7) >= 5
    profile = profile * torch.where(weekend, 1 - _u(gen, (n, 1), 0.0, 0.5, device), 1.0)
    load = (profile * torch.exp(_ar1(gen, n, total, device, torch.where(fine, 0.99, 0.95),
                                     _u(gen, (n, 1), 0.05, 0.4, device)))).clamp_min(0)
    quad = _u(gen, (n, 1), device=device) < 0.5
    heat = torch.where(quad, load**2, load) * _u(gen, (n, 1), 0.02, 0.5, device)
    r12 = _logu(gen, (n, 1), 0.01, 0.3, device)
    r1a = _logu(gen, (n, 1), 0.003, 0.08, device)
    r2a = _logu(gen, (n, 1), 0.001, 0.05, device)
    cap = _logu(gen, (n, 1), 0.1, 1.0, device)
    cooled = _u(gen, (n, 1), device=device) < 0.8
    prop = _u(gen, (n, 1), device=device) < 0.3
    setpoint = ambient.mean(1, keepdim=True) + _u(gen, (n, 1), 2.0, 15.0, device)
    band = _u(gen, (n, 1), 0.3, 3.0, device)
    power = (heat.mean(1, keepdim=True) + r1a * 10) * _u(gen, (n, 1), 1.2, 3.0, device) * cooled
    t1 = ambient[:, 0] + 5
    t2 = ambient[:, 0] + 5
    on = torch.zeros(n, device=device, dtype=torch.bool)
    r12, r1a, r2a, cap, setpoint, band, power, prop = (
        v[:, 0] for v in (r12, r1a, r2a, cap, setpoint, band, power, prop))
    temp = torch.empty(n, length, device=device)
    energy = torch.empty(n, length, device=device)
    for step in range(total):
        on = (t1 > setpoint + band) | (on & (t1 > setpoint - band))
        cool = torch.where(prop, power * ((t1 - setpoint + band) / (2 * band)).clamp(0, 1),
                           power * on)
        amb = ambient[:, step]
        t1, t2 = (t1 + r12 * (t2 - t1) + r1a * (amb - t1) + heat[:, step] - cool,
                  t2 + r12 * cap * (t1 - t2) + r2a * (amb - t2))
        if step >= burn:
            temp[:, step - burn] = t1
            energy[:, step - burn] = cool + heat[:, step]
    return torch.stack([temp, energy], -1)


def queue_family(gen, n, length, device):
    """Return float32 fluid-queue observables [n,length] on `device`, drawn from `gen`."""
    t = torch.arange(length, device=device, dtype=torch.float32)[None]
    day = torch.tensor([288.0, 96.0, 24.0], device=device)[
        torch.randint(0, 3, (n, 1), generator=gen, device=device)]
    hour = 2 * math.pi * (t + _u(gen, (n, 1), 0.0, 1e4, device)) / day
    logp = (_u(gen, (n, 1), 0.0, 1.0, device) * torch.cos(hour)
            + _u(gen, (n, 1), 0.0, 0.4, device)
            * torch.cos(2 * hour + _u(gen, (n, 1), 0.0, 6.3, device))
            - _u(gen, (n, 1), 0.0, 0.6, device) * ((((t / day).floor()) % 7) >= 5)
            + _ar1(gen, n, length, device, _u(gen, (n, 1), 0.9, 0.999, device),
                   _u(gen, (n, 1), 0.02, 0.3, device)))
    hazard = _logu(gen, (n, 1), 1e-3, 3e-2, device)
    regime = ((_u(gen, (n, length), device=device) < hazard).long().cumsum(-1) % 2).bool()
    regime = regime & (_u(gen, (n, 1), device=device) < 0.6)
    logp = logp + torch.log(_logu(gen, (n, 1), 1.2, 3.0, device)) * regime
    profile = torch.exp(logp - logp.mean(1, keepdim=True))
    cap = _logu(gen, (n, 1), 5.0, 500.0, device)
    dip = ((_u(gen, (n, length), device=device) < 5e-4).long().cumsum(-1) % 2).bool()
    dip = dip & (_u(gen, (n, 1), device=device) < 0.3)
    cap_t = cap * torch.where(dip, _u(gen, (n, 1), 0.3, 0.8, device), 1.0)
    rho = _u(gen, (n, 1), 0.5, 1.1, device)
    arrivals = torch.poisson((cap * rho * profile).clamp(0, 1e6), generator=gen)
    s = (arrivals - cap_t).double().cumsum(-1)
    backlog = (s - s.cummin(-1).values.clamp(max=0)).float()
    before = torch.cat([torch.zeros(n, 1, device=device), backlog[:, :-1]], 1)
    served = torch.minimum(before + arrivals, cap_t)
    util = served / cap_t
    latency = (backlog / cap_t + 1) * torch.exp(0.1 * _randn(gen, (n, length), device))
    occ = ((arrivals + backlog) / (cap * _u(gen, (n, 1), 1.5, 3.0, device))).clamp(0.01, 0.98)
    crit = _u(gen, (n, 1), 0.2, 0.5, device)
    speed = (torch.where(occ < crit, 1.0, crit / (1 - crit) * (1 - occ) / occ)
             * _u(gen, (n, 1), 40, 120, device))
    kind = torch.randint(0, 5, (n, 1), generator=gen, device=device)
    out = torch.where(kind == 0, util, torch.where(kind == 1, backlog, torch.where(
        kind == 2, latency, torch.where(kind == 3, served, speed))))
    return out


SIMULATORS = {"chaos": _chaos_bank, "epi": _epi_bank, "hydro": _hydro_bank,
              "thermal": _thermal_bank}


def _valid(x):
    x0 = x[..., 0].float()
    ok = torch.isfinite(x0).all(1) & (x0.abs().amax(1) < 6e4)
    x0 = torch.where(torch.isfinite(x0), x0, 0)
    tail = x0[:, x0.shape[1] // 2 :].std(1)
    return ok & (tail > 0.05 * x0.std(1)) & (tail > 1e-6 * (x0.abs().mean(1) + 1e-6))


def _simulate(name, gen, n, device):
    spare = n + max(1, int(n * SPARE))
    with torch.no_grad(), torch.autocast(device_type=torch.device(device).type, enabled=False):
        previous = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            x = SIMULATORS[name](gen, spare, BANK_LENGTH, device)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous
        ok = _valid(x)
        order = torch.argsort((~ok).to(torch.int8), stable=True)[:n]
        x = x[order].float()
        x = torch.where(torch.isfinite(x), x, 0).clamp(-6e4, 6e4)
        rms = x.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-12)
        x = (x / rms).clamp(-6e4, 6e4).half()
    return x


def _bank(name, gen, device):
    key = (name, torch.device(device).type)  # cuda and cuda:0 must share one bank.
    state = _STATE.get(key)
    if state is None:
        state = _STATE[key] = {"bank": _simulate(name, gen, BANK[name], device), "calls": 0,
                               "next": 0}
    state["calls"] += 1
    if REFRESH > 0 and state["calls"] % REFRESH == 0:
        bank = state["bank"]
        part = max(1, len(bank) // PART)
        lo = state["next"] * part
        bank[lo : lo + part] = _simulate(name, gen, part, device)
        state["next"] = (state["next"] + 1) % PART
    return state["bank"]


def _crop(gen, bank, n, length, device):
    rows, size, channels = bank.shape
    if length > 2 * size:
        raise ValueError(f"{length}-step series need mech_length >= {(length + 1) // 2} (the bank "
                         f"has {size} steps)")
    span = min(length, size)
    idx = torch.randint(0, rows, (n, 1), generator=gen, device=device)
    ch = torch.randint(0, channels, (n, 1), generator=gen, device=device)
    max_stride = max(1, min(4, size // span))
    stride = torch.randint(1, max_stride + 1, (n, 1), generator=gen, device=device)
    pool = _u(gen, (n, 1), device=device) < 0.5
    room = (size - stride * span).clamp_min(0)
    offset = (_u(gen, (n, 1), device=device) * (room + 1)).long().clamp(max=size - 1)
    pos = offset + stride * torch.arange(span, device=device)[None]
    x = torch.zeros(n, span, device=device)
    for j in range(max_stride):
        use = torch.where(pool, (j < stride).float() / stride, float(j == 0))
        p = (pos + j).clamp(max=size - 1)
        x = x + use * bank[idx, p, ch].float()
    if span < length:
        x = torch.cat([x.flip(-1), x], -1)[:, -length:]
    return x


def _observe(gen, x, nonlinear):
    n, length = x.shape
    device = x.device
    z = _standardise(x)
    if nonlinear > 0:
        kind = torch.randint(0, 3, (n, 1), generator=gen, device=device)
        a = _u(gen, (n, 1), 0.3, 1.0, device)
        y = torch.where(kind == 0, torch.exp(a * z),
                        torch.where(kind == 1, torch.tanh(a * z),
                                    torch.nn.functional.softplus(2 * a * z)))
        z = torch.where(_u(gen, (n, 1), device=device) < nonlinear, _standardise(y), z)
    level = _u(gen, (n, 1), 0.0, 0.1, device) * (_u(gen, (n, 1), device=device) < 0.7)
    eps = _randn(gen, (n, length), device)
    mult = _u(gen, (n, 1), device=device) < 0.3
    base = x / x.std(1, keepdim=True).clamp_min(1e-6)
    out = torch.where(mult, base * (1 + level * eps), z + level * eps)
    t = torch.linspace(0, 1, length, device=device)[None]
    trend = (_u(gen, (n, 1), -1.0, 1.0, device) * t
             + _u(gen, (n, 1), 0.0, 0.5, device)
             * torch.sin(2 * math.pi * t * _u(gen, (n, 1), 0.2, 1.5, device)))
    return out + trend * (_u(gen, (n, 1), device=device) < 0.2)


def _maker(name):

    def make(gen, n, length, device):
        """Draw float32 [n,length] family series; cached banks support length <= 2 * BANK_LENGTH."""
        if name == "queue":
            x = queue_family(gen, n, length, device)
            x = _observe(gen, torch.where(torch.isfinite(x), x, 0), 0.0)
        else:
            x = _observe(gen, _crop(gen, _bank(name, gen, device), n, length, device),
                         0.25 if name == "chaos" else 0.0)
        return torch.where(torch.isfinite(x), x, 0)

    return make


NAMES = ("chaos", "epi", "hydro", "queue", "thermal")
MAKERS = {k: _maker(k) for k in NAMES}
