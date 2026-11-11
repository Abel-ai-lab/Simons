"""Heavy-tailed walks, innovations ETS and intermittent pulses, batched on device."""

import torch

from simons.synth.base import (
    PERIOD_BANK,
    _categorical,
    _choice,
    _logu,
    _periodic,
    _randn,
    _seasonal_cumsum,
    _smooth,
    _standardise,
    _u,
)
from simons.synth.nonstat import _laplace, _ou_scan

ETS_BANK = (4, 7, 12, 24, 48, 52, 96, 168, 288, 336, 365, 672)
ETS_MULT = (7, 4, 12, 52)
CLIP = 50.0


def _gamma(alpha, gen):
    """Draw Gamma(alpha,1) with fixed rejection rounds, boosting alpha below 1."""
    small = alpha < 1
    a = torch.where(small, alpha + 1, alpha)
    d = a - 1.0 / 3.0
    c = 1.0 / torch.sqrt(9.0 * d)
    out = d.clone()
    done = torch.zeros_like(a, dtype=torch.bool)
    for _ in range(6):
        z = torch.randn(a.shape, generator=gen, device=a.device)
        u = torch.rand(a.shape, generator=gen, device=a.device)
        v = (1 + c * z) ** 3
        ok = (v > 0) & (torch.log(u.clamp_min(1e-12))
                        < 0.5 * z**2 + d - d * v + d * torch.log(v.clamp_min(1e-12)))
        out = torch.where(ok & ~done, d * v, out)
        done |= ok
    boost = torch.rand(a.shape, generator=gen, device=a.device).clamp_min(1e-12) ** (
        1 / alpha.clamp_min(1e-3))
    return torch.where(small, out * boost, out)


def _student(gen, shape, nu, device):
    """Draw unit-variance Student-t values [n,L] with per-row degrees of freedom nu [n,1] > 2."""
    half = (nu / 2).expand(shape).contiguous()
    g = _gamma(half, gen) / half
    z = _randn(gen, shape, device) / g.clamp_min(1e-4).sqrt()
    return z * ((nu - 2) / nu).sqrt()


def _pareto(gen, shape, alpha, device):
    """Draw Pareto(alpha) on [1,inf), flooring uniforms to bound values near 1e4."""
    u = _u(gen, shape, device=device).clamp_min(1e-6)
    return u ** (-1 / alpha)


def _segments(gen, n, length, device, max_breaks):
    """Return segment indices [n,L] from random break times."""
    count = torch.randint(0, max_breaks + 1, (n, 1), generator=gen, device=device)
    positions = torch.randint(1, length, (n, max_breaks), generator=gen, device=device)
    valid = torch.arange(max_breaks, device=device)[None] < count
    positions = torch.where(valid, positions, 0)
    marks = torch.zeros((n, length), device=device).scatter_(1, positions, 1.0)
    marks[:, 0] = 1.0
    return marks.long().cumsum(-1) - 1


def _finish(x):
    x = torch.where(torch.isfinite(x), x, 0.0)
    return _standardise(x).clamp(-CLIP, CLIP)


def levy_family(gen, n, length, device):
    """Return float32 heavy-tailed random walks [n,length], standardized and clipped to +/-CLIP."""
    use = lambda p: _u(gen, (n, 1), device=device) < p  # noqa: E731
    nu = _u(gen, (n, 1), 2.5, 8.0, device)
    eps = _student(gen, (n, length), nu, device)
    persist = 1 - _logu(gen, (n, 1), 1e-3, 0.05, device)
    sv_sd = _u(gen, (n, 1), 0.2, 1.0, device)
    sv_drive = sv_sd * (1 - persist**2).sqrt() * _randn(gen, (n, length), device)
    garch = use(0.3)
    beta = _u(gen, (n, 1), 0.9, 0.995, device)
    alpha = _u(gen, (n, 1), 0.05, 0.25, device)
    lagged = torch.cat([torch.zeros_like(eps[:, :1]), eps[:, :-1]], -1)
    g_drive = alpha * (lagged.abs() - lagged.abs().mean(-1, keepdim=True))
    a = torch.where(garch, beta, persist).expand(n, length).double()
    drive = torch.where(garch, g_drive, sv_drive).double()
    h0 = (sv_sd * _randn(gen, (n, 1), device))[:, 0].double() * (~garch[:, 0])
    log_sigma = _ou_scan(a, drive, h0).float().clamp(-4, 4)
    sigma = torch.exp(log_sigma - log_sigma.mean(-1, keepdim=True))
    seg = _segments(gen, n, length, device, 3)
    scale = _logu(gen, (n, 1), 0.01, 0.5, device) * use(0.85)
    rho = _u(gen, (n, 1), -0.5, 1.0, device)
    mu = [_randn(gen, (n, 1), device) * scale]
    for _ in range(3):
        mu.append(rho * mu[-1] + _randn(gen, (n, 1), device) * scale)
    drift = torch.cat(mu, -1).gather(1, seg)
    rate = _logu(gen, (n, 1), 1e-4, 1e-2, device) * use(0.6)
    hits = _u(gen, (n, length), device=device) < rate
    size = _logu(gen, (n, 1), 2.0, 20.0, device)
    tail = _u(gen, (n, 1), 1.5, 3.0, device)
    sign = torch.where(_u(gen, (n, length), device=device) < 0.5, -1.0, 1.0)
    pareto = sign * _pareto(gen, (n, length), tail, device).clamp_max(100.0)
    magnitude = torch.where(use(0.5), _laplace(gen, (n, length), device), pareto)
    jumps = hits * size * magnitude
    increments = drift + sigma * eps + jumps
    revert = use(0.3)
    a_mr = torch.where(revert, _u(gen, (n, 1), 0.98, 1.0, device), 1.0).expand(n, length)
    y = _ou_scan(a_mr.double(), increments.double(),
                 torch.zeros(n, device=device, dtype=torch.float64)).float()
    unit = sigma.mean(-1, keepdim=True)
    y = (y + use(0.3) * _logu(gen, (n, 1), 0.1, 2.0, device) * unit
         * _randn(gen, (n, length), device))
    period = _choice(gen, PERIOD_BANK, (n,), device)
    y = (y + use(0.5) * _logu(gen, (n, 1), 0.3, 10.0, device) * unit
         * _periodic(gen, period, length, device))
    logspace = use(0.3)
    span = _u(gen, (n, 1), 0.2, 1.5, device)
    z = _standardise(torch.where(torch.isfinite(y), y, 0.0))
    y = torch.where(logspace, torch.exp((span * z).clamp(-20, 20)), y)
    return _finish(y)


def ets_family(gen, n, length, device):
    """Return float32 innovations ETS paths [n,length], standardized and clipped to +/-CLIP."""
    use = lambda p: _u(gen, (n, 1), device=device) < p  # noqa: E731
    sigma = _logu(gen, (n, 1), 0.005, 0.1, device)
    heavy = use(0.3)
    nu = _u(gen, (n, 1), 3.0, 10.0, device)
    e = torch.where(heavy, _student(gen, (n, length), nu, device), _randn(gen, (n, length), device))
    e = sigma * e
    alpha = _u(gen, (n, 1), 0.02, 0.5, device)
    beta = alpha * _u(gen, (n, 1), 0.01, 0.3, device)
    gamma = _u(gen, (n, 1), 0.001, 0.3, device) * (1 - alpha)
    trend = _categorical(gen, (0.35, 0.25, 0.4), n, device)[:, None]
    phi = torch.where(trend == 2, _u(gen, (n, 1), 0.8, 0.98, device), 1.0)
    beta = torch.where(trend == 0, 0.0, beta)
    b0 = _randn(gen, (n, 1), device) * sigma * _logu(gen, (n, 1), 0.01, 1.0, device)
    b0 = torch.where(trend == 0, 0.0, b0)
    b = _ou_scan(phi.expand(n, length).double(), (beta * e).double(), b0[:, 0].double()).float()
    b_prev = torch.cat([b0, b[:, :-1]], -1)
    level = (phi * b_prev + alpha * e).cumsum(-1)
    seasonal = _categorical(gen, (0.3, 0.7), n, device)[:, None] > 0
    m1 = _choice(gen, ETS_BANK, (n,), device)
    m2 = (m1 * _choice(gen, ETS_MULT, (n,), device))
    m2 = torch.where(m2 > 700, m1, m2)
    second = use(0.35) & (m2 != m1)[:, None]
    amp1 = _logu(gen, (n, 1), 0.02, 0.6, device)
    amp2 = _logu(gen, (n, 1), 0.02, 0.4, device)
    share = torch.where(second, _u(gen, (n, 1), 0.3, 0.7, device), 1.0)
    s1 = amp1 * _periodic(gen, m1, length, device) + _seasonal_cumsum(
        share * gamma * e, m1.long(), seasonal[:, 0], ETS_BANK)
    s2 = amp2 * _periodic(gen, m2, length, device) + _seasonal_cumsum(
        (1 - share) * gamma * e, m2.long(), (seasonal & second)[:, 0], ETS_BANK)
    s = seasonal * (s1 + second * s2)
    # Remove the current level innovation before forming the observation.
    y = level - alpha * e + s + e
    mult = use(0.45)
    span = _logu(gen, (n, 1), 0.1, 1.0, device)
    log_y = span * _standardise(torch.where(torch.isfinite(y), y, 0.0))
    y = torch.where(mult, torch.exp(log_y.clamp(-20, 20)), y)
    return _finish(y)


def _occurrence(gen, n, length, device):
    """Draw a two-state Markov on/off chain as alternating geometric runs [n,L]."""
    mean_on = _logu(gen, (n, 1), 1.0, 30.0, device)
    mean_off = _logu(gen, (n, 1), 1.0, 300.0, device)
    k = torch.arange(length, device=device)[None]
    start_on = torch.randint(0, 2, (n, 1), generator=gen, device=device)
    is_on = (k + start_on) % 2 == 0
    mean = torch.where(is_on, mean_on, mean_off)
    p_end = (1 / mean).clamp(1e-4, 1.0)
    u = _u(gen, (n, length), device=device).clamp(1e-7, 1 - 1e-7)
    duration = torch.ceil(torch.log(u) / torch.log1p(-p_end.clamp_max(1 - 1e-6))).clamp(1, length)
    duration = torch.where(p_end >= 1.0, 1.0, duration)
    ends = duration.cumsum(-1).clamp_max(length).long()
    marks = torch.zeros((n, length + 1), device=device).scatter_add_(
        1, ends, torch.ones_like(duration))[:, :length]
    run = marks.cumsum(-1).long()
    return (run + start_on) % 2 == 0


def pulse_family(gen, n, length, device):
    """Return float32 heavy-tailed pulses [n,length], standardized and clipped to +/-CLIP."""
    use = lambda p: _u(gen, (n, 1), device=device) < p  # noqa: E731
    t = torch.arange(length, device=device, dtype=torch.float32)[None]
    on = _occurrence(gen, n, length, device)
    period = _choice(gen, PERIOD_BANK, (n,), device)
    logit = _u(gen, (n, 1), -2.0, 2.0, device) + use(0.5) * _u(gen, (n, 1), 0.5, 3.0, device) * \
        _periodic(gen, period, length, device)
    occur = on & (_u(gen, (n, length), device=device) < torch.sigmoid(logit))
    kind = _categorical(gen, (0.4, 0.3, 0.3), n, device)[:, None]
    shape = _logu(gen, (n, 1), 0.3, 3.0, device)
    gam = _gamma(shape.expand(n, length).contiguous(), gen) / shape
    logn = torch.exp(_u(gen, (n, 1), 0.3, 1.5, device) * _randn(gen, (n, length), device))
    par = _pareto(gen, (n, length), _u(gen, (n, 1), 1.5, 3.0, device), device).clamp_max(1e3)
    amount = torch.where(kind == 0, gam, torch.where(kind == 1, logn, par))
    envelope = torch.exp(use(0.5) * _u(gen, (n, 1), 0.2, 1.0, device)
                         * _smooth(gen, n, length, device))
    x = occur * amount * envelope
    sub = _categorical(gen, (0.45, 0.35, 0.2), n, device)[:, None]
    lam = _logu(gen, (n, 1), 0.3, 20.0, device)
    counts = torch.poisson(x * lam, generator=gen)
    raw = torch.where(use(0.35), counts, x)
    a1 = _u(gen, (n, 1), 0.5, 0.995, device)
    a2 = _u(gen, (n, 1), 0.5, 0.995, device)
    zero = torch.zeros(n, device=device, dtype=torch.float64)
    q1 = _ou_scan(a1.expand(n, length).double(), ((1 - a1) * x).double(), zero)
    q2 = _ou_scan(a2.expand(n, length).double(), ((1 - a2) * q1.float()).double(), zero)
    both = use(0.5)
    w = _u(gen, (n, 1), 0.3, 0.9, device)
    q = torch.where(both, (w * q1.float() + (1 - w) * q2.float()), q1.float())
    mean_q = q.mean(-1, keepdim=True)
    base = mean_q * _logu(gen, (n, 1), 0.01, 2.0, device) * (
        1 + use(0.5) * 0.5 * _periodic(gen, _choice(gen, PERIOD_BANK, (n,), device), length, device)
    ).clamp_min(0)
    reservoir = (q + base + mean_q * _logu(gen, (n, 1), 1e-3, 0.05, device)
                 * _randn(gen, (n, length), device))
    total = x.cumsum(-1)
    exclusive = total - x
    reset = _choice(gen, PERIOD_BANK, (n,), device)[:, None]
    start = (torch.floor(t / reset) * reset).long().clamp_max(length - 1)
    within = total - exclusive.gather(1, start)
    capacity = total[:, -1:].clamp_min(1e-6) / _logu(gen, (n, 1), 2.0, 50.0, device)
    inventory = capacity - torch.remainder(total, capacity)
    accumulated = torch.where(use(0.5), within, inventory)
    out = torch.where(sub == 0, raw, torch.where(sub == 1, reservoir, accumulated))
    return _finish(out)


MAKERS = {"levy": levy_family, "ets": ets_family, "pulse": pulse_family}
