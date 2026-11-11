"""Nonlinear causal DAGs, beat layers, AM/FM waves and multiplicative calendar priors."""

import math

import torch

from simons.synth.base import (
    PERIOD_BANK,
    _bank,
    _categorical,
    _choice,
    _logu,
    _periods,
    _randn,
    _spectral,
    _standardise,
    _u,
    kernel_family,
)
from simons.synth.nonstat import _ou_scan, _trend

MAX_NODES = 8
MAX_ROOTS = 4


def _slow(gen, n, length, device, lo=0.02, hi=0.5):
    scale = _logu(gen, (n, 1), lo, hi, device) * length
    return _spectral(gen, length, device, lambda w: torch.exp(-((w[None] * scale) ** 2) / 2))


def cauker_family(gen, n, length, device):
    """Return float32 causal-DAG series [n,length]; replace constant outputs with a root."""
    t = torch.arange(length, device=device, dtype=torch.float32)[None] / length
    roots = _standardise(kernel_family(gen, MAX_ROOTS * n, length, device, max_components=3))
    roots = roots.view(MAX_ROOTS, n, length)
    m = MAX_ROOTS * n
    kind = torch.randint(0, 3, (m, 1), generator=gen, device=device)
    use = _u(gen, (m, 1), device=device) < 0.5
    a, b = _randn(gen, (m, 1), device), _randn(gen, (m, 1), device)
    linear = a * 2 * (t - 0.5) + 0.5 * b
    rate = _u(gen, (m, 1), -2.0, 2.0, device)
    expo = a * torch.exp(rate * t)
    expo = expo - expo.mean(-1, keepdim=True)
    impulses = ((_u(gen, (m, length), device=device) < 0.002)
                * _u(gen, (m, length), -5.0, 5.0, device))
    mean = torch.where(kind == 0, linear, torch.where(kind == 1, expo, impulses))
    roots = roots + (use * mean).view(MAX_ROOTS, n, length)
    count = torch.randint(2, MAX_ROOTS + 1, (n, 1), generator=gen, device=device)
    nodes = []
    for v in range(MAX_NODES):
        root = roots[v] if v < MAX_ROOTS else None
        if v < 2:
            nodes.append(root)
            continue
        stack = torch.stack(nodes, 1)  # [n, v, L]
        parents = torch.randint(1, 4, (n, 1), generator=gen, device=device)
        index = (_u(gen, (n, 3), device=device) * v).long().clamp_max(v - 1)
        valid = (torch.arange(3, device=device)[None] < parents).float()
        weight = _randn(gen, (n, 3), device) * valid / parents.float().sqrt()
        picked = stack.gather(1, index[..., None].expand(n, 3, length))
        u = (weight[..., None] * picked).sum(1) + _randn(gen, (n, 1), device)
        act = _categorical(gen, (0.2, 0.25, 0.15, 0.2, 0.15, 0.05), n, device)[:, None]
        slope = _u(gen, (n, 1), 0.01, 0.3, device)
        k = _logu(gen, (n, 1), 0.5, 3.0, device)
        omega = _logu(gen, (n, 1), 0.5, 3.0, device)
        modulus = _u(gen, (n, 1), 1.0, 5.0, device)
        z = torch.where(act == 0, u,
            torch.where(act == 1, u.clamp_min(0),
            torch.where(act == 2, torch.where(u > 0, u, slope * u),
            torch.where(act == 3, torch.sigmoid(k * u),
            torch.where(act == 4, torch.sin(omega * u), torch.remainder(u, modulus))))))
        z = _standardise(z)
        if root is not None:
            z = torch.where(v < count, root, z)
        nodes.append(z)
    stack = torch.stack(nodes, 1)
    pick = count + (_u(gen, (n, 1), device=device) * (MAX_NODES - count)).long()
    pick = pick.clamp_max(MAX_NODES - 1)
    out = stack.gather(1, pick[..., None].expand(n, 1, length))[:, 0]
    out = torch.where(out.std(-1, keepdim=True) < 1e-3, roots[0], out)
    noise = _logu(gen, (n, 1), 0.003, 0.1, device) * (_u(gen, (n, 1), device=device) < 0.5)
    return out + noise * _randn(gen, (n, length), device)


def rhythm_family(gen, n, length, device, layers=5):
    """Return float32 polyrhythms [n,length]; without drift/noise, pre-beat values are zero."""
    tf = torch.arange(length, device=device, dtype=torch.float32)[None]
    bank = _choice(gen, PERIOD_BANK, (n, 1), device)
    period = torch.where(_u(gen, (n, 1), device=device) < 0.6, bank,
                         _logu(gen, (n, 1), 4.0, 400.0, device))
    count = torch.randint(3, layers + 1, (n, 1), generator=gen, device=device)
    m = n * layers
    p = period.repeat(layers, 1)  # [layers*n, 1], layer-major
    k = _choice(gen, (1.0, 2.0, 3.0, 4.0, 6.0, 8.0), (m, 1), device)
    spacing = (p / k).clamp_min(2.0)
    offset = _u(gen, (m, 1), device=device) * spacing
    beat = torch.floor((tf - offset) / spacing)
    fired = beat > torch.floor((tf - 1 - offset) / spacing)
    q = _u(gen, (m, 1), 0.3, 1.0, device)
    fired = fired & (_u(gen, (m, length), device=device) < q)
    amp = _logu(gen, (m, 1), 0.2, 5.0, device)
    jitter = torch.exp(0.25 * _randn(gen, (m, length), device)
                       * (_u(gen, (m, 1), device=device) < 0.5))
    impulse = fired * amp * jitter
    tau = torch.exp(_u(gen, (m, 1), 0.0, 1.0, device) * torch.log(spacing.clamp_min(1.0)))
    tau = tau.clamp_min(1.0)  # e^(-1/tau) >= e^-1 keeps the chunked scan in range.
    decay = torch.exp(-1.0 / tau).expand(m, length).double()
    env = _ou_scan(decay, impulse.double(), torch.zeros(m, device=device, dtype=torch.float64))
    env = env.float().view(layers, n, length)
    active = (torch.arange(layers, device=device)[:, None, None] < count[None])
    y = (env * active).sum(0)
    carrier = 1 + 0.5 * torch.sin(2 * math.pi * tf
                                  / (period * _choice(gen, (2.0, 4.0, 7.0, 12.0), (n, 1), device))
                                  + _u(gen, (n, 1), 0, 2 * math.pi, device))
    y = torch.where(_u(gen, (n, 1), device=device) < 0.3, y * carrier, y)
    scale = y.std(-1, keepdim=True).clamp_min(1e-3)
    drift = torch.where(_u(gen, (n, 1), device=device) < 0.5, _trend(gen, n, length, device),
                        _slow(gen, n, length, device))
    y = (y + (_u(gen, (n, 1), device=device) < 0.4) * _u(gen, (n, 1), 0.1, 1.0, device) * scale
         * drift)
    noisy = _u(gen, (n, 1), device=device) < 0.6
    y = (y + noisy * _logu(gen, (n, 1), 0.003, 0.2, device) * scale
         * _randn(gen, (n, length), device))
    return y


def wave_family(gen, n, length, device, components=3):
    """Return float32 AM/FM waveform mixtures [n,length] on `device`, drawn from `gen`."""
    m = n * components
    tf = torch.arange(length, device=device, dtype=torch.float32)[None]
    period = _periods(gen, m, device)[:, None].clamp_min(3.0)
    delta = _u(gen, (m, 1), 0.0, 0.3, device)
    eps = _u(gen, (m, 1), 0.0, 0.5, device)
    freq = torch.exp(delta * _slow(gen, m, length, device)) / period
    cycles = torch.remainder(freq.cumsum(-1) + _u(gen, (m, 1), device=device), 1.0)
    phase = 2 * math.pi * cycles
    shape = torch.randint(0, 4, (m, 1), generator=gen, device=device)
    kappa = _logu(gen, (m, 1), 1.0, 10.0, device)
    sine = torch.sin(phase)
    square = torch.tanh(kappa * sine) / torch.tanh(kappa)
    triangle = 2 / math.pi * torch.asin(sine.clamp(-1, 1))
    saw = 2 * cycles - 1
    wave = torch.where(shape == 0, sine, torch.where(shape == 1, square,
                       torch.where(shape == 2, triangle, saw)))
    amplitude = (_logu(gen, (m, 1), 0.2, 2.0, device)
                 * torch.exp(eps * _slow(gen, m, length, device)))
    count = torch.randint(1, components + 1, (n, 1), generator=gen, device=device)
    active = (torch.arange(components, device=device)[:, None, None] < count[None])
    y = ((amplitude * wave).view(components, n, length) * active).sum(0)
    trend = _randn(gen, (n, 1), device) * (tf / length - 0.5) * 2
    y = y + (_u(gen, (n, 1), device=device) < 0.5) * trend
    return y + _logu(gen, (n, 1), 0.005, 0.3, device) * _randn(gen, (n, length), device)


_GRANULARITY = ((24.0, 168.0), (7.0, 365.25), (52.18, 0.0), (12.0, 0.0), (24.0, 0.0), (7.0, 0.0))
_FLAT = tuple(x for pair in _GRANULARITY for x in pair)


def fpfn_family(gen, n, length, device, harmonics=3):
    """Return float32 positive trend * seasonal * (1 + noise) series [n,length]."""
    tf = torch.arange(length, device=device, dtype=torch.float32)[None]
    s = tf / length
    g = torch.randint(0, len(_GRANULARITY), (n,), generator=gen, device=device)
    periods = _bank(_FLAT, device).view(-1, 2)[g]  # [n, 2]
    m_l = _randn(gen, (n, 1), device).clamp_min(-0.9)
    log_me = _randn(gen, (n, 1), device) * 0.7
    trend = (1 + m_l * s) * torch.exp(log_me * s)
    trend = torch.where(_u(gen, (n, 1), device=device) < 0.2, torch.ones_like(trend), trend)
    season = torch.ones((n, length), device=device)
    start = _u(gen, (n, 1), 0.0, 1000.0, device)
    for j in range(2):
        p = periods[:, j : j + 1]
        used = p > 0
        p = p.clamp_min(1.0)
        level = _u(gen, (n, 1), 0.0, 0.4, device)
        acc = torch.zeros((n, length), device=device)
        for f in range(1, harmonics + 1):
            a = _randn(gen, (n, 1), device) * level / f
            b = _randn(gen, (n, 1), device) * level / f
            arg = 2 * math.pi * f * torch.remainder((tf + start) / p, 1.0)
            acc = acc + a * torch.sin(arg) + b * torch.cos(arg)
        season = season * torch.where(used, (1 + acc).clamp_min(0.05), 1.0)
    shape = _u(gen, (n, 1), 1.0, 5.0, device)
    u = _u(gen, (n, length), 1e-6, 1 - 1e-6, device)
    weibull = (-torch.log1p(-u)) ** (1 / shape)
    mean = torch.exp(torch.lgamma(1 + 1 / shape))
    level = _logu(gen, (n, 1), 0.01, 0.3, device)
    noise = (1 + level * (weibull - mean)).clamp_min(0.01)
    return trend * season * noise


MAKERS = {"cauker": cauker_family, "rhythm": rhythm_family, "wave": wave_family,
          "fpfn": fpfn_family}
