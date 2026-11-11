"""Non-stationary OU paths, changepoints and spike priors, batched on device."""

import math

import torch
import torch.nn.functional as F

from simons.synth.base import (
    PERIOD_BANK,
    _categorical,
    _choice,
    _logu,
    _periodic,
    _randn,
    _smooth,
    _spectral,
    _standardise,
    _u,
)

CHUNK = 256


def _laplace(gen, shape, device):
    u = _u(gen, shape, -0.5 + 1e-6, 0.5 - 1e-6, device)
    return -torch.sign(u) * torch.log1p(-2 * u.abs())


def _trend(gen, n, length, device):
    s = torch.arange(length, device=device, dtype=torch.float32)[None] / length
    kind = torch.randint(0, 4, (n, 1), generator=gen, device=device)
    c1, c2 = _randn(gen, (n, 1), device), _randn(gen, (n, 1), device)
    poly = c1 * (s - 0.5) * 2 + c2 * ((s - 0.5) * 2) ** 2
    centre, rate = _u(gen, (n, 1), 0.1, 0.9, device), _logu(gen, (n, 1), 5.0, 50.0, device)
    logistic = 2 * c1 * torch.sigmoid(rate * (s - centre))
    cycles = _logu(gen, (n, 1), 0.3, 3.0, device)
    sine = c1 * torch.sin(2 * math.pi * cycles * s + _u(gen, (n, 1), 0, 2 * math.pi, device))
    knot = _u(gen, (n, 1), 0.1, 0.9, device)
    piece = c1 * s + (c2 - c1) * (s - knot).clamp_min(0)
    return torch.where(kind == 0, poly, torch.where(kind == 1, logistic,
                       torch.where(kind == 2, sine, piece)))


def _ou_scan(a, drive, y0):
    """Solve y_t = a_t y_{t-1} + drive_t from y_{-1} = y0 in float64 chunks."""
    n, length = a.shape
    pad = (-length) % CHUNK
    a = F.pad(a, (0, pad), value=1.0).view(n, -1, CHUNK)
    drive = F.pad(drive, (0, pad)).view(n, -1, CHUNK)
    # Chunked exp(-log_p) stays finite in float64 for a >= e^-2.7; callers use a >= e^-1.
    log_p = torch.log(a).cumsum(-1)
    p = torch.exp(log_p)
    local = p * (drive * torch.exp(-log_p)).cumsum(-1)
    carry = y0
    starts = []
    for c in range(a.shape[1]):
        starts.append(carry)
        carry = local[:, c, -1] + p[:, c, -1] * carry
    start = torch.stack(starts, 1)[..., None]
    return (local + p * start).flatten(1)[:, :length]


def sde_family(gen, n, length, device):
    """Return float32 regime-switching OU paths [n,length] with varying drift and volatility."""
    t = torch.arange(length, device=device, dtype=torch.float32)[None]
    hazard = _logu(gen, (n, 1), 1 / 8192, 1 / 256, device)
    switches = (_u(gen, (n, length), device=device) < hazard).long().cumsum(-1)
    first = torch.randint(0, 2, (n, 1), generator=gen, device=device)
    regime = ((switches + first) % 2).bool()
    theta = torch.where(regime, _u(gen, (n, 1), 0.05, 0.5, device),
                        _u(gen, (n, 1), 1.0, 5.0, device))
    side = torch.where(_u(gen, (n, 1), device=device) < 0.5, -1.0, 1.0)
    mu0 = side * 2 + _randn(gen, (n, 1), device)
    mu1 = -side * 2 + _randn(gen, (n, 1), device)
    mu = torch.where(regime, mu1, mu0)
    s0 = 0.3 * torch.exp(0.5 * _randn(gen, (n, 1), device))
    s1 = 1.5 * torch.exp(0.5 * _randn(gen, (n, 1), device))
    sigma = torch.where(regime, s1, s0)
    use = lambda p: _u(gen, (n, 1), device=device) < p  # noqa: E731
    mu = mu + use(0.7) * 2 * _trend(gen, n, length, device)
    period = _choice(gen, PERIOD_BANK, (n,), device)
    mu = (mu + use(0.6) * _logu(gen, (n, 1), 0.2, 2.0, device)
          * _periodic(gen, period, length, device))
    theta = theta * torch.exp(use(0.2) * _randn(gen, (n, 1), device) * (t / length - 0.5) * 2)
    sigma = sigma * torch.exp(use(0.3) * 0.7 * _trend(gen, n, length, device).clamp(-2, 2))
    period2 = _choice(gen, PERIOD_BANK, (n,), device)
    phase = _u(gen, (n, 1), 0, 2 * math.pi, device)
    sigma = sigma * torch.exp(use(0.3) * _u(gen, (n, 1), 0.2, 0.8, device)
                              * torch.sin(2 * math.pi * t / period2[:, None] + phase))
    sigma = sigma * torch.exp(use(0.5) * _u(gen, (n, 1), 0.0, 0.7, device)
                              * _smooth(gen, n, length, device))
    hurst = _u(gen, (n, 1), 0.3, 0.8, device)
    fgn = _spectral(gen, length, device,
                    lambda w: w[None].clamp_min(1e-4) ** (1 - 2 * hurst))
    eps = torch.where(use(0.2), fgn, _randn(gen, (n, length), device))
    dt = _logu(gen, (n, 1), 0.005, 0.05, device)
    theta = torch.minimum(theta.clamp_min(1e-3), 0.25 / dt).double()
    a = torch.exp(-theta * dt.double())
    drive = ((1 - a) * mu.double()
             + sigma.double() * torch.sqrt((1 - a**2) / (2 * theta)) * eps.double())
    y0 = mu[:, 0].double() + (sigma[:, 0] / (2 * theta[:, 0].float()).sqrt()).double() * _randn(
        gen, (n,), device).double()
    y = _ou_scan(a, drive, y0)
    return y.float()


def step_family(gen, n, length, device, max_changes=10):
    """Return float32 changepoint series [n,length] with optional intermittency and seasonality."""
    t = torch.arange(length, device=device)[None]
    tf = t.float()
    count = torch.randint(1, max_changes + 1, (n, 1), generator=gen, device=device)
    positions = torch.randint(1, length, (n, max_changes), generator=gen, device=device)
    valid = torch.arange(max_changes, device=device)[None] < count
    positions = torch.where(valid, positions, 0)
    marks = torch.zeros((n, length), device=device).scatter_(1, positions, 1.0)
    marks[:, 0] = 1.0
    segment = marks.long().cumsum(-1) - 1
    start = torch.where(marks > 0, t, 0).cummax(-1).values
    slots = max_changes + 1
    levels = _laplace(gen, (n, slots), device).cumsum(-1)
    kind = _categorical(gen, (0.5, 0.3, 0.2), n * slots, device).view(n, slots)
    slope = _randn(gen, (n, slots), device) * _logu(gen, (n, 1), 0.3, 3.0, device) / length
    walk = (_randn(gen, (n, length), device) * _logu(gen, (n, 1), 0.01, 0.1, device)).cumsum(-1)
    seg_kind = kind.gather(1, segment)
    since = tf - start.float()
    x = levels.gather(1, segment)
    x = x + torch.where(seg_kind == 1, slope.gather(1, segment) * since * 3, 0.0)
    x = x + torch.where(seg_kind == 2, walk - walk.gather(1, start), 0.0)
    onoff = _u(gen, (n, 1), device=device) < 0.2
    on = (segment % 2 == 1) ^ (_u(gen, (n, 1), device=device) < 0.5)
    x = torch.where(onoff, on * (x.abs() + 0.5), x)
    smooth = _u(gen, (n, 1), device=device) < 0.3
    width = _logu(gen, (n, 1), 1.0, 30.0, device)
    pad = length // 2
    padded = F.pad(x[:, None], (pad, pad), mode="replicate")[:, 0]
    grid = padded.shape[-1]
    omega = 2 * math.pi * torch.arange(grid // 2 + 1, device=device, dtype=torch.float32) / grid
    blurred = torch.fft.irfft(torch.fft.rfft(padded) * torch.exp(-(omega[None] * width) ** 2 / 2),
                              n=grid)[:, pad : pad + length]
    x = torch.where(smooth, blurred, x)
    unit = x.std(-1, keepdim=True).clamp_min(0.3)
    period = _choice(gen, PERIOD_BANK, (n,), device)
    seasonal = _u(gen, (n, 1), device=device) < 0.5
    x = (x + seasonal * unit * _logu(gen, (n, 1), 0.05, 0.7, device)
         * _periodic(gen, period, length, device))
    x = x + unit * _logu(gen, (n, 1), 0.01, 0.4, device) * _randn(gen, (n, length), device)
    return x


def _nearest_event_distance(events, t):
    big = 10**6
    left = torch.where(events, t, -big).cummax(-1).values
    right = torch.where(events, t, 2 * big).flip(-1).cummin(-1).values.flip(-1)
    return torch.minimum(t - left, right - t).float()


def spike_family(gen, n, length, device):
    """Return float32 spike, sawtooth and anomaly series [n,length] in equal shares."""
    t = torch.arange(length, device=device)[None]
    tf = t.float()
    sub = torch.randint(0, 3, (n, 1), generator=gen, device=device)
    period = _choice(gen, PERIOD_BANK, (n,), device)[:, None]
    phase = _u(gen, (n, 1), device=device)
    m = torch.remainder(tf / period + phase, 1.0) * period
    periodic_d = torch.minimum(m, period - m)
    rate = _logu(gen, (n, 1), 1 / 500, 1 / 20, device)
    burst = (_u(gen, (n, length), device=device) < rate)
    random_d = _nearest_event_distance(burst, t)
    is_burst = _u(gen, (n, 1), device=device) < 0.4
    d = torch.where(is_burst, random_d, periodic_d)
    halfwidth = torch.minimum(_logu(gen, (n, 1), 1.0, 40.0, device), period / 3).clamp_min(1.0)
    shape = _categorical(gen, (0.4, 0.3, 0.3), n, device)[:, None]
    tri = (1 - d / halfwidth).clamp_min(0)
    ramp = _u(gen, (n, 1), 0.1, 0.5, device)
    plateau = ((1 - d / halfwidth) / ramp).clamp(0, 1)
    pulse = torch.where(shape == 0, tri, torch.where(shape == 1, -tri, plateau))
    amplitude = _logu(gen, (n, 1), 1.0, 10.0, device)
    modulation = torch.exp(0.3 * _smooth(gen, n, length, device))
    baseline = _randn(gen, (n, 1), device) * (tf / length - 0.5)
    white = _randn(gen, (n, length), device)
    brown = _standardise(white.cumsum(-1))
    pink = _spectral(gen, length, device, lambda w: 1 / w[None].clamp_min(1e-4))
    noise = torch.where(_u(gen, (n, 1), device=device) < 0.5, brown, pink)
    spikes = (baseline + amplitude * modulation * pulse
              + _logu(gen, (n, 1), 0.02, 0.5, device) * noise)
    direction = torch.where(_u(gen, (n, 1), device=device) < 0.5, -1.0, 1.0)
    saw = direction * torch.remainder(tf / period + phase, 1.0)
    sine_period = _choice(gen, PERIOD_BANK, (n,), device)[:, None]
    saw = (saw + _randn(gen, (n, 1), device) * 0.5 * (tf / length - 0.5)
           + _u(gen, (n, 1), 0.0, 0.2, device) * torch.sin(2 * math.pi * tf / sine_period)
           + _logu(gen, (n, 1), 0.005, 0.1, device) * white)
    spike_w = torch.randint(1, 4, (n, 1), generator=gen, device=device).float()
    hit = (periodic_d < spike_w / 2 + 0.5).float()
    s = tf / length
    trend = 1 + _u(gen, (n, 1), -0.8, 2.0, device) * s
    cycle = 1 + 0.6 * torch.sin(2 * math.pi * s * _logu(gen, (n, 1), 0.5, 4.0, device)
                                + _u(gen, (n, 1), 0, 2 * math.pi, device))
    magnitude = torch.where(_u(gen, (n, 1), device=device) < 0.5, trend, cycle).clamp_min(0.05)
    sign = torch.where(_u(gen, (n, 1), device=device) < 0.7, 1.0, -1.0)
    anomaly = sign * amplitude * magnitude * hit + _logu(gen, (n, 1), 0.01, 0.3, device) * white
    return torch.where(sub == 0, spikes, torch.where(sub == 1, saw, anomaly))


MAKERS = {"sde": sde_family, "step": step_family, "spike": spike_family}
