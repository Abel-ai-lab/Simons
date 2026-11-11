"""Calendar series with workday/weekend profiles, holidays and seasonal variation."""

import math

import torch
import torch.nn.functional as F

from simons.synth.base import _bank, _categorical, _logu, _spectral, _standardise, _u

GRID = 288
STEPS_PER_DAY = (12.0, 24.0, 48.0, 96.0, 144.0, 288.0, 1.0)
STEPS_WEIGHTS = (0.08, 0.26, 0.12, 0.14, 0.06, 0.12, 0.22)
BUMPS = 4
HARMONICS = 6


def _circ(h, c):
    """Return signed circular distance on [0,1)."""
    return torch.remainder(h - c + 0.5, 1.0) - 0.5


def _profiles(gen, n, device):
    """Return positive workday/weekend profiles [n,GRID], including the weekend level factor."""
    h = (torch.arange(GRID, device=device, dtype=torch.float32) / GRID)[None, None]  # [1,1,G]
    count = torch.randint(1, BUMPS + 1, (n, 1), generator=gen, device=device)
    live = (torch.arange(BUMPS, device=device)[None] < count).float()  # [n, B]
    centre = _u(gen, (n, BUMPS), 0.18, 0.92, device)
    width = _logu(gen, (n, BUMPS), 0.015, 0.12, device)
    height = _logu(gen, (n, BUMPS), 0.2, 2.0, device) * live
    action = _categorical(gen, (0.25, 0.3, 0.25, 0.2), n * BUMPS, device).view(n, BUMPS)
    shift = _u(gen, (n, BUMPS), 1.0, 3.0, device) / 24
    e_centre = torch.where(action == 2, centre + shift, centre)
    e_width = torch.where(action == 3, width * _u(gen, (n, BUMPS), 1.5, 3.0, device), width)
    e_height = torch.where(action == 1, 0.0, torch.where(
        action == 3, height * _u(gen, (n, BUMPS), 0.3, 0.7, device), height))
    e_height = e_height * _u(gen, (n, BUMPS), 0.6, 1.2, device)

    def bumps(c, w, a):
        d = _circ(h, c[..., None]) / w[..., None]
        return (a[..., None] * torch.exp(-0.5 * d * d)).sum(1)  # [n, G]

    k = torch.arange(1, HARMONICS + 1, device=device, dtype=torch.float32)[None]
    nh = torch.randint(2, HARMONICS + 1, (n, 1), generator=gen, device=device)
    var = torch.exp(-k / _u(gen, (n, 1), 0.8, 2.5, device)) * (k <= nh)
    amp = _u(gen, (n, 1), 0.0, 0.6, device)
    a = torch.randn((n, HARMONICS), generator=gen, device=device) * var.sqrt() * amp
    b = torch.randn((n, HARMONICS), generator=gen, device=device) * var.sqrt() * amp
    ang = 2 * math.pi * k[..., None] * h[0]  # [1, K, G]
    fourier = (a[..., None] * ang.cos() + b[..., None] * ang.sin()).sum(1)
    e_fourier = fourier * _u(gen, (n, 1), 0.3, 1.0, device)
    work = bumps(centre, width, height) + fourier
    wend = bumps(e_centre, e_width, e_height) + e_fourier
    low = torch.minimum(work.amin(-1, keepdim=True), wend.amin(-1, keepdim=True))
    base = _u(gen, (n, 1), 0.05, 1.0, device)
    work, wend = work - low + base, wend - low + base
    level = torch.where(_u(gen, (n, 1), device=device) < 0.25,
                        _logu(gen, (n, 1), 1.05, 1.8, device),
                        _logu(gen, (n, 1), 0.35, 0.95, device))
    return work, wend * level


def calendar_family(gen, n, length, device):
    """Return float32 standardized calendar series [n,length] on `device`, drawn from `gen`."""
    t = torch.arange(length, device=device, dtype=torch.float32)[None]
    days = length  # Allocate enough days for the daily-data case P = 1.
    dd = torch.arange(days, device=device, dtype=torch.float32)[None]
    P = _bank(STEPS_PER_DAY, device)[_categorical(gen, STEPS_WEIGHTS, n, device)][:, None]
    sub = P > 1.5
    offset = torch.floor(_u(gen, (n, 1), device=device) * P)
    pos = t + offset
    day = torch.floor(pos / P)
    hour = (pos - day * P) / P
    day_l = day.long().clamp(0, days - 1)
    week_phase = torch.randint(0, 7, (n, 1), generator=gen, device=device).float()
    dow_axis = torch.remainder(dd + week_phase, 7.0)
    year_phase = _u(gen, (n, 1), 0.0, 365.25, device)
    doy_axis = torch.remainder(dd + year_phase, 365.25)
    two_types = _u(gen, (n, 1), device=device) < 0.4
    saturday = torch.where(two_types, 1.0, _u(gen, (n, 1), 0.2, 0.8, device))
    s = torch.where(dow_axis >= 5.5, 1.0, torch.where(dow_axis >= 4.5, saturday, 0.0))
    s = torch.where((_u(gen, (n, 1), device=device) < 0.1) & (dow_axis < 5.5), 0.0, s)
    workday = s < 0.5
    rate = _logu(gen, (n, 1), 0.005, 0.05, device)
    random_hol = (_u(gen, (n, days), device=device) < rate) & workday
    fixed = torch.floor(_u(gen, (n, 8), 0.0, 365.0, device))
    use_fixed = torch.arange(8, device=device)[None] < torch.randint(0, 9, (n, 1), generator=gen,
                                                                     device=device)
    fixed_hol = ((torch.floor(doy_axis)[..., None] == fixed[:, None]) & use_fixed[:, None]).any(-1)
    holiday = (random_hol | fixed_hol) & (_u(gen, (n, 1), device=device) < 0.85)
    nxt = F.pad(holiday[:, 1:], (0, 1))
    prv = F.pad(holiday[:, :-1], (1, 0))
    bridge = ((nxt & (dow_axis < 0.5)) | (prv & (dow_axis > 3.5) & (dow_axis < 4.5))) & ~holiday
    s = torch.where(holiday, 1.0, s)
    s = torch.where(bridge, torch.maximum(s, _u(gen, (n, 1), 0.3, 0.7, device)), s)
    n_days = torch.ceil(length / P)
    ev_live = torch.arange(3, device=device)[None] < torch.randint(0, 4, (n, 1), generator=gen,
                                                                   device=device)
    ev_len = _logu(gen, (n, 3), 3.0, 14.0, device)
    ev_start = _u(gen, (n, 3), device=device) * (n_days + ev_len) - ev_len
    ev_ramp = _u(gen, (n, 3), 0.5, 2.5, device)
    ev_w = ((torch.minimum(dd[..., None] - ev_start[:, None] + 1,
                           ev_start[:, None] + ev_len[:, None] - dd[..., None])
             / ev_ramp[:, None]).clamp(0, 1) * ev_live[:, None])  # [n, D, 3]
    ev_level = torch.log(_logu(gen, (n, 3), 0.5, 1.6, device))
    ev_type = _u(gen, (n, 3), 0.0, 0.9, device)
    log_level = (ev_w * ev_level[:, None]).sum(-1)
    s = torch.maximum(s, (ev_w * ev_type[:, None]).amax(-1))
    wd = torch.exp(_u(gen, (n, 1), 0.0, 0.12, device) * torch.randn((n, 7), generator=gen,
                                                                    device=device))
    log_level = log_level + torch.log(wd.gather(1, dow_axis.long().clamp(0, 6)))
    tau_d = _logu(gen, (n, 1), 1.0, 30.0, device)
    ou_d = _spectral(gen, days, device, lambda w: 1.0 / (1.0 + (w[None] * tau_d) ** 2))
    log_level = log_level + _u(gen, (n, 1), 0.0, 0.12, device) * ou_d
    ya = _u(gen, (n, 1), 0.0, 0.4, device) * (_u(gen, (n, 1), device=device) < 0.7)
    yp = _u(gen, (n, 2), 0.0, 2 * math.pi, device)
    ang = 2 * math.pi * doy_axis / 365.25
    log_level = log_level + ya * (torch.sin(ang + yp[:, :1]) + 0.4 * torch.sin(2 * ang + yp[:, 1:]))
    jitter = (_u(gen, (n, 1), 0.0, 0.012, device)
              * torch.randn((n, days), generator=gen, device=device))
    work, wend = _profiles(gen, n, device)
    s_t = s.gather(1, day_l)
    h = torch.remainder(hour + jitter.gather(1, day_l), 1.0)
    idx = (h * GRID).long().clamp(0, GRID - 1)
    hl = torch.where(sub, idx, 0)  # Daily data uses the mean profile instead of an intraday point.
    prof_w = torch.where(sub, work.gather(1, hl), work.mean(-1, keepdim=True))
    prof_e = torch.where(sub, wend.gather(1, hl), wend.mean(-1, keepdim=True))
    activity = ((1 - s_t) * prof_w + s_t * prof_e) * torch.exp(log_level.gather(1, day_l))
    night_c = _u(gen, (n, 1), -0.08, 0.08, device)
    half = _u(gen, (n, 1), 0.1, 0.27, device)
    dl = torch.where(P <= 48, _u(gen, (n, 1), 0.0, 0.35, device), 0.0)
    half = half * (1 + dl * torch.sin(ang + yp[:, :1]).gather(1, day_l))
    night = (_circ(hour, night_c).abs() < half) & sub & (_u(gen, (n, 1), device=device) < 0.3)
    floor = torch.where(_u(gen, (n, 1), device=device) < 0.5, 0.0,
                        activity.amin(-1, keepdim=True) * _u(gen, (n, 1), 0.0, 1.0, device))
    activity = torch.where(night, floor, activity)
    sd = activity.std(-1, keepdim=True).clamp_min(1e-6)
    tau_n = _logu(gen, (n, 1), 1.0, 30.0, device)
    noise = _spectral(gen, length, device, lambda w: 1.0 / (1.0 + (w[None] * tau_n) ** 2))
    sigma = _logu(gen, (n, 1), 0.01, 0.2, device)
    tau_s = _logu(gen, (n, 1), 0.05, 0.5, device) * length
    slow = _spectral(gen, length, device, lambda w: 1.0 / (1.0 + (w[None] * tau_s) ** 2))
    slow = (slow * _u(gen, (n, 1), 0.0, 0.25, device)
            + _u(gen, (n, 1), -0.5, 0.5, device) * (t / length - 0.5))
    mode = _categorical(gen, (0.45, 0.35, 0.2), n, device)[:, None]
    additive = activity + sd * (sigma * noise + slow)
    multiplicative = activity * torch.exp(sigma * noise + 0.5 * slow)
    capacity_at = activity.mean(-1, keepdim=True) + sd * _u(gen, (n, 1), 0.2, 1.5, device)
    g = _logu(gen, (n, 1), 1.0, 10.0, device) / sd
    capacity = (-F.softplus(g * (activity - capacity_at)) / g
                + sd * (0.3 * sigma * noise + 0.3 * slow))
    capacity = torch.where(night, capacity.amax(-1, keepdim=True), capacity)
    y = torch.where(mode == 0, additive, torch.where(mode == 1, multiplicative, capacity))
    y = torch.where(torch.isfinite(y), y, 0)
    return _standardise(y)


MAKERS = {"calendar": calendar_family}
