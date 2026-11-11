"""Augment standardized synthetic series with mixing, seasonality, filtering and quantization."""

import math

import torch

from simons.synth.base import _logu, _standardise, _u

CATEGORIES = ("mix", "season", "filter", "lattice")
CONFIG = None


def configure(categories, rate, weights):
    global CONFIG
    cats = tuple(c for c in str(categories or "").split(",") if c)
    if not cats or rate <= 0:
        CONFIG = None
        return
    if not set(cats) <= set(CATEGORIES):
        raise ValueError(f"augmentation categories must be some of {CATEGORIES}")
    if weights is None:
        w = [1.0] * len(cats)
    else:
        w = [float(v) for v in str(weights).split(",")]
        if len(w) != len(cats) or min(w) < 0 or sum(w) <= 0:
            raise ValueError("weights needs one nonnegative weight per category")
    CONFIG = {"categories": cats, "rate": float(rate), "weights": tuple(w)}


def _dirichlet15(gen, n, k, device):
    """Draw Dirichlet(1.5) rows using Gamma(1.5) = Exp(1) + z^2 / 2."""
    g = -torch.log(_u(gen, (n, k), device=device).clamp_min(1e-12))
    g = g + 0.5 * torch.randn((n, k), generator=gen, device=device) ** 2
    return g / g.sum(-1, keepdim=True)


def _mix(gen, x):
    n, length = x.shape
    device = x.device
    p1 = torch.randperm(n, generator=gen, device=device)
    p2 = torch.randperm(n, generator=gen, device=device)
    three = _u(gen, (n, 1), device=device) < 0.5
    wa = _dirichlet15(gen, n, 3, device)
    wb = _dirichlet15(gen, n, 3, device)
    two = torch.tensor([1.0, 1.0, 0.0], device=device)
    wa = torch.where(three, wa, wa * two)
    wb = torch.where(three, wb, wb * two)
    wa, wb = wa / wa.sum(-1, keepdim=True), wb / wb.sum(-1, keepdim=True)
    t = torch.arange(length, device=device, dtype=torch.float32)[None] / length
    centre = _u(gen, (n, 1), 0.2, 0.8, device)
    width = _logu(gen, (n, 1), 0.01, 0.2, device)
    s = torch.sigmoid((t - centre) / width)
    s = torch.where(_u(gen, (n, 1), device=device) < 0.5, s, 0.0)
    y = (wa[:, :1] * (1 - s) + wb[:, :1] * s) * x
    y = y + (wa[:, 1:2] * (1 - s) + wb[:, 1:2] * s) * x[p1]
    return y + (wa[:, 2:] * (1 - s) + wb[:, 2:] * s) * x[p2]


_DAY_STEPS = (24.0, 96.0, 288.0, 48.0, 1.0)


def _season(gen, x):
    n, length = x.shape
    device = x.device
    t = torch.arange(length, device=device, dtype=torch.float32)[None]
    c = _u(gen, (n, 1), 0.2, 2.0, device)
    a = _u(gen, (n, 1), 0.2, 0.8, device)
    day = torch.tensor(_DAY_STEPS, device=device)[
        torch.randint(len(_DAY_STEPS), (n, 1), generator=gen, device=device)]
    phase = _u(gen, (n, 1), device=device)
    week_pos = torch.remainder(t / (7 * day) + phase, 1.0)
    weekend = (week_pos >= 5 / 7).float()
    up = torch.where(_u(gen, (n, 1), device=device) < 0.25, 1.0, -1.0)
    square = 1 + up * a * weekend
    cycles = _u(gen, (n, 1), 1.0, 4.0, device)
    sine = 1 + a * torch.sin(2 * math.pi * (cycles * t / length + phase))
    m = torch.where(_u(gen, (n, 1), device=device) < 0.5, square, sine)
    base = x - x.amin(-1, keepdim=True) + c
    return base * m


def _filter(gen, x, taps=32):
    n, length = x.shape
    device = x.device
    size = torch.randint(2, taps + 1, (n, 1), generator=gen, device=device)
    j = torch.arange(taps, device=device)[None]
    live = j < size
    smooth_w = -torch.log(_u(gen, (n, taps), device=device).clamp_min(1e-12)) * live
    smooth_w = smooth_w / smooth_w.sum(-1, keepdim=True)
    signed_w = torch.randn((n, taps), generator=gen, device=device) * live
    signed_w = signed_w / signed_w.abs().sum(-1, keepdim=True).clamp_min(1e-6)
    w = torch.where(_u(gen, (n, 1), device=device) < 0.7, smooth_w, signed_w)
    # Repeat the first value on the left to avoid a start-up transient.
    padded = torch.cat([x[:, :1].expand(n, taps - 1), x], -1)
    y = torch.zeros_like(x)
    for k in range(taps):
        y = y + w[:, k : k + 1] * padded[:, taps - 1 - k : taps - 1 - k + length]
    diff = torch.cat([torch.zeros_like(x[:, :1]), x[:, 1:] - x[:, :-1]], -1)
    integ = x.cumsum(-1)
    kind = _u(gen, (n, 1), device=device)
    return torch.where(kind < 0.15, diff, torch.where(kind < 0.3, integ, y))


def _lattice(gen, x):
    n, length = x.shape
    device = x.device
    delta = _logu(gen, (n, 1), 0.05, 0.5, device)
    offset = _u(gen, (n, 1), device=device)
    quant = delta * torch.floor(x / delta + offset)
    q = _u(gen, (n, 1), 0.05, 0.5, device)
    ordered = x.sort(-1).values
    floor = ordered.gather(-1, (q * (length - 1)).long())
    censored = torch.maximum(x, floor)
    return torch.where(_u(gen, (n, 1), device=device) < 0.4, censored, quant)


_MAKERS = {"mix": _mix, "season": _season, "filter": _filter, "lattice": _lattice}


def apply(gen, x):
    """Augment standardised [n, L] series x; returns standardised series."""
    if CONFIG is None:
        return x
    n, _ = x.shape
    device = x.device
    cats, weights = CONFIG["categories"], CONFIG["weights"]
    active = _u(gen, (n, 1), device=device) < CONFIG["rate"]
    count = torch.where(_u(gen, (n, 1), device=device) < 0.3, 2, 1)
    # Gumbel top-k samples weighted categories without replacement.
    w = torch.tensor(weights, device=device)[None]
    logw = torch.log(w.clamp_min(1e-12))
    gumbel = -torch.log(-torch.log(_u(gen, (n, len(cats)), device=device).clamp(1e-12, 1 - 1e-7)))
    keys = logw + gumbel
    rank = keys.argsort(-1, descending=True).argsort(-1)
    chosen = active & (rank < count) & (w > 0)  # [n, C]
    for c in CATEGORIES:
        if c not in cats:
            continue
        col = cats.index(c)
        y = _standardise(_MAKERS[c](gen, x))
        y = torch.where(torch.isfinite(y), y, 0)
        x = torch.where(chosen[:, col : col + 1], y, x)
    return x
