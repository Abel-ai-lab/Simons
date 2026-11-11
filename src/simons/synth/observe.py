"""Map synthetic latent paths to recorded values while preserving count rows."""

import torch

from simons.synth.base import _logu, _standardise, _u

CATEGORIES = ("link", "mult", "duty")
CONFIG = None
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
SEGMENTS = 256


def make_config(categories, rate=0.5, duty_periodic=0.0):
    """Return observation settings for comma-separated categories, or None when disabled."""
    cats = tuple(c for c in str(categories or "").split(",") if c)
    if not cats or rate <= 0:
        return None
    if not set(cats) <= set(CATEGORIES):
        raise ValueError(f"synthetic_observe must list some of {CATEGORIES}")
    if "mult" in cats and "link" not in cats:
        raise ValueError("synthetic_observe: mult needs link")
    if not 0 <= duty_periodic <= 1:
        raise ValueError("synthetic_observe_duty_periodic must be from 0 to 1")
    return {"categories": cats, "rate": float(rate), "duty_periodic": float(duty_periodic)}


def choose(gen, n, device, is_count):
    """[n, C] bool: the configured categories each row takes."""
    cats = CONFIG["categories"]
    active = (_u(gen, (n, 1), device=device) < CONFIG["rate"]) & ~is_count
    count = torch.where(_u(gen, (n, 1), device=device) < 0.4, 2, 1)
    # Gumbel top-k samples categories uniformly without replacement.
    gumbel = -torch.log(-torch.log(_u(gen, (n, len(cats)), device=device).clamp(1e-12, 1 - 1e-7)))
    rank = gumbel.argsort(-1, descending=True).argsort(-1)
    chosen = active & (rank < count)
    if "mult" in cats:
        mult, link = cats.index("mult"), cats.index("link")
        chosen[:, link] |= chosen[:, mult]
    return chosen


def has(chosen, name):
    """[n, 1] bool column of category `name` (False if not configured)."""
    cats = CONFIG["categories"]
    if name not in cats:
        return torch.zeros_like(chosen[:, :1])
    i = cats.index(name)
    return chosen[:, i : i + 1]


def link(gen, x, use_mult):
    """Positive inverse Box-Cox of the standardised path (unit level), with optional mult noise."""
    n, length = x.shape
    device = x.device
    z = _standardise(torch.where(torch.isfinite(x), x, 0))
    lam = torch.tensor(LAMBDAS, device=device)[
        torch.randint(len(LAMBDAS), (n, 1), generator=gen, device=device)]
    c = _logu(gen, (n, 1), 0.05, 1.0, device)
    safe = lam.clamp_min(1e-3)
    base = (1 + safe * c * z).clamp_min(0)
    power = torch.where(base > 0, base.clamp_min(1e-30) ** (1 / safe), torch.zeros_like(base))
    y = torch.where(lam < 1e-3, torch.exp((c * z).clamp_max(30)), power)
    sigma = _logu(gen, (n, 1), 0.01, 0.3, device)
    noise = torch.exp(sigma * torch.randn((n, length), generator=gen, device=device))
    y = torch.where(use_mult, y * noise, y)
    return y / y.mean(-1, keepdim=True).clamp_min(1e-12)


def _duty(gen, x):
    n, length = x.shape
    device = x.device
    mean_on = _logu(gen, (n, 1), 4.0, 512.0, device)
    mean_off = _logu(gen, (n, 1), 4.0, 512.0, device)
    k = torch.arange(SEGMENTS, device=device)[None]
    start_on = _u(gen, (n, 1), device=device) < mean_on / (mean_on + mean_off)
    on_k = (k % 2 == 0) == start_on
    mean = torch.where(on_k, mean_on, mean_off)
    p = (1 / mean).clamp(1e-6, 1 - 1e-6)
    u = _u(gen, (n, SEGMENTS), device=device).clamp_min(1e-12)
    runs = torch.floor(torch.log(u) / torch.log1p(-p)) + 1
    edges = runs.cumsum(-1).contiguous()
    t = torch.arange(length, device=device, dtype=edges.dtype)[None].expand(n, length).contiguous()
    seg = torch.searchsorted(edges, t, right=True).clamp_max(SEGMENTS - 1)
    on = on_k.gather(-1, seg)
    if CONFIG["duty_periodic"] > 0:  # Skip draws at zero to preserve the Markov-only stream.
        on = torch.where(_u(gen, (n, 1), device=device) < CONFIG["duty_periodic"],
                         _periodic_on(gen, n, length, device), on)
    shifted = x - x.amin(-1, keepdim=True).clamp_max(0)
    return shifted * on


DUTY_PERIODS = (24.0, 24.0, 48.0, 96.0, 96.0, 144.0, 288.0, 168.0, 12.0, 7.0, 52.0, 60.0)


def _periodic_on(gen, n, length, device):
    """Draw periodic on/off masks with switch-off jitter up to 5% of the period."""
    period = torch.tensor(DUTY_PERIODS, device=device)[
        torch.randint(len(DUTY_PERIODS), (n, 1), generator=gen, device=device)]
    share = _u(gen, (n, 1), 0.3, 0.8, device)
    phase = _u(gen, (n, 1), device=device)
    t = torch.arange(length, device=device, dtype=torch.float32)[None]
    cycle_pos = t / period + phase
    cycle = torch.floor(cycle_pos)
    frac = cycle_pos - cycle
    # Hash cycle indices for shared within-cycle jitter without another [n,L] draw.
    seed = _u(gen, (n, 1), 0.0, 1000.0, device)
    jitter = 0.05 * torch.sin(12.9898 * cycle + seed)
    return frac < (share + jitter)


def record(gen, x, chosen):
    """`duty` on the rows that chose it; x in original units, no NaN."""
    if "duty" not in CONFIG["categories"]:
        return x
    y = _duty(gen, x)
    y = torch.where(torch.isfinite(y), y, x)
    return torch.where(has(chosen, "duty"), y, x)
