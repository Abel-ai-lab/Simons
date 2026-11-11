"""Generate synthetic series on device and mix them into training batches."""

import math

import numpy as np
import torch
import torch.nn.functional as F

PERIOD_BANK = (4, 6, 7, 10, 12, 14, 24, 26, 30, 40, 48, 52, 60, 96, 168, 288, 336, 365, 672, 730)
SEASON_BANK = (4, 6, 7, 10, 12, 14, 24, 26, 30, 48, 52, 60, 96, 168, 288, 336, 365)
MULTIPLES = (2, 4, 7, 12, 24, 52)
SYNTHETIC_ID = 1_000_000


def _u(gen, shape, lo=0.0, hi=1.0, device="cpu"):
    return torch.rand(shape, generator=gen, device=device) * (hi - lo) + lo


def _logu(gen, shape, lo, hi, device):
    return torch.exp(_u(gen, shape, math.log(lo), math.log(hi), device))


def _randn(gen, shape, device):
    return torch.randn(shape, generator=gen, device=device)


_BANKS = {}


def _bank(values, device):
    """Cache device copies to avoid blocking host-to-device copies on each step."""
    key = (values, str(device))
    if key not in _BANKS:
        _BANKS[key] = torch.tensor(values, dtype=torch.float32, device=device)
    return _BANKS[key]


def _choice(gen, values, shape, device):
    bank = _bank(values, device)
    return bank[torch.randint(len(values), shape, generator=gen, device=device)]


def _categorical(gen, probabilities, n, device):
    edges = _bank(tuple(float(x) for x in torch.tensor(probabilities).cumsum(0)[:-1]), device)
    return (_u(gen, (n, 1), device=device) > edges).sum(-1)


def _standardise(x):
    x = x - x.mean(-1, keepdim=True)
    return x / x.std(-1, keepdim=True).clamp_min(1e-6)


def _periods(gen, n, device):
    bank = _choice(gen, PERIOD_BANK, (n,), device)
    free = _logu(gen, (n,), 2.0, 700.0, device)
    return torch.where(_u(gen, (n,), device=device) < 0.6, bank, free)


def _periodic(gen, period, length, device, harmonics=12):
    """Draw Fourier series with min(12, p/2) harmonics and Gaussian-decaying variance."""
    n = period.shape[0]
    k = torch.arange(1, harmonics + 1, device=device, dtype=torch.float32)
    smooth = _u(gen, (n, 1), 1.0, 6.0, device)
    var = torch.exp(-(k**2) / (2 * smooth**2)) * (k[None] <= (period[:, None] / 2).floor())
    a = torch.randn((n, harmonics), generator=gen, device=device) * var.sqrt()
    b = torch.randn((n, harmonics), generator=gen, device=device) * var.sqrt()
    t = torch.arange(length, device=device, dtype=torch.float32)
    phase = 2 * math.pi * torch.remainder(t[None, None] * k[None, :, None]
                                          / period[:, None, None], 1)
    x = (a[..., None] * phase.cos() + b[..., None] * phase.sin()).sum(1)
    return _standardise(x)


def _spectral(gen, length, device, spectrum):
    """Sample a stationary path with power spectrum `spectrum(omega)` [n,L+1] on a 2L grid."""
    grid = 2 * length
    omega = 2 * math.pi * torch.arange(grid // 2 + 1, device=device, dtype=torch.float32) / grid
    amp = spectrum(omega).clamp_min(0).sqrt()
    amp[:, 0] = 0
    xi = torch.complex(
        torch.randn(amp.shape, generator=gen, device=device),
        torch.randn(amp.shape, generator=gen, device=device),
    )
    return _standardise(torch.fft.irfft(amp * xi, n=grid)[:, :length])


def _smooth(gen, n, length, device):
    """Sample an RBF or Matern GP with nu in {0.5, 1.5, 2.5}; nu=0.5 gives OU paths."""
    scale = _logu(gen, (n, 1), 0.003, 1.0, device) * length
    nu = _choice(gen, (0.5, 1.5, 2.5, 100.0), (n, 1), device)

    def spectrum(omega):
        w = omega[None] * scale
        matern = (1 + w**2 / (2 * nu.clamp_max(10))) ** (-(nu.clamp_max(10) + 0.5))
        rbf = torch.exp(-(w**2) / 2)
        return torch.where(nu > 10, rbf, matern)

    return _spectral(gen, length, device, spectrum)


def kernel_family(gen, n, length, device, max_components=5):
    """Return float32 KernelSynth series [n,length] from random periodic, GP, noise and trends."""
    m, slots = n * max_components, max_components
    t = torch.arange(length, device=device, dtype=torch.float32) / length
    count = torch.randint(1, slots + 1, (n,), generator=gen, device=device)
    kind = _categorical(gen, (0.4, 0.25, 0.15, 0.2), m, device)
    period = _periods(gen, m, device).view(slots, n)
    multiple = period[:1] * _choice(gen, MULTIPLES, (slots, n), device)
    harmonic = _u(gen, (slots, n), device=device) < 0.4
    harmonic[0] = False
    period = torch.where(harmonic, multiple, period).flatten()
    periodic = _periodic(gen, period, length, device)
    smooth = _smooth(gen, m, length, device)
    sigma = torch.where(_u(gen, (m, 1), device=device) < 0.5, 0.1, 1.0)
    white = torch.randn((m, length), generator=gen, device=device) * sigma
    slope = torch.randn((m, 1), generator=gen, device=device) * 2
    linear = slope * (t[None] - 0.5) * 2
    ramp = _u(gen, (m, 1), 0.05, 1.0, device) + t[None] * torch.sign(slope)
    ramp = ramp - torch.minimum(ramp.amin(-1, keepdim=True), torch.zeros_like(ramp[:, :1]))
    amplitude = _logu(gen, (m, 1), 0.1, 3.0, device)
    k = kind[:, None]
    component = torch.where(
        k == 0, periodic * amplitude,
        torch.where(k == 1, smooth * amplitude, torch.where(k == 2, white, linear)),
    )
    factor = torch.where(k == 3, ramp, component / amplitude)
    component, factor = component.view(slots, n, length), factor.view(slots, n, length)
    product = (_u(gen, (slots, n, 1), device=device) < 0.2)
    acc = component[0]
    for slot in range(1, slots):
        combined = torch.where(product[slot], acc * factor[slot], acc + component[slot])
        acc = torch.where((slot < count)[:, None], combined, acc)
    return acc


def _arma_response(gen, n, grid, device, seasonal_period):
    """Return a random stable SARMA filter response [n,grid/2+1] (complex)."""
    omega = 2 * math.pi * torch.arange(grid // 2 + 1, device=device, dtype=torch.float32) / grid
    z = torch.polar(torch.ones_like(omega), -omega)[None]
    use_ar = _u(gen, (n, 1), device=device) < 0.7
    use_seasonal = _u(gen, (n, 1), device=device) < 0.6
    use_seasonal |= ~use_ar
    response = torch.ones((n, omega.shape[0]), dtype=torch.complex64, device=device)
    pairs = torch.randint(0, 4, (n, 1), generator=gen, device=device)
    for j in range(3):
        r = _u(gen, (n, 1), 0.0, 0.95, device) * ((j < pairs) & use_ar)
        theta = _u(gen, (n, 1), 0.0, math.pi, device)
        response = response / (1 - 2 * r * theta.cos() * z + r**2 * z**2)
    real = _u(gen, (n, 1), -0.95, 0.95, device) * (_u(gen, (n, 1), device=device) < 0.5) * use_ar
    response = response / (1 - real * z)
    zeros = torch.randint(0, 3, (n, 1), generator=gen, device=device)
    for j in range(2):
        r = _u(gen, (n, 1), 0.0, 1.0, device) * (j < zeros)
        theta = _u(gen, (n, 1), 0.0, math.pi, device)
        response = response * (1 - 2 * r * theta.cos() * z + r**2 * z**2)
    s = seasonal_period[:, None]
    zs = torch.polar(torch.ones_like(omega[None] * s), -omega[None] * s)
    phi = _u(gen, (n, 1), 0.2, 0.9, device) * use_seasonal
    theta_s = (_u(gen, (n, 1), -0.6, 0.6, device) * use_seasonal
               * (_u(gen, (n, 1), device=device) < 0.5))
    response = response * (1 - theta_s * zs) / (1 - phi * zs)
    d = _u(gen, (n, 1), 0.0, 1.0, device) * (_u(gen, (n, 1), device=device) < 0.3)
    diff = (1 - z).abs().clamp_min(1e-4)
    angle = torch.angle(1 - z)
    magnitude, phase = torch.broadcast_tensors(diff ** (-d), -d * angle)
    response = response * torch.polar(magnitude.contiguous(), phase.contiguous())
    response[:, 0] = 0
    return response


def _seasonal_cumsum(x, period, active, bank=SEASON_BANK):
    """Compute y_t = y_{t-p} + x_t for active rows with p in `bank`; preserve other rows."""
    out = x
    length = x.shape[-1]
    for s in bank:
        pad = (-length) % s
        y = F.pad(x, (0, pad)).unflatten(-1, (-1, s)).cumsum(-2).flatten(-2)[:, :length]
        out = torch.where((active & (period == s))[:, None], y, out)
    return out


def _sarima_paths(gen, n, length, device):
    season = _choice(gen, SEASON_BANK, (n,), device)
    grid = 2 * length
    response = _arma_response(gen, n, grid, device, season)
    noise = torch.fft.rfft(torch.randn((n, grid), generator=gen, device=device))
    y = _standardise(torch.fft.irfft(noise * response, n=grid)[:, length:])
    integrate = _u(gen, (n, 1), device=device) < 0.4
    y = torch.where(integrate, _standardise(y.cumsum(-1)), y)
    seasonal = _u(gen, (n,), device=device) < 0.3
    y = _seasonal_cumsum(y, season, seasonal)
    return _standardise(y)


def _stretch(y, factor):
    """Interpolate [n,L] paths at t / factor for each row."""
    length = y.shape[-1]
    position = torch.arange(length, device=y.device, dtype=torch.float32)[None] / factor[:, None]
    low = position.floor().long().clamp_max(length - 1)
    high = (low + 1).clamp_max(length - 1)
    frac = position - low
    return y.gather(-1, low) * (1 - frac) + y.gather(-1, high) * frac


def sarima_family(gen, n, length, device):
    """Return float32 seasonal ARIMA series [n,length] with optional slow envelopes."""
    paths = _sarima_paths(gen, 2 * n, length, device)
    base, extra = paths[:n], paths[n:]
    factor = torch.randint(4, 65, (n,), generator=gen, device=device).float()
    extra = _stretch(extra, factor)
    interact = _u(gen, (n, 1), device=device) < 0.7
    multiplicative = _u(gen, (n, 1), device=device) < 0.5
    lo, hi = extra.amin(-1, keepdim=True), extra.amax(-1, keepdim=True)
    unit = 2 * (extra - lo) / (hi - lo).clamp_min(1e-6) - 1
    weight = _u(gen, (n, 1), device=device)
    level = _u(gen, (n, 1), 0.0, 3.0, device)  # Let the envelope act on nonzero means too.
    mixed = torch.where(
        multiplicative,
        (1 + weight * unit) * (base + level),
        base + _logu(gen, (n, 1), 0.2, 3.0, device) * extra,
    )
    return torch.where(interact, mixed, base)


def _noisers(gen, x, probability):
    """Return (values, is_count) with Poisson count rates lambda0 * minmax(x)."""
    n, length = x.shape
    device = x.device
    kind = torch.randint(0, 4, (n, 1), generator=gen, device=device)
    active = _u(gen, (n, 1), device=device) < probability
    lo, hi = x.amin(-1, keepdim=True), x.amax(-1, keepdim=True)
    unit = (x - lo) / (hi - lo).clamp_min(1e-6)
    rate = _logu(gen, (n, 1), 0.05, 50.0, device) * unit
    rate = rate * torch.where(
        kind == 1, torch.exp(0.5 * torch.randn((n, length), generator=gen, device=device)), 1.0
    )
    counts = torch.poisson(rate, generator=gen)
    sigma = _u(gen, (n, 1), 0.05, 0.5, device)
    shocks = (unit + 0.1) * torch.exp(sigma * torch.randn((n, length), generator=gen,
                                                          device=device))
    spikes = (torch.rand((n, length), generator=gen, device=device) < 0.01) * torch.randn(
        (n, length), generator=gen, device=device
    ) * _u(gen, (n, 1), 3.0, 10.0, device)
    out = torch.where(kind <= 1, counts, torch.where(kind == 2, shocks, x + spikes))
    is_count = active & (kind <= 1)
    return torch.where(active, out, x), is_count


def draw(n, length, gen, device, families=("kernel", "sarima"), weights=None):
    """Return standardized float32 [n,length] series from weighted families, then augment them."""
    if n < 1:
        raise ValueError("draw needs n >= 1")
    if weights is None:
        split = [n // len(families)] * len(families)
    else:
        total = sum(weights)
        split = [int(n * w / total) for w in weights]
    split[0] += n - sum(split)
    from simons.synth import augment

    makers = family_makers(families)
    parts = [makers[f](gen, k, length, device) for f, k in zip(families, split) if k]
    x = torch.cat(parts)
    x = x[torch.randperm(n, generator=gen, device=device)]
    x = torch.where(torch.isfinite(x), x, 0)
    x = _standardise(x)
    return augment.apply(gen, x)


def family_makers(families) -> dict:
    """Return {name: maker(gen,n,length,device)} for the requested families."""
    from simons.synth.calendar import MAKERS as CALENDAR
    from simons.synth.causal import MAKERS as CAUSAL
    from simons.synth.heavy import MAKERS as HEAVY
    from simons.synth.mech import MAKERS as MECH
    from simons.synth.nonstat import MAKERS as NONSTAT

    makers = {"kernel": kernel_family, "sarima": sarima_family, **NONSTAT, **CAUSAL, **HEAVY,
              **MECH, **CALENDAR}
    unknown = sorted(set(families) - set(makers))
    if unknown:
        raise ValueError(f"unknown synthetic families {unknown}")
    return makers


def post(x, gen, noise=0.0, pad=0.4, missing=0.1, min_observed=96, group=None,
         return_flip=False):
    """Apply observation effects, random units, padding and gaps to standardized x [n,L].
    `group` [n] shares flips/padding; `return_flip` also returns time-flip flags [n,1]."""
    n, length = x.shape
    device = x.device

    def shared(v):
        return v if group is None else v[group]

    flip = shared(_u(gen, (n, 1), device=device) < 0.25)
    x = torch.where(flip, x.flip(-1), x)
    x, is_count = _noisers(gen, x, noise) if noise > 0 else (x, torch.zeros_like(flip))
    from simons.synth import observe

    chosen = None
    if observe.CONFIG is not None:
        chosen = observe.choose(gen, n, device, is_count)
        linked = observe.has(chosen, "link")
        x = torch.where(linked, observe.link(gen, x, observe.has(chosen, "mult")), x)
    sign = torch.where(_u(gen, (n, 1), device=device) < 0.25, -1.0, 1.0)
    scale = _logu(gen, (n, 1), 1e-2, 1e4, device)
    offset = _u(gen, (n, 1), -3.0, 3.0, device)
    positive = _u(gen, (n, 1), device=device) < 0.3
    y = sign * x
    y = torch.where(positive, y - y.amin(-1, keepdim=True), y + offset)
    if chosen is not None:
        y = torch.where(linked, x, y)  # Linked rows are already positive, so apply scaling only.
    x = torch.where(is_count, x, y * scale)
    x = torch.where(torch.isfinite(x), x, 0)
    if chosen is not None:
        x = observe.record(gen, x, chosen)
    t = torch.arange(length, device=device)[None]
    observed = shared(_logu(gen, (n, 1), min_observed, length, device).long())
    short = shared(_u(gen, (n, 1), device=device) < pad)
    x = torch.where(short & (t < length - observed), float("nan"), x)
    holes = _u(gen, (n, 1), device=device) < missing
    rate = _logu(gen, (n, 1), 0.001, 0.02, device)
    span = torch.randint(1, 33, (n, 1), generator=gen, device=device)
    starts = torch.rand((n, length), generator=gen, device=device) < rate
    latest = torch.where(starts, t, -10**6).cummax(-1).values
    hole = holes & (t - latest < span)
    hole[:, -32:] = False  # Keep the last patch observed.
    x = torch.where(hole, float("nan"), x)
    if return_flip:
        return x.float(), flip
    return x.float()


def generate(
    n: int,
    length: int,
    gen: torch.Generator,
    device,
    families=("kernel", "sarima"),
    noise: float = 0.0,
    pad: float = 0.4,
    missing: float = 0.1,
    min_observed: int = 96,
    weights=None,
) -> torch.Tensor:
    """Return float32 [n,length] synthetic series in random units; NaN marks missing values."""
    x = draw(n, length, gen, device, families, weights)
    return post(x, gen, noise, pad, missing, min_observed)


class SyntheticMixer:
    """Replace the last round(fraction * rows) batch rows from a reusable synthetic-series pool."""

    def __init__(self, fraction, gen, families=("kernel", "sarima"), noise=0.0, pad=0.4,
                 pool_steps=1, weights=None, multi=None, observe=None):
        self.fraction, self.gen, self.families = fraction, gen, tuple(families)
        self.multi, self.pool_ids, self.last_ids = multi, None, None
        self.noise, self.pad, self.pool_steps = noise, pad, max(1, int(pool_steps))
        self.weights = weights
        self.observe = observe
        self.pool, self.used = None, 0
        self.track_first, self.pool_first, self.last_first = 0, None, None
        self.first_event = None
        self.first_offset = 0

    def _build(self, size, length, device):
        from simons.synth import observe

        saved = observe.CONFIG
        observe.CONFIG = self.observe
        try:
            pool_ids = None
            if self.multi is None:
                pool = generate(size, length, self.gen, device, self.families, self.noise,
                                self.pad, weights=self.weights)
            else:
                built = self.multi.build(size, length, self.gen, device, self.families,
                                         self.noise, self.pad, self.weights)
                pool, pool_ids = built[0], tuple(built[1:])
        finally:
            observe.CONFIG = saved
        pool_first, first_event = None, None
        if self.track_first:
            seen = torch.isfinite(pool).unflatten(-1, (-1, self.track_first)).any(-1)
            position = torch.arange(seen.shape[-1], device=seen.device)
            first = torch.where(seen, position, seen.shape[-1]).amin(-1)
            if first.is_cuda:  # first_rows waits for this asynchronous copy only when needed.
                pool_first = torch.empty(first.shape, dtype=first.dtype, pin_memory=True)
                pool_first.copy_(first, non_blocking=True)
                first_event = torch.cuda.Event()
                first_event.record()
            else:
                pool_first = first
        return pool, pool_ids, pool_first, first_event

    def take(self, n, length, device):
        if self.pool is None or self.pool.shape[1] != length or self.used + n > len(self.pool):
            self.pool, ids, self.pool_first, self.first_event = self._build(
                n * self.pool_steps, length, device)
            if ids is not None:
                self.pool_ids = ids
            self.used = 0
        out = self.pool[self.used : self.used + n]
        if self.multi is not None:
            self.last_ids = tuple(a[self.used : self.used + n] for a in self.pool_ids)
        if self.track_first:
            self.last_first = (self.pool_first, self.first_event, self.used, self.used + n)
        self.used += n
        return out

    def last_meta(self):
        """Device [n, 3] covariate-role metadata of the last call's series, or None."""
        if self.multi is None or self.last_ids is None or len(self.last_ids) < 3:
            return None
        return self.last_ids[2]

    def host_ids(self, count, width):
        """Host [count, width] series IDs of the last call's synthetic rows."""
        if self.multi is not None:
            return self.last_ids[0].reshape(count, width)
        return SYNTHETIC_ID + np.arange(count * width).reshape(count, width)

    def first_rows(self):
        """Host first-observed patches of the last call's synthetic series (numpy), or None."""
        if self.last_first is None:
            return None
        first, event, start, stop = self.last_first
        if event is not None:
            event.synchronize()
        return first[start:stop].numpy() + self.first_offset

    def __call__(self, sequence, series_ids):
        rows, width, length = sequence.shape
        count = round(self.fraction * rows)
        if count <= 0:
            self.last_first = None
            return sequence, series_ids
        synthetic = self.take(count * width, length, sequence.device)
        sequence = sequence.clone()
        series_ids = series_ids.clone()
        sequence[rows - count :] = synthetic.view(count, width, length).to(sequence.dtype)
        if self.multi is not None:
            ids = self.last_ids[1].to(series_ids.device)
        else:
            ids = SYNTHETIC_ID + torch.arange(count * width, device=series_ids.device)
        series_ids[rows - count :] = ids.view(count, width)
        return sequence, series_ids
