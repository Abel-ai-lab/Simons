"""Couple synthetic series into groups sharing a series ID."""

import math

import numpy as np
import torch

from simons.synth.base import SYNTHETIC_ID, _spectral, _standardise, draw, post

MECHANISMS = ("identity", "mix", "lag", "coint", "func", "event")
WIDTH = 32
PULSE_PERIODS = (7, 12, 24, 48, 52, 96, 168, 288, 336)


def fft_length(n):
    """Return the smallest even 7-smooth length >= n for efficient cuFFT grids."""
    m = n + (n % 2)
    while True:
        r = m
        for p in (2, 3, 5, 7):
            while r % p == 0:
                r //= p
        if r == 1:
            return m
        m += 2


def _logu(rng, lo, hi):
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def _group_sizes(rng, max_variates, width=WIDTH):
    sizes, used = [], 0
    while used < width:
        q = min(int(_logu(rng, 2, max_variates + 1)), max_variates, width - used)
        sizes.append(q)
        used += q
    return sizes


def plan(rng, chunks, max_variates, mechanisms, lag_max):
    """Return per-slot, per-edge and per-driver numpy arrays for `chunks` coupled 32-slot rows.
    Each group owns two driver rows; edge parents index [drivers; slots], with -1 for no parent."""
    mech_ids = [MECHANISMS.index(m) for m in mechanisms]
    cum = np.cumsum(np.full(len(mech_ids), 1.0 / len(mech_ids)))
    n = chunks * WIDTH
    group = np.zeros(n, np.int64)
    mech = np.zeros(n, np.int64)
    own_kind = np.zeros(n, np.int64)
    ou = np.ones(n, np.float32)
    rho = np.zeros(n, np.float32)
    fkind = np.zeros(n, np.int64)
    fparam = np.zeros((n, 4), np.float32)
    parent = np.full((n, 3), -1, np.int64)
    tau = np.zeros((n, 3), np.float32)
    decay = np.zeros((n, 3), np.float32)
    coef = np.zeros((n, 3), np.float32)
    dkind = np.full(n, -1, np.int64)
    dparam = np.zeros((n, 3), np.float32)
    root = np.zeros(n, bool)
    sizes = [q for _ in range(chunks) for q in _group_sizes(rng, max_variates)]
    groups = len(sizes)
    slot = 0
    for g, q in enumerate(sizes):
        m = (mech_ids[min(int(np.searchsorted(cum, rng.random(), side='right')), len(cum) - 1)]
             if q > 1 else 0)
        d0 = slot
        k_drivers = 1 if m in (4, 5) else int(rng.integers(1, 3))
        for k in range(2):
            if m == 0 or k >= k_drivers:
                kind = -1
            elif m == 3:
                kind = 1 if rng.random() < 0.6 else 2
            elif m == 5:
                kind = 3 if rng.random() < 0.6 else 4
            else:
                kind = 0
            if kind >= 0:
                dkind[d0 + k] = kind
                dparam[d0 + k] = (_logu(rng, 0.003, 0.05),
                                  float(PULSE_PERIODS[int(rng.integers(len(PULSE_PERIODS)))]),
                                  float(rng.integers(1, 5)))
        members = np.arange(slot, slot + q)
        group[members] = g
        mech[members] = m
        plain = []
        observed_root = m == 2 or m == 5 or (m == 4 and rng.random() < 0.5)
        for i, j in enumerate(members):
            if m == 0:
                continue
            if observed_root and i < (k_drivers if m == 2 else 1):
                parent[j, 0], coef[j, 0] = d0 + i, 1.0
                rho[j] = rng.uniform(0.85, 1.0)
                root[j] = True
                plain.append(j)
                continue
            if m == 1:
                for k in range(k_drivers):
                    parent[j, k] = d0 + k
                    coef[j, k] = rng.normal() * (1.0 if k == 0 else rng.uniform(0.2, 1.0))
                rho[j] = rng.uniform(0.3, 0.95)
            elif m == 2:
                if plain and rng.random() < 0.4:
                    parent[j, 0] = n + plain[int(rng.integers(len(plain)))]
                else:
                    parent[j, 0] = d0 + int(rng.integers(k_drivers))
                    plain.append(j)
                tau[j, 0] = round(_logu(rng, 4, lag_max))
                decay[j, 0] = 0.0 if rng.random() < 0.6 else _logu(rng, 1, 48)
                coef[j, 0] = (1.0 if rng.random() < 0.5 else -1.0) * rng.uniform(0.5, 1.5)
                if rng.random() < 0.3:
                    parent[j, 1] = d0 + int(rng.integers(k_drivers))
                    tau[j, 1] = round(_logu(rng, 4, lag_max))
                    decay[j, 1] = 0.0 if rng.random() < 0.6 else _logu(rng, 1, 48)
                    coef[j, 1] = (1.0 if rng.random() < 0.5 else -1.0) * rng.uniform(0.3, 1.0)
                rho[j] = rng.uniform(0.4, 0.95)
            elif m == 3:
                for k in range(k_drivers):
                    parent[j, k] = d0 + k
                    tau[j, k] = 0.0 if rng.random() < 0.7 else round(_logu(rng, 1, 64))
                    coef[j, k] = (1.0 if rng.random() < 0.5 else -1.0) * rng.uniform(0.5, 2.0)
                own_kind[j], ou[j] = 1, _logu(rng, 2, 100)
                rho[j] = rng.uniform(0.6, 0.97)
            elif m == 4:
                parent[j, 0], coef[j, 0] = d0, 1.0
                tau[j, 0] = 0.0 if rng.random() < 0.4 else round(_logu(rng, 4, lag_max))
                decay[j, 0] = 0.0 if rng.random() < 0.7 else _logu(rng, 1, 32)
                fkind[j] = int(rng.integers(1, 7))
                fparam[j] = (_logu(rng, 0.5, 3.0), rng.uniform(-1, 1), rng.uniform(-1, 1),
                             rng.uniform(-2, 2))
                rho[j] = rng.uniform(0.5, 0.95)
            else:
                for e in range(1 + (rng.random() < 0.3)):
                    parent[j, e] = d0
                    tau[j, e] = round(_logu(rng, 1, 128)) if rng.random() < 0.8 else 0.0
                    decay[j, e] = _logu(rng, 1, 100)
                    coef[j, e] = (1.0 if rng.random() < 0.5 else -1.0) * rng.uniform(0.5, 2.0)
                rho[j] = rng.uniform(0.3, 0.9)
        slot += q
    return dict(group=group, mech=mech, own_kind=own_kind, ou=ou, rho=rho, fkind=fkind,
                fparam=fparam, parent=parent, tau=tau, decay=decay, coef=coef,
                dkind=dkind, dparam=dparam, root=root,
                groups=groups)


def _to_device(array, device):
    t = torch.from_numpy(np.ascontiguousarray(array))
    if torch.device(device).type == "cuda":
        return t.pin_memory().to(device, non_blocking=True)
    return t


def _transform(y, kind, p):
    """Apply per-row transforms; p [n,4] holds gain, b, s0, s1; kind 0 is identity."""
    g, b, s0, s1 = (p[:, i : i + 1] for i in range(4))
    out = y
    out = torch.where(kind == 1, torch.tanh(g * y), out)
    out = torch.where(kind == 2, torch.nn.functional.softplus(g * y), out)
    out = torch.where(kind == 3, y.abs(), out)
    clip = 0.2 + b.abs() * 1.3
    out = torch.where(kind == 4, torch.minimum(torch.maximum(y, -clip), clip), out)
    out = torch.where(kind == 5, torch.round(y * (1 + 3 * b.abs())) / (1 + 3 * b.abs()), out)
    out = torch.where(kind == 6, s0 * y + s1 * torch.relu(y - b), out)
    return out


def couple(structure, gen, length, device, families, weights, lag_max, extra=0):
    """Return standardized coupled [n_c,length] and plain [extra,length] float32 series.
    `structure` comes from `plan`; both outputs share the same base draw."""
    s = {k: (_to_device(v, device) if isinstance(v, np.ndarray) else v)
         for k, v in structure.items()}
    n = s["group"].shape[0]
    full = fft_length(length + lag_max)
    # Reuse bases round-robin to keep pool shapes fixed for FFT-plan reuse.
    quota_d, quota_o = max(1, n // 4), max(1, n // 3)
    bases = draw(extra + n + quota_d, full, gen, device, families, weights)
    plain = _standardise(bases[:extra, -length:]) if extra else bases[:0, -length:]
    own = bases[extra : extra + n]
    use_base = structure["dkind"] == 0
    driver_src = np.zeros(n, np.int64)
    driver_src[use_base] = np.arange(int(use_base.sum())) % quota_d
    dbase = bases[extra + n :][_to_device(driver_src, device)]
    t = torch.arange(full, device=device, dtype=torch.float32)[None]
    dk = s["dkind"][:, None]
    rate, period, width = (s["dparam"][:, i : i + 1] for i in range(3))
    steps = torch.randn((n, full), generator=gen, device=device)
    walk = _standardise(steps.cumsum(-1))
    drift = _standardise((steps + 4 * (rate - 0.025) / 0.025 * 0.05).cumsum(-1))
    impulses = (torch.rand((n, full), generator=gen, device=device) < rate) * torch.exp(
        0.5 * steps)
    phase = torch.rand((n, 1), generator=gen, device=device) * period
    pulses = (torch.remainder(t + phase, period) < width) * torch.exp(0.3 * steps)
    drivers = torch.where(dk == 1, walk, torch.where(dk == 2, drift, dbase))
    drivers = torch.where(dk == 3, _standardise(impulses), drivers)
    drivers = _standardise(torch.where(dk == 4, _standardise(pulses), drivers))
    is_ou = structure["own_kind"] == 1
    ou_src = np.zeros(n, np.int64)
    ou_src[is_ou] = np.arange(int(is_ou.sum())) % quota_o
    ou_scale = np.ones(quota_o, np.float32)
    ou_scale[ou_src[is_ou][::-1]] = structure["ou"][is_ou][::-1]
    scale = _to_device(ou_scale, device)[:, None]
    noise = _spectral(gen, full, device, lambda w: 1.0 / (1.0 + (w[None] * scale) ** 2))
    own = torch.where(s["own_kind"][:, None] == 1, noise[_to_device(ou_src, device)], own)
    grid = 2 * full
    omega = 2 * math.pi * torch.arange(grid // 2 + 1, device=device, dtype=torch.float32) / grid
    tau, decay, coef = s["tau"], s["decay"], s["coef"]
    r = torch.where(decay > 0, torch.exp(-1.0 / decay.clamp_min(1e-3)), torch.zeros_like(decay))
    delay = torch.polar(torch.ones_like(tau[..., None] * omega), -tau[..., None] * omega)
    smooth = (1 - r)[..., None] / (1 - r[..., None] * torch.polar(
        torch.ones_like(omega), -omega))
    weight = torch.where(s["parent"] >= 0, coef, torch.zeros_like(coef))[..., None]
    response = weight * delay * smooth  # [n, 3, F]
    parent = s["parent"].clamp_min(0)
    rho = s["rho"][:, None]
    spec_d = torch.fft.rfft(drivers, n=grid)

    def combine(slots):
        table = torch.cat([spec_d, torch.fft.rfft(slots, n=grid)])
        mixed = (table[parent] * response).sum(1)
        signal = torch.fft.irfft(mixed, n=grid)[:, :full]
        signal = _standardise(_transform(signal, s["fkind"][:, None], s["fparam"]))
        return rho.sqrt() * signal + (1 - rho).sqrt() * own

    x = combine(combine(own))  # A second pass lets chain followers read their parents' outputs.
    return _standardise(x[:, -length:]), plain


class MultiPool:
    """Build 32-slot pools with a `fraction` of rows containing coupled groups."""

    def __init__(self, fraction, max_variates=16, mechanisms=MECHANISMS, lag_max=256, seed=0):
        if not mechanisms or not set(mechanisms) <= set(MECHANISMS):
            raise ValueError(f"mechanisms must list some of {MECHANISMS}")
        if int(lag_max) != lag_max or lag_max < 4:
            raise ValueError("lag_max must be an integer of at least 4")
        self.fraction, self.max_variates = fraction, int(max_variates)
        self.mechanisms, self.lag_max = tuple(mechanisms), int(lag_max)
        self.rng = np.random.default_rng(seed)
        self.roles = None

    def build(self, n, length, gen, device, families, noise, pad, weights):
        """Return (pool [N,length], host IDs [N], device IDs [N][, role metadata [N,3]]).
        N >= n is divisible by 32; metadata holds role, end block and future cut; NaN marks gaps."""
        chunks = -(-n // WIDTH)
        coupled = round(self.fraction * chunks)
        structure_host = plan(self.rng, coupled, self.max_variates, self.mechanisms,
                              self.lag_max)
        n_c, n_u = coupled * WIDTH, (chunks - coupled) * WIDTH
        x_c, x_u = couple(structure_host, gen, length, device, families, weights,
                          self.lag_max, extra=n_u)
        kinds = self.rng.permutation(np.r_[np.ones(coupled, bool), np.zeros(chunks - coupled,
                                                                           bool)])
        order = np.empty(chunks * WIDTH, np.int64)
        group = np.empty(chunks * WIDTH, np.int64)
        c = u = 0
        base_u = structure_host["groups"]
        for k, is_coupled in enumerate(kinds):
            dest = slice(k * WIDTH, (k + 1) * WIDTH)
            if is_coupled:
                order[dest] = np.arange(c * WIDTH, (c + 1) * WIDTH)
                group[dest] = structure_host["group"][c * WIDTH : (c + 1) * WIDTH]
                c += 1
            else:
                order[dest] = n_c + np.arange(u * WIDTH, (u + 1) * WIDTH)
                group[dest] = base_u + np.arange(u * WIDTH, (u + 1) * WIDTH)
                u += 1
        x = torch.cat([x_c, x_u])[_to_device(order, device)]
        x, flip = post(x, gen, noise, pad, group=_to_device(group, device), return_flip=True)
        ids = SYNTHETIC_ID + group
        if self.roles is None:
            return x, ids, _to_device(ids, device)
        meta = _to_device(self.roles.plan_pool(structure_host, kinds, length, len(group)), device)
        # Time-flipped groups reverse causality, so they receive no covariate roles.
        return x, ids, _to_device(ids, device), meta * (~flip).long()
