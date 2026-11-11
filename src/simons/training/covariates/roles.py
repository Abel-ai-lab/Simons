"""Add known-future covariate roles to coupled synthetic groups and effect rows."""

import math

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from simons.covariates import ROLE_BINARY, ROLE_KNOWN, known_rows
from simons.synth.multi import MECHANISMS

WIDTH = 32
PATCH = 32
EFFECT_ID = 50_000_000
MECH = {name: i for i, name in enumerate(MECHANISMS)}

DEFAULTS = {
    "boundaries": [0.27, 0.53],
    "p_role": [0.10, 0.20, 0.40],
    "p_eff": [0.0, 0.0, 0.0],
    "root_p": 0.8,
    "sibling_p": [0.2, 0.5],
    "max_cov": 8,
    "partial_future": 0.15,
    "end_block": 0.5,
    "end_block_max": 16,
    "effect_workers": 3,
    "effect_horizons": [1, 2, 3, 4, 6, 8, 12, 16],
    "effect_pad": 0.4,
    "effect_min_observed": 1024,
    "effect_flip": 0.3,
    "base_families": "kernel,sarima,sde,step,spike,cauker,rhythm,wave,fpfn,ets,levy,pulse",
    "base_weights": "0.21056,0.21056,0.12024,0.05512,0.05512,0.08016,0.04008,0.03008,0.03008,"
                    "0.068,0.015,0.015",
    "synth_calendar": 0.3,
    "effect_mix": None,
}


def settings(spec):
    """The role settings: DEFAULTS updated by `spec` (validated)."""
    s = {**DEFAULTS, **(spec or {})}
    unknown = set(spec or {}) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"covariate roles: unknown fields {sorted(unknown)}")
    if len(s["p_role"]) != len(s["boundaries"]) + 1 or len(s["p_eff"]) != len(s["p_role"]):
        raise ValueError("role_share and effect_share need one value per role phase")
    if max(s["p_eff"]) > 0 and s["effect_mix"] is None:
        raise ValueError("effect rows (effect_share > 0) need effect_mix")
    if s["effect_mix"] is not None:
        from simons.synth.covariates import families

        families.parse_mix(s["effect_mix"])
    return s


def phase_value(values, boundaries, fraction):
    return values[int(np.searchsorted(np.asarray(boundaries, float), fraction, side="right"))]


class RolePlanner:
    """Assign covariate roles and future visibility to coupled synthetic groups."""

    def __init__(self, s, seed: int, stage_steps: int):
        self.s = s
        self.rng = np.random.default_rng([int(seed), 4417])
        self.stage_steps = max(1, int(stage_steps))
        self.step = 0
        self.stats = {"pools": 0, "groups": 0, "role_groups": 0, "cov_slots": 0,
                      "target_slots": 0, "end_blocks": 0, "cuts": 0, "by_mech": {}}

    def fraction(self):
        return self.step / self.stage_steps

    def p_role(self):
        return float(phase_value(self.s["p_role"], self.s["boundaries"], self.fraction()))

    def plan_pool(self, structure, kinds, length, n_pool):
        """Return [n_pool,3]: role, target end-block patches, future cut step (0: none).
        `kinds` follows pool row order; True marks coupled rows."""
        s, rng = self.s, self.rng
        group, mech, root = structure["group"], structure["mech"], structure["root"]
        n_c = len(group)
        meta_c = np.zeros((n_c, 3), np.int64)
        p = self.p_role()
        self.stats["pools"] += 1
        if n_c:
            starts = np.flatnonzero(np.r_[True, group[1:] != group[:-1]])
            ends = np.r_[starts[1:], n_c]
        else:
            starts = ends = np.zeros(0, np.int64)
        for a, b in zip(starts.tolist(), ends.tolist()):
            q = b - a
            if q < 2:
                continue
            self.stats["groups"] += 1
            if rng.random() >= p:
                continue
            m = int(mech[a])
            r = root[a:b]
            if m in (MECH["lag"], MECH["event"]) or (m == MECH["func"] and r.any()):
                cov = r & (rng.random(q) < s["root_p"])
            else:
                share = rng.uniform(*s["sibling_p"])
                cov = rng.random(q) < share
            if cov.all():
                cov[int(rng.integers(q))] = False
            if cov.sum() > s["max_cov"]:
                on = np.flatnonzero(cov)
                cov[:] = False
                cov[rng.choice(on, s["max_cov"], replace=False)] = True
            if not cov.any():
                continue
            block = meta_c[a:b]
            block[cov, 0] = 1
            if rng.random() < s["end_block"]:
                block[~cov, 1] = int(rng.integers(1, s["end_block_max"] + 1))
                self.stats["end_blocks"] += 1
            cut = cov & (rng.random(q) < s["partial_future"])
            if cut.any():
                block[cut, 2] = rng.integers(int(0.75 * length), length, size=int(cut.sum()))
                self.stats["cuts"] += int(cut.sum())
            self.stats["role_groups"] += 1
            self.stats["cov_slots"] += int(cov.sum())
            self.stats["target_slots"] += int((~cov).sum())
            key = MECHANISMS[m]
            self.stats["by_mech"][key] = self.stats["by_mech"].get(key, 0) + 1
        meta = np.zeros((n_pool, 3), np.int64)
        c = 0
        for k, is_coupled in enumerate(kinds):
            if is_coupled:
                meta[k * WIDTH:(k + 1) * WIDTH] = meta_c[c * WIDTH:(c + 1) * WIDTH]
                c += 1
        return meta


def effect_row(s, rng, bases, length):
    """Return a 32-slot row: values [32,L], meta [32,3], group [32], first [32], family counts.
    Metadata holds (role,end block,cut); group -1 marks empty slots; first is an observed patch."""
    from simons.synth.covariates import families, generic

    values = np.full((WIDTH, length), np.nan, np.float32)
    meta = np.zeros((WIDTH, 3), np.int64)
    group = np.full(WIDTH, -1, np.int64)
    slot, g, tries = 0, 0, 0
    names, p = families.parse_mix(s["effect_mix"])
    counts = np.zeros(len(families.FAMILIES), np.int64)
    while slot < WIDTH - 1 and tries < 64:
        tries += 1
        family = names[int(rng.choice(len(names), p=p))]
        k = (min(generic.MAX_COV, int(rng.geometric(0.35)), WIDTH - slot - 1)
             if family == "generic" else min(generic.MAX_COV, WIDTH - slot - 1))
        H = families.horizon(family, rng, s["effect_horizons"]) * PATCH
        observed = length
        if rng.random() < s["effect_pad"]:
            lo = max(s["effect_min_observed"], H + 256)
            observed = int(math.exp(rng.uniform(math.log(lo), math.log(length))))
            observed = min(length, max(lo, observed // PATCH * PATCH))
        hist = observed - H
        b = bases.window(observed)
        if np.isfinite(b[:hist]).sum() < 64 or np.isfinite(b[hist:]).sum() < H // 4:
            continue
        y, c = families.make_task(family, b, rng, hist, H, k, bases.window)
        if len(c) == 0 or y.shape[-1] != observed or c.shape[-1] != observed:
            continue
        k = len(c)
        if not np.isfinite(y[:hist]).any():
            continue
        counts[families.FAMILIES.index(family)] += 1
        if rng.random() < s["effect_flip"]:
            y = -y
        start = length - observed
        values[slot, start:] = y
        values[slot + 1:slot + 1 + k, start:] = c
        meta[slot, 1] = H // PATCH
        meta[slot + 1:slot + 1 + k, 0] = 1
        for i in range(k):
            if rng.random() < s["partial_future"]:
                meta[slot + 1 + i, 2] = start + hist + int(rng.integers(1, H))
        group[slot:slot + 1 + k] = g
        g += 1
        slot += 1 + k
    seen = np.isfinite(values).reshape(WIDTH, -1, PATCH).any(-1)
    patches = seen.shape[1]
    first = np.where(seen.any(1), seen.argmax(1), patches).astype(np.int64)
    return {"values": torch.from_numpy(values), "meta": torch.from_numpy(meta),
            "group": torch.from_numpy(group), "first": torch.from_numpy(first),
            "families": torch.from_numpy(counts)}


class SynthBases:
    """Draw synthetic base windows for effect rows using a private CPU torch generator."""

    def __init__(self, s, gen):
        self.gen = gen
        self.families = tuple(str(s["base_families"]).split(","))
        self.weights = [float(w) for w in str(s["base_weights"]).split(",")]
        self.calendar = float(s["synth_calendar"])
        self.pool, self.used = None, 0

    def refill(self, n, length):
        from simons.synth import draw

        n_cal = round(n * self.calendar)
        parts = [draw(n - n_cal, length, self.gen, "cpu", self.families, self.weights)]
        if n_cal:
            parts.append(draw(n_cal, length, self.gen, "cpu", ("calendar",)))
        x = torch.cat(parts)
        self.pool = x[torch.randperm(len(x), generator=self.gen)].double().numpy()
        self.used = 0

    def window(self, length):
        if self.pool is None or self.used >= len(self.pool) or self.pool.shape[1] < length:
            self.refill(64, max(length, 1024))
        x = self.pool[self.used]
        self.used += 1
        return x[len(x) - length:]


class EffectRows(IterableDataset):
    """Stream synthetic effect rows with one numpy generator per loader worker."""

    def __init__(self, s, seed, first_step, length):
        from simons.synth import augment

        self.s, self.seed, self.first_step, self.length = s, seed, first_step, length
        self.augment = augment.CONFIG

    def __iter__(self):
        from simons.synth import augment

        augment.CONFIG = self.augment
        info = get_worker_info()
        worker = info.id if info is not None else 0
        rng = np.random.default_rng([int(self.seed), 6113, worker, int(self.first_step)])
        torch.set_num_threads(1)
        gen = torch.Generator()
        gen.manual_seed(int(rng.integers(2 ** 62)))
        bases = SynthBases(self.s, gen)
        bases.refill(64, self.length)
        while True:
            if bases.used >= len(bases.pool):
                bases.refill(64, self.length)
            yield effect_row(self.s, rng, bases, self.length)


class RoleBatch:
    """Hold one batch's covariate roles, end blocks and visibility masks on device."""

    def __init__(self, role, role_ids, endk, fixed):
        self.role, self.role_ids, self.endk, self.fixed = role, role_ids, endk, fixed

    def hidden(self, hidden):
        """CPM mask [B,V,P] with the role groups' end blocks hidden and covariates never hidden."""
        patches = hidden.shape[-1]
        position = torch.arange(patches, device=hidden.device)
        end = (self.endk[..., None] > 0) & (position >= patches - self.endk[..., None])
        return (hidden | end) & ~self.role[..., None]

    def target(self, target):
        """Loss targets [B,V,S,patch]: none on covariate rows."""
        return target.masked_fill(self.role[..., None, None], float("nan"))

    def token_roles(self, size):
        return self.role_ids[..., None].expand(*self.role_ids.shape, size)


class Hook:
    """Plan synthetic covariate roles and effect rows, then prepare packed batches."""

    def __init__(self, spec, seed: int, first_step: int, device, stage_steps: int, length: int):
        self.s = settings(spec)
        self.device = torch.device(device)
        self.stage_steps = max(1, int(stage_steps))
        self.length = int(length)
        self.planner = RolePlanner(self.s, seed * 31 + 4417 + first_step, stage_steps)
        self.rng = np.random.default_rng([int(seed), 9049, int(first_step)])
        self.effects = None
        if max(self.s["p_eff"]) > 0:
            workers = int(self.s["effect_workers"])
            private = torch.Generator()
            private.manual_seed(int(seed) * 1013 + 9049 + int(first_step))
            loader = DataLoader(EffectRows(self.s, seed, first_step, self.length),
                                batch_size=None, generator=private, num_workers=workers,
                                pin_memory=self.device.type == "cuda",
                                persistent_workers=workers > 0,
                                prefetch_factor=8 if workers > 0 else None)
            self.effects = iter(loader)
        self.stats = {"steps": 0, "effect_rows": 0, "effect_groups": 0, "binary_rows": 0,
                      "role_slots": 0}
        self._pending = []

    def p_eff(self, step):
        return float(phase_value(self.s["p_eff"], self.s["boundaries"], step / self.stage_steps))

    def prepare(self, step, mixer, sequence, series_ids, host_slots, far_offset=0):
        """Return (sequence,series_ids,host_slots,RoleBatch) at `step` from role-phase start.
        `far_offset` is the NaN prefix in steps before synthetic rows; zero in stage 2."""
        rows, width, length = sequence.shape
        device = sequence.device
        meta = torch.zeros((rows, width, 3), dtype=torch.long, device=device)
        got = mixer.last_meta()
        count = round(mixer.fraction * rows)
        if got is not None and count:
            meta[rows - count:] = got.view(count, width, 3).to(device)
        n_eff = 0
        if self.effects is not None:
            share = self.p_eff(step) * rows
            n_eff = int(math.floor(share) + (self.rng.random() < share - math.floor(share)))
            n_eff = min(n_eff, rows)
        eff_ids = eff_first = None
        if n_eff:
            parts = [next(self.effects) for _ in range(n_eff)]
            values = torch.stack([p["values"] for p in parts])
            if values.shape[-1] != length:
                raise ValueError(f"effect rows of length {values.shape[-1]} for a {length} step")
            group = torch.stack([p["group"] for p in parts])  # [n, W]
            offset = (torch.arange(n_eff)[:, None] * WIDTH)
            ids = torch.where(group >= 0, EFFECT_ID + offset + group, -1)
            eff_ids, eff_first = ids.numpy(), torch.stack([p["first"] for p in parts]).numpy()
            sequence = sequence.clone()
            series_ids = series_ids.clone()
            sequence[:n_eff] = values.to(device, sequence.dtype, non_blocking=True)
            series_ids[:n_eff] = ids.to(device, series_ids.dtype, non_blocking=True)
            meta[:n_eff] = torch.stack([p["meta"] for p in parts]).to(device, non_blocking=True)
            self.stats["effect_rows"] += n_eff
            self.stats["effect_groups"] += int(sum(int(p["group"].max()) + 1 for p in parts))
            from simons.synth.covariates.families import FAMILIES

            total = torch.stack([p["families"] for p in parts]).sum(0).tolist()
            fam = self.stats.setdefault("families", dict.fromkeys(FAMILIES, 0))
            for f, n in zip(FAMILIES, total):
                fam[f] += int(n)
        role = meta[..., 0] > 0
        cut = meta[..., 2]
        if far_offset:
            cut = torch.where(cut > 0, cut + int(far_offset), cut)
        position = torch.arange(length, device=device)
        drop = role[..., None] & (cut[..., None] > 0) & (position >= cut[..., None])
        sequence = sequence.masked_fill(drop, float("nan"))
        shifted, two = known_rows(sequence, role, True, PATCH)
        sequence = torch.where(role[..., None], shifted, sequence)
        role_ids = (ROLE_KNOWN + (ROLE_BINARY - ROLE_KNOWN) * two.long()) * role.long()
        self.stats["steps"] += 1
        self._pending.append(torch.stack([role.sum(), two.sum()]))
        if len(self._pending) >= 200:
            self._flush()

        if eff_ids is not None:
            inner = host_slots

            def host_slots(inner=inner, eff_ids=eff_ids, eff_first=eff_first, n=n_eff):
                ids, first = inner()
                ids[:n] = eff_ids
                first[:n] = eff_first
                return ids, first

        batch = RoleBatch(role, role_ids, meta[..., 1], two)
        return sequence, series_ids, host_slots, batch

    def _flush(self):
        if self._pending:
            total = torch.stack(self._pending).sum(0).tolist()
            self.stats["role_slots"] += int(total[0])
            self.stats["binary_rows"] += int(total[1])
            self._pending = []

    def summary(self):
        self._flush()
        return {"batch": dict(self.stats), "plan": dict(self.planner.stats)}
