"""Augment real windows on device with mixup, censoring, amplitude, sign and time flips."""

import torch


class RealAugmenter:
    def __init__(self, gen, censor=0.0, amplitude=0.0, sign_flip=0.0, time_flip=0.0, mixup=0.0,
                 mixup_alpha=1.5):
        self.gen = gen
        self.censor, self.amplitude = censor, amplitude
        self.sign_flip, self.time_flip = sign_flip, time_flip
        self.mixup, self.mixup_alpha = mixup, mixup_alpha

    @property
    def active(self):
        return max(self.censor, self.amplitude, self.sign_flip, self.time_flip, self.mixup) > 0

    def _u(self, shape, device):
        return torch.rand(shape, generator=self.gen, device=device)

    def __call__(self, sequence, series_ids, real_rows):
        if not self.active or real_rows <= 0:
            return sequence
        _, width, length = sequence.shape
        device = sequence.device
        x = sequence[:real_rows].reshape(real_rows * width, length).float()
        ids = series_ids[:real_rows].reshape(-1)
        n = x.shape[0]
        # Draw per series so related variates share transforms; id -1 uses a padding bucket.
        buckets = int(series_ids.shape[0] * width) + 1
        index = torch.where(ids >= 0, ids, buckets - 1).clamp(max=buckets - 1)
        draws = self._u((buckets, 8), device)[index]  # [n, 8]
        finite = torch.isfinite(x)
        valid = (ids >= 0) & finite.any(-1)
        position = torch.arange(length, device=device)
        first = torch.where(finite, position, length).amin(-1)

        if self.mixup > 0:
            alpha = torch.full((buckets,), float(self.mixup_alpha), device=device)
            g1 = _gamma(alpha, self.gen)
            g2 = _gamma(alpha, self.gen)
            lam = (g1 / (g1 + g2)).clamp(0.2, 0.8)[index]
            # Require a partner whose observed span covers this slot.
            key = torch.where(valid, first, length + 1).float() + 1e-3 * self._u((n,), device)
            order = torch.argsort(key)
            rank = torch.empty_like(order)
            rank[order] = torch.arange(n, device=device)
            pick = (self._u((n,), device) * (rank + 1).float()).floor().long().clamp(max=n - 1)
            partner = order[pick]
            xb = x[partner]
            sa = _mean_abs(x, finite)
            sb = _mean_abs(xb, torch.isfinite(xb))
            mixed = sa[:, None] * (lam[:, None] * x / sa[:, None]
                                   + (1 - lam[:, None]) * xb / sb[:, None])
            use = ((draws[:, 0] < self.mixup) & valid & valid[partner]
                   & (partner != torch.arange(n, device=device)))
            x = torch.where(use[:, None], mixed, x)
            finite = torch.isfinite(x)
            first = torch.where(finite, position, length).amin(-1)

        if self.censor > 0:
            count = finite.sum(-1)
            ordered = torch.sort(torch.where(finite, x, float("inf")), dim=-1).values
            bottom = self._u((n,), device) < 0.5
            q = torch.where(bottom, 0.4 * self._u((n,), device), 0.6 + 0.4 * self._u((n,), device))
            rank = (q * (count - 1).clamp_min(0).float()).round().long()
            c = ordered.gather(-1, rank[:, None].clamp(max=length - 1))
            censored = torch.where(bottom[:, None], torch.maximum(x, c), torch.minimum(x, c))
            use = (self._u((n,), device) < self.censor) & valid & (count > 1)
            x = torch.where(use[:, None] & finite, censored, x)

        if self.amplitude > 0:
            knots = torch.sort(self._u((buckets, 2), device), -1).values
            pos = torch.cat([torch.zeros(buckets, 1, device=device), knots,
                             torch.ones(buckets, 1, device=device)], -1)[index]
            val = torch.exp(0.3 * torch.randn((buckets, 4), generator=self.gen,
                                              device=device))[index]
            span = (length - first).clamp_min(1).float()
            t = ((position[None] - first[:, None]).float() / span[:, None]).clamp(0, 1)
            seg = torch.searchsorted(pos[:, 1:3].contiguous(), t.contiguous(), right=True)
            p0, p1 = pos.gather(-1, seg), pos.gather(-1, seg + 1)
            v0, v1 = val.gather(-1, seg), val.gather(-1, seg + 1)
            w = ((t - p0) / (p1 - p0).clamp_min(1e-6)).clamp(0, 1)
            envelope = v0 + w * (v1 - v0)
            use = (draws[:, 1] < self.amplitude) & valid
            x = torch.where(use[:, None], x * envelope, x)

        if self.sign_flip > 0:
            use = (draws[:, 2] < self.sign_flip) & valid
            x = torch.where(use[:, None], -x, x)

        if self.time_flip > 0:
            start = torch.full((buckets,), length, device=device, dtype=torch.long)
            start = start.scatter_reduce(0, index, first, reduce="amin")[index]
            source = torch.where(position[None] >= start[:, None],
                                 start[:, None] + (length - 1 - position[None]), position[None])
            reversed_ = x.gather(-1, source.clamp(0, length - 1))
            use = (draws[:, 3] < self.time_flip) & valid
            x = torch.where(use[:, None], reversed_, x)

        sequence = sequence.clone()
        sequence[:real_rows] = x.view(real_rows, width, length).to(sequence.dtype)
        return sequence


def _mean_abs(x, finite):
    total = torch.where(finite, x.abs(), 0.0).sum(-1)
    return (total / finite.sum(-1).clamp_min(1)).clamp_min(1e-6)


def _gamma(alpha, gen):
    """Draw Gamma(alpha,1) for alpha >= 1 using fixed rejection rounds on device."""
    d = alpha - 1.0 / 3.0
    c = 1.0 / torch.sqrt(9.0 * d)
    out = d.clone()
    done = torch.zeros_like(alpha, dtype=torch.bool)
    for _ in range(8):
        z = torch.randn(alpha.shape, generator=gen, device=alpha.device)
        u = torch.rand(alpha.shape, generator=gen, device=alpha.device)
        v = (1 + c * z) ** 3
        ok = (v > 0) & (torch.log(u.clamp_min(1e-12))
                        < 0.5 * z**2 + d - d * v + d * torch.log(v.clamp_min(1e-12)))
        out = torch.where(ok & ~done, d * v, out)
        done |= ok
    return out
