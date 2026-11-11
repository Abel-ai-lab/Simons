"""Masked next-patch pinball loss on packed training rows."""

import math

import numpy as np
import torch
from torch.nn import functional as F

from simons.model import QUANTILES
from simons.training import network, packing
from simons.training.data import MAX_VARIATES

MIN_CONTEXT_PATCHES = 1


def contiguous_patch_mask(shape, device, rate_max: float, span_max: int) -> torch.Tensor:
    """[..., patches] hidden flags: per row a rate ~ U(0, rate_max) and span ~ U{1..span_max};
    spans start at each patch with probability rate / span."""
    patches = shape[-1]
    span = torch.randint(1, span_max + 1, (*shape[:-1], 1), device=device)
    rate = torch.rand(*shape[:-1], 1, device=device) * rate_max
    starts = torch.rand(shape, device=device) < rate / span
    position = torch.arange(patches, device=device)
    latest = torch.where(starts, position, -span_max).cummax(dim=-1).values
    return position - latest < span


def pinball_parts(prediction: torch.Tensor, target: torch.Tensor):
    """(sum of pinball losses over observed targets and quantiles, their count), no host sync."""
    observed = torch.isfinite(target)
    error = torch.where(observed, target, 0.0).unsqueeze(-1) - prediction
    levels = prediction.new_tensor(QUANTILES)
    loss = torch.maximum(levels * error, (levels - 1) * error)
    return (loss * observed.unsqueeze(-1)).sum(), observed.sum() * len(QUANTILES)


def pinball_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape[:-1] != target.shape or prediction.shape[-1] != len(QUANTILES):
        raise ValueError("prediction and target shapes differ")
    total, count = pinball_parts(prediction, target)
    if not bool(count):
        raise ValueError("Need at least one observed target")
    return total / count


def token_first(raw_first, far_len: int, patch: int, pool: int, far_tokens: int):
    """Map first observed raw-patch indices to tokens in the [far,near,future] layout.
    When far tokens are dropped, far steps map to the first near token."""
    step = raw_first * patch
    far = step // (patch * pool) if far_tokens else step * 0
    near = far_tokens + (step - far_len) // patch
    return np.where(step < far_len, far, near)


def packed_loss(model, sequence, host_slots, mask_rate_max, mask_span_max, far_on=True,
                row_tokens=0, checkpoint_blocks=0, roles=None, micro_batches=1, rank=None):
    """Return (mean next-patch pinball loss, observed target count) for sequence [B,V,T].
    `host_slots()` returns numpy window IDs and first observed patch indices [B,V].
    With `micro_batches` > 1, this runs backward internally and returns a detached loss;
    the caller must not back-propagate again."""
    patch = model.config.patch_length
    batch, variates, length = sequence.shape
    spec = network.layout(model.config, length)
    far_len = spec[0] if spec is not None else 0
    near = sequence[..., far_len:]
    patches = near.shape[-1] // patch
    # Far tokens supply context only and must remain unmasked.
    hidden = contiguous_patch_mask((batch, variates, patches), sequence.device, mask_rate_max,
                                   mask_span_max)
    if roles is not None:
        hidden = roles.hidden(hidden)
    observed = torch.isfinite(near).unflatten(-1, (-1, patch)).any(-1)
    hidden &= observed.cumsum(-1) > MIN_CONTEXT_PATCHES
    mask = hidden.repeat_interleave(patch, -1)
    if far_len:
        mask = F.pad(mask, (far_len, 0), value=False)
    inputs = sequence.masked_fill(mask, float("nan"))
    tokens, rotary, started, loc, scale, far, far_tokens = network.front(
        model, inputs, far_on, fixed=None if roles is None else roles.fixed)
    loc, scale = loc[..., far_tokens:], scale[..., far_tokens:]
    target = near.unflatten(-1, (-1, patch))[:, :, 1:].double()
    target = torch.asinh((target - loc[..., :-1, None]) / scale[..., :-1, None]).float()
    seen = (observed & ~hidden).cummax(-1).values
    target = target.masked_fill(~seen[..., :-1, None], float("nan"))
    # Position i predicts patch i + 1; the last position and far tokens carry no target.
    target = torch.cat([target, target.new_full((batch, variates, 1, patch), float("nan"))], 2)
    if far_tokens:
        target = torch.cat([target.new_full((batch, variates, far_tokens, patch), float("nan")),
                            target], 2)
    if roles is not None:
        target = roles.target(target)
    size = far_tokens + patches

    series_ids, slot_first = host_slots()
    if far_len:
        slot_first = token_first(slot_first, far_len, patch, model.config.far_pool, far_tokens)
    source, segment, rows = packing.plan(series_ids.reshape(-1), slot_first.reshape(-1), size,
                                         row_tokens=row_tokens)
    unpacked, size = size, max(size, row_tokens)
    source, segment = (torch.from_numpy(a) for a in (source, segment))
    if sequence.is_cuda:
        source, segment = (a.pin_memory().to(sequence.device, non_blocking=True)
                           for a in (source, segment))
    shape = (rows, MAX_VARIATES, size)
    gather = packing.gather
    packed_tokens = gather(tokens.reshape(-1, tokens.shape[-1]), source).view(*shape, -1)
    cos, sin = (gather(t.reshape(-1, t.shape[-1]), source) for t in rotary)
    packed_rotary = tuple(t.view(rows * MAX_VARIATES, 1, size, -1) for t in (cos, sin))
    packed_started = gather(started.reshape(-1), source, False).view(shape)
    packed_target = gather(target.reshape(-1, patch), source, float("nan")).view(*shape, patch)
    packed_far = None
    if far is not None:
        packed_far = gather(far.expand(batch, variates, unpacked).reshape(-1), source,
                            False).view(shape)
    packed_roles = None
    if roles is not None:
        packed_roles = gather(roles.token_roles(unpacked).reshape(-1), source, 0).view(shape)
    segment = segment.view(shape)
    if micro_batches <= 1:
        prediction = network.trunk(model, packed_tokens, packed_rotary, packed_started, segment,
                                   packed_far, checkpoint_blocks, roles=packed_roles,
                                   fp32_residual=True)
        return pinball_loss(prediction, packed_target), torch.isfinite(packed_target).sum()
    count = torch.isfinite(packed_target).sum()
    world = rank.world if rank is not None and rank.active else 1
    total = count.detach().to(torch.float64).reshape(1)
    if world > 1:
        import torch.distributed as dist

        total = total.clone()
        dist.all_reduce(total)
    total = total[0].clamp_min(1.0)
    chunks = chunk_bounds(rows, micro_batches)
    counts = [int(torch.isfinite(packed_target[a:b]).sum()) for a, b in chunks]
    summed = torch.zeros((), dtype=torch.float64, device=sequence.device)
    for (a, b), n in zip(chunks, counts):
        if n == 0:
            continue
        cut = slice(a * MAX_VARIATES, b * MAX_VARIATES)
        prediction = network.trunk(
            model, packed_tokens[a:b], tuple(t[cut] for t in packed_rotary),
            packed_started[a:b], segment[a:b],
            None if packed_far is None else packed_far[a:b], checkpoint_blocks,
            roles=None if packed_roles is None else packed_roles[a:b], fp32_residual=True)
        part, _ = pinball_parts(prediction, packed_target[a:b])
        (part * (world / (total * len(QUANTILES))).to(part.dtype)).backward()
        summed += part.detach().to(torch.float64)
    if world > 1:
        import torch.distributed as dist

        dist.all_reduce(summed)
    return (summed / (total * len(QUANTILES))).float(), count


def chunk_bounds(rows: int, chunks: int):
    """Contiguous [a, b) row ranges, as equal as possible, none empty (at most `rows`)."""
    chunks = max(1, min(chunks, rows))
    base, extra = divmod(rows, chunks)
    out, a = [], 0
    for c in range(chunks):
        b = a + base + (1 if c < extra else 0)
        out.append((a, b))
        a = b
    return out


@torch.no_grad()
def validation_loss(model, batches, device, autocast) -> float | None:
    """Mean pinball loss of the forecast patches of held-out (history, target, ids) batches."""
    if not batches:
        return None
    was_training = model.training
    model.eval()
    patch, total, count = model.config.patch_length, 0.0, 0.0
    for history, target, series_ids in batches:
        history, target = history.to(device), target.to(device)
        series_ids = series_ids.to(device)
        future = math.ceil(target.shape[-1] / patch)
        values = F.pad(history, (0, future * patch), value=float("nan"))
        with autocast:
            tokens, rotary, started, loc, scale, far, _ = network.front(model, values)
            segments = series_ids[:, :, None].expand(*started.shape)
            prediction = network.trunk(model, tokens, rotary, started, segments, far)
        first = tokens.shape[2] - future - 1
        window = slice(first, first + future)
        goal = F.pad(target, (0, future * patch - target.shape[-1]), value=float("nan"))
        goal = goal.unflatten(-1, (future, patch)).double()
        goal = torch.asinh((goal - loc[..., window, None]) / scale[..., window, None]).float()
        part, n = pinball_parts(prediction[:, :, window], goal)
        total += float(part)
        count += float(n)
    model.train(was_training)
    return total / max(count, 1.0)
