"""Packed and covariate training passes over the shared Simons model."""

import math

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from simons.model import QUANTILES, front, future_steps, layout  # noqa: F401


def trunk(model, tokens: torch.Tensor, rotary, started: torch.Tensor, segments: torch.Tensor,
          far: torch.Tensor | None = None, checkpoint_blocks: int = 0, roles=None,
          fp32_residual: bool = False) -> torch.Tensor:
    """Return packed quantiles [B,V,S,patch,9] in asinh units.
    Inputs: tokens [B,V,S,2p], rotary [B*V,1,S,half], started/segments [B,V,S] (-1: filler).
    Far flags: [B,V,S], [S] or None; roles: [B,V,S], [B,V] or None."""
    batch, variates, patches, _ = tokens.shape
    x = model.embed(tokens) + model.embed_mlp(tokens)
    if far is not None:
        x = x + far[..., None].to(x.dtype) * model.far_type
    if fp32_residual:
        x = x.float()
    read_after = -1
    if roles is not None:
        vectors = model.covariates.role_vectors(roles)
        x = x + (vectors if roles.dim() == 3 else vectors[:, :, None]).to(x.dtype)
        if model.covariates.read is not None:
            read_after = model.config.read_after
    x = x.reshape(batch * variates, patches, -1)
    segment = segments.reshape(batch * variates, patches)
    start = started.reshape(batch * variates, patches)
    position = torch.arange(patches, device=tokens.device)
    query, key = position[:, None], position[None, :]
    allowed = (segment[:, :, None] == segment[:, None, :]) & (key <= query)[None]
    allowed &= start[:, None, :] | (key == query)[None]
    mask = allowed[:, None]
    ids = segments.transpose(1, 2)  # [B,S,V]
    related = (ids[..., :, None] == ids[..., None, :]).reshape(batch * patches, 1, variates,
                                                               variates)
    for index, block in enumerate(model.time_blocks):
        if index < checkpoint_blocks and torch.is_grad_enabled():
            x = checkpoint(block, x, mask, rotary, use_reentrant=False)
        else:
            x = block(x, mask, rotary=rotary)
        if index == read_after:
            if roles.dim() == 3:
                x = packed_read(model.covariates.read, x, segments, roles)
            else:
                x = model.covariates.read_rows(x, related, batch, variates, patches, roles)
    x = x.reshape(batch, variates, patches, -1).transpose(1, 2).reshape(batch * patches,
                                                                        variates, -1)
    x = model.variate_block(x, related)
    x = x.reshape(batch, patches, variates, -1).transpose(1, 2)
    return quantiles(model, x)


def packed_read(block, x, segments, roles):
    """Update target tokens x [R*V,S,D] from covariates of the same window and position.
    `segments` and `roles` have shape [R,V,S]."""
    from simons.covariates import ROLE_BINARY, ROLE_FIT, ROLE_KNOWN, ROLE_TARGET

    rows, variates, size = segments.shape
    xm = x.reshape(rows, variates, size, -1).transpose(1, 2).reshape(rows * size, variates, -1)
    ids = segments.transpose(1, 2)  # [R,S,V]
    r = roles.transpose(1, 2)
    keys = (r == ROLE_KNOWN) | (r == ROLE_FIT) | (r == ROLE_BINARY)
    queries = (r == ROLE_TARGET) & (ids >= 0)
    related = ids[..., :, None] == ids[..., None, :]
    cross = queries[..., :, None] & keys[..., None, :] & related  # [R,S,V,V]
    update = cross.any(-1).reshape(rows * size, variates, 1)
    eye = torch.eye(variates, dtype=torch.bool, device=segments.device)
    mask = (cross | eye).reshape(rows * size, 1, variates, variates)
    y = block(xm, mask)
    out = xm + update.to(xm.dtype) * (y - xm)
    return out.reshape(rows, size, variates, -1).transpose(1, 2).reshape(rows * variates, size,
                                                                         -1)


def covariate_forward(model, values: torch.Tensor, series_ids: torch.Tensor,
                      covariates: torch.Tensor, fit_keep: torch.Tensor | None = None):
    """Return next-patch quantiles [B,V,P,patch,9], loc/scale [B,V,P]; P = T / patch.
    Values [B,V,T] and covariates [B,C,T] span history and future; `fit_keep` [B] selects fits."""
    from simons.covariates import ROLE_BINARY, covariate_rows

    c = model.config
    patch = c.patch_length
    variates = values.shape[1]
    grown = covariate_rows(c, values, series_ids, covariates, fit_keep)
    roles, fixed = None, None
    if grown is not None:
        values, series_ids, roles = grown
        fixed = (roles == ROLE_BINARY) if c.binary_rows else None
    tokens, rotary, started, loc, scale, far, far_tokens = front(model, values, fixed=fixed)
    batch, rows, size = started.shape
    # Attention masks isolate skipped prefixes; retain at least the last history token.
    last = size - math.ceil(c.prediction_length / patch) - 1
    any_started = started.reshape(-1, size).any(0)
    crop = int(torch.where(any_started, torch.arange(size, device=values.device), size)
               .amin().clamp_max(last))
    segments = series_ids[:, :, None].expand(batch, rows, size - crop)
    q = trunk(model, tokens[:, :, crop:], tuple(t[:, :, crop:] for t in rotary),
              started[:, :, crop:], segments, None if far is None else far[crop:], roles=roles)
    q = F.pad(q, (0, 0, 0, 0, crop, 0), value=float("nan"))
    if far_tokens:
        pool = c.far_pool
        q, loc, scale = (torch.cat([t[:, :, :far_tokens].repeat_interleave(pool, 2),
                                    t[:, :, far_tokens:]], 2) for t in (q, loc, scale))
    return q[:, :variates], loc[:, :variates], scale[:, :variates]


def quantiles(model, x: torch.Tensor) -> torch.Tensor:
    """[..., width] features -> [..., patch, 9] sorted quantiles (asinh units), unclamped."""
    raw = model.head(model.norm(x)).float()
    return raw.unflatten(-1, (model.config.patch_length, len(QUANTILES))).sort(dim=-1).values
