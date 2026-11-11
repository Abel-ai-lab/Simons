"""Average checkpoints uniformly or interpolate an anchor and model with WiSE."""

import hashlib
from pathlib import Path

import torch

# Forecast settings do not affect weights, so soup members may differ in them.
FORECAST_FIELDS = ("context_length", "views", "tail_paths", "sign_clip", "integer_snap",
                   "fit_row", "xreg", "xreg_gate", "xreg_square", "xreg_guard",
                   "xreg_guard_margin", "xreg_guard_min", "rows_only_history")


def sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path):
    from simons.training.trainer import read_checkpoint

    checkpoint = read_checkpoint(path)
    checkpoint.pop("optimizer", None)
    return checkpoint


def _check(members, paths):
    first = members[0]
    reference = {k: v for k, v in first["model_config"].items() if k not in FORECAST_FIELDS}
    for path, member in zip(paths[1:], members[1:]):
        config = {k: v for k, v in member["model_config"].items() if k not in FORECAST_FIELDS}
        if config != reference:
            differ = sorted(k for k in set(config) | set(reference)
                            if config.get(k) != reference.get(k))
            raise ValueError(f"{path}: model config differs in {differ}")
        if set(member["model"]) != set(first["model"]):
            raise ValueError(f"{path}: state keys differ")
        for key, tensor in first["model"].items():
            other = member["model"][key]
            if other.shape != tensor.shape or other.dtype != tensor.dtype:
                raise ValueError(f"{path}: tensor {key} {tuple(other.shape)}/{other.dtype} vs "
                                 f"{tuple(tensor.shape)}/{tensor.dtype}")


def merge(paths, wise_alpha: float | None = None) -> dict:
    """Return an averaged checkpoint without optimizer state.
    With `wise_alpha=a`, paths [anchor,model] yield (1-a)*anchor+a*model."""
    paths = [str(p) for p in paths]
    if not paths:
        raise ValueError("a soup needs at least one member")
    if wise_alpha is not None:
        if len(paths) != 2 or not 0.0 <= wise_alpha <= 1.0:
            raise ValueError("WiSE needs two members (anchor, model) and an alpha in [0, 1]")
        anchor, model = (_load(p) for p in paths)
        _check([model, anchor], [paths[1], paths[0]])
        state = {}
        for key, tensor in model["model"].items():
            if torch.is_floating_point(tensor):
                mix = (1.0 - wise_alpha) * anchor["model"][key].double() \
                    + wise_alpha * tensor.double()
                state[key] = mix.to(tensor.dtype)
            else:
                state[key] = tensor.clone()
        merged = {**model, "model": state}
        recipe = {"wise_alpha": float(wise_alpha), "anchor": paths[0], "member": paths[1]}
    else:
        members = [_load(p) for p in paths]
        _check(members, paths)
        state = {}
        for key, tensor in members[0]["model"].items():
            if torch.is_floating_point(tensor) and len(members) > 1:
                total = sum(member["model"][key].double() for member in members)
                state[key] = (total / len(members)).to(tensor.dtype)
            else:
                state[key] = tensor.clone()
        merged = {**members[0], "model": state}
        recipe = {"uniform": paths}
    merged["merge"] = {**recipe, "sha256": [sha256(p) for p in paths]}
    return merged


def write(paths, output, wise_alpha: float | None = None) -> Path:
    """Write `merge(paths, wise_alpha)` to `output` (a checkpoint.pt for `export`)."""
    from simons.training.trainer import write_checkpoint

    merged = merge(paths, wise_alpha)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_checkpoint(output, merged)
    return output
