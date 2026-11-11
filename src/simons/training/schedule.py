"""Context-curriculum phases, learning-rate multipliers and synthetic shares."""

import math

import numpy as np
import torch


def parse_schedule(spec, steps: int, final_context: int, patch: int):
    """[(context, start_step, end_step)] per curriculum phase, covering [0, steps); None gives
    one phase. Every phase must get at least one step."""
    if spec is None:
        return [(final_context, 0, steps)]
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError("context_schedule must be a string like '1024:0.8,4096:0.2'")
    stages = []
    for part in spec.split(","):
        context, fraction = part.split(":")
        context, fraction = int(context), float(fraction)
        if context <= 0 or context % patch or not 0 < fraction <= 1:
            raise ValueError(f"Bad context_schedule stage {part!r}")
        stages.append((context, fraction))
    if abs(sum(f for _, f in stages) - 1) > 1e-6:
        raise ValueError("context_schedule fractions must sum to 1")
    if stages[-1][0] != final_context:
        raise ValueError("The last context_schedule stage must use the model's context_length")
    if max(c for c, _ in stages) > final_context:
        raise ValueError("No stage may exceed the model's context_length")
    out, cumulative, start = [], 0.0, 0
    for index, (context, fraction) in enumerate(stages):
        cumulative += fraction
        end = steps if index == len(stages) - 1 else round(cumulative * steps)
        if end <= start:
            raise ValueError(f"context_schedule phase {index} ({context}:{fraction}) gets no "
                             f"step of the {steps}-step plan")
        out.append((context, start, end))
        start = end
    return out


def base_multiplier(step: int, s) -> float:
    """The schedule's LR multiplier at `step` (0-based), before any warm-start ramp."""
    steps = s["steps"]
    warmup = warmup_steps(s)
    if step < warmup:
        return (step + 1) / warmup
    final = s["final_lr_fraction"]
    if s["lr_schedule"] == "warmup_decay":
        decay = max(1, s["decay_steps"])
        start = steps - decay
        if step < start:
            return 1.0
        p = (step - start + 1) / decay
        shape = 1 - math.sqrt(p) if s["decay_shape"] == "sqrt" else 1 - p
        return final + (1 - final) * shape
    if s["lr_schedule"] == "wsd":
        start = steps - max(1, int(s["decay_fraction"] * steps))
        if step < start:
            return 1.0
        progress = (step - start) / max(1, steps - start)
        shape = 1 - math.sqrt(progress) if s["decay_shape"] == "sqrt" else 1 - progress
        return final + (1 - final) * shape
    progress = (step - warmup) / max(1, steps - warmup)
    return final + (1 - final) * (1 - progress)


def warmup_steps(s) -> int:
    if s["lr_schedule"] == "warmup_decay":
        return max(1, s["warmup_steps"])
    return max(1, int(s["warmup_fraction"] * s["steps"]))


def multiplier(step: int, s, first_step: int = 0, warm: bool = False) -> float:
    """Return the LR multiplier at `step`, including warm-start scaling and a re-warm ramp."""
    if not warm:
        return base_multiplier(step, s)
    value = base_multiplier(max(step, warmup_steps(s)), s)
    ramp = s["rewarm_steps"]
    if ramp > 0:
        t = min(1.0, (step - first_step + 1) / ramp)
        value *= s["rewarm_from"] + (1.0 - s["rewarm_from"]) * t
    return value * s.get("warm_lr_scale", 1.0)


def share_function(stages, synthetic_stages: int, ramp: int, base: float):
    """step -> synthetic share: 1 through the first `synthetic_stages` phases, then a linear ramp
    to `base` over `ramp` steps, then `base`."""
    if not synthetic_stages:
        return lambda step: base
    switch = stages[synthetic_stages - 1][2]

    def share(step):
        if step < switch:
            return 1.0
        if ramp and step < switch + ramp:
            return 1.0 + (base - 1.0) * (step - switch + 1) / ramp
        return base

    return share


def stage_shares(share, stages):
    """Mean synthetic share over each (context, start, end) phase's steps (for the data plan)."""
    return [float(np.mean([share(t) for t in range(start, end)])) for _, start, end in stages]


def ramp_rows(rows: int, share: float, base: float) -> torch.Tensor:
    """Return kept row indices at `share`, retaining the regular synthetic-row count.
    A ramp step carries no more tokens than a regular step."""
    synthetic = round(base * rows)
    real = min(rows - synthetic, round((1 - share) * rows))
    return torch.cat([torch.arange(real), torch.arange(rows - synthetic, rows)])


def ramp_fraction(rows_before: int, rows_after: int, base: float) -> float:
    """The mixer fraction that makes round(fraction * rows_after) the regular synthetic rows."""
    return round(base * rows_before) / max(1, rows_after)
