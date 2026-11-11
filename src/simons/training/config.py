"""Training YAML: `model` contains SimonsConfig fields; `training` contains the keys below.

`stage1` trains on synthetic series; `stage2` mixes real and synthetic data.
Keys ending in `_stage` or `_stages` refer to context-curriculum phases within a run.
See docs/TRAINING.md for the four-stage recipe."""

from pathlib import Path

import yaml

from simons.config import SimonsConfig

FAMILIES = ("kernel", "sarima", "sde", "step", "spike", "cauker", "rhythm", "wave", "fpfn", "levy",
            "ets", "pulse", "chaos", "epi", "hydro", "queue", "thermal", "calendar")

DEFAULTS = {
    "stage": None,  # stage1 (synthetic) | stage2 (real + synthetic)
    "steps": None,  # optimizer steps of the full plan
    "seed": None,
    "device": "auto",  # auto | cuda | cpu
    "max_steps": None,  # stop early after this many steps of this run (smoke tests; None: none)
    "checkpoint_every": 1000,
    "log_every": 25,
    # Optimizer (optim.py)
    "muon_lr": 0.002,  # peak LR of the hidden matrices (NorMuon)
    "adam_lr": 0.0005,  # peak LR of embeddings, head, biases and norms (AdamW)
    "muon_weight_decay": 0.05,  # decoupled, scaled with the LR, hidden matrices only
    "mup_base_width": 704,  # width transfer base d0
    # Learning-rate schedule (schedule.py)
    "lr_schedule": "warmup_decay",  # warmup_decay | wsd | linear
    "warmup_steps": 1000,  # warmup_decay
    "decay_steps": 5000,  # warmup_decay
    "decay_shape": "linear",  # warmup_decay and wsd: linear | sqrt
    "warmup_fraction": 0.05,  # wsd, linear
    "decay_fraction": 0.3,  # wsd
    "final_lr_fraction": 0.0,
    # Objective (objective.py)
    "mask_rate_max": 0.4,
    "mask_span_max": 8,
    # Packing (packing.py); exact, so this changes speed only
    "pack_row_tokens": 288,  # packed row length in stages with far tokens (0: window length)
    # Recompute the first K time blocks during backward to save memory (0: off).
    "checkpoint_blocks": 0,
    # Split each GPU's packed rows into chunks to save memory (1: off).
    "micro_batches": 1,
    # Synthetic series (simons.synth)
    "synthetic_families": "kernel,sarima,sde,step,spike,cauker,rhythm,wave,fpfn",
    "synthetic_family_weights": None,  # one weight per family (None: equal)
    "synthetic_noise": 0.3,  # share of series through the count / shock / spike noisers
    "synthetic_pad": 0.4,  # share of series with a short (left-padded) history
    "synthetic_pool_steps": 6,  # steps of series drawn per pool
    "synthetic_pool_steps_long": 2,  # the same in stages after the first
    "synthetic_augment": "mix,season,filter,lattice",  # simons.synth.augment ("" = off)
    "synthetic_augment_rate": 0.4,
    "synthetic_augment_weights": "0.35,0.25,0.2,0.2",
    "synthetic_observe": "",  # simons.synth.observe categories ("" = off)
    "synthetic_observe_rate": 0.1,
    "synthetic_observe_duty_periodic": 0.5,
    "synthetic_multi_fraction": 0.5,  # share of synthetic rows holding coupled groups
    "mech_bank": 16384,  # simons.synth.mech: chaos trajectories (other banks: half)
    "mech_length": 9216,
    "mech_refresh": 400,  # maker calls between bank refreshes (0: never)
    "mech_refresh_part": 1,  # a refresh re-simulates 1/part of the bank
    # `stage1` runs (synthetic)
    "synthetic_rows": 36,  # rows of 32 series per step, over all devices
    "synthetic_context": 8192,  # history steps of the synthetic windows
    # `stage2` runs (real + synthetic): real data (data.py, mixture.py)
    "data": None,  # corpus folder (train-*.parquet) or "toy"
    "validation_data": None,  # folder with validation-*.parquet (None: no validation)
    "batch_size": 784,  # real windows per step, over all devices
    "context_schedule": None,  # phases, e.g. "1024:0.15,7680:0.45,32768:0.40" (None: one)
    "synthetic_fraction": 0.4,  # share of packed rows that are synthetic
    "synthetic_only_stages": 0,  # leading curriculum phases replaced by `stage1` runs
    "synthetic_ramp_steps": 0,  # share ramp 1 -> synthetic_fraction after them
    "mixture_cap_epochs": 6.0,  # UniMax over (domain, subdomain) groups; max passes over a source
    "stitch_from_stage": -1,  # phases >= k stitch consecutive segments (-1: never)
    "far_from_stage": 0,  # phases >= k keep far tokens; other phases use far history for stats
    "augment_downsample": 0.4,  # augment.py / warp.py: real-window augmentation rates
    "augment_downsample_max": 8,
    "augment_stretch": 0.25,
    "augment_stretch_up": 0.5,
    "augment_warp": 0.2,
    "augment_warp_convex": 0.65,
    "augment_censor": 0.2,
    "augment_amplitude": 0.2,
    "augment_sign_flip": 0.2,
    "augment_time_flip": 0.1,
    "augment_mixup": 0.2,
    "augment_mixup_alpha": 1.5,
    "loader_workers": 16,
    # Shape-cluster sampling (clusters.py): draw share, mass cap, labels and build settings.
    "shape_mix": 0.0,
    "shape_cap": 10.0,
    "shape_clusters": None,
    "shape_k": 1024,
    "shape_anchors": 200000,
    "shape_context": 1024,
    "shape_table_seed": 0,
    "shape_shards": 1,
    # Warm start
    "warm_start": None,  # checkpoint.pt or model folder to load
    "warm_start_stage": 1,  # the run starts at this phase of its context curriculum
    "rewarm_from": 0.15,
    "rewarm_steps": 300,
    # Start step (None: warm_start_stage's first step for stage2, 0 for stage1); LR scale.
    "warm_start_step": None,
    "warm_lr_scale": 1.0,
    # Read initialization: keep saved weights, or copy the variate block with zero writes.
    # Missing read weights use the variate initialization.
    "covariate_read_init": "keep",
    # Role/effect shares per phase (fractions of run steps); stage1 supports effect rows.
    # effect_mix orders generic, classical, energy, retail, lagged. Worker count affects draws.
    "role_phases": None,
    "role_share": "0",
    "effect_share": "0",
    "effect_mix": None,
    "effect_workers": 4,
    # Main covariate pass (stage2): weight 0 disables it; mix orders synthetic, sibling, real.
    # Base data: first train file or first rows per dataset subfolder (None: data).
    # Source weights: {source: weight} JSON; exclusions: {"items": {source: [item_id, ...]}}.
    # Norm: global weighted mean or mean of rank means. Worker count affects draws.
    "covariate_weight": 0.0,
    "covariate_data": None,
    "covariate_base_data": None,
    "covariate_mix": "0.63,0.25,0.12",
    "covariate_tokens": 4096,  # per step over all GPUs, split evenly between them
    "covariate_max_batch": 128,  # windows per GPU and step
    "covariate_workers": 3,
    "covariate_history_weight": 0.1,
    "covariate_source_weights": None,
    "covariate_exclude": None,
    "covariate_norm": "global",
    # Long pass: real series meeting length/observed-share limits; weight 0 disables it.
    # long_tokens=None uses covariate_tokens.
    "long_weight": 0.0,
    "long_tokens": None,
    "long_workers": 2,
    "long_min_length": 16384,
    "long_min_observed": 0.75,
    # Export
    "export_context_length": None,  # context_length of the exported model (None: the model's)
}

FRACTIONS = ("synthetic_noise", "synthetic_pad", "synthetic_augment_rate", "synthetic_observe_rate",
             "synthetic_observe_duty_periodic", "synthetic_multi_fraction", "synthetic_fraction",
             "final_lr_fraction", "warmup_fraction", "decay_fraction", "mask_rate_max",
             "augment_downsample", "augment_stretch", "augment_stretch_up", "augment_warp",
             "augment_warp_convex", "augment_censor", "augment_amplitude", "augment_sign_flip",
             "augment_time_flip", "augment_mixup", "rewarm_from", "covariate_history_weight",
             "long_min_observed")


def weights(value, count: int, name: str):
    """A "w1,w2,..." string as floats (None stays None), one per item."""
    if value is None:
        return None
    out = [float(w) for w in str(value).split(",")]
    if len(out) != count or min(out) < 0 or sum(out) <= 0:
        raise ValueError(f"{name} needs {count} nonnegative weights")
    return out


def shares(value, name: str) -> list[float]:
    """A "p1,p2,..." string (or one number) as a list of fractions."""
    out = [float(v) for v in str(value).split(",")]
    if min(out) < 0 or max(out) > 1:
        raise ValueError(f"{name} must hold fractions from 0 to 1")
    return out


def covariate_roles(s) -> dict | None:
    """The role settings of `covariates.roles` from the config keys, or None when off."""
    role = shares(s["role_share"], "role_share")
    effect = shares(s["effect_share"], "effect_share")
    if max(role) <= 0 and max(effect) <= 0:
        return None
    phases = shares(s["role_phases"], "role_phases") if s["role_phases"] is not None else []
    if len(effect) == 1 and len(role) > 1:
        effect = effect * len(role)
    mix = None
    if s["effect_mix"] is not None:
        from simons.synth.covariates.families import FAMILIES

        mix = dict(zip(FAMILIES, weights(s["effect_mix"], len(FAMILIES), "effect_mix")))
    return {"boundaries": phases, "p_role": role, "p_eff": effect, "effect_mix": mix,
            "effect_workers": int(s["effect_workers"])}


def check_far_region(model, context: int):
    """A training context longer than near_context needs a far region of whole far tokens."""
    block = model.patch_length * model.far_pool
    if context > model.near_context and (context - model.near_context) % block:
        raise ValueError(f"context {context}: context - near_context must be a multiple of "
                         f"patch_length * far_pool ({block})")


def load(path: Path, overrides: dict | None = None):
    """(SimonsConfig, training settings dict) from a YAML file, validated; `overrides` values
    that are not None replace the file's training keys."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"model", "training"}:
        raise ValueError("The config must contain exactly a model and a training section")
    model_keys, training = raw["model"] or {}, raw["training"] or {}
    if not isinstance(model_keys, dict) or not isinstance(training, dict):
        raise ValueError("The model and training sections must be mappings of keys")
    model = SimonsConfig(**model_keys)
    unknown = set(training) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown training keys: {sorted(unknown)}")
    given = {k: v for k, v in (overrides or {}).items() if v is not None}
    return model, validate(model, {**DEFAULTS, **training, **given})


def validate(model, s: dict) -> dict:
    if s["stage"] not in ("stage1", "stage2"):
        raise ValueError("stage must be stage1 (synthetic) or stage2 (real + synthetic)")
    for key in ("steps", "seed", "checkpoint_every", "log_every"):
        low = 0 if key == "seed" else 1
        if isinstance(s[key], bool) or not isinstance(s[key], int) or s[key] < low:
            kind = "nonnegative" if low == 0 else "positive"
            raise ValueError(f"{key} must be a {kind} integer, got {s[key]!r}")
    if s["max_steps"] is not None and (isinstance(s["max_steps"], bool)
                                       or not isinstance(s["max_steps"], int)
                                       or s["max_steps"] < 1):
        raise ValueError(f"max_steps must be a positive integer or null, got {s['max_steps']!r}")
    for key in FRACTIONS:
        if not isinstance(s[key], (int, float)) or not 0 <= s[key] <= 1:
            raise ValueError(f"{key} must be a number from 0 to 1")
    if s["augment_mixup"] > 0 and not s["augment_mixup_alpha"] >= 1:
        raise ValueError("augment_mixup_alpha must be at least 1")
    if s["lr_schedule"] not in ("warmup_decay", "wsd", "linear"):
        raise ValueError("lr_schedule must be warmup_decay, wsd or linear")
    if s["decay_shape"] not in ("linear", "sqrt"):
        raise ValueError("decay_shape must be linear or sqrt")
    if s["warm_start_step"] is not None and not 0 <= s["warm_start_step"] < s["steps"]:
        raise ValueError("warm_start_step must be a step of the plan")
    if s["warm_lr_scale"] <= 0:
        raise ValueError("warm_lr_scale must be positive")
    width = s["mup_base_width"]
    if isinstance(width, bool) or not isinstance(width, (int, float)) or width <= 0:
        raise ValueError(f"mup_base_width must be a positive number, got {width!r}")
    if s["covariate_read_init"] not in ("keep", "variate"):
        raise ValueError("covariate_read_init must be keep or variate")
    if s["covariate_norm"] not in ("global", "rank"):
        raise ValueError("covariate_norm must be global or rank")
    roles = covariate_roles(s)
    passes = s["covariate_weight"] > 0 or s["long_weight"] > 0
    if (roles or passes) and not model.covariates:
        raise ValueError("covariate roles and passes need a model with covariates: true")
    if (roles or passes) and model.patch_length != 32:
        raise ValueError("covariate roles and passes need a model with patch_length 32")
    if roles:
        from simons.training.covariates.roles import settings as role_settings

        role_settings(roles)
        if s["stage"] == "stage2" and max(roles["p_eff"]) > 0:
            raise ValueError("effect rows belong to stage1 runs (effect_share 0 in stage2 runs)")
        if s["synthetic_multi_fraction"] <= 0:
            raise ValueError("covariate roles need coupled synthetic groups "
                             "(synthetic_multi_fraction > 0)")
        if model.read_after < 0:
            raise ValueError("covariate roles need a model with the one-way read "
                             "(read_after >= 0)")
        if not model.binary_rows:
            raise ValueError("covariate roles need a model with binary_rows: true")
    if passes and s["stage"] != "stage2":
        raise ValueError("the covariate passes belong to stage2 runs")
    if passes and model.prediction_length < 512:
        raise ValueError("the covariate passes draw horizons of up to 512 steps: they need "
                         "prediction_length >= 512")
    if s["long_weight"] > 0 and s["covariate_weight"] <= 0:
        raise ValueError("the long pass runs with the main covariate pass")
    weights(s["covariate_mix"], 3, "covariate_mix")
    if not isinstance(s["micro_batches"], int) or s["micro_batches"] < 1:
        raise ValueError("micro_batches must be a positive integer")
    if (not isinstance(s["checkpoint_blocks"], int)
            or not 0 <= s["checkpoint_blocks"] <= model.layers):
        raise ValueError("checkpoint_blocks must be from 0 to the model's layers")
    augment = [c for c in str(s["synthetic_augment"] or "").split(",") if c]
    if augment and s["synthetic_augment_rate"] > 0:
        from simons.synth.augment import CATEGORIES

        if not set(augment) <= set(CATEGORIES):
            raise ValueError(f"synthetic_augment must list some of {CATEGORIES}")
        weights(s["synthetic_augment_weights"], len(augment), "synthetic_augment_weights")
    families = s["synthetic_families"].split(",")
    if not set(families) <= set(FAMILIES):
        raise ValueError(f"synthetic_families must list some of {FAMILIES}")
    weights(s["synthetic_family_weights"], len(families), "synthetic_family_weights")
    if s["stage"] == "stage1":
        if (s["synthetic_context"] > model.context_length
                or s["synthetic_context"] % model.patch_length):
            raise ValueError("synthetic_context must be a multiple of patch_length within the "
                             "model's context")
        check_far_region(model, s["synthetic_context"])
        if s["synthetic_rows"] < 1:
            raise ValueError("synthetic_rows must be positive")
    else:
        if not s["data"]:
            raise ValueError("a stage2 run needs data")
        from simons.training.schedule import parse_schedule

        stages = parse_schedule(s["context_schedule"], s["steps"], model.context_length,
                                model.patch_length)
        for context, _, _ in stages:
            check_far_region(model, context)
        if s["synthetic_only_stages"] >= len(stages):
            raise ValueError("synthetic_only_stages must leave at least one real stage")
        if s["warm_start"] is not None and not 0 <= s["warm_start_stage"] < len(stages):
            raise ValueError("warm_start_stage must be a phase of the context schedule")

        if not 0 <= s["shape_mix"] <= 1 or s["shape_cap"] <= 0:
            raise ValueError("shape_mix must be from 0 to 1 and shape_cap positive")
        if s["shape_mix"] > 0 and not s["shape_clusters"]:
            raise ValueError("shape_mix needs shape_clusters (python -m simons.training "
                             "clusters writes it)")
        if s["shape_mix"] > 0 and (s["shape_context"] % model.patch_length
                                   or s["shape_context"] > model.near_context):
            raise ValueError("shape_context must be a multiple of patch_length within "
                             "near_context")
    return s
