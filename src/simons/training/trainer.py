"""Train synthetic or mixed real/synthetic batches and export model checkpoints."""

import copy
import json
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from simons.model import Simons
from simons.synth import SyntheticMixer, mech, observe
from simons.synth import augment as synthetic_augment
from simons.training import config as config_module
from simons.training import network, objective, schedule
from simons.training.augment import RealAugmenter
from simons.training.data import (
    MAX_VARIATES,
    StepIndices,
    Windows,
    collate,
    load,
    pack_windows,
    write_json,
)
from simons.training.distributed import Rank
from simons.training.optim import NorMuonAdamW
from simons.training.warp import power_warp

CHECKPOINT_FORMAT = "simons-train-1"
VALIDATION_WINDOWS = 1024


def device_for(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def read_checkpoint(path) -> dict:
    """A training checkpoint.pt (or soup) on the CPU; raises unless it has the format."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a Simons training checkpoint")
    return checkpoint


def write_checkpoint(path: Path, payload: dict):
    """Save `payload` to `path` atomically (a temporary file, then a rename)."""
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def save_checkpoint(path: Path, model, optimizer, settings, step, seconds):
    write_checkpoint(path, {
        "format": CHECKPOINT_FORMAT,
        "model_config": model.config.to_dict(),
        "training": settings,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "elapsed_seconds": seconds,
    })


def load_weights(model, path, read_init: str = "keep") -> dict:
    """Load checkpoint/model-folder weights, allowing missing covariate modules.
    `read_init="variate"` resets the read from the variate block; "keep" retains saved weights."""
    path = Path(path)
    if path.is_dir():
        from safetensors.torch import load_file

        state = load_file(path / "model.safetensors")
        info = {"source": str(path)}
    else:
        checkpoint = read_checkpoint(path)
        state = checkpoint["model"]
        info = {"source": str(path), "step": checkpoint["step"]}
    if model.covariates is None:
        model.load_state_dict(state, strict=True)
        return info
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not k.startswith("covariates.") for k in missing):
        raise ValueError(f"{path}: missing weights {missing[:4]}, unexpected {unexpected[:4]}")
    read = model.covariates.read
    if read is not None:
        saved = any(k.startswith("covariates.read.") for k in state)
        if read_init == "variate" or not saved:
            init_read(read, model.variate_block)
            info["read"] = "from the variate block"
        else:
            info["read"] = "kept"
    return info


@torch.no_grad()
def init_read(read, variate_block):
    """Copy the variate block into the one-way read with zero output writes."""
    read.load_state_dict(variate_block.state_dict())
    torch.nn.init.zeros_(read.out.weight)
    torch.nn.init.zeros_(read.mlp[2].weight)
    torch.nn.init.zeros_(read.mlp[2].bias)


def multi_pool(s, seed):
    """Return a coupled multivariate synthetic pool, or None when disabled."""
    if s["synthetic_multi_fraction"] <= 0:
        return None
    from simons.synth.multi import MultiPool

    return MultiPool(s["synthetic_multi_fraction"], seed=seed * 6271 + 29)


def make_mixer(s, device, seed, first_step, pool_scale, fraction):
    families = s["synthetic_families"].split(",")
    gen = torch.Generator(device=device)
    offset = (7883, 29) if s["stage"] == "stage1" else (7919, 17)
    gen.manual_seed(seed * offset[0] + offset[1] + first_step)
    view = None
    if s["synthetic_observe"]:
        view = observe.make_config(s["synthetic_observe"], s["synthetic_observe_rate"],
                                   s["synthetic_observe_duty_periodic"])
    return SyntheticMixer(
        fraction, gen, families, s["synthetic_noise"], s["synthetic_pad"],
        s["synthetic_pool_steps"] * pool_scale,
        weights=config_module.weights(s["synthetic_family_weights"], len(families),
                                      "synthetic_family_weights"),
        multi=multi_pool(s, seed), observe=view,
    )


def train(config_path: Path, output: Path, data: str | None = None,
          warm_start: str | None = None, resume: Path | None = None,
          covariate_data: str | None = None, max_steps: int | None = None,
          shape_clusters: str | None = None):
    """Train one YAML-configured stage and return its final checkpoint.pt path.
    `output` must be new or empty; `max_steps` limits execution without changing the plan.
    `warm_start` loads weights with a fresh optimizer; `resume` restores optimizer and step
    but restarts random streams and role phases over the remaining steps."""
    # Expandable CUDA segments avoid fragmentation from packed shapes changing every step.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    model_config, s = config_module.load(config_path, {
        "data": data, "warm_start": warm_start, "covariate_data": covariate_data,
        "max_steps": max_steps, "shape_clusters": shape_clusters})
    device = device_for(s["device"])
    rank = Rank.from_env(device.type)
    if rank.active and device.type == "cuda":
        device = torch.device("cuda", rank.local)
    stage1 = s["stage"] == "stage1"

    torch.manual_seed(s["seed"])
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    model = Simons(model_config).to(device)
    warm = s["warm_start"] is not None
    warm_info = None
    if warm and resume is None:
        warm_info = load_weights(model, s["warm_start"], s["covariate_read_init"])
    rank.broadcast_model(model)
    if not rank.main:
        torch.manual_seed(rank.stream_seed(s["seed"]))
    optimizer = NorMuonAdamW(model, muon_lr=s["muon_lr"], adam_lr=s["adam_lr"],
                             weight_decay=s["muon_weight_decay"],
                             mup_base_width=s["mup_base_width"])

    patch = model_config.patch_length
    future = network.future_steps(model_config)
    if stage1:
        stages = [(s["synthetic_context"], 0, s["steps"])]
    else:
        stages = schedule.parse_schedule(s["context_schedule"], s["steps"],
                                         model_config.context_length, patch)
    # Retain the re-warm origin on resume to preserve the learning-rate schedule.
    warm_first = 0
    if warm and s["warm_start_step"] is not None:
        warm_first = s["warm_start_step"]
    elif warm and not stage1:
        warm_first = stages[s["warm_start_stage"]][1]
    first_step, previous_seconds = warm_first, 0.0
    if resume is not None:
        state = read_checkpoint(resume)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        first_step, previous_seconds = state["step"], state["elapsed_seconds"]
        del state
    if first_step >= s["steps"]:
        raise ValueError("Nothing to train: the start step is past the plan's steps")
    if rank.main:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("Use a fresh output directory, including for resume")
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "config.json", {"model": model_config.to_dict(), "training": s,
                                            "first_step": first_step, "warm_start": warm_info})

    synthetic_augment.configure(s["synthetic_augment"], s["synthetic_augment_rate"],
                                s["synthetic_augment_weights"])
    mech.configure(s["mech_bank"], s["mech_length"], s["mech_refresh"], s["mech_refresh_part"])
    stream = rank.stream_seed(s["seed"])
    base_share = 1.0 if stage1 else s["synthetic_fraction"]
    mixer = None
    if base_share > 0:
        mixer = make_mixer(s, device, stream, first_step, rank.world, base_share)
        mixer.track_first = patch
    share = schedule.share_function(stages, 0 if stage1 else s["synthetic_only_stages"],
                                    s["synthetic_ramp_steps"], base_share)

    roles_hook, passes = None, []
    role_spec = config_module.covariate_roles(s)
    if role_spec is not None:
        from simons.training.covariates.roles import Hook

        if mixer is None or mixer.multi is None:
            raise ValueError("covariate roles need the run's synthetic mixer with coupled groups")
        roles_hook = Hook(role_spec, stream, first_step, device, s["steps"] - first_step,
                          s["synthetic_context"] + future)
        mixer.multi.roles = roles_hook.planner
    if s["covariate_weight"] > 0:
        passes = covariate_passes(s, model_config, rank, stream, first_step, device)

    global_batch = s["synthetic_rows"] if stage1 else s["batch_size"]
    rank_offset, rank_batch = rank.split(global_batch)
    windows, mixture, validation = None, None, []
    if not stage1:
        train_set = load(s["data"], "train")
        windows = Windows(train_set, model_config.context_length + future - 1, 1,
                          s["steps"] * global_batch, s["seed"],
                          downsample=s["augment_downsample"],
                          downsample_max=s["augment_downsample_max"],
                          stretch=s["augment_stretch"], stretch_up=s["augment_stretch_up"],
                          stitch_index=s["stitch_from_stage"] >= 0)
        mixture = plan_mixture(s, model_config, windows, train_set, stages, share, future,
                               global_batch)
        if s["shape_mix"] > 0:
            from simons.training import clusters

            windows.shape_labels = clusters.load(s["shape_clusters"], windows)
        if s["validation_data"] and rank.main:
            held_out = Windows(load(s["validation_data"], "validation"),
                               model_config.context_length, model_config.prediction_length,
                               VALIDATION_WINDOWS, 0)
            samples = [held_out[i] for i in range(len(held_out))]
            validation = [pack_windows(samples[i: i + 64]) for i in range(0, len(samples), 64)]

    def stage_loader(index, context, start, end):
        if stage1:
            rows = (rank_batch, MAX_VARIATES, context + future)
            empty = (torch.full(rows, float("nan")), torch.full(rows[:2], -1, dtype=torch.long),
                     torch.zeros(rows[:2], dtype=torch.long))
            return (empty for _ in range(max(start, first_step), end))
        staged = stage_window_sampler(s, model_config, windows, mixture, index, context, future)
        return DataLoader(
            Subset(staged, StepIndices(max(start, first_step), end, global_batch, rank_offset,
                                       rank_batch)),
            batch_size=rank_batch, collate_fn=collate(patch),
            num_workers=rank.loader_workers(s["loader_workers"]) if device.type == "cuda" else 0,
            pin_memory=device.type == "cuda", generator=torch.Generator(),
        )

    def batches():
        for index, (context, start, end) in enumerate(stages):
            if end > first_step:
                for item in stage_loader(index, context, start, end):
                    yield index, item

    augmenter = RealAugmenter(None, s["augment_censor"], s["augment_amplitude"],
                              s["augment_sign_flip"], s["augment_time_flip"], s["augment_mixup"],
                              s["augment_mixup_alpha"])
    warp_gen = None
    if not stage1:
        if augmenter.active:
            augmenter.gen = torch.Generator(device=device)
            augmenter.gen.manual_seed(stream * 104729 + 31 + first_step)
        if s["augment_warp"] > 0:
            warp_gen = torch.Generator(device=device)
            warp_gen.manual_seed(stream * 7907 + 43 + first_step)

    bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False)
    autocast = torch.autocast("cuda", torch.bfloat16, enabled=bf16)
    model.train()
    started, paused = time.monotonic(), 0.0
    step, logged, pass_logged = first_step, [], {}
    nonfinite = torch.zeros((), dtype=torch.bool, device=device)

    def check_finite():
        if bool(nonfinite) or not all(bool(torch.isfinite(x)) for x in logged):
            raise FloatingPointError(f"Non-finite loss or gradient by step {step}")

    def elapsed():
        return previous_seconds + time.monotonic() - started - paused

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    log = (output / "training.jsonl").open("w") if rank.main else None

    def write_log(rate, stage):
        """Log interval means with every rank participating in covariate-loss reductions."""
        nonlocal logged, pass_logged
        check_finite()
        losses = torch.stack(logged).float()
        record = {"step": step, "loss": float(losses.mean())}
        for name, values in pass_logged.items():
            value = torch.stack(values).float().mean()
            if rank.active:
                value = rank.mean_value(value)
            record[f"{name}_loss"] = float(value)
        if rank.main:
            log.write(json.dumps({**record, "lr_multiplier": round(rate, 6), "phase": stage,
                                  "seconds": round(elapsed(), 1)}) + "\n")
            log.flush()
        logged, pass_logged = [], {}

    try:
        for stage, (sequence, series_ids, slot_first) in batches():
            far_on = stage1 or stage >= s["far_from_stage"]
            if mixer is not None:
                if stage > 0:
                    mixer.pool_steps = s["synthetic_pool_steps_long"] * rank.world
                mixer.fraction = share(step)
                keep = None
                if mixer.fraction > base_share:
                    rows = len(sequence)
                    keep = schedule.ramp_rows(rows, mixer.fraction, base_share)
                if keep is not None and len(keep):
                    sequence, series_ids, slot_first = (sequence[keep], series_ids[keep],
                                                        slot_first[keep])
                    mixer.fraction = schedule.ramp_fraction(rows, len(keep), base_share)
            if roles_hook is not None:
                # Prefetched pools use the preceding step's role phase, except at the first step.
                roles_hook.planner.step = max(step - 1, first_step) - first_step
            host_ids = series_ids.numpy().copy()
            host_first = slot_first.numpy().copy()
            sequence = sequence.to(device, non_blocking=True)
            series_ids = series_ids.to(device, non_blocking=True)
            synthetic_rows = round(mixer.fraction * len(sequence)) if mixer is not None else 0
            if warp_gen is not None:
                sequence = power_warp(sequence, series_ids, len(sequence) - synthetic_rows,
                                      warp_gen, s["augment_warp"],
                                      convex=s["augment_warp_convex"])
            if augmenter.active:
                sequence = augmenter(sequence, series_ids, len(sequence) - synthetic_rows)
            far_len = 0
            if not stage1:
                spec = network.layout(model_config, sequence.shape[-1])
                far_len = spec[0] if spec is not None else 0
            if mixer is not None and far_len:
                rows = len(sequence)
                tail, series_ids = mixer(sequence[..., far_len:], series_ids)
                sequence = torch.cat([sequence[..., :far_len], tail], -1)
                sequence[rows - round(mixer.fraction * rows):, :, :far_len] = float("nan")
                mixer.first_offset = far_len // patch
            elif mixer is not None:
                mixer.first_offset = 0
                sequence, series_ids = mixer(sequence, series_ids)

            def host_slots(host_ids=host_ids, host_first=host_first):
                if mixer is not None and mixer.last_first is not None:
                    rows, width = host_ids.shape
                    count = round(mixer.fraction * rows)
                    host_ids[rows - count:] = mixer.host_ids(count, width)
                    host_first[rows - count:] = mixer.first_rows().reshape(count, width)
                return host_ids, host_first

            roles = None
            if roles_hook is not None:
                sequence, series_ids, host_slots, roles = roles_hook.prepare(
                    step - first_step, mixer, sequence, series_ids, host_slots,
                    far_offset=far_len)
            rate = schedule.multiplier(step, s, warm_first, warm)
            for group in optimizer.param_groups:
                group["lr"] = group["peak_lr"] * rate
            optimizer.zero_grad(set_to_none=True)
            with autocast:
                loss, count = objective.packed_loss(
                    model, sequence, host_slots, s["mask_rate_max"], s["mask_span_max"],
                    far_on=far_on, row_tokens=s["pack_row_tokens"] if far_on else 0,
                    checkpoint_blocks=s["checkpoint_blocks"], roles=roles,
                    micro_batches=s["micro_batches"], rank=rank)
            if s["micro_batches"] > 1:
                mean_loss = loss
            else:
                objective_loss, mean_loss = rank.scale_loss(loss, count)
                objective_loss.backward()
            for name, cov_pass in passes:
                with autocast:
                    cov_loss = cov_pass.loss(model)
                (cov_pass.weight * pass_objective(cov_loss, cov_pass, s, rank, name)).backward()
                pass_logged.setdefault(name, []).append(cov_loss.detach())
            rank.average_gradients(model)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            nonfinite |= ~torch.isfinite(norm)
            optimizer.step()
            step += 1
            logged.append(mean_loss.detach())
            if step % s["log_every"] == 0 or step == s["steps"]:
                write_log(rate, stage)
            if s["max_steps"] is not None and step - first_step >= s["max_steps"]:
                break
            if step % s["checkpoint_every"] == 0 and step < s["steps"]:
                sync()
                check_finite()
                pause = time.monotonic()
                if rank.main:
                    checkpoint(output, model, optimizer, s, step, elapsed(), validation, device,
                               autocast, log)
                rank.barrier()
                paused += time.monotonic() - pause
        sync()
        if logged:
            write_log(rate, stage)
        check_finite()
        if rank.main:
            checkpoint(output, model, optimizer, s, step, elapsed(), validation, device, autocast,
                       log)
            covariates = {}
            if roles_hook is not None:
                covariates["roles"] = roles_hook.summary()
            for name, cov_pass in passes:
                covariates[name] = cov_pass.summary()
            write_json(output / "run.json", {
                "status": "completed" if step == s["steps"] else "partial",
                "steps": step, "steps_this_run": step - first_step,
                "elapsed_seconds": elapsed(),
                "parameters": sum(p.numel() for p in model.parameters()),
                **({"peak_memory_gb": round(torch.cuda.max_memory_allocated(device) / 2**30, 2)}
                   if device.type == "cuda" else {}),
                **({"covariates": covariates} if covariates else {}),
            })
        rank.barrier()
    finally:
        if log is not None:
            log.close()
        for _, cov_pass in passes:
            cov_pass.close()
        rank.close()
    return output / "checkpoint.pt"


def covariate_passes(s, model_config, rank, stream_seed, first_step, device):
    """[(name, CovariateStream)] of a stage2 run: the main pass and, with `long_weight`, the
    long pass. Each GPU draws its share of the pass's tokens from its own random stream."""
    from simons.training.covariates import stream

    mix = config_module.weights(s["covariate_mix"], 3, "covariate_mix")
    main = {
        "future": network.future_steps(model_config), "fovea_near": model_config.near_context,
        "fovea_pool": model_config.far_pool,
        "base": s["covariate_base_data"] or s["data"], "real": s["covariate_data"],
        "weight": s["covariate_weight"], "history_weight": s["covariate_history_weight"],
        "mix": dict(zip(("synthetic", "sibling", "real"), mix)),
        "tokens": -(-int(s["covariate_tokens"]) // rank.world),
        "max_batch": int(s["covariate_max_batch"]), "workers": int(s["covariate_workers"]),
        "source_weights": s["covariate_source_weights"], "exclude": s["covariate_exclude"],
    }
    # Keep worker count fixed: each worker contributes its own random stream.
    out = [("covariate", stream.CovariateStream(main, stream_seed, first_step, device))]
    if s["long_weight"] > 0:
        tokens = s["long_tokens"] or s["covariate_tokens"]
        long = stream.long_spec(main, {
            "weight": s["long_weight"], "tokens": -(-int(tokens) // rank.world),
            "workers": int(s["long_workers"]), "min_length": int(s["long_min_length"]),
            "min_observed": float(s["long_min_observed"])})
        out.append(("long", stream.CovariateStream(long, stream_seed, first_step, device)))
    return out


def pass_objective(loss, cov_pass, s, rank, name):
    """Scale covariate loss for gradient averaging using a global weighted mean or rank means."""
    if not rank.active or (name == "covariate" and s["covariate_norm"] == "rank"):
        return loss
    import torch.distributed as dist

    local = cov_pass.last_weight_sum.to(torch.float64).reshape(1)
    total = local.clone()
    dist.all_reduce(total)
    return loss * (rank.world * local[0] / total[0].clamp_min(1.0)).to(loss.dtype)


def checkpoint(output, model, optimizer, s, step, seconds, validation, device, autocast, log):
    if validation:
        loss = objective.validation_loss(model, validation, device, autocast)
        log.write(json.dumps({"step": step, "validation_loss": loss}) + "\n")
        log.flush()
    save_checkpoint(output / "checkpoint.pt", model, optimizer, s, step, seconds)


def stage_window_sampler(s, model_config, windows, mixture, index, context, future):
    """The window sampler of curriculum phase `index` (context `context`)."""
    staged = copy.copy(windows)
    staged.context = context + future - 1
    near = model_config.near_context
    if context > near:
        staged.end_min = near + future
    if 0 <= s["stitch_from_stage"] <= index:
        staged.stitch = True
    if mixture is not None:
        staged.weights, _ = mixture.weights(*stage_spec(staged, context, future, near))
    labels = getattr(windows, "shape_labels", None)
    if labels is not None:
        from simons.training.clusters import StageTable

        staged.shape = StageTable(labels[0], windows, staged.weights, labels[1], s["shape_cap"],
                                  s["shape_mix"])
    return staged


def stage_spec(staged, context, future, near):
    """Mixture.weights arguments for a phase: window length, window ends, downsampling,
    loss-bearing steps and stretching."""
    trained = near + future if context > near else context + future
    return (context + future, staged.end_min, staged.downsample, staged.downsample_max,
            staged.downsample_min_points, trained, staged.stretch, staged.stretch_up)


def plan_mixture(s, model_config, windows, train_set, stages, share, future, global_batch):
    """Build the UniMax mixture matching the run's expected real-token budget."""
    from simons.training.mixture import Mixture, plan_budget, source_meta

    meta = source_meta(train_set)

    def make(budget):
        return Mixture(windows.names, windows.shapes, meta, float(s["mixture_cap_epochs"]), budget)

    specs = []
    near = model_config.near_context
    for index, (context, start, end) in enumerate(stages):
        staged = stage_window_sampler({**s, "stitch_from_stage": -1}, model_config, windows,
                                      None, index, context, future)
        specs.append((end - start, stage_spec(staged, context, future, near)))
    budget, _ = plan_budget(make, specs, global_batch, schedule.stage_shares(share, stages))
    return make(float(budget))


def build_clusters(config_path: Path, embedder: Path, output: Path, data: str | None = None):
    """Write shape-cluster labels for a stage2 corpus using a frozen checkpoint/model folder."""
    from simons.training import clusters

    model_config, s = config_module.load(config_path,
                                         {"data": None if data is None else str(data)})
    device = device_for(s["device"])
    model = Simons(model_config).to(device)
    info = {"embedder": str(embedder), **load_weights(model, embedder)}
    future = network.future_steps(model_config)
    windows = Windows(load(s["data"], "train"), model_config.context_length + future - 1, 1, 1,
                      0)
    labels, built = clusters.build_labels(model, windows, s["shape_k"], s["shape_anchors"],
                                          s["shape_context"], s["shape_table_seed"],
                                          s["shape_shards"], device)
    clusters.save(output, labels, {**built, **info}, windows)
    return output


def export(checkpoint_path: Path, output: Path, context_length: int | None = None):
    """Export a checkpoint or soup to config.json and model.safetensors; return `output`."""
    from dataclasses import replace

    from simons.config import SimonsConfig

    checkpoint = read_checkpoint(checkpoint_path)
    known = {k: v for k, v in checkpoint["model_config"].items() if k != "model_type"}
    config = SimonsConfig(**known)
    model = Simons(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    context_length = context_length or checkpoint["training"].get("export_context_length")
    if context_length:
        model.config = replace(config, context_length=int(context_length))
    model.save_pretrained(output)
    return output

