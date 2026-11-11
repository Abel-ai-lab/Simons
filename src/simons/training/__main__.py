"""Command-line entry point for training, clustering, checkpoint averaging and export."""

import argparse
from pathlib import Path

from simons.training.trainer import build_clusters, export, train


def main():
    parser = argparse.ArgumentParser(prog="python -m simons.training")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("train", help="run one stage of a recipe")
    run.add_argument("--config", type=Path, required=True,
                     help="the stage's YAML config (model and training sections)")
    run.add_argument("--output", type=Path, required=True,
                     help="run folder for config.json, training.jsonl, checkpoint.pt and "
                          "run.json; must be new or empty, also with --resume")
    run.add_argument("--data",
                     help="override the config's corpus folder (train-*.parquet, or toy); it is "
                          "also the covariate pass's base corpus unless covariate_base_data is "
                          "set")
    run.add_argument("--warm-start",
                     help="override the config's warm start (a checkpoint.pt or model folder): "
                          "its weights only, with a fresh optimizer, from warm_start_step "
                          "(default: the first step of phase warm_start_stage in a stage2 run, "
                          "step 0 in a stage1 run), with the re-warm LR ramp")
    run.add_argument("--resume", type=Path,
                     help="continue an interrupted run from its checkpoint.pt, with the same "
                          "config and overrides: restores weights, optimizer, step and elapsed "
                          "time; LR schedule and context phases continue from that step, the "
                          "role phases start again over the remaining steps; random streams "
                          "start afresh, as for a run that starts at that step")
    run.add_argument("--covariate-data",
                     help="override the config's covariate corpus folder (the real windows of "
                          "the covariate pass)")
    run.add_argument("--shape-clusters",
                     help="override the config's shape-cluster labels file (written by the "
                          "clusters command)")
    run.add_argument("--max-steps", type=int,
                     help="stop after this many steps of this run and write the checkpoint; the "
                          "plan and its schedules are unchanged (smoke tests; run.json says "
                          "partial when the run stops before the plan's last step)")
    cl = commands.add_parser("clusters", help="build the shape-cluster labels of a config")
    cl.add_argument("--config", type=Path, required=True)
    cl.add_argument("--embedder", type=Path, required=True,
                    help="frozen embedder: the run's warm start (checkpoint.pt) or a model folder")
    cl.add_argument("--output", type=Path, required=True, help="labels file (.npz)")
    cl.add_argument("--data", help="override the config's corpus folder")
    mix = commands.add_parser("soup", help="average trained checkpoints into one checkpoint")
    mix.add_argument("members", type=Path, nargs="+", help="checkpoint.pt files")
    mix.add_argument("--output", type=Path, required=True, help="checkpoint file to write")
    mix.add_argument("--wise", type=float, metavar="ALPHA",
                     help="two members ANCHOR MODEL: (1 - ALPHA) * ANCHOR + ALPHA * MODEL "
                          "instead of the uniform average")
    out = commands.add_parser("export", help="write a checkpoint as a model folder")
    out.add_argument("--checkpoint", type=Path, required=True)
    out.add_argument("--output", type=Path, required=True)
    out.add_argument("--context-length", type=int,
                     help="history steps the exported model reads (default: the config's "
                          "export_context_length, else the model's context)")
    args = parser.parse_args()
    if args.command == "clusters":
        print(build_clusters(args.config, args.embedder, args.output, args.data))
    elif args.command == "soup":
        from simons.training.soup import write

        print(write(args.members, args.output, args.wise))
    elif args.command == "train":
        print(train(args.config, args.output, data=args.data, warm_start=args.warm_start,
                    resume=args.resume, covariate_data=args.covariate_data,
                    max_steps=args.max_steps, shape_clusters=args.shape_clusters))
    else:
        print(export(args.checkpoint, args.output, args.context_length))


if __name__ == "__main__":
    main()
