#!/usr/bin/env bash
# Train one Simons size end to end: scripts/train.sh SIZE CORPUS COVARIATES [GPUS]
# SIZE is 25m, 60m, 178m or 297m. MAX_STEPS=N stops each stage after N steps. See docs/TRAINING.md.
# RUNS=dir sets the output folder (default runs).
set -euo pipefail

size=${1:?size: 25m, 60m, 178m or 297m}
data=${2:?corpus folder with train-*.parquet}
covdata=${3:?covariate corpus folder (format: simons.training.covariates.windows)}
gpus=${4:-$(python -c "import torch; print(max(1, torch.cuda.device_count()))")}
runs=${RUNS:-runs}
smoke=()
if [[ -n ${MAX_STEPS:-} ]]; then
    smoke=(--max-steps "$MAX_STEPS")
fi
covargs=(--covariate-data "$covdata")

launch() {
    torchrun --standalone --nproc-per-node "$gpus" -m simons.training train \
        ${smoke[@]+"${smoke[@]}"} "$@"
}

clusters() {
    python -m simons.training clusters --config "$1" --data "$data" --embedder "$2" --output "$3"
}

if [[ $size == 297m ]]; then
    # The final --max-steps overrides the smoke-test flag for this leg.
    leg=$(( ${MAX_STEPS:-20240} < 20240 ? ${MAX_STEPS:-20240} : 20240 ))
    launch --config "configs/simons-297m-stage1.yaml" --output "$runs/simons-297m-stage1-leg1" \
        --max-steps "$leg"
    launch --config "configs/simons-297m-stage1-ext.yaml" \
        --warm-start "$runs/simons-297m-stage1-leg1/checkpoint.pt" \
        --output "$runs/simons-297m-stage1"
else
    launch --config "configs/simons-$size-stage1.yaml" --output "$runs/simons-$size-stage1"
fi

launch --config "configs/simons-$size-stage2.yaml" \
    --warm-start "$runs/simons-$size-stage1/checkpoint.pt" --output "$runs/simons-$size-stage2"
shape=()
if [[ $size == 297m ]]; then
    clusters "configs/simons-$size-stage3.yaml" "$runs/simons-$size-stage2/checkpoint.pt" \
        "$runs/simons-$size-stage3-clusters.npz"
    shape=(--shape-clusters "$runs/simons-$size-stage3-clusters.npz")
fi
launch --config "configs/simons-$size-stage3.yaml" --data "$data" "${covargs[@]}" \
    --warm-start "$runs/simons-$size-stage2/checkpoint.pt" ${shape[@]+"${shape[@]}"} \
    --output "$runs/simons-$size-stage3"
if [[ $size == 297m ]]; then
    clusters "configs/simons-$size-stage4.yaml" "$runs/simons-$size-stage3/checkpoint.pt" \
        "$runs/simons-$size-stage4-clusters.npz"
    shape=(--shape-clusters "$runs/simons-$size-stage4-clusters.npz")
fi
launch --config "configs/simons-$size-stage4.yaml" --data "$data" "${covargs[@]}" \
    --warm-start "$runs/simons-$size-stage3/checkpoint.pt" ${shape[@]+"${shape[@]}"} \
    --output "$runs/simons-$size-stage4"
if [[ $size == 297m ]]; then
    python -m simons.training soup --wise 0.5 --output "$runs/simons-$size-final/checkpoint.pt" \
        "$runs/simons-$size-stage3/checkpoint.pt" "$runs/simons-$size-stage4/checkpoint.pt"
else
    launch --config "configs/simons-$size-stage4-long.yaml" --data "$data" "${covargs[@]}" \
        --warm-start "$runs/simons-$size-stage3/checkpoint.pt" \
        --output "$runs/simons-$size-stage4-long"
    python -m simons.training soup --output "$runs/simons-$size-final/checkpoint.pt" \
        "$runs/simons-$size-stage4/checkpoint.pt" "$runs/simons-$size-stage4-long/checkpoint.pt"
fi
python -m simons.training export --checkpoint "$runs/simons-$size-final/checkpoint.pt" \
    --output "$runs/simons-$size"
echo "model folder: $runs/simons-$size (load with Simons.from_pretrained)"
