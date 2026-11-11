# Training

The training code and configs reproduce the released models' recipe end to end. The recipe is a
chain of four training stages, followed by a weight average and an export to a model folder.
`scripts/train.sh` runs the whole chain for one model size.

## Run the recipe

Training needs CUDA GPUs. The configs target 80 GB GPUs. On the example corpus, peak memory
across stages is 27 GB per GPU for the 25M and 70 GB for the 60M.
[GPU count and memory](#gpu-count-and-memory) gives the peak of every size.

The public training data is the Hugging Face dataset
[abel-lab/Simons-Pretrain](https://huggingface.co/datasets/abel-lab/Simons-Pretrain), which
`scripts/train.sh` reads by name in place of the two folders below. [DATA.md](DATA.md) describes all
the training data. The code reads any corpus in the [formats below](#corpus-format).
`scripts/prepare_example_data.py` builds a small public example corpus and an example covariate
corpus. It downloads about 330 MB from the Hugging Face Hub and needs about 7 GB of memory.
The script writes `<output>/train`, `<output>/validation` and `<output>/covariates`, and setting
`validation_data: <output>/validation` logs a validation loss.

```shell
pip install -e .

# A small public example corpus and an example covariate corpus:
python scripts/prepare_example_data.py --output data/example

# The whole chain, then a model folder for Simons.from_pretrained (runs/simons-25m):
bash scripts/train.sh 25m data/example/train data/example/covariates   # optional 4th: GPUs
# A smoke test of every stage (each stops after a few steps of its plan):
MAX_STEPS=5 bash scripts/train.sh 25m data/example/train data/example/covariates
# The released public data, downloaded on first use:
bash scripts/train.sh 25m abel-lab/Simons-Pretrain abel-lab/Simons-Pretrain
```

## Stages

Each row is one training run:

| Stage | Config in `configs/` | `stage` key | Starts from | Purpose |
|---|---|---|---|---|
| 1 | `simons-<size>-stage1.yaml` | `stage1` | random weights | synthetic pretraining |
| 2 | `simons-<size>-stage2.yaml` | `stage1` | stage 1 | the synthetic covariate curriculum |
| 3 | `simons-<size>-stage3.yaml` | `stage2` | stage 2 | post-training on real data |
| 4 | `simons-<size>-stage4.yaml` | `stage2` | stage 3 | a continuation of stage 3's last steps |
| 4-long | `simons-<size>-stage4-long.yaml` | `stage2` | stage 3 | a longer continuation |

Terms in this table and below:

- The **`stage` key** of a config sets the kind of run. A `stage1` run trains on synthetic series
  only. A `stage2` run trains on real data plus 40% synthetic series.
- A **warm start** loads checkpoint weights with a fresh optimizer. Stages 2 to 4 use warm starts.
- A **continuation** trains stage 3's last steps again, from stage 3's final weights.
- A **covariate pass** is an extra loss pass in each training step, on windows that pair a target
  with covariates.
- The **covariate rows** are the model's input rows for known-future covariates. In the
  **one-way read**, target rows read covariate rows and never the reverse. **XReg** is a
  forecast-time regression of the target on its covariates.

The chain, in order:

1. **Stage 1.** The 297M's stage 1 runs in two legs. The second leg uses
   `simons-297m-stage1-ext.yaml`.
2. **Stage 2.** Its synthetic data carries two additions, which train the covariate rows and the
   one-way read:
   - covariate roles in coupled synthetic series;
   - synthetic covariate-effect rows from five generator families: generic, classical, energy,
     retail and lagged.
3. **Stage 3.** Its synthetic rows carry covariate roles. It adds a covariate pass at loss weight
   0.025 on three kinds of windows:
   - synthetic effects;
   - sibling pairs from the corpus;
   - real known-future-covariate data.
4. **Stage 4.** Each continuation adds a second, long-history pass on real covariate data. This
   page calls it the long pass, and stage 3's pass the main covariate pass. Every size has
   `simons-<size>-stage4.yaml`. Every size except the 297M also has the longer
   `-stage4-long.yaml`.
5. **Final weights.** They are one model, with no ensemble at forecast time:
   - 25M, 60M and 178M: the uniform average of stage 4 and stage 4-long (Model soups);
   - 297M: 0.5 x stage 3 + 0.5 x stage 4 (WiSE-FT).
6. **Export.** It writes a model folder. The folder's config carries the covariate settings and
   the forecast-time rules: XReg with its backtest guard, and covariate rows only for long
   histories. The 25M and 60M are exported to read the last 16,384 steps (`export_context_length`).

Stages 1 to 4 train with the quantile (pinball) loss, NorMuon and AdamW.

[Fine-tune](GETTING_STARTED.md#fine-tune) trains a released model further on your own data with the
stage-4 config.

## Running the stages yourself

Each stage is one command. `python -m simons.training {train,clusters,soup,export} --help` lists
a command's flags. Pick one name from the braces.

```shell
torchrun --nproc-per-node 2 -m simons.training train --config configs/simons-178m-stage2.yaml \
    --warm-start runs/simons-178m-stage1/checkpoint.pt --output runs/simons-178m-stage2
torchrun --nproc-per-node 2 -m simons.training train --config configs/simons-178m-stage3.yaml \
    --data path/to/corpus --covariate-data path/to/covariates \
    --warm-start runs/simons-178m-stage2/checkpoint.pt --output runs/simons-178m-stage3
# ... stage4 and stage4-long from runs/simons-178m-stage3/checkpoint.pt, then
python -m simons.training soup --output runs/simons-178m-final/checkpoint.pt \
    runs/simons-178m-stage4/checkpoint.pt runs/simons-178m-stage4-long/checkpoint.pt
python -m simons.training export --checkpoint runs/simons-178m-final/checkpoint.pt \
    --output models/178m
```

### Shape-cluster labels of the 297M

Stages 3 and 4 of the 297M draw half of their real windows through shape-cluster sampling. The
cluster labels are built from embeddings of the corpus series. The model that the run warm-starts
from computes the embeddings. Build the labels before the run:

- for stage 3, from stage 2's model;
- for stage 4, from stage 3's model.

`python -m simons.training clusters --help` lists the options. `scripts/train.sh 297m ...` builds
the labels for you and passes the file with `--shape-clusters`.

## Configs and GPUs

GPU counts for the training recipe:

| Size | Configs (`configs/simons-<size>-...`) | GPUs, stage 1 | GPUs, stages 2 to 4 |
|---|---|---|---|
| 25M | stage1, stage2, stage3, stage4, stage4-long | 1 | 1 |
| 60M | stage1, stage2, stage3, stage4, stage4-long | 1 | 1 |
| 178M | stage1, stage2, stage3, stage4, stage4-long | 7 | 2 |
| 297M | stage1, stage1-ext, stage2, stage3, stage4 | 7 | 2 |

Training steps per stage:

| Size | Stage 1 | Stage 2 | Stage 3 | Stage 4 | Stage 4-long |
|---|---|---|---|---|---|
| 25M | 65,000 | 12,000 | 31,208 | 4,000 | 8,000 |
| 60M | 55,000 | 12,000 | 9,341 | 1,200 | 2,400 |
| 178M | 36,000 | 7,500 | 46,750 | 6,000 | 9,000 |
| 297M | 36,000 in two legs | 6,000 | 46,750 | 6,000 | none |

Stage settings:

| Stage | Sizes | Detail |
|---|---|---|
| 1 | all | 17 generator families; never-repeating series of 8,704 steps |
| 1 | 60M, 178M, 297M | a calendar family at 35% as well |
| 2 | 178M, 297M | weighted toward energy-like effects |
| 3 | all | batches of 784 windows; context up to 32,768 |
| 3 | 297M | shape-cluster sampling |
| 4 | all | stage 3's last steps at twice the learning rate |

## GPU count and memory

The number of GPUs is part of a run. Random streams and the split of each step's batch depend on
it. Another GPU count gives a different run.

Two settings of the 297M's stage 2 are per GPU and follow its 2 GPUs: `mech_bank` and
`mech_refresh`. See the config.

Each GPU holds its share of the step's batch. Two settings trade speed for memory without changing
the results:

- `checkpoint_blocks` checkpoints the activations of the first K time blocks. The values and
  gradients stay the same.
- `micro_batches` runs each GPU's packed rows forward and backward in M chunks. The chunks'
  gradients add up to the gradient of the step's global mean loss. The gradient is the same up to
  float summation order.

As configured, stages 2 to 4 of the 25M and 60M fit one GPU of 80 GB, and those of the 178M and
297M fit two.

Peak memory per GPU for a few steps of each stage on the example corpus:

| Size | Setup | Peak per GPU |
|---|---|---|
| 25M | one A100 80 GB | 27 GB |
| 60M | stage 2, one GPU | 70 GB |
| 178M | stage 4, two GPUs | 34 GB |
| 297M | stage 4 at the 32,768 context, two GPUs | 21 GB |
| 297M | the same without checkpointing | 43 GB |

A corpus of longer series packs more tokens per step. With less GPU memory, raise `micro_batches`
or use more GPUs. `checkpoint_blocks` can rise only up to the model's number of layers. The
configs of stages 3 and 4 already set it there.

## Corpus format

The training corpus is a folder of `train-*.parquet` files. `validation-*.parquet` files are
optional. The columns:

| Column | Type | Meaning |
|---|---|---|
| `item_id` | string | series id |
| `source` | string | dataset name, the unit of the data mixture |
| `freq` | string | pandas frequency string (`"H"`, `"15T"`, `"D"`, ...) |
| `domain`, `subdomain` | string, optional | groups balanced by the mixture |
| `target` | float32 `[variate][time]` | values, NaN = missing |

A window uses at most 32 variates. Name segments of one long series `<base>@<offset>` to let
long-context phases of stages 3 and 4 stitch them together.

The real-data mixture is hierarchical UniMax. It splits the token budget equally over
`(domain, subdomain)` groups, then equally over the sources of each group. A cap limits the passes
per source. The `mixture_cap_epochs` key in the stage-3 and stage-4 configs sets the cap.

### Covariate corpus

The covariate passes of the training code (`simons.training.covariates`) read two corpora:

- the training corpus, for synthetic effects on its series and sibling pairs of its multivariate
  rows;
- a covariate corpus, for real covariate data.

The covariate corpus is a folder of `train-*.parquet` files with one row per series:

| Column | Type | Meaning |
|---|---|---|
| `item_id` | string | series id |
| `source` | string | collection name |
| `freq` | string | pandas frequency string (`"h"`, `"30min"`, `"D"`, ...) |
| `target` | float `[time]` | the target values, NaN = missing |
| `covariates` | float `[covariate][time]` | the known-future covariates, NaN = unknown |
| `covariate_names` | list of string, optional | names, for your own bookkeeping; not read |

Three columns carry rules:

- `source`: windows are drawn by source first, then by series. Each source is drawn equally in
  the main covariate pass, unless `covariate_source_weights` says otherwise. The long pass draws
  every source equally.
- `covariates`: each covariate is as long as `target` and aligned with it step by step. Another
  length is an error at load. Rows without covariates are skipped.
- `freq`: the string sets the forecast horizons drawn for the series, in patches of 32 steps.

| `freq` ends in | Horizons drawn |
|---|---|
| `MIN`, `T` or `S` | 1, 2, 3, 4, 6 or 8 patches |
| `H` | 1, 2, 3, 4 or 6 patches |
| anything else | 1 or 2 patches |

The match ignores case. Month-start and quarter-start strings such as `"MS"` and `"QS"` end in
`S`, so they also fall in the first row.

`scripts/prepare_example_data.py` writes a small example covariate corpus. It holds calendar
covariates of public series and synthetic energy-like series with issued forecasts.

#### Training windows

A training window is a random stretch of a series: a history, then a horizon.

- The history is 128 to 4,096 steps in stages 3 and 4. In stage 4's long pass it is 2,048 or
  4,096 steps.
- The horizon is 1 to 8 patches of 32 steps.
- The trainer hides the target over the horizon. It shows every covariate over the whole window,
  history and horizon.
- The first 16 covariates of a row are used. Each is dropped at random in some windows.

#### Known at forecast time

At every time step, the data must already hold the covariate value that was available when
forecasting that time step. This must hold for any forecast origin up to a horizon before it.

- **Inputs fixed in advance** can be stored as they happened. Examples: calendar features,
  holidays, schedules, timetables, planned prices, promotions, closures and control set-points.
- **Issued forecasts** must be stored as an archive of a fixed lead. Examples: weather, load,
  renewable output and prices. Store, for example, at each hour the day-ahead forecast issued the
  day before (a "stitched" archive). Never store the later analysis or the observed value. If
  several vintages exist, keep one per column: one column per lead.

The trainer does not read issue times. It takes the stored value as known over the whole window.

Label sources that use observed values as forecast proxies so their sampling weights can be
controlled.

Cut every series before your evaluation period. Check it against your test data, as you check the
training corpus.

#### Source weights and held-out series

Two optional JSON files shape the draws. A key of the stage-3 and stage-4 configs names each file:

- `covariate_source_weights`: `{source: weight}`. Weight 0 removes a source from the main
  covariate pass; the long pass ignores the weights.
- `covariate_exclude`: `{"items": {source: [item_id, ...]}}`. These series stay out of every
  covariate pass, for example for evaluation.

`configs/covariate-source-weights.example.json` excludes the example corpus's synthetic series
from the main covariate pass with weight 0.

#### Base series

The base series of the covariate passes' synthetic and sibling windows come from the training
corpus. The trainer reads the first `train-*.parquet` file of `covariate_base_data`. The default
is the run's `data`. A sibling window takes a multivariate row with channels of equal length: one
channel as the target and the others as covariates.

## Reproducing the released models

The released weights are on Hugging Face, and the benchmark code is in `benchmarks/`.
Stages 1 and 2 use synthetic data and the released configs. Stages 3 and 4 require the full
training corpus to reproduce the released weights; [Simons-Pretrain](DATA.md) contains its public
part. Training on another corpus produces different weights.

The configs default to the example corpus in `data/example/train` and `data/example/covariates`.
Set the data paths, per-source weights and held-out series for your corpus. Remove benchmark test
series before training; the trainer does not check benchmark exclusions.

`mixture_cap_epochs: 6.0` caps the passes per source. If the token budget exceeds six passes over
the whole corpus, all source caps increase by the same factor.

The 297M's main covariate pass uses `covariate_norm: rank`: the mean of each GPU's mean loss.
The long pass and the other sizes use the global weighted mean. These agree on one GPU.

Training uses eager `torch.distributed` data parallelism with one gradient all-reduce per step.
See [GPU count and memory](#gpu-count-and-memory) for memory settings and GPU-count effects.
