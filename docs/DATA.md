# Training data

This page summarizes the training data of the Simons models:
[Simons-25M](https://huggingface.co/abel-lab/Simons-25M),
[Simons-60M](https://huggingface.co/abel-lab/Simons-60M),
[Simons-178M](https://huggingface.co/abel-lab/Simons-178M), and
[Simons-297M](https://huggingface.co/abel-lab/Simons-297M).

The models train on synthetic data from the [open generators](../src/simons/synth) and on
real-world data from public and private datasets.

**Released data.**
[abel-lab/Simons-Pretrain](https://huggingface.co/datasets/abel-lab/Simons-Pretrain)
contains public datasets from the models' real training data. It uses the format the training
code reads, with one folder per dataset. Its
[manifest.json](https://huggingface.co/datasets/abel-lab/Simons-Pretrain/blob/main/manifest.json)
lists each dataset with its source, frequency, and license.

The share-alike folders hold their own LICENSE.md: CC BY-SA 4.0 for `bdg-2`, `museval-cgm`,
`nmt-eeg`, `hai_setpoints`, `nyc_subway_events`,
`wikipedia_sports_fixtures` and `wikipedia_tv_episodes`, and the ODbL 1.0 for `worldtrace` and
`bcb_focus_inflation`.

## Public and private data

The split covers all the real training data, after the benchmark exclusions.

| | Public | Private data |
|---|---|---|
| Share of real training tokens | 66.02% | 33.97% |
| Datasets or sources | 70 dataset families | 111 sources |
| Covariate collections | 61 | 59 |

## Summary by domain

The table covers all the real training data, public and private, after the benchmark
exclusions. Synthetic data isn't counted.

| Domain | Share of real training tokens |
|---|---|
| Transport | 15.34% |
| Environment | 15.27% |
| Health | 11.86% |
| Computing | 11.67% |
| Energy | 11.47% |
| Industry | 10.95% |
| Human activity | 7.64% |
| Economics and finance | 7.63% |
| Media attention | 7.62% |
| Commerce | 0.41% |
| Physical sciences | 0.12% |
| Society | 0.01% |
| Multidomain | <0.01% |
| Agriculture | <0.01% |
| **Total** | 100.00% |

## Benchmark exclusions

The training data excludes the test data of three benchmarks:
[fev-bench](https://huggingface.co/spaces/autogluon/fev-bench),
[GIFT-Eval](https://huggingface.co/spaces/Salesforce/GIFT-Eval), and
[TIME](https://github.com/zqiao11/TIME).

## Licenses

Each dataset keeps its own license and terms. The
[NOTICE.md](https://huggingface.co/datasets/abel-lab/Simons-Pretrain/blob/main/NOTICE.md) of
Simons-Pretrain gives each dataset's license with a link to its source and its text, the
attributions the licenses require, and credits for the public datasets the models trained on that
it doesn't redistribute. [NOTICE](../NOTICE) holds the notices for the model weights.
