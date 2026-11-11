"""Build example training and covariate corpora from public datasets and synthetic energy series."""

import argparse
from pathlib import Path

import numpy as np

REPO = "autogluon/chronos_datasets"
DEFAULT = {
    "ushcn_daily": ("nature", "weather-stations"),
    "uber_tlc_hourly": ("transport", "ride-hailing"),
}
# Exclude GIFT-Eval datasets and collections containing copies to avoid evaluation overlap.
GIFT_EVAL_TEST = ("m4", "m3", "tourism", "car_parts", "electricity", "solar", "kdd_cup", "saugeen",
                  "temperature_rain", "covid_deaths", "hospital", "us_births", "ett", "jena",
                  "loop_seattle", "sz_taxi", "m_dense", "bitbrains", "bizitobs", "restaurant",
                  "hierarchical_sales")
# Exclude collections containing TIME evaluation series.
TIME_OVERLAP = ("mexico_city_bikes", "taxi_30min", "wiki_daily")
SEGMENT = 8192
MIN_POINTS = 64


def frequency(stamps) -> str:
    import pandas as pd

    found = pd.infer_freq(pd.DatetimeIndex(stamps[: min(len(stamps), 64)]))
    return getattr(found, "freqstr", found) or "unknown"


def calendar(stamps, freq: str) -> tuple[list[str], np.ndarray]:
    """Calendar covariates of a timestamp array: (names, values [C, T])."""
    import pandas as pd

    index = pd.DatetimeIndex(stamps)
    names, rows = [], []
    if freq[-1:].upper() in ("H", "T") or freq.lower().endswith(("min", "h")):
        names.append("hour_of_day")
        rows.append(index.hour.to_numpy(dtype=np.float32))
    names += ["day_of_week", "weekend"]
    rows += [index.day_of_week.to_numpy(dtype=np.float32),
             (index.day_of_week >= 5).astype(np.float32)]
    return names, np.stack(rows)


def segments_of(record, columns, config, domain, subdomain, covariate_rows=None):
    """Return segment rows; append the first channel and calendar inputs to `covariate_rows`."""
    values = np.array([record[c] for c in columns], dtype=np.float32)  # [variate, time]
    if values.shape[1] < MIN_POINTS:
        return []
    freq = frequency(record["timestamp"])
    segments = []
    for offset in range(0, values.shape[1], SEGMENT):
        segment = values[:, offset: offset + SEGMENT]
        if np.isfinite(segment).sum(-1).max() < MIN_POINTS:
            continue
        segments.append({"item_id": f"{config}/{record['id']}@{offset}", "source": config,
                         "freq": freq, "domain": domain, "subdomain": subdomain,
                         "target": list(segment.astype(np.float64))})
        if covariate_rows is not None and np.isfinite(segment[0]).sum() >= MIN_POINTS:
            names, known = calendar(record["timestamp"][offset: offset + SEGMENT], freq)
            covariate_rows.append({
                "item_id": f"{config}/{record['id']}@{offset}", "source": f"{config}-calendar",
                "freq": freq, "target": segment[0].astype(np.float64),
                "covariates": list(known.astype(np.float64)),
                "covariate_names": names})
    return segments


def series_of(config, domain, subdomain, covariate_rows=None):
    """Return segment rows grouped by series in the dataset config."""
    from datasets import load_dataset

    dataset = load_dataset(REPO, config, split="train")
    columns = [name for name, feature in dataset.features.items()
               if name not in ("id", "timestamp") and getattr(feature, "feature", None) is not None
               and str(getattr(feature.feature, "dtype", "")).startswith(("float", "int"))]
    for record in dataset:
        segments = segments_of(record, columns, config, domain, subdomain, covariate_rows)
        if segments:
            yield segments


def synthetic_energy(count: int, length: int, seed: int = 0) -> list[dict]:
    """Generate hourly energy targets and day-ahead driver forecasts in covariate corpus format."""
    from simons.synth.covariates import energy

    rng = np.random.default_rng(seed)
    config = {"spd": {24: 1.0}}
    rows = []
    while len(rows) < count:
        target, covariates = energy.sample(None, rng, length, 32, config)
        target, covariates = target[:length], covariates[:, :length]
        if np.isfinite(target).sum() < MIN_POINTS or not len(covariates):
            continue
        rows.append({"item_id": f"synthetic-energy/{len(rows)}", "source": "synthetic-energy",
                     "freq": "h", "target": np.asarray(target, dtype=np.float64),
                     "covariates": list(np.asarray(covariates, dtype=np.float64)),
                     "covariate_names": [f"forecast_{i}" for i in range(len(covariates))]})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--configs", nargs="*", default=list(DEFAULT),
                        help=f"configs of {REPO} (default: {' '.join(DEFAULT)})")
    parser.add_argument("--validation-share", type=float, default=0.02,
                        help="share of series (with all their segments) held out for validation")
    parser.add_argument("--synthetic-covariate-series", type=int, default=200,
                        help="synthetic energy-like series in the example covariate corpus")
    args = parser.parse_args()
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(0)
    covariates = args.output / "covariates"
    covariates.mkdir(parents=True, exist_ok=True)
    for config in args.configs:
        if any(word in config.lower() for word in GIFT_EVAL_TEST):
            raise SystemExit(f"{config}: overlaps a GIFT-Eval test dataset")
        if any(word in config.lower() for word in TIME_OVERLAP):
            raise SystemExit(f"{config}: holds series of TIME test datasets")
        domain, subdomain = DEFAULT.get(config, ("other", config))
        splits = {"train": [], "validation": []}
        calendar_rows = []
        for segments in series_of(config, domain, subdomain, calendar_rows):
            held = rng.random() < args.validation_share
            splits["validation" if held else "train"].extend(segments)
        for split, rows in splits.items():
            if rows:
                folder = args.output / split
                folder.mkdir(parents=True, exist_ok=True)
                pq.write_table(pa.Table.from_pylist(rows), folder / f"{split}-{config}.parquet")
        held_out = {row["item_id"] for row in splits["validation"]}
        calendar_rows = [row for row in calendar_rows if row["item_id"] not in held_out]
        if calendar_rows:
            pq.write_table(pa.Table.from_pylist(calendar_rows),
                           covariates / f"train-calendar-{config}.parquet")
        print(f"{config}: {len(splits['train'])} train and {len(splits['validation'])} "
              f"validation segments, {len(calendar_rows)} covariate rows")
    if args.synthetic_covariate_series:
        rows = synthetic_energy(args.synthetic_covariate_series, 17520)
        pq.write_table(pa.Table.from_pylist(rows), covariates / "train-synthetic-energy.parquet")
    print(f"corpus: {args.output / 'train'}; covariate corpus: {covariates}")


if __name__ == "__main__":
    main()
