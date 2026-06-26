#!/usr/bin/env python3
"""Aggregate sim_transfer_analysis.py outputs across seeds.

Example:
python evaluation_kit/transfer_probe/sim_transfer_analysis_aggregate.py \
  "log/probe-experiments/sim_transfer_stats_seed*.csv"
"""

import argparse
import glob

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or more CSV paths or glob patterns produced by sim_transfer_analysis.py.",
    )
    parser.add_argument("--out", default=None, help="Optional CSV path for the aggregate summary.")
    args = parser.parse_args()

    matched = []
    for pattern in args.paths:
        matches = sorted(glob.glob(pattern))
        if matches:
            matched.extend(matches)
        else:
            matched.append(pattern)

    matched = list(dict.fromkeys(matched))
    if not matched:
        raise SystemExit("No input files matched.")

    frames = []
    missing = []
    for path in matched:
        try:
            frames.append(pd.read_csv(path))
        except FileNotFoundError:
            missing.append(path)

    if missing:
        raise SystemExit("Missing input files:\n" + "\n".join(missing))
    if not frames:
        raise SystemExit("No readable input files.")

    df = pd.concat(frames, ignore_index=True)
    summary = df.groupby("metric")["value"].agg(["mean", "std", "count"])
    summary["ci95"] = 1.96 * summary["std"] / summary["count"] ** 0.5

    print(f"Aggregated over {len(frames)} files")
    print(summary)
    if args.out:
        summary.reset_index().to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
