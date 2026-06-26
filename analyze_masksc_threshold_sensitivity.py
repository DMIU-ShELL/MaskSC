#!/usr/bin/env python3
"""Post-hoc Mask-SC similarity-threshold sensitivity from recorded logs.

This script replays threshold-plus-top-k selection over the pairwise cosine
similarities stored in ``task_similarities.csv``. It does not rerun RL.

Example:

    python analyze_masksc_threshold_sensitivity.py \
      --method-root log/checklist_runs/ct28-interleaved-MaskSC-3-thesis \
      --task-order-config env_configs/ct28/seed1/meta_ctgraph_ct28_interleaved.json \
      --selection-cap 3 \
      --thresholds 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
      --analysis-min-depth 3 \
      --out threshold_sensitivity_ct28.csv
"""

import argparse
import os

import numpy as np
import pandas as pd

import analyze_amsc_selection as common
import analyze_masksc_selection as masksc


METRICS = (
    "support_size",
    "same_family_precision",
    "same_family_recall",
    "any_same_family",
    "parent_selected",
    "jaccard_stability",
)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--method-root", required=True)
    parser.add_argument("--task-order-config", required=True)
    parser.add_argument("--selection-cap", type=int, required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", required=True)
    parser.add_argument("--analysis-min-depth", type=int, default=3)
    parser.add_argument("--family-stride", type=int, default=None)
    parser.add_argument("--family-names", nargs="*", default=None)
    parser.add_argument("--min-depth", type=int, default=2)
    parser.add_argument("--method-name", default="Mask-SC")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    metadata = common.load_task_metadata(
        args.task_order_config,
        family_stride=args.family_stride,
        family_names=args.family_names,
        min_depth=args.min_depth,
    )
    runs = common.discover_runs(args.method_root)
    if not runs:
        raise ValueError(
            f"No task_similarities.csv files found under {args.method_root}"
        )

    rows = []
    for threshold in args.thresholds:
        task_frames = []
        for run_index, run_dir in enumerate(runs):
            seed = common.extract_seed(run_dir)
            if seed is None:
                seed = run_index
            events = masksc.read_selection_events(
                os.path.join(run_dir, "task_similarities.csv"),
                threshold=threshold,
                cap=args.selection_cap,
            )
            # The logged ``selected`` field reflects the threshold used during
            # training. Replace it with the support reconstructed under the
            # post-hoc threshold so all downstream metrics use the replayed
            # selector.
            for event in events:
                event["selected"] = set(event["raw_selected"])
            _, task_df, _, _ = masksc.analyze_run_events(
                events,
                metadata,
                args.method_name,
                run_dir,
                seed,
                args.selection_cap,
                threshold,
                "posthoc_similarity",
            )
            task_frames.append(task_df)

        tasks = pd.concat(task_frames, ignore_index=True)
        tasks = tasks[tasks["depth"] >= args.analysis_min_depth].copy()
        seed_means = (
            tasks.groupby("seed", as_index=False)[list(METRICS)]
            .mean(numeric_only=True)
        )
        row = {
            "threshold": threshold,
            "selection_cap": args.selection_cap,
            "n_seeds": seed_means["seed"].nunique(),
            "n_seed_tasks": len(tasks),
        }
        for metric in METRICS:
            values = seed_means[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            row[metric] = float(np.mean(values)) if len(values) else np.nan
        rows.append(row)

    result = pd.DataFrame(rows).sort_values("threshold")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    result.to_csv(args.out, index=False)
    print(result.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nWrote {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
