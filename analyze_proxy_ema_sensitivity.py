#!/usr/bin/env python3
"""Summarize EMA sensitivity from proxy online-update comparison logs."""

import argparse
import os

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--topk", type=int, required=True)
    parser.add_argument("--family-stride", type=int, default=4)
    parser.add_argument("--min-task-idx", type=int, default=4)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    data = pd.read_csv(args.input).sort_values(
        ["update_mode", "task_idx", "prev_idx", "task_batch"]
    )
    required = {
        "update_mode",
        "task_idx",
        "task_batch",
        "prev_idx",
        "similarity",
        "same_family",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing columns in {args.input}: {missing}")

    variation = (
        data.groupby(["update_mode", "task_idx", "prev_idx"])["similarity"]
        .apply(
            lambda values: np.mean(np.abs(np.diff(values.to_numpy())))
            if len(values) > 1
            else np.nan
        )
        .groupby("update_mode")
        .mean()
    )
    within_pair_std = (
        data.groupby(["update_mode", "task_idx", "prev_idx"])["similarity"]
        .std()
        .groupby("update_mode")
        .mean()
    )

    event_rows = []
    for (mode, task_idx, task_batch), group in data.groupby(
        ["update_mode", "task_idx", "task_batch"]
    ):
        if task_idx < args.min_task_idx:
            continue
        selected_rows = group[group["similarity"] > args.threshold].nlargest(
            args.topk, "similarity"
        )
        selected = set(selected_rows["prev_idx"].astype(int))
        relevant = set(
            group.loc[group["same_family"] == 1, "prev_idx"].astype(int)
        )
        parent = task_idx - args.family_stride
        event_rows.append(
            {
                "update_mode": mode,
                "support_size": len(selected),
                "same_family_precision": (
                    len(selected & relevant) / len(selected)
                    if selected
                    else np.nan
                ),
                "same_family_recall": (
                    len(selected & relevant) / len(relevant)
                    if relevant
                    else np.nan
                ),
                "any_same_family": float(bool(selected & relevant)),
                "parent_selected": float(parent in selected),
            }
        )

    event_summary = (
        pd.DataFrame(event_rows)
        .groupby("update_mode", as_index=False)
        .mean(numeric_only=True)
    )
    event_summary["mean_absolute_similarity_change"] = event_summary[
        "update_mode"
    ].map(variation)
    event_summary["mean_within_pair_similarity_std"] = event_summary[
        "update_mode"
    ].map(within_pair_std)
    labels = {
        "no_ema": r"$\gamma_{\mathrm{ema}}=0$",
        "masksc": r"$\gamma_{\mathrm{ema}}=0.5$",
        "unit_input_ema": r"unit-input EMA, $\gamma_{\mathrm{ema}}=0.5$",
    }
    event_summary.insert(
        1,
        "update_label",
        event_summary["update_mode"].map(labels).fillna(
            event_summary["update_mode"]
        ),
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    event_summary.to_csv(args.out, index=False)
    print(
        event_summary.to_string(
            index=False, float_format=lambda value: f"{value:.4f}"
        )
    )
    print(f"\nWrote {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
