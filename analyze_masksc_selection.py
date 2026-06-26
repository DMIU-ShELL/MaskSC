#!/usr/bin/env python3
"""Analyze Mask-SC retrieval quality, stability, and mask composition.

Oracle-relevant priors are earlier tasks from the same family. For similarity
selection, the script reconstructs the threshold-plus-top-k support before any
performance filter and compares it with the final support recorded in
``task_similarities.csv``.

Examples:

    python analyze_masksc_selection.py \
      --methods Mask-SC-3=log/checklist_runs/ct28-interleaved-MaskSC-3-thesis \
      --task-order-config env_configs/ct28/seed1/meta_ctgraph_ct28_interleaved.json \
      --analysis-min-depth 3 \
      --out-dir selection_analysis_ct28_masksc3 \
      --composition-analysis --per-task-table

    python analyze_masksc_selection.py \
      --methods Mask-SC-4=log/checklist_runs/mg16-interleaved-MaskSC-4-thesis \
      --task-order-config env_configs/minigrid_object_remap_seed86.json \
      --analysis-min-depth 3 \
      --out-dir selection_analysis_mg16_masksc4
"""

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd

import analyze_amsc_selection as common


EVENT_MEAN_METRICS = [
    "support_size",
    "support_fraction",
    "same_family_precision",
    "same_family_recall",
    "any_same_family",
    "parent_selected",
    "family_pure_support",
    "jaccard_stability",
    "cap_utilization",
    "raw_support_size",
    "raw_same_family_precision",
    "raw_same_family_recall",
    "raw_parent_selected",
    "raw_cap_utilization",
    "performance_filter_retention",
    "selected_similarity_mean",
    "raw_selected_similarity_mean",
    "threshold_qualified_count",
]

SUMMARY_METRICS = EVENT_MEAN_METRICS + [
    "first_correct_found",
    "selection_events_until_first_correct",
    "steps_until_first_correct",
]


def _as_float(value, default=np.nan):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result


def load_run_selection_config(
    run_dir,
    threshold_override=None,
    cap_override=None,
):
    config = {}
    config_path = os.path.join(run_dir, "config.json")
    if os.path.isfile(config_path):
        with open(config_path, "r") as handle:
            config = json.load(handle)

    threshold = threshold_override
    if threshold is None:
        threshold = _as_float(config.get("COS_TH"), np.nan)

    cap = cap_override
    if cap is None:
        raw_cap = config.get("detect_topk")
        if raw_cap not in (None, "None", "null", ""):
            try:
                cap = int(raw_cap)
            except (TypeError, ValueError):
                cap = None

    return {
        "threshold": threshold,
        "cap": cap,
        "strategy": str(config.get("select_strategy", "unknown")),
        "performance_gate_enabled": (
            config.get("selection_prior_min_perf") not in (None, "None", "null")
            or str(config.get("selection_require_prior_better_than_current", False))
            .lower() == "true"
        ),
    }


def read_selection_events(
    csv_path,
    threshold=np.nan,
    cap=None,
):
    grouped = defaultdict(list)
    with open(csv_path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "selected" not in reader.fieldnames:
            raise ValueError(f"Unsupported task-similarity CSV: {csv_path}")
        for row in reader:
            task_idx = common._as_int(row.get("task_idx"), -1)
            if task_idx < 0:
                continue
            key = (
                common._as_int(row.get("learn_block")),
                task_idx,
                common._as_int(row.get("iteration")),
                common._as_int(row.get("total_steps")),
            )
            grouped[key].append(row)

    events = []
    for key, rows in grouped.items():
        learn_block, task_idx, iteration, total_steps = key
        candidates = []
        final_selected = set()
        eligible = {}
        for row in rows:
            prior_idx = common._as_int(row.get("prev_idx"), -1)
            if prior_idx < 0 or prior_idx >= task_idx:
                continue
            similarity = _as_float(row.get("similarity"))
            candidates.append((prior_idx, similarity))
            eligible[prior_idx] = common._as_int(row.get("eligible"), 1) == 1
            if common._as_int(row.get("selected")) == 1:
                final_selected.add(prior_idx)

        finite_candidates = [
            (idx, similarity)
            for idx, similarity in candidates
            if np.isfinite(similarity)
        ]
        can_reconstruct_raw = np.isfinite(threshold) and bool(finite_candidates)
        if can_reconstruct_raw:
            qualified = sorted(
                [
                    (idx, similarity)
                    for idx, similarity in finite_candidates
                    if similarity > threshold
                ],
                key=lambda pair: pair[1],
                reverse=True,
            )
            raw_selected = {
                idx for idx, _ in (qualified if cap is None else qualified[:cap])
            }
            threshold_qualified_count = len(qualified)
        else:
            raw_selected = set(final_selected)
            threshold_qualified_count = np.nan

        events.append(
            {
                "learn_block": learn_block,
                "task_idx": task_idx,
                "iteration": iteration,
                "total_steps": total_steps,
                "selected": final_selected,
                "raw_selected": raw_selected,
                "similarities": dict(candidates),
                "eligible": eligible,
                "threshold_qualified_count": threshold_qualified_count,
                "raw_reconstructed": can_reconstruct_raw,
            }
        )

    events.sort(
        key=lambda event: (
            event["learn_block"],
            event["task_idx"],
            event["iteration"],
            event["total_steps"],
        )
    )
    return events


def _support_metrics(selected, relevant, parent_idx, available_count):
    correct = selected & relevant
    support_size = len(selected)
    return {
        "support_size": support_size,
        "support_fraction": (
            support_size / available_count if available_count else np.nan
        ),
        "same_family_precision": (
            len(correct) / support_size if support_size else np.nan
        ),
        "same_family_recall": (
            len(correct) / len(relevant) if relevant else np.nan
        ),
        "any_same_family": float(bool(correct)) if relevant else np.nan,
        "parent_selected": (
            float(parent_idx in selected) if parent_idx is not None else np.nan
        ),
        "family_pure_support": (
            float(bool(selected) and selected <= relevant) if relevant else np.nan
        ),
        "correct_prior_count": len(correct),
    }


def analyze_run_events(
    events,
    metadata,
    method,
    run_dir,
    seed,
    cap,
    threshold,
    strategy,
):
    meta_by_idx = {task.idx: task for task in metadata}
    rows = []
    final_confusion = []
    raw_confusion = []
    previous_support = {}
    event_number = defaultdict(int)
    first_event_step = {}
    first_correct = {}

    first_iterations = {}
    for event in events:
        task_idx = event["task_idx"]
        first_iterations[task_idx] = min(
            first_iterations.get(task_idx, event["iteration"]),
            event["iteration"],
        )
    iteration_gaps = [
        first_iterations[idx] - first_iterations[idx - 1]
        for idx in sorted(first_iterations)
        if idx - 1 in first_iterations
        and first_iterations[idx] > first_iterations[idx - 1]
    ]
    iterations_per_task = (
        float(np.median(iteration_gaps)) if iteration_gaps else np.nan
    )
    steps_per_iteration_values = [
        event["total_steps"] / event["iteration"]
        for event in events
        if event["iteration"] > 0 and event["total_steps"] > 0
    ]
    steps_per_iteration = (
        float(np.median(steps_per_iteration_values))
        if steps_per_iteration_values
        else np.nan
    )

    for event in events:
        task_idx = event["task_idx"]
        if task_idx not in meta_by_idx:
            continue
        task = meta_by_idx[task_idx]
        selected = {idx for idx in event["selected"] if idx < task_idx}
        raw_selected = {idx for idx in event["raw_selected"] if idx < task_idx}
        relevant = {
            idx
            for idx in range(task_idx)
            if meta_by_idx[idx].family == task.family
        }
        parent_idx = max(relevant) if relevant else None

        final_metrics = _support_metrics(
            selected, relevant, parent_idx, task_idx
        )
        raw_metrics = _support_metrics(
            raw_selected, relevant, parent_idx, task_idx
        )

        event_number[task_idx] += 1
        first_event_step.setdefault(task_idx, event["total_steps"])
        if final_metrics["any_same_family"] == 1.0 and task_idx not in first_correct:
            if np.isfinite(iterations_per_task) and np.isfinite(steps_per_iteration):
                task_start_step = task_idx * iterations_per_task * steps_per_iteration
                elapsed_steps = max(0.0, event["total_steps"] - task_start_step)
            else:
                elapsed_steps = event["total_steps"] - first_event_step[task_idx]
            first_correct[task_idx] = {
                "event": event_number[task_idx],
                "steps": elapsed_steps,
            }

        previous = previous_support.get(task_idx)
        jaccard = (
            common._jaccard(previous, selected)
            if previous is not None
            else np.nan
        )
        previous_support[task_idx] = selected

        selected_sims = [
            event["similarities"].get(idx, np.nan) for idx in selected
        ]
        raw_selected_sims = [
            event["similarities"].get(idx, np.nan) for idx in raw_selected
        ]
        selected_sims = [value for value in selected_sims if np.isfinite(value)]
        raw_selected_sims = [
            value for value in raw_selected_sims if np.isfinite(value)
        ]

        row = {
            "method": method,
            "seed": seed,
            "run_dir": run_dir,
            "strategy": strategy,
            "task_idx": task_idx,
            "task_name": task.name,
            "family": task.family,
            "depth": task.depth,
            "learn_block": event["learn_block"],
            "iteration": event["iteration"],
            "total_steps": event["total_steps"],
            "event_number": event_number[task_idx],
            "selection_cap": cap,
            "similarity_threshold": threshold,
            **final_metrics,
            "jaccard_stability": jaccard,
            "cap_utilization": (
                len(selected) / min(cap, task_idx)
                if cap is not None and min(cap, task_idx) > 0
                else np.nan
            ),
            "raw_support_size": len(raw_selected),
            "raw_same_family_precision": raw_metrics["same_family_precision"],
            "raw_same_family_recall": raw_metrics["same_family_recall"],
            "raw_parent_selected": raw_metrics["parent_selected"],
            "raw_cap_utilization": (
                len(raw_selected) / min(cap, task_idx)
                if cap is not None and min(cap, task_idx) > 0
                else np.nan
            ),
            "performance_filter_retention": (
                len(selected) / len(raw_selected) if raw_selected else np.nan
            ),
            "selected_similarity_mean": (
                float(np.mean(selected_sims)) if selected_sims else np.nan
            ),
            "raw_selected_similarity_mean": (
                float(np.mean(raw_selected_sims))
                if raw_selected_sims
                else np.nan
            ),
            "threshold_qualified_count": event["threshold_qualified_count"],
            "raw_reconstructed": event["raw_reconstructed"],
            "selected_indices": " ".join(map(str, sorted(selected))),
            "raw_selected_indices": " ".join(map(str, sorted(raw_selected))),
        }
        rows.append(row)

        for prior_idx in selected:
            final_confusion.append(
                {
                    "method": method,
                    "seed": seed,
                    "task_idx": task_idx,
                    "depth": task.depth,
                    "current_family": task.family,
                    "prior_family": meta_by_idx[prior_idx].family,
                }
            )
        for prior_idx in raw_selected:
            raw_confusion.append(
                {
                    "method": f"{method}-raw",
                    "seed": seed,
                    "task_idx": task_idx,
                    "depth": task.depth,
                    "current_family": task.family,
                    "prior_family": meta_by_idx[prior_idx].family,
                }
            )

    event_df = pd.DataFrame(rows)
    task_rows = []
    for task in metadata:
        task_events = event_df[event_df["task_idx"] == task.idx]
        if task_events.empty:
            continue
        relevant_count = sum(
            prior.family == task.family for prior in metadata[: task.idx]
        )
        row = {
            "method": method,
            "seed": seed,
            "run_dir": run_dir,
            "strategy": strategy,
            "task_idx": task.idx,
            "task_name": task.name,
            "family": task.family,
            "depth": task.depth,
            "selection_cap": cap,
            "similarity_threshold": threshold,
            "num_selection_events": len(task_events),
            "relevant_prior_count": relevant_count,
        }
        for metric in EVENT_MEAN_METRICS:
            row[metric] = float(task_events[metric].mean())

        found = task.idx in first_correct
        row["first_correct_found"] = (
            float(found) if relevant_count else np.nan
        )
        row["selection_events_until_first_correct"] = (
            float(first_correct[task.idx]["event"]) if found else np.nan
        )
        row["steps_until_first_correct"] = (
            float(first_correct[task.idx]["steps"]) if found else np.nan
        )
        task_rows.append(row)

    return (
        event_df,
        pd.DataFrame(task_rows),
        pd.DataFrame(final_confusion),
        pd.DataFrame(raw_confusion),
    )


def summarize_by_group(task_df, group_columns):
    rows = []
    seed_group = (
        task_df.groupby(
            ["method", "seed"] + group_columns, dropna=False
        )[SUMMARY_METRICS]
        .mean(numeric_only=True)
        .reset_index()
    )
    for keys, group in seed_group.groupby(
        ["method"] + group_columns, dropna=False
    ):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(["method"] + group_columns, keys))
        for metric in SUMMARY_METRICS:
            mean, low, high, count = common._mean_ci(
                group[metric].to_numpy()
            )
            if metric in {
                "support_fraction",
                "same_family_precision",
                "same_family_recall",
                "any_same_family",
                "parent_selected",
                "family_pure_support",
                "jaccard_stability",
                "cap_utilization",
                "raw_same_family_precision",
                "raw_same_family_recall",
                "raw_parent_selected",
                "raw_cap_utilization",
                "performance_filter_retention",
                "first_correct_found",
            }:
                low = max(0.0, low) if np.isfinite(low) else low
                high = min(1.0, high) if np.isfinite(high) else high
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "n_seeds": count,
                }
            )
    return pd.DataFrame(rows)


def _format(value):
    return "--" if not np.isfinite(value) else f"{value:.3f}"


def render_latex_table(summary, row_columns, caption, label):
    metrics = [
        ("support_size", "Support"),
        ("support_fraction", "Library frac."),
        ("same_family_precision", "Precision"),
        ("same_family_recall", "Recall"),
        ("any_same_family", "Any correct"),
        ("parent_selected", "Parent"),
        ("jaccard_stability", "Jaccard"),
        ("cap_utilization", "Cap util."),
    ]
    index_columns = ["method"] + row_columns
    pivot = summary.pivot_table(
        index=index_columns,
        columns="metric",
        values="mean",
        aggfunc="first",
    ).reset_index()
    lines = [
        r"\begin{table}[ht]",
        f"    \\caption{{{caption}}}",
        f"    \\label{{{label}}}",
        r"    \centering",
        r"    \scriptsize",
        r"    \begin{tabular}{"
        + "l" * len(index_columns)
        + "r" * len(metrics)
        + "}",
        r"    \toprule",
        " & ".join(
            [column.replace("_", " ").title() for column in index_columns]
            + [title for _, title in metrics]
        )
        + r" \\",
        r"    \midrule",
    ]
    for _, row in pivot.iterrows():
        cells = [
            str(row[column]).replace("_", r"\_")
            for column in index_columns
        ]
        cells.extend(_format(row.get(metric, np.nan)) for metric, _ in metrics)
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"    \bottomrule", r"    \end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def write_definitions(out_dir):
    text = """Mask-SC selection-analysis definitions

Oracle-relevant prior:
  An earlier task from the same family as the target task.

Parent prior:
  The most recent earlier same-family task. In an interleaved hierarchical
  curriculum, this is the preceding depth from the same family.

Raw support:
  Similarity candidates above the configured threshold, sorted by similarity
  and limited by the configured top-k cap, before any performance filter.

Final support:
  The selected support recorded by the trainer after optional performance
  filtering. This is the support used by the mask-composition layers.

Cap utilization:
  Final support size divided by min(k, number of available priors).

Precision and recall:
  Same-family selected priors divided by selected priors, and same-family
  selected priors divided by all available earlier same-family priors.

Jaccard stability:
  Jaccard similarity between final supports at consecutive selection events.

First-correct latency:
  Selection events and inferred environment steps until the first event that
  selects at least one same-family prior.
"""
    with open(os.path.join(out_dir, "analysis_definitions.txt"), "w") as handle:
        handle.write(text)


def plot_support_distribution(event_df, output_path):
    plt = common._setup_matplotlib()
    methods = list(event_df["method"].drop_duplicates())
    figure, axes = plt.subplots(
        len(methods),
        1,
        figsize=(6.4, max(3.0, 2.4 * len(methods))),
        squeeze=False,
    )
    for axis, method in zip(axes[:, 0], methods):
        values = event_df.loc[event_df["method"] == method, "support_size"]
        bins = (
            np.arange(-0.5, values.max() + 1.5, 1.0)
            if len(values)
            else [-0.5, 0.5]
        )
        axis.hist(values, bins=bins, density=True, alpha=0.8)
        axis.set_title(method)
        axis.set_ylabel("Probability")
        axis.grid(axis="y", alpha=0.2)
    axes[-1, 0].set_xlabel("Selected prior count")
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def run_analysis(
    methods,
    task_order_config,
    out_dir,
    family_stride=None,
    family_names=None,
    min_depth=2,
    analysis_min_depth=None,
    threshold_override=None,
    cap_override=None,
    per_task_table=False,
    plots=True,
    composition_analysis=False,
):
    os.makedirs(out_dir, exist_ok=True)
    write_definitions(out_dir)
    metadata = common.load_task_metadata(
        task_order_config,
        family_stride=family_stride,
        family_names=family_names,
        min_depth=min_depth,
    )

    event_frames = []
    task_frames = []
    final_confusion_frames = []
    raw_confusion_frames = []
    beta_frames = []
    skipped = []

    for method, root in methods.items():
        runs = common.discover_runs(root)
        if not runs:
            skipped.append(f"{method}: no task_similarities.csv under {root}")
            continue
        for run_index, run_dir in enumerate(runs):
            seed = common.extract_seed(run_dir)
            if seed is None:
                seed = run_index
            selection_config = load_run_selection_config(
                run_dir,
                threshold_override=threshold_override,
                cap_override=cap_override,
            )
            events = read_selection_events(
                os.path.join(run_dir, "task_similarities.csv"),
                threshold=selection_config["threshold"],
                cap=selection_config["cap"],
            )
            event_df, task_df, final_confusion, raw_confusion = (
                analyze_run_events(
                    events,
                    metadata,
                    method,
                    run_dir,
                    seed,
                    selection_config["cap"],
                    selection_config["threshold"],
                    selection_config["strategy"],
                )
            )
            event_frames.append(event_df)
            task_frames.append(task_df)
            final_confusion_frames.append(final_confusion)
            raw_confusion_frames.append(raw_confusion)

            if composition_analysis:
                logged_path = os.path.join(run_dir, "beta_composition.csv")
                if os.path.isfile(logged_path):
                    beta_df = common.read_logged_beta_composition(
                        logged_path, metadata, method, run_dir, seed
                    )
                else:
                    beta_df = common.reconstruct_beta_composition(
                        run_dir, events, metadata, method, seed
                    )
                if not beta_df.empty:
                    beta_frames.append(beta_df)

    if not event_frames:
        raise ValueError("No Mask-SC selection logs found.\n" + "\n".join(skipped))

    event_df = pd.concat(event_frames, ignore_index=True)
    task_df = pd.concat(task_frames, ignore_index=True)
    final_confusion_df = pd.concat(
        final_confusion_frames, ignore_index=True
    )
    raw_confusion_df = pd.concat(raw_confusion_frames, ignore_index=True)
    beta_df = (
        pd.concat(beta_frames, ignore_index=True)
        if beta_frames
        else pd.DataFrame()
    )

    if analysis_min_depth is not None:
        event_df = event_df[event_df["depth"] >= analysis_min_depth].copy()
        task_df = task_df[task_df["depth"] >= analysis_min_depth].copy()
        final_confusion_df = final_confusion_df[
            final_confusion_df["depth"] >= analysis_min_depth
        ].copy()
        raw_confusion_df = raw_confusion_df[
            raw_confusion_df["depth"] >= analysis_min_depth
        ].copy()
        if not beta_df.empty:
            beta_df = beta_df[beta_df["depth"] >= analysis_min_depth].copy()
        if task_df.empty:
            raise ValueError(
                f"No tasks remain after --analysis-min-depth {analysis_min_depth}"
            )

    event_df.to_csv(os.path.join(out_dir, "selection_events.csv"), index=False)
    task_df.to_csv(
        os.path.join(out_dir, "selection_by_run_task.csv"), index=False
    )

    depth_summary = summarize_by_group(task_df, ["depth"])
    overall_summary = summarize_by_group(task_df, [])
    per_task_summary = summarize_by_group(
        task_df, ["task_idx", "task_name"]
    )
    depth_summary.to_csv(
        os.path.join(out_dir, "selection_by_depth.csv"), index=False
    )
    overall_summary.to_csv(
        os.path.join(out_dir, "selection_summary.csv"), index=False
    )
    per_task_summary.to_csv(
        os.path.join(out_dir, "selection_by_task.csv"), index=False
    )

    support_distribution = (
        event_df.groupby(["method", "seed", "task_idx", "support_size"])
        .size()
        .rename("count")
        .reset_index()
    )
    support_distribution.to_csv(
        os.path.join(out_dir, "support_size_distribution.csv"), index=False
    )

    if not final_confusion_df.empty:
        common.write_confusion_outputs(
            final_confusion_df, metadata, out_dir
        )
    if not raw_confusion_df.empty:
        raw_dir = os.path.join(out_dir, "raw_support")
        os.makedirs(raw_dir, exist_ok=True)
        common.write_confusion_outputs(raw_confusion_df, metadata, raw_dir)

    overall_tex = render_latex_table(
        overall_summary,
        [],
        "Mask-SC selection analysis. Oracle-relevant priors are earlier "
        "tasks from the same family.",
        "tbl:masksc_selection",
    )
    with open(os.path.join(out_dir, "selection_summary.tex"), "w") as handle:
        handle.write(overall_tex + "\n")

    depth_tex = render_latex_table(
        depth_summary,
        ["depth"],
        "Mask-SC selection analysis by task depth.",
        "tbl:masksc_selection_depth",
    )
    with open(os.path.join(out_dir, "selection_by_depth.tex"), "w") as handle:
        handle.write(depth_tex + "\n")

    if per_task_table:
        task_tex = render_latex_table(
            per_task_summary,
            ["task_idx", "task_name"],
            "Mask-SC selection analysis by target task.",
            "tbl:masksc_selection_task",
        )
        with open(os.path.join(out_dir, "selection_by_task.tex"), "w") as handle:
            handle.write(task_tex + "\n")

    if plots:
        plot_specs = [
            ("support_size", "Selected prior count", "support_size_by_depth.pdf"),
            (
                "support_fraction",
                "Fraction of available library",
                "support_fraction_by_depth.pdf",
            ),
            (
                "same_family_precision",
                "Same-family precision",
                "same_family_precision_by_depth.pdf",
            ),
            (
                "same_family_recall",
                "Same-family recall",
                "same_family_recall_by_depth.pdf",
            ),
            (
                "parent_selected",
                "Immediate-parent retrieval probability",
                "parent_retrieval_by_depth.pdf",
            ),
            (
                "any_same_family",
                "Probability of at least one same-family prior",
                "any_same_family_by_depth.pdf",
            ),
            (
                "jaccard_stability",
                "Consecutive-event Jaccard",
                "jaccard_stability_by_depth.pdf",
            ),
            (
                "cap_utilization",
                "Fraction of top-k capacity used",
                "cap_utilization_by_depth.pdf",
            ),
            (
                "performance_filter_retention",
                "Fraction retained after performance filter",
                "performance_filter_retention_by_depth.pdf",
            ),
            (
                "raw_support_size",
                "Pre-filter selected prior count",
                "raw_support_size_by_depth.pdf",
            ),
            (
                "raw_same_family_precision",
                "Pre-filter same-family precision",
                "raw_same_family_precision_by_depth.pdf",
            ),
            (
                "selected_similarity_mean",
                "Mean selected cosine similarity",
                "selected_similarity_by_depth.pdf",
            ),
            (
                "selection_events_until_first_correct",
                "Selection events until first correct retrieval",
                "first_correct_selection_event_by_depth.pdf",
            ),
            (
                "steps_until_first_correct",
                "Steps from inferred task start until correct retrieval",
                "first_correct_steps_by_depth.pdf",
            ),
            (
                "first_correct_found",
                "Probability correct family is ever retrieved",
                "first_correct_found_by_depth.pdf",
            ),
        ]
        for metric, ylabel, filename in plot_specs:
            if depth_summary.loc[
                depth_summary["metric"] == metric, "mean"
            ].notna().any():
                common.plot_metric(
                    depth_summary,
                    metric,
                    ylabel,
                    os.path.join(out_dir, filename),
                )
        plot_support_distribution(
            event_df,
            os.path.join(out_dir, "support_size_distribution.pdf"),
        )

    beta_task_df = pd.DataFrame()
    if not beta_df.empty:
        beta_df.to_csv(
            os.path.join(out_dir, "beta_composition_components.csv"),
            index=False,
        )
        beta_task_df = common.summarize_composition(beta_df)
        beta_task_df.to_csv(
            os.path.join(out_dir, "beta_composition_by_run_task.csv"),
            index=False,
        )
        beta_depth = (
            beta_task_df.groupby(["method", "seed", "depth"])[
                [
                    "same_family_beta_mass",
                    "cross_family_beta_mass",
                    "current_beta_mass",
                    "selected_prior_count",
                    "active_component_count",
                    "effective_n",
                ]
            ]
            .mean()
            .reset_index()
        )
        beta_depth.to_csv(
            os.path.join(out_dir, "beta_composition_by_depth.csv"),
            index=False,
        )
        with open(
            os.path.join(out_dir, "beta_composition_summary.tex"), "w"
        ) as handle:
            handle.write(common.render_composition_latex(beta_task_df) + "\n")
        if plots:
            common.plot_composition(beta_task_df, out_dir)

    print(overall_tex)
    print(f"\nWrote Mask-SC selection analysis to {os.path.abspath(out_dir)}")
    for message in skipped:
        print(f"Skipped {message}")
    return {
        "events": event_df,
        "tasks": task_df,
        "depth_summary": depth_summary,
        "overall_summary": overall_summary,
        "beta_tasks": beta_task_df,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Mask-SC fixed-cap policy retrieval.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--methods", nargs="+", required=True, help="NAME=PATH")
    parser.add_argument("--task-order-config", required=True)
    parser.add_argument("--out-dir", default="masksc_selection_analysis")
    parser.add_argument("--family-stride", type=int, default=None)
    parser.add_argument("--family-names", nargs="*", default=None)
    parser.add_argument("--min-depth", type=int, default=2)
    parser.add_argument(
        "--analysis-min-depth",
        type=int,
        default=None,
        help="Use 3 to exclude cold-start depth-2 tasks.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=None,
        help="Override the threshold read from each run's config.json.",
    )
    parser.add_argument(
        "--selection-cap",
        type=int,
        default=None,
        help="Override detect_topk read from each run's config.json.",
    )
    parser.add_argument("--per-task-table", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--composition-analysis", action="store_true")
    args = parser.parse_args()

    run_analysis(
        common.parse_methods(args.methods),
        args.task_order_config,
        args.out_dir,
        family_stride=args.family_stride,
        family_names=args.family_names,
        min_depth=args.min_depth,
        analysis_min_depth=args.analysis_min_depth,
        threshold_override=args.similarity_threshold,
        cap_override=args.selection_cap,
        per_task_table=args.per_task_table,
        plots=not args.no_plots,
        composition_analysis=args.composition_analysis,
    )


if __name__ == "__main__":
    main()
