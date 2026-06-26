#!/usr/bin/env python3
"""Analyze AMSC retrieval support and mask-composition coefficients.

The selection analysis reads ``task_similarities.csv`` files. Oracle-relevant
priors are defined as earlier tasks from the same family.

The optional composition analysis first uses ``beta_composition.csv`` when it
is present. For older runs, it reconstructs the task-completion composition
from the final model checkpoint and the final selected support recorded for
each task.

Example:

    python analyze_amsc_selection.py \
      --methods AMSC-norm=<log_root> AMSC-no-norm=<log_root> \
      --task-order-config env_configs/ct28/seed1/meta_ctgraph_ct28_interleaved.json \
      --out-dir selection_analysis_ct28 \
      --family-stride 4 --family-names A B C D \
      --composition-analysis --per-task-table
"""

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TaskMeta:
    idx: int
    name: str
    family: str
    depth: int


SELECTION_METRICS = [
    "support_size",
    "support_fraction",
    "same_family_precision",
    "same_family_recall",
    "any_same_family",
    "jaccard_stability",
    "first_correct_found",
    "selection_events_until_first_correct",
    "steps_until_first_correct",
]


def parse_methods(items):
    methods = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected NAME=PATH, got {item!r}")
        name, path = item.split("=", 1)
        methods[name] = os.path.abspath(os.path.expanduser(path))
    return methods


def extract_seed(path):
    patterns = [
        r"supermask-(\d+)(?:-|/)",
        r"(?:^|[/_-])seed[_-]?(\d+)(?:[/_-]|$)",
        r"(?:^|[/_-])run[_-]?(\d+)(?:[/_-]|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, path, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def discover_runs(root):
    runs = []
    for dirpath, _, files in os.walk(root):
        if "task_similarities.csv" in files:
            runs.append(dirpath)
    return sorted(runs)


def _parse_minigrid_task(task_idx, task_name):
    match = re.search(r"Env-(.+)-R(\d+)-v\d+$", task_name)
    if not match:
        return None
    return TaskMeta(task_idx, task_name, match.group(1), int(match.group(2)))


def load_task_metadata(
    config_path,
    family_stride=None,
    family_names=None,
    min_depth=2,
):
    with open(config_path, "r") as handle:
        config = json.load(handle)

    tasks = config.get("tasks")
    if isinstance(tasks, list) and tasks:
        parsed = [_parse_minigrid_task(idx, str(name)) for idx, name in enumerate(tasks)]
        if all(item is not None for item in parsed):
            return parsed

    num_tasks = int(config.get("num_tasks", len(config.get("filter_tasks", []))))
    if num_tasks <= 0:
        raise ValueError(
            "Could not determine task count from the task-order configuration"
        )

    if family_stride is None:
        image_seeds = set()
        for path in config.get("config_paths", []):
            match = re.search(r"imgseed(\d+)", str(path), flags=re.IGNORECASE)
            if match:
                image_seeds.add(int(match.group(1)))
        if image_seeds and num_tasks % len(image_seeds) == 0:
            family_stride = len(image_seeds)

    if family_stride is None or family_stride <= 0:
        raise ValueError(
            "Could not infer task families. Pass --family-stride; for CT28 use 4."
        )

    if family_names is None or len(family_names) == 0:
        family_names = [f"F{idx + 1}" for idx in range(family_stride)]
    if len(family_names) != family_stride:
        raise ValueError(
            f"Expected {family_stride} family names, got {len(family_names)}"
        )

    metadata = []
    for task_idx in range(num_tasks):
        family_idx = task_idx % family_stride
        depth = min_depth + task_idx // family_stride
        family = str(family_names[family_idx])
        metadata.append(
            TaskMeta(task_idx, f"{family}-D{depth}", family, depth)
        )
    return metadata


def _as_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def read_selection_events(csv_path, selection_column="selected"):
    grouped = defaultdict(list)
    with open(csv_path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header in {csv_path}")
        if selection_column not in reader.fieldnames:
            raise ValueError(
                f"{csv_path} has no {selection_column!r} column; "
                f"available columns: {reader.fieldnames}"
            )
        for row in reader:
            task_idx = _as_int(row.get("task_idx"), -1)
            if task_idx < 0:
                continue
            key = (
                _as_int(row.get("learn_block")),
                task_idx,
                _as_int(row.get("iteration")),
                _as_int(row.get("total_steps")),
            )
            grouped[key].append(row)

    events = []
    for key, rows in grouped.items():
        learn_block, task_idx, iteration, total_steps = key
        selected = {
            _as_int(row.get("prev_idx"), -1)
            for row in rows
            if _as_int(row.get(selection_column)) == 1
        }
        selected.discard(-1)
        events.append(
            {
                "learn_block": learn_block,
                "task_idx": task_idx,
                "iteration": iteration,
                "total_steps": total_steps,
                "selected": selected,
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


def _jaccard(left, right):
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def analyze_run_events(events, metadata, method, run_dir, seed):
    meta_by_idx = {task.idx: task for task in metadata}
    rows = []
    confusion_rows = []
    previous_support = {}
    event_number = defaultdict(int)
    first_event_step = {}
    first_correct = {}
    event_count = defaultdict(int)
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
        relevant = {
            idx
            for idx in range(task_idx)
            if meta_by_idx[idx].family == task.family
        }
        correct = selected & relevant

        event_number[task_idx] += 1
        event_count[task_idx] += 1
        first_event_step.setdefault(task_idx, event["total_steps"])
        if correct and task_idx not in first_correct:
            if np.isfinite(iterations_per_task) and np.isfinite(steps_per_iteration):
                inferred_task_start_step = (
                    task_idx * iterations_per_task * steps_per_iteration
                )
                steps_from_task_start = max(
                    0.0, event["total_steps"] - inferred_task_start_step
                )
            else:
                steps_from_task_start = (
                    event["total_steps"] - first_event_step[task_idx]
                )
            first_correct[task_idx] = {
                "event": event_number[task_idx],
                "steps": steps_from_task_start,
                "steps_after_first_selection": (
                    event["total_steps"] - first_event_step[task_idx]
                ),
            }

        support_size = len(selected)
        precision = len(correct) / support_size if support_size else np.nan
        recall = len(correct) / len(relevant) if relevant else np.nan
        any_correct = float(bool(correct)) if relevant else np.nan
        previous = previous_support.get(task_idx)
        jaccard = _jaccard(previous, selected) if previous is not None else np.nan
        previous_support[task_idx] = selected

        rows.append(
            {
                "method": method,
                "seed": seed,
                "run_dir": run_dir,
                "task_idx": task_idx,
                "task_name": task.name,
                "family": task.family,
                "depth": task.depth,
                "learn_block": event["learn_block"],
                "iteration": event["iteration"],
                "total_steps": event["total_steps"],
                "event_number": event_number[task_idx],
                "support_size": support_size,
                "support_fraction": support_size / task_idx if task_idx else np.nan,
                "same_family_precision": precision,
                "same_family_recall": recall,
                "any_same_family": any_correct,
                "jaccard_stability": jaccard,
                "relevant_prior_count": len(relevant),
                "correct_prior_count": len(correct),
                "selected_indices": " ".join(map(str, sorted(selected))),
            }
        )
        for prior_idx in selected:
            confusion_rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "current_family": task.family,
                    "prior_family": meta_by_idx[prior_idx].family,
                    "task_idx": task_idx,
                    "depth": task.depth,
                }
            )

    event_df = pd.DataFrame(rows)
    task_rows = []
    for task in metadata:
        task_events = event_df[event_df["task_idx"] == task.idx]
        if task_events.empty and task.idx != 0:
            continue
        relevant_count = sum(
            prior.family == task.family for prior in metadata[: task.idx]
        )
        row = {
            "method": method,
            "seed": seed,
            "run_dir": run_dir,
            "task_idx": task.idx,
            "task_name": task.name,
            "family": task.family,
            "depth": task.depth,
            "num_selection_events": len(task_events),
            "relevant_prior_count": relevant_count,
        }
        if task_events.empty:
            row.update(
                {
                    "support_size": 0.0,
                    "support_fraction": np.nan,
                    "same_family_precision": np.nan,
                    "same_family_recall": np.nan,
                    "any_same_family": np.nan,
                    "jaccard_stability": np.nan,
                }
            )
        else:
            for metric in [
                "support_size",
                "support_fraction",
                "same_family_precision",
                "same_family_recall",
                "any_same_family",
                "jaccard_stability",
            ]:
                row[metric] = float(task_events[metric].mean())
        found = task.idx in first_correct
        row["first_correct_found"] = float(found) if relevant_count else np.nan
        row["selection_events_until_first_correct"] = (
            float(first_correct[task.idx]["event"]) if found else np.nan
        )
        row["steps_until_first_correct"] = (
            float(first_correct[task.idx]["steps"]) if found else np.nan
        )
        row["steps_after_first_selection_until_first_correct"] = (
            float(first_correct[task.idx]["steps_after_first_selection"])
            if found
            else np.nan
        )
        task_rows.append(row)

    return (
        event_df,
        pd.DataFrame(task_rows),
        pd.DataFrame(confusion_rows),
    )


def _mean_ci(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan, 0
    mean = float(values.mean())
    if values.size == 1:
        return mean, mean, mean, 1
    half = 1.96 * float(values.std(ddof=1)) / math.sqrt(values.size)
    return mean, mean - half, mean + half, int(values.size)


def summarize_by_group(task_df, group_columns):
    rows = []
    seed_group = (
        task_df.groupby(["method", "seed"] + group_columns, dropna=False)[
            SELECTION_METRICS
        ]
        .mean(numeric_only=True)
        .reset_index()
    )
    for keys, group in seed_group.groupby(["method"] + group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(["method"] + group_columns, keys))
        for metric in SELECTION_METRICS:
            mean, low, high, count = _mean_ci(group[metric].to_numpy())
            if metric in {
                "support_fraction",
                "same_family_precision",
                "same_family_recall",
                "any_same_family",
                "jaccard_stability",
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


def _format_value(value, digits=3):
    return "--" if not np.isfinite(value) else f"{value:.{digits}f}"


def render_latex_table(summary, row_columns, caption, label):
    metrics = [
        ("support_size", "Support"),
        ("support_fraction", "Library frac."),
        ("same_family_precision", "Precision"),
        ("same_family_recall", "Recall"),
        ("any_same_family", "Any correct"),
        ("jaccard_stability", "Jaccard"),
        ("first_correct_found", "First found"),
        ("selection_events_until_first_correct", "Events to first"),
        ("steps_until_first_correct", "Steps to first"),
    ]
    index_cols = ["method"] + row_columns
    pivot = summary.pivot_table(
        index=index_cols, columns="metric", values="mean", aggfunc="first"
    ).reset_index()
    lines = [
        r"\begin{table}[ht]",
        f"    \\caption{{{caption}}}",
        f"    \\label{{{label}}}",
        r"    \centering",
        r"    \scriptsize",
        r"    \begin{tabular}{"
        + "l" * len(index_cols)
        + "r" * len(metrics)
        + "}",
        r"    \toprule",
        " & ".join(
            [column.replace("_", " ").title() for column in index_cols]
            + [label for _, label in metrics]
        )
        + r" \\",
        r"    \midrule",
    ]
    for _, row in pivot.iterrows():
        cells = [str(row[column]).replace("_", r"\_") for column in index_cols]
        cells.extend(_format_value(row.get(metric, np.nan)) for metric, _ in metrics)
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"    \bottomrule", r"    \end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def _setup_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-amsc-selection")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/amsc-selection-cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def write_definitions(out_dir):
    definitions = """AMSC selection-analysis definitions

Oracle-relevant prior:
  An earlier task from the same task family as the current task.

Support size:
  Number of prior masks selected at one selection event. Depth summaries first
  average events within each run/task and then average tasks within each seed.

Support fraction:
  Selected support size divided by the number of prior tasks available.

Same-family precision:
  Selected same-family priors divided by selected priors. Undefined for an
  empty selected support.

Same-family recall:
  Selected same-family priors divided by all earlier same-family priors.
  Undefined when no earlier same-family prior exists.

Any same-family:
  Indicator that at least one earlier same-family prior is selected.

Jaccard stability:
  Jaccard similarity between supports at consecutive selection events for the
  same target task. Two consecutive empty supports have stability 1.

First-correct latency:
  Number of selection events until a same-family prior is first selected.
  Environment-step latency is inferred from the fixed per-task iteration budget
  observed in the log. The CSV also reports steps after the first logged
  selection opportunity. Failed retrievals are reported separately by
  first_correct_found.

Family confusion:
  Counts and row-normalized fractions of selected edges from each current-task
  family to each selected-prior family.

Composition reconstruction:
  For older runs, beta weights are reconstructed by softmaxing each saved beta
  row only over the final selected prior support plus the current task mask.
  New runs use beta_composition.csv logged immediately before consolidation.
"""
    with open(os.path.join(out_dir, "analysis_definitions.txt"), "w") as handle:
        handle.write(definitions)


def plot_metric(summary, metric, ylabel, output_path):
    plt = _setup_matplotlib()
    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    subset = summary[summary["metric"] == metric]
    for method, group in subset.groupby("method"):
        group = group.sort_values("depth")
        axis.plot(group["depth"], group["mean"], marker="o", label=method)
        axis.fill_between(
            group["depth"].to_numpy(dtype=float),
            group["ci_low"].to_numpy(dtype=float),
            group["ci_high"].to_numpy(dtype=float),
            alpha=0.16,
        )
    axis.set_xlabel("Task depth")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def plot_support_distribution(event_df, output_path):
    plt = _setup_matplotlib()
    methods = list(event_df["method"].drop_duplicates())
    figure, axes = plt.subplots(
        len(methods), 1, figsize=(6.4, max(3.0, 2.4 * len(methods))), squeeze=False
    )
    for axis, method in zip(axes[:, 0], methods):
        values = event_df.loc[event_df["method"] == method, "support_size"]
        bins = np.arange(-0.5, values.max() + 1.5, 1.0) if len(values) else [-0.5, 0.5]
        axis.hist(values, bins=bins, density=True, alpha=0.8)
        axis.set_title(method)
        axis.set_ylabel("Probability")
        axis.grid(axis="y", alpha=0.2)
    axes[-1, 0].set_xlabel("Selected prior count")
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def write_confusion_outputs(confusion_df, metadata, out_dir):
    if confusion_df.empty:
        return
    plt = _setup_matplotlib()
    families = list(dict.fromkeys(task.family for task in metadata))
    for method, group in confusion_df.groupby("method"):
        matrix = pd.crosstab(group["current_family"], group["prior_family"])
        matrix = matrix.reindex(index=families, columns=families, fill_value=0)
        normalized = matrix.div(matrix.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", method)
        matrix.to_csv(os.path.join(out_dir, f"{stem}_family_confusion_counts.csv"))
        normalized.to_csv(
            os.path.join(out_dir, f"{stem}_family_confusion_normalized.csv")
        )

        figure, axis = plt.subplots(figsize=(5.0, 4.3))
        image = axis.imshow(normalized.to_numpy(), vmin=0, vmax=1, cmap="Blues")
        axis.set_xticks(range(len(families)), families, rotation=35, ha="right")
        axis.set_yticks(range(len(families)), families)
        axis.set_xlabel("Selected prior family")
        axis.set_ylabel("Current task family")
        axis.set_title(method)
        for row in range(len(families)):
            for column in range(len(families)):
                value = normalized.iloc[row, column]
                axis.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value > 0.55 else "black",
                )
        figure.colorbar(image, ax=axis, label="Fraction of selected edges")
        figure.tight_layout()
        figure.savefig(
            os.path.join(out_dir, f"{stem}_family_confusion.pdf")
        )
        plt.close(figure)


def _find_model_checkpoint(run_dir):
    candidates = []
    for name in os.listdir(run_dir):
        if name.endswith(".bin") and "-model-" in name:
            candidates.append(os.path.join(run_dir, name))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _final_support_by_task(events):
    final = {}
    for event in events:
        final[event["task_idx"]] = sorted(
            idx for idx in event["selected"] if idx < event["task_idx"]
        )
    return final


def _softmax(values):
    values = np.asarray(values, dtype=float)
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / exp.sum()


def reconstruct_beta_composition(run_dir, events, metadata, method, seed):
    checkpoint = _find_model_checkpoint(run_dir)
    if checkpoint is None:
        return pd.DataFrame()
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required for composition analysis") from error

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "network" in state and isinstance(state["network"], dict):
        state = state["network"]
    beta_items = [
        (key[: -len(".betas")], tensor.detach().cpu().numpy())
        for key, tensor in state.items()
        if key.endswith(".betas") and getattr(tensor, "ndim", 0) == 2
    ]
    if not beta_items:
        return pd.DataFrame()

    meta_by_idx = {task.idx: task for task in metadata}
    final_support = _final_support_by_task(events)
    rows = []
    for task in metadata:
        selected = final_support.get(task.idx, [])
        active = selected + [task.idx]
        for layer, betas in beta_items:
            if task.idx >= betas.shape[0] or max(active) >= betas.shape[1]:
                continue
            logits = betas[task.idx, active]
            weights = _softmax(logits)
            effective_n = float(1.0 / np.square(weights).sum())
            for component_idx, logit, weight in zip(active, logits, weights):
                if component_idx == task.idx:
                    role = "current"
                elif meta_by_idx[component_idx].family == task.family:
                    role = "same_family"
                else:
                    role = "cross_family"
                rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "run_dir": run_dir,
                        "task_idx": task.idx,
                        "task_name": task.name,
                        "family": task.family,
                        "depth": task.depth,
                        "layer": layer,
                        "component_idx": component_idx,
                        "component_role": role,
                        "beta_logit": float(logit),
                        "beta_weight": float(weight),
                        "selected_prior_count": len(selected),
                        "effective_n": effective_n,
                        "source": "checkpoint_reconstruction",
                    }
                )
    return pd.DataFrame(rows)


def read_logged_beta_composition(path, metadata, method, run_dir, seed):
    frame = pd.read_csv(path)
    meta_by_idx = {task.idx: task for task in metadata}
    rows = []
    for _, row in frame.iterrows():
        task_idx = int(row["task_idx"])
        component_idx = int(row["component_idx"])
        if task_idx not in meta_by_idx:
            continue
        task = meta_by_idx[task_idx]
        if component_idx == task_idx:
            role = "current"
        elif meta_by_idx[component_idx].family == task.family:
            role = "same_family"
        else:
            role = "cross_family"
        rows.append(
            {
                **row.to_dict(),
                "method": method,
                "seed": seed,
                "run_dir": run_dir,
                "task_name": task.name,
                "family": task.family,
                "depth": task.depth,
                "component_role": role,
                "source": "task_end_log",
            }
        )
    return pd.DataFrame(rows)


def summarize_composition(beta_df):
    if beta_df.empty:
        return pd.DataFrame()
    role_mass = (
        beta_df.groupby(
            [
                "method",
                "seed",
                "run_dir",
                "task_idx",
                "task_name",
                "family",
                "depth",
                "layer",
                "selected_prior_count",
                "effective_n",
            ],
            dropna=False,
        )
        .apply(
            lambda group: pd.Series(
                {
                    "same_family_beta_mass": group.loc[
                        group["component_role"] == "same_family", "beta_weight"
                    ].sum(),
                    "cross_family_beta_mass": group.loc[
                        group["component_role"] == "cross_family", "beta_weight"
                    ].sum(),
                    "current_beta_mass": group.loc[
                        group["component_role"] == "current", "beta_weight"
                    ].sum(),
                }
            )
        )
        .reset_index()
    )
    summary = (
        role_mass.groupby(
            [
                "method",
                "seed",
                "run_dir",
                "task_idx",
                "task_name",
                "family",
                "depth",
            ],
            dropna=False,
        )[
            [
                "same_family_beta_mass",
                "cross_family_beta_mass",
                "current_beta_mass",
                "selected_prior_count",
                "effective_n",
            ]
        ]
        .mean()
        .reset_index()
    )
    summary["active_component_count"] = summary["selected_prior_count"] + 1.0
    return summary


def render_composition_latex(beta_task_df):
    metrics = [
        ("same_family_beta_mass", "Same-family mass"),
        ("cross_family_beta_mass", "Cross-family mass"),
        ("current_beta_mass", "Current mass"),
        ("selected_prior_count", "Selected priors"),
        ("active_component_count", "Active components"),
        ("effective_n", r"$N_{\mathrm{eff}}$"),
    ]
    seed_means = (
        beta_task_df.groupby(["method", "seed"])[[key for key, _ in metrics]]
        .mean()
        .reset_index()
    )
    method_means = seed_means.groupby("method")[[key for key, _ in metrics]].mean()
    lines = [
        r"\begin{table}[ht]",
        r"    \caption{AMSC task-completion mask-composition analysis.}",
        r"    \label{tbl:amsc_composition}",
        r"    \centering",
        r"    \scriptsize",
        r"    \begin{tabular}{lrrrrrr}",
        r"    \toprule",
        "Method & " + " & ".join(label for _, label in metrics) + r" \\",
        r"    \midrule",
    ]
    for method, row in method_means.iterrows():
        values = [_format_value(row[key]) for key, _ in metrics]
        lines.append(str(method).replace("_", r"\_") + " & " + " & ".join(values) + r" \\")
    lines.extend([r"    \bottomrule", r"    \end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def plot_composition(beta_task_df, out_dir):
    if beta_task_df.empty:
        return
    plt = _setup_matplotlib()
    seed_depth = (
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
    depth_mean = seed_depth.groupby(["method", "depth"]).mean(numeric_only=True).reset_index()

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for method, group in depth_mean.groupby("method"):
        group = group.sort_values("depth")
        axes[0].plot(
            group["depth"], group["same_family_beta_mass"], marker="o",
            label=f"{method}: same"
        )
        axes[0].plot(
            group["depth"], group["cross_family_beta_mass"], marker="s",
            linestyle="--", label=f"{method}: cross"
        )
        axes[0].plot(
            group["depth"], group["current_beta_mass"], marker="^",
            linestyle=":", label=f"{method}: current"
        )
        axes[1].plot(
            group["depth"], group["selected_prior_count"], marker="o",
            label=f"{method}: selected priors"
        )
        axes[1].plot(
            group["depth"], group["active_component_count"], marker="^",
            linestyle=":", label=f"{method}: active incl. current"
        )
        axes[1].plot(
            group["depth"], group["effective_n"], marker="s",
            linestyle="--", label=f"{method}: effective"
        )
    axes[0].set_xlabel("Task depth")
    axes[0].set_ylabel("Mean beta mass")
    axes[1].set_xlabel("Task depth")
    axes[1].set_ylabel("Number of components")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(os.path.join(out_dir, "composition_by_depth.pdf"))
    plt.close(figure)


def run_analysis(
    methods,
    task_order_config,
    out_dir,
    family_stride=None,
    family_names=None,
    min_depth=2,
    selection_column="selected",
    analysis_min_depth=None,
    per_task_table=False,
    plots=True,
    composition_analysis=False,
):
    if composition_analysis and selection_column != "selected":
        raise ValueError(
            "Composition analysis requires --selection-column selected because "
            "the beta coefficients were optimized over the actual support."
        )
    os.makedirs(out_dir, exist_ok=True)
    write_definitions(out_dir)
    metadata = load_task_metadata(
        task_order_config,
        family_stride=family_stride,
        family_names=family_names,
        min_depth=min_depth,
    )

    event_frames = []
    task_frames = []
    confusion_frames = []
    beta_frames = []
    skipped = []

    for method, root in methods.items():
        runs = discover_runs(root)
        if not runs:
            skipped.append(f"{method}: no task_similarities.csv under {root}")
            continue
        for run_index, run_dir in enumerate(runs):
            seed = extract_seed(run_dir)
            if seed is None:
                seed = run_index
            similarity_path = os.path.join(run_dir, "task_similarities.csv")
            events = read_selection_events(similarity_path, selection_column)
            event_df, task_df, confusion_df = analyze_run_events(
                events, metadata, method, run_dir, seed
            )
            event_frames.append(event_df)
            task_frames.append(task_df)
            confusion_frames.append(confusion_df)

            if composition_analysis:
                logged_path = os.path.join(run_dir, "beta_composition.csv")
                if os.path.isfile(logged_path):
                    beta_df = read_logged_beta_composition(
                        logged_path, metadata, method, run_dir, seed
                    )
                else:
                    beta_df = reconstruct_beta_composition(
                        run_dir, events, metadata, method, seed
                    )
                if not beta_df.empty:
                    beta_frames.append(beta_df)

    if not event_frames:
        details = "\n".join(skipped)
        raise ValueError(f"No selection logs were found.\n{details}")

    event_df = pd.concat(event_frames, ignore_index=True)
    task_df = pd.concat(task_frames, ignore_index=True)
    confusion_df = pd.concat(confusion_frames, ignore_index=True)
    beta_df = pd.concat(beta_frames, ignore_index=True) if beta_frames else pd.DataFrame()

    if analysis_min_depth is not None:
        event_df = event_df[event_df["depth"] >= analysis_min_depth].copy()
        task_df = task_df[task_df["depth"] >= analysis_min_depth].copy()
        confusion_df = confusion_df[
            confusion_df["depth"] >= analysis_min_depth
        ].copy()
        if not beta_df.empty:
            beta_df = beta_df[beta_df["depth"] >= analysis_min_depth].copy()
        if task_df.empty:
            raise ValueError(
                f"No tasks remain after --analysis-min-depth {analysis_min_depth}"
            )

    event_df.to_csv(os.path.join(out_dir, "selection_events.csv"), index=False)
    task_df.to_csv(os.path.join(out_dir, "selection_by_run_task.csv"), index=False)

    depth_summary = summarize_by_group(task_df, ["depth"])
    overall_summary = summarize_by_group(task_df, [])
    per_task_summary = summarize_by_group(task_df, ["task_idx", "task_name"])
    depth_summary.to_csv(os.path.join(out_dir, "selection_by_depth.csv"), index=False)
    overall_summary.to_csv(os.path.join(out_dir, "selection_summary.csv"), index=False)
    per_task_summary.to_csv(os.path.join(out_dir, "selection_by_task.csv"), index=False)

    support_distribution = (
        event_df.groupby(["method", "seed", "task_idx", "support_size"])
        .size()
        .rename("count")
        .reset_index()
    )
    support_distribution.to_csv(
        os.path.join(out_dir, "support_size_distribution.csv"), index=False
    )
    write_confusion_outputs(confusion_df, metadata, out_dir)

    overall_tex = render_latex_table(
        overall_summary,
        [],
        "AMSC selection analysis. Oracle-relevant priors are earlier tasks from the same family.",
        "tbl:amsc_selection",
    )
    with open(os.path.join(out_dir, "selection_summary.tex"), "w") as handle:
        handle.write(overall_tex + "\n")

    depth_tex = render_latex_table(
        depth_summary,
        ["depth"],
        "AMSC selection analysis by task depth.",
        "tbl:amsc_selection_depth",
    )
    with open(os.path.join(out_dir, "selection_by_depth.tex"), "w") as handle:
        handle.write(depth_tex + "\n")

    if per_task_table:
        task_tex = render_latex_table(
            per_task_summary,
            ["task_idx", "task_name"],
            "AMSC selection analysis by target task.",
            "tbl:amsc_selection_task",
        )
        with open(os.path.join(out_dir, "selection_by_task.tex"), "w") as handle:
            handle.write(task_tex + "\n")

    if plots:
        plot_metric(
            depth_summary, "support_size", "Selected prior count",
            os.path.join(out_dir, "support_size_by_depth.pdf")
        )
        plot_metric(
            depth_summary, "support_fraction", "Fraction of available library",
            os.path.join(out_dir, "support_fraction_by_depth.pdf")
        )
        plot_metric(
            depth_summary, "same_family_precision", "Same-family precision",
            os.path.join(out_dir, "same_family_precision_by_depth.pdf")
        )
        plot_metric(
            depth_summary, "same_family_recall", "Same-family recall",
            os.path.join(out_dir, "same_family_recall_by_depth.pdf")
        )
        plot_metric(
            depth_summary, "any_same_family",
            "Probability of at least one same-family prior",
            os.path.join(out_dir, "any_same_family_by_depth.pdf")
        )
        plot_metric(
            depth_summary, "jaccard_stability", "Consecutive-event Jaccard",
            os.path.join(out_dir, "jaccard_stability_by_depth.pdf")
        )
        plot_metric(
            depth_summary, "selection_events_until_first_correct",
            "Selection events until first correct retrieval",
            os.path.join(out_dir, "first_correct_selection_event_by_depth.pdf")
        )
        plot_metric(
            depth_summary, "steps_until_first_correct",
            "Steps from inferred task start until correct retrieval",
            os.path.join(out_dir, "first_correct_steps_by_depth.pdf")
        )
        plot_metric(
            depth_summary, "first_correct_found",
            "Probability correct family is ever retrieved",
            os.path.join(out_dir, "first_correct_found_by_depth.pdf")
        )
        plot_support_distribution(
            event_df, os.path.join(out_dir, "support_size_distribution.pdf")
        )

    beta_task_df = pd.DataFrame()
    if not beta_df.empty:
        beta_df.to_csv(os.path.join(out_dir, "beta_composition_components.csv"), index=False)
        beta_task_df = summarize_composition(beta_df)
        beta_task_df.to_csv(
            os.path.join(out_dir, "beta_composition_by_run_task.csv"), index=False
        )
        beta_summary = (
            beta_task_df.groupby(["method", "seed"])[
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
        beta_summary.to_csv(
            os.path.join(out_dir, "beta_composition_summary.csv"), index=False
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
            os.path.join(out_dir, "beta_composition_by_depth.csv"), index=False
        )
        composition_tex = render_composition_latex(beta_task_df)
        with open(os.path.join(out_dir, "beta_composition_summary.tex"), "w") as handle:
            handle.write(composition_tex + "\n")
        if plots:
            plot_composition(beta_task_df, out_dir)

    print(overall_tex)
    print(f"\nWrote selection analysis to {os.path.abspath(out_dir)}")
    for message in skipped:
        print(f"Skipped {message}")
    if composition_analysis and not beta_frames:
        print("No beta composition data or compatible model checkpoints were found.")
    return {
        "events": event_df,
        "tasks": task_df,
        "depth_summary": depth_summary,
        "overall_summary": overall_summary,
        "beta_tasks": beta_task_df,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze AMSC sparse retrieval and mask composition.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--methods", nargs="+", required=True, help="NAME=PATH")
    parser.add_argument("--task-order-config", required=True)
    parser.add_argument("--out-dir", default="amsc_selection_analysis")
    parser.add_argument("--family-stride", type=int, default=None)
    parser.add_argument("--family-names", nargs="*", default=None)
    parser.add_argument("--min-depth", type=int, default=2)
    parser.add_argument(
        "--analysis-min-depth",
        type=int,
        default=None,
        help=(
            "Exclude target tasks below this depth from all summaries, plots, "
            "confusion matrices, and composition analysis. Use 3 to remove D2."
        ),
    )
    parser.add_argument(
        "--selection-column",
        choices=["selected", "pre_shuffle_selected"],
        default="selected",
    )
    parser.add_argument("--per-task-table", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--composition-analysis", action="store_true")
    args = parser.parse_args()
    run_analysis(
        parse_methods(args.methods),
        args.task_order_config,
        args.out_dir,
        family_stride=args.family_stride,
        family_names=args.family_names,
        min_depth=args.min_depth,
        selection_column=args.selection_column,
        analysis_min_depth=args.analysis_min_depth,
        per_task_table=args.per_task_table,
        plots=not args.no_plots,
        composition_analysis=args.composition_analysis,
    )


if __name__ == "__main__":
    main()
