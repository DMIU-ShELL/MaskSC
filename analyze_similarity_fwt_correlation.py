#!/usr/bin/env python3
"""Relate early Mask-SC retrieval similarity to per-task forward transfer.

The analysis unit is one (RL seed, target task) pair. Per-task FWT is computed
with the same functions used by ``significance_test_metrics_latex.py``. Event
similarities are first aggregated within each seed and target task, avoiding
pseudoreplication from assigning one FWT value to every prior or selection
event.

The primary inferential summary computes one Spearman correlation across tasks
for each RL seed. It then reports the mean seed-level correlation, a bootstrap
confidence interval over seeds, and an exact sign-flip p-value. A depth-adjusted
partial Spearman analysis is also reported when a ``depth`` column is present.
Pooled and task-mean correlations are descriptive secondary summaries.

Example:

    python analyze_similarity_fwt_correlation.py \
      --method-root log/checklist_runs/ct28-interleaved-MaskSC-3-thesis \
      --expert-root log/ct28/ct28-interleaved-single-task-experts-PPO \
      --task-order-config env_configs/ct28/seed1/meta_ctgraph_ct28_interleaved.json \
      --selection-events selection_analysis_ct28_masksc3/selection_events.csv \
      --early-events 10 \
      --out-dir similarity_fwt_correlation_ct28
"""

import argparse
import json
import os
import re
from types import SimpleNamespace

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import significance_test_metrics_latex as metric_utils


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Correlate early Mask-SC selected cosine similarity with per-task "
            "forward transfer."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--method-root", required=True)
    parser.add_argument("--expert-root", required=True)
    parser.add_argument("--task-order-config", required=True)
    parser.add_argument(
        "--selection-events",
        required=True,
        help="selection_events.csv produced by analyze_masksc_selection.py",
    )
    parser.add_argument("--out-dir", default="similarity_fwt_correlation")
    parser.add_argument(
        "--similarity-column",
        default="selected_similarity_mean",
        help="Event-level similarity column to aggregate.",
    )
    parser.add_argument(
        "--early-events",
        type=int,
        default=10,
        help="Use the first N selection events of each seed/task pair.",
    )
    parser.add_argument(
        "--aggregate",
        choices=["mean", "median", "max", "first"],
        default="mean",
        help="Aggregate event-level similarities within each seed/task pair.",
    )
    parser.add_argument(
        "--minimum-events",
        type=int,
        default=1,
        help="Minimum number of finite early similarity events required.",
    )
    parser.add_argument(
        "--exclude-task-ids",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Optional target-task indices to exclude from the correlation. "
            "Tasks without selection events are absent automatically."
        ),
    )
    parser.add_argument(
        "--control-column",
        default="depth",
        help="Categorical control used for the adjusted partial Spearman result.",
    )
    parser.add_argument(
        "--performance-normalization",
        choices=["auto", "scalar", "minigrid_shortest_path", "none"],
        default="auto",
    )
    parser.add_argument("--max-return", type=float, default=1.0)
    parser.add_argument(
        "--minigrid-shortest-path-seeds",
        type=int,
        nargs="+",
        default=None,
    )
    parser.add_argument("--min-denominator", type=float, default=1e-3)
    parser.add_argument("--bootstrap-iters", type=int, default=10000)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument(
        "--alternative",
        choices=["two-sided", "greater", "less"],
        default="two-sided",
        help="Alternative for the sign-flip test over seed-level correlations.",
    )
    parser.add_argument("--plot-title", default=None)
    return parser.parse_args()


def ordered_task_names(task_names, task_order, num_tasks):
    return metric_utils.task_names_in_eval_order(
        task_names, task_order, num_tasks
    )


def make_normalization_args(args):
    return SimpleNamespace(
        performance_normalization=args.performance_normalization,
        max_return=args.max_return,
        minigrid_shortest_path_seeds=args.minigrid_shortest_path_seeds,
        task_order_config=args.task_order_config,
    )


def load_expert_aucs(
    expert_root,
    task_order,
    task_names,
    return_scales,
    normalization_mode,
):
    expert_runs = metric_utils.find_runs(expert_root)
    if not expert_runs:
        raise ValueError(f"No expert eval_metrics files found under {expert_root}")

    def load_expert_set(paths):
        id_to_path = {}
        for path in paths:
            task_id = metric_utils.extract_expert_task_id(path, task_names)
            if task_id is None:
                raise ValueError(f"Could not identify expert task for {path}")
            if task_id in id_to_path:
                raise ValueError(
                    f"Duplicate expert task {task_id}: "
                    f"{id_to_path[task_id]} and {path}"
                )
            id_to_path[task_id] = path

        if task_order is not None:
            expected = set(task_order)
            observed = set(id_to_path)
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            if missing or extra:
                raise ValueError(
                    "Expert task ids do not match --task-order-config. "
                    f"Missing: {missing}; unexpected: {extra}"
                )
            ordered_paths = [id_to_path[task_id] for task_id in task_order]
        else:
            ordered_paths = [id_to_path[key] for key in sorted(id_to_path)]

        if len(ordered_paths) != len(return_scales):
            raise ValueError(
                f"Expected {len(return_scales)} expert tasks, found "
                f"{len(ordered_paths)}"
            )

        aucs = []
        for position, path in enumerate(ordered_paths):
            matrix = metric_utils.load_eval_matrix(path)
            if matrix.shape[1] != 1:
                raise ValueError(
                    f"Expected a single-task expert matrix in {path}, "
                    f"got {matrix.shape}"
                )
            aucs.append(
                metric_utils.normalized_single_task_auc(
                    matrix,
                    return_scales[position],
                    clip=normalization_mode != "none",
                )
            )
        return np.asarray(aucs, dtype=float)

    grouped = {}
    ungrouped = []
    for path in expert_runs:
        match = re.search(r"(?:^|[/\\])seed(\d+)(?:[/\\]|$)", path)
        if match:
            grouped.setdefault(match.group(1), []).append(path)
        else:
            ungrouped.append(path)

    if grouped and ungrouped:
        raise ValueError(
            "Expert root mixes seed-grouped and ungrouped runs; use a "
            "consistent expert directory layout."
        )
    if grouped:
        return {
            seed: load_expert_set(paths)
            for seed, paths in sorted(grouped.items())
        }, None
    return None, load_expert_set(expert_runs)


def compute_per_task_fwt(args):
    task_order, task_names = metric_utils.load_task_order_config(
        args.task_order_config
    )
    method_runs = metric_utils.find_runs(args.method_root)
    if not method_runs:
        raise ValueError(
            f"No eval_metrics files found under {args.method_root}"
        )

    first_matrix = metric_utils.load_eval_matrix(method_runs[0])
    num_tasks = first_matrix.shape[1]
    if task_order is not None and len(task_order) != num_tasks:
        raise ValueError(
            f"Task config contains {len(task_order)} tasks, but method runs "
            f"contain {num_tasks} task columns."
        )

    normalization_args = make_normalization_args(args)
    return_scales, normalization_mode = metric_utils.resolve_return_scales(
        normalization_args,
        task_names,
        task_order,
        num_tasks,
    )
    expert_by_seed, default_expert = load_expert_aucs(
        args.expert_root,
        task_order,
        task_names,
        return_scales,
        normalization_mode,
    )
    names = ordered_task_names(task_names, task_order, num_tasks)

    records = []
    observed_seeds = set()
    for run_dir in method_runs:
        seed = metric_utils.extract_seed_generic(run_dir)
        if seed is None:
            raise ValueError(f"Could not identify RL seed for {run_dir}")
        seed = str(seed)
        if seed in observed_seeds:
            raise ValueError(
                f"Multiple method runs found for seed {seed} under "
                f"{args.method_root}"
            )
        observed_seeds.add(seed)

        matrix = metric_utils.load_eval_matrix(run_dir)
        if matrix.shape[1] != num_tasks:
            raise ValueError(
                f"Inconsistent task count in {run_dir}: {matrix.shape[1]} "
                f"instead of {num_tasks}"
            )
        continual_auc = metric_utils.normalized_own_task_auc(
            matrix,
            return_scales,
            clip=normalization_mode != "none",
        )
        if expert_by_seed is not None:
            if seed not in expert_by_seed:
                raise ValueError(f"No expert set found for RL seed {seed}")
            expert_auc = expert_by_seed[seed]
        else:
            expert_auc = default_expert

        fwt = metric_utils.forward_transfer(
            continual_auc,
            expert_auc,
            min_den=args.min_denominator,
        )
        for task_idx in range(num_tasks):
            task_name = names[task_idx] if names else f"task{task_idx}"
            curriculum_task_id = (
                task_order[task_idx] if task_order is not None else task_idx
            )
            records.append(
                {
                    "seed": seed,
                    "task_idx": task_idx,
                    "curriculum_task_id": curriculum_task_id,
                    "task_name_config": task_name,
                    "continual_auc": continual_auc[task_idx],
                    "expert_auc": expert_auc[task_idx],
                    "fwt": fwt[task_idx],
                    "method_run_dir": run_dir,
                }
            )
    return pd.DataFrame.from_records(records), normalization_mode


def aggregate_early_similarity(args):
    events = pd.read_csv(args.selection_events)
    required = {
        "seed",
        "task_idx",
        "event_number",
        args.similarity_column,
    }
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(
            f"{args.selection_events} is missing required columns: {missing}"
        )
    if args.early_events <= 0:
        raise ValueError("--early-events must be positive")

    events = events.copy()
    events["seed"] = events["seed"].astype(str)
    events["task_idx"] = pd.to_numeric(
        events["task_idx"], errors="raise"
    ).astype(int)
    events["event_number"] = pd.to_numeric(
        events["event_number"], errors="raise"
    ).astype(int)
    events[args.similarity_column] = pd.to_numeric(
        events[args.similarity_column], errors="coerce"
    )
    events = events[
        (events["event_number"] >= 1)
        & (events["event_number"] <= args.early_events)
    ]

    metadata_columns = [
        column
        for column in ("task_name", "family", "depth")
        if column in events.columns
    ]
    grouped = events.groupby(["seed", "task_idx"], sort=True)

    def aggregate_values(values):
        values = values[np.isfinite(values)]
        if values.size < args.minimum_events:
            return np.nan
        if args.aggregate == "mean":
            return float(np.mean(values))
        if args.aggregate == "median":
            return float(np.median(values))
        if args.aggregate == "max":
            return float(np.max(values))
        return float(values[0])

    rows = []
    for (seed, task_idx), group in grouped:
        finite_values = group[args.similarity_column].to_numpy(dtype=float)
        row = {
            "seed": str(seed),
            "task_idx": int(task_idx),
            "early_similarity": aggregate_values(finite_values),
            "similarity_event_count": int(np.isfinite(finite_values).sum()),
            "first_event_number": int(group["event_number"].min()),
            "last_event_number": int(group["event_number"].max()),
        }
        for column in metadata_columns:
            non_null = group[column].dropna()
            row[column] = non_null.iloc[0] if len(non_null) else np.nan
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def spearman(x, y):
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 3:
        return np.nan, np.nan, int(finite.sum())
    rho, p_value = stats.spearmanr(
        np.asarray(x)[finite],
        np.asarray(y)[finite],
    )
    return float(rho), float(p_value), int(finite.sum())


def partial_spearman(x, y, controls):
    frame = pd.DataFrame({"x": x, "y": y, "control": controls}).dropna()
    if len(frame) < 4 or frame["control"].nunique() < 2:
        return np.nan, np.nan, len(frame)

    ranked_x = stats.rankdata(frame["x"].to_numpy(dtype=float))
    ranked_y = stats.rankdata(frame["y"].to_numpy(dtype=float))
    control_dummies = pd.get_dummies(
        frame["control"].astype(str),
        drop_first=True,
        dtype=float,
    ).to_numpy()
    design = np.column_stack([np.ones(len(frame)), control_dummies])
    residual_x = ranked_x - design @ np.linalg.lstsq(
        design, ranked_x, rcond=None
    )[0]
    residual_y = ranked_y - design @ np.linalg.lstsq(
        design, ranked_y, rcond=None
    )[0]
    if np.std(residual_x) <= 1e-12 or np.std(residual_y) <= 1e-12:
        return np.nan, np.nan, len(frame)
    rho, p_value = stats.pearsonr(residual_x, residual_y)
    return float(rho), float(p_value), len(frame)


def bootstrap_mean(values, iters, rng):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    samples = rng.choice(values, size=(iters, values.size), replace=True)
    means = samples.mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def summarize_seed_correlations(values, args, rng):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "num_seeds": 0,
            "mean_rho": np.nan,
            "median_rho": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "sign_flip_p": np.nan,
        }
    ci_low, ci_high = bootstrap_mean(
        values, args.bootstrap_iters, rng
    )
    p_value, exact = metric_utils.paired_sign_flip_test(
        values,
        alternative=args.alternative,
        exact_max_pairs=20,
        rng=rng,
        iters=args.bootstrap_iters,
    )
    return {
        "num_seeds": int(values.size),
        "mean_rho": float(np.mean(values)),
        "median_rho": float(np.median(values)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "sign_flip_p": float(p_value),
        "sign_flip_exact": bool(exact),
    }


def compute_correlations(merged, args):
    per_seed_rows = []
    has_control = (
        args.control_column
        and args.control_column in merged.columns
        and merged[args.control_column].notna().any()
    )
    for seed, group in merged.groupby("seed", sort=True):
        rho, p_value, count = spearman(
            group["early_similarity"].to_numpy(),
            group["fwt"].to_numpy(),
        )
        row = {
            "seed": seed,
            "num_tasks": count,
            "spearman_rho": rho,
            "spearman_p": p_value,
        }
        if has_control:
            adjusted_rho, adjusted_p, adjusted_count = partial_spearman(
                group["early_similarity"].to_numpy(),
                group["fwt"].to_numpy(),
                group[args.control_column].to_numpy(),
            )
            row.update(
                {
                    "adjusted_num_tasks": adjusted_count,
                    "partial_spearman_rho": adjusted_rho,
                    "partial_spearman_p": adjusted_p,
                }
            )
        per_seed_rows.append(row)
    per_seed = pd.DataFrame.from_records(per_seed_rows)

    rng = np.random.default_rng(args.rng_seed)
    summary = {
        "analysis_unit": "seed_target_task",
        "similarity_column": args.similarity_column,
        "similarity_aggregate": args.aggregate,
        "early_events": args.early_events,
        "num_seed_task_pairs": int(len(merged)),
        "num_seeds": int(merged["seed"].nunique()),
        "num_target_tasks": int(merged["task_idx"].nunique()),
        "seed_level_spearman": summarize_seed_correlations(
            per_seed["spearman_rho"].to_numpy(),
            args,
            rng,
        ),
    }

    pooled_rho, pooled_p, pooled_n = spearman(
        merged["early_similarity"].to_numpy(),
        merged["fwt"].to_numpy(),
    )
    summary["pooled_descriptive_spearman"] = {
        "rho": pooled_rho,
        "naive_p": pooled_p,
        "n": pooled_n,
    }

    task_means = (
        merged.groupby("task_idx", as_index=False)
        .agg(
            early_similarity=("early_similarity", "mean"),
            fwt=("fwt", "mean"),
        )
        .dropna()
    )
    task_rho, task_p, task_n = spearman(
        task_means["early_similarity"].to_numpy(),
        task_means["fwt"].to_numpy(),
    )
    summary["task_mean_descriptive_spearman"] = {
        "rho": task_rho,
        "naive_p": task_p,
        "n": task_n,
    }

    if has_control:
        summary["control_column"] = args.control_column
        summary["seed_level_partial_spearman"] = summarize_seed_correlations(
            per_seed["partial_spearman_rho"].to_numpy(),
            args,
            rng,
        )
    return per_seed, summary


def format_number(value, digits=3):
    if value is None or not np.isfinite(value):
        return "---"
    return f"{value:.{digits}f}"


def write_latex_table(summary, output_path):
    rows = []
    unadjusted = summary["seed_level_spearman"]
    rows.append(
        (
            "Spearman",
            unadjusted,
            summary["pooled_descriptive_spearman"]["rho"],
        )
    )
    adjusted = summary.get("seed_level_partial_spearman")
    if adjusted is not None:
        rows.append(
            (
                f"Partial Spearman ({summary['control_column']})",
                adjusted,
                None,
            )
        )

    lines = [
        r"\begin{table}[t]",
        r"    \centering",
        r"    \small",
        r"    \begin{tabular}{lcccc}",
        r"        \toprule",
        (
            r"        Analysis & Mean seed $\rho$ & 95\% BCI & "
            r"Sign-flip $p$ & Pooled $\rho$ \\"
        ),
        r"        \midrule",
    ]
    for label, result, pooled in rows:
        ci = (
            f"[{format_number(result['ci_low'])}, "
            f"{format_number(result['ci_high'])}]"
        )
        lines.append(
            "        "
            + " & ".join(
                [
                    label,
                    format_number(result["mean_rho"]),
                    ci,
                    format_number(result["sign_flip_p"]),
                    format_number(pooled),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabular}",
            (
                r"    \caption{Association between early selected cosine "
                r"similarity and per-task forward transfer. One correlation "
                r"is computed across target tasks for each RL seed. The "
                r"confidence interval resamples the seed-level coefficients, "
                r"and the p-value uses an exact sign-flip test. The pooled "
                r"coefficient is descriptive.}"
            ),
            r"    \label{tbl:similarity_fwt_correlation}",
            r"\end{table}",
        ]
    )
    with open(output_path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def plot_correlation(merged, summary, args, output_base):
    figure, axis = plt.subplots(figsize=(6.4, 4.6))
    if "depth" in merged.columns and merged["depth"].notna().any():
        depth_values = sorted(merged["depth"].dropna().unique())
        colour_map = plt.get_cmap("viridis")
        for position, depth in enumerate(depth_values):
            subset = merged[merged["depth"] == depth]
            colour = colour_map(
                position / max(1, len(depth_values) - 1)
            )
            axis.scatter(
                subset["early_similarity"],
                subset["fwt"],
                s=24,
                alpha=0.35,
                color=colour,
                label=f"Depth {depth:g}",
                edgecolors="none",
            )
    else:
        axis.scatter(
            merged["early_similarity"],
            merged["fwt"],
            s=24,
            alpha=0.35,
            color="#2878B5",
            edgecolors="none",
        )

    task_means = (
        merged.groupby("task_idx", as_index=False)
        .agg(
            early_similarity=("early_similarity", "mean"),
            fwt=("fwt", "mean"),
        )
        .dropna()
    )
    axis.scatter(
        task_means["early_similarity"],
        task_means["fwt"],
        s=42,
        facecolors="white",
        edgecolors="black",
        linewidths=0.8,
        label="Task mean",
        zorder=3,
    )
    axis.axhline(0.0, color="0.45", linewidth=0.8, linestyle="--")
    axis.set_xlabel(
        f"{args.aggregate.capitalize()} selected cosine similarity "
        f"(first {args.early_events} events)"
    )
    axis.set_ylabel("Per-task FWT")
    if args.plot_title:
        axis.set_title(args.plot_title)
    else:
        seed_result = summary["seed_level_spearman"]
        axis.set_title(
            "Early retrieval similarity and forward transfer\n"
            f"mean seed Spearman $\\rho$="
            f"{format_number(seed_result['mean_rho'])}"
        )
    axis.grid(alpha=0.2)
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(
            handles,
            labels,
            frameon=False,
            fontsize=8,
            ncol=2,
        )
    figure.tight_layout()
    figure.savefig(output_base + ".pdf", bbox_inches="tight")
    figure.savefig(output_base + ".png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    fwt, normalization_mode = compute_per_task_fwt(args)
    similarity = aggregate_early_similarity(args)
    merged = similarity.merge(
        fwt,
        on=["seed", "task_idx"],
        how="inner",
        validate="one_to_one",
    )
    if args.exclude_task_ids:
        merged = merged[
            ~merged["task_idx"].isin(set(args.exclude_task_ids))
        ]
    merged = merged[
        np.isfinite(merged["early_similarity"])
        & np.isfinite(merged["fwt"])
    ].copy()
    if merged.empty:
        raise ValueError(
            "No finite seed/task pairs remain after joining similarity and FWT."
        )

    per_seed, summary = compute_correlations(merged, args)
    summary.update(
        {
            "method_root": os.path.abspath(args.method_root),
            "expert_root": os.path.abspath(args.expert_root),
            "selection_events": os.path.abspath(args.selection_events),
            "task_order_config": os.path.abspath(args.task_order_config),
            "performance_normalization": normalization_mode,
            "num_fwt_seed_task_pairs_before_selection_join": int(len(fwt)),
            "num_selection_seed_task_pairs_before_fwt_join": int(
                len(similarity)
            ),
            "excluded_task_ids": args.exclude_task_ids or [],
            "minimum_events": args.minimum_events,
        }
    )

    fwt_path = os.path.join(
        args.out_dir, "per_task_fwt_all_valid_tasks.csv"
    )
    merged_path = os.path.join(
        args.out_dir, "similarity_fwt_seed_task.csv"
    )
    per_seed_path = os.path.join(
        args.out_dir, "similarity_fwt_per_seed_correlations.csv"
    )
    summary_path = os.path.join(
        args.out_dir, "similarity_fwt_summary.json"
    )
    latex_path = os.path.join(
        args.out_dir, "similarity_fwt_correlation.tex"
    )
    plot_base = os.path.join(
        args.out_dir, "similarity_fwt_scatter"
    )

    fwt.sort_values(["seed", "task_idx"]).to_csv(fwt_path, index=False)
    merged.sort_values(["seed", "task_idx"]).to_csv(
        merged_path, index=False
    )
    per_seed.sort_values("seed").to_csv(per_seed_path, index=False)
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    write_latex_table(summary, latex_path)
    plot_correlation(merged, summary, args, plot_base)

    seed_result = summary["seed_level_spearman"]
    print(
        "Seed-level Spearman: "
        f"mean rho={seed_result['mean_rho']:.4f}, "
        f"95% BCI=[{seed_result['ci_low']:.4f}, "
        f"{seed_result['ci_high']:.4f}], "
        f"sign-flip p={seed_result['sign_flip_p']:.4g}, "
        f"seeds={seed_result['num_seeds']}"
    )
    adjusted = summary.get("seed_level_partial_spearman")
    if adjusted is not None:
        print(
            f"Depth-adjusted partial Spearman: "
            f"mean rho={adjusted['mean_rho']:.4f}, "
            f"95% BCI=[{adjusted['ci_low']:.4f}, "
            f"{adjusted['ci_high']:.4f}], "
            f"sign-flip p={adjusted['sign_flip_p']:.4g}"
        )
    pooled = summary["pooled_descriptive_spearman"]
    task_mean = summary["task_mean_descriptive_spearman"]
    print(
        f"Descriptive pooled rho={pooled['rho']:.4f} "
        f"(n={pooled['n']}); task-mean rho={task_mean['rho']:.4f} "
        f"(n={task_mean['n']})"
    )
    print(
        f"Computed FWT for {len(fwt)} seed/task pairs; "
        f"{len(merged)} pairs had finite selection similarity and entered "
        "the correlation."
    )
    print(f"Wrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
