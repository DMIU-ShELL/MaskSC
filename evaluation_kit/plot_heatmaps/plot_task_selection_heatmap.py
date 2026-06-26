#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def resolve_experiment_path(experiment_path, log_root):
    path = Path(experiment_path).expanduser()
    if path.exists():
        return path
    path = Path(log_root).expanduser() / experiment_path
    if path.exists():
        return path
    raise FileNotFoundError(f"Could not find experiment path: {experiment_path}")


def discover_csvs(experiment_dir, filename):
    csvs = sorted(experiment_dir.rglob(filename))
    if not csvs:
        raise FileNotFoundError(f"No {filename} files found under {experiment_dir}")
    return csvs


def infer_num_tasks(csvs):
    max_task = -1
    for csv_path in csvs:
        with csv_path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                max_task = max(max_task, int(row["task_idx"]), int(row["prev_idx"]))
    if max_task < 0:
        raise ValueError("No task rows found in CSV files")
    return max_task + 1


def load_seed_matrix(csv_path, num_tasks):
    selected_sum = np.zeros((num_tasks, num_tasks), dtype=np.float64)
    opportunities = np.zeros((num_tasks, num_tasks), dtype=np.float64)

    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"task_idx", "prev_idx", "selected"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")

        for row in reader:
            target = int(row["task_idx"])
            prior = int(row["prev_idx"])
            if target >= num_tasks or prior >= num_tasks:
                continue
            selected = int(float(row["selected"]))
            selected_sum[target, prior] += selected
            opportunities[target, prior] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        frequency = selected_sum / opportunities
    frequency[opportunities == 0] = np.nan
    return selected_sum, opportunities, frequency


def aggregate(csvs, num_tasks, metric):
    seed_values = []
    selected_total = np.zeros((num_tasks, num_tasks), dtype=np.float64)
    opportunity_total = np.zeros((num_tasks, num_tasks), dtype=np.float64)

    for csv_path in csvs:
        selected_sum, opportunities, frequency = load_seed_matrix(csv_path, num_tasks)
        selected_total += selected_sum
        opportunity_total += opportunities
        seed_values.append(frequency if metric == "frequency" else selected_sum)

    if metric == "frequency":
        stacked = np.stack(seed_values, axis=0)
        valid = ~np.isnan(stacked)
        counts = valid.sum(axis=0)
        sums = np.nansum(stacked, axis=0)
        matrix = np.full((num_tasks, num_tasks), np.nan, dtype=np.float64)
        matrix[counts > 0] = sums[counts > 0] / counts[counts > 0]
    elif metric == "count":
        matrix = selected_total
        matrix[opportunity_total == 0] = np.nan
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    return matrix, selected_total, opportunity_total


def save_matrix_csv(matrix, output_csv):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["target_task"] + [f"prior_{idx}" for idx in range(matrix.shape[1])])
        for task_idx, row in enumerate(matrix):
            writer.writerow([task_idx] + ["" if np.isnan(v) else f"{v:.6f}" for v in row])


def oracle_correct(target, prior, family_size):
    return prior < target and target % family_size == prior % family_size


def build_family_matrix(selected_total, family_size):
    family_counts = np.zeros((family_size, family_size), dtype=np.float64)
    target_family_totals = np.zeros((family_size,), dtype=np.float64)

    for target in range(selected_total.shape[0]):
        target_family = target % family_size
        for prior in range(selected_total.shape[1]):
            count = selected_total[target, prior]
            if count <= 0:
                continue
            prior_family = prior % family_size
            family_counts[target_family, prior_family] += count
            target_family_totals[target_family] += count

    family_frequency = np.full_like(family_counts, np.nan)
    nonzero = target_family_totals > 0
    family_frequency[nonzero] = family_counts[nonzero] / target_family_totals[nonzero, None]
    return family_frequency, family_counts


def build_oracle_alignment(selected_total, family_size):
    num_tasks = selected_total.shape[0]
    correct = np.zeros(num_tasks, dtype=np.float64)
    total = np.zeros(num_tasks, dtype=np.float64)

    for target in range(num_tasks):
        for prior in range(num_tasks):
            count = selected_total[target, prior]
            if count <= 0:
                continue
            total[target] += count
            if oracle_correct(target, prior, family_size):
                correct[target] += count

    alignment = np.full(num_tasks, np.nan, dtype=np.float64)
    alignment[total > 0] = correct[total > 0] / total[total > 0]
    return alignment, correct, total


def save_vector_csv(values, totals, output_csv):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["target_task", "oracle_alignment", "selected_count"])
        for task_idx, value in enumerate(values):
            writer.writerow([task_idx, "" if np.isnan(value) else f"{value:.6f}", f"{totals[task_idx]:.0f}"])


def plot_family_heatmap(family_matrix, output_path, title, annotate=True):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    cmap = plt.cm.Blues
    cmap.set_bad(color=mcolors.CSS4_COLORS['whitesmoke'])
    masked = np.ma.masked_invalid(family_matrix)
    im = ax.imshow(masked, interpolation="nearest", aspect="equal", cmap=cmap, vmin=0.0, vmax=1.0)

    ticks = np.arange(family_matrix.shape[0])
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticks)
    ax.set_yticklabels(ticks)
    ax.set_xlabel("Retrieved prior family")
    ax.set_ylabel("Target family")
    ax.set_title(title)

    if annotate:
        for row in range(family_matrix.shape[0]):
            for col in range(family_matrix.shape[1]):
                val = family_matrix[row, col]
                if np.isnan(val):
                    continue
                ax.text(col, row, f"{val:.2f}", ha="center", va="center", fontsize=10, color="white" if val > 0.55 else "black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Selection frequency")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_oracle_alignment(alignment, totals, output_path, title):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    tasks = np.arange(alignment.shape[0])
    valid = ~np.isnan(alignment)

    ax.bar(tasks[valid], alignment[valid], color="#2f6fb0", width=0.8)
    ax.scatter(tasks[~valid], np.zeros((~valid).sum()), marker="x", color="black", s=24, label="No selections")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.5, alignment.shape[0] - 0.5)
    ax.set_xlabel("Target task")
    ax.set_ylabel("Oracle-aligned selections")
    ax.set_title(title)
    ax.set_xticks(tasks)
    ax.tick_params(axis="x", labelrotation=90)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)

    selected_total = totals.sum()
    valid_total = totals[valid].sum()
    overall = np.nansum(alignment[valid] * totals[valid]) / valid_total if valid_total > 0 else np.nan
    if not np.isnan(overall):
        ax.axhline(overall, color="#111111", linestyle="--", linewidth=1.2, label=f"Overall {overall:.2f}")
        ax.legend(loc="lower right", frameon=False)

    ax.text(0.01, 0.96, f"Selected events: {int(selected_total)}", transform=ax.transAxes, va="top", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_combined(family_matrix, alignment, totals, output_path, title):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={"width_ratios": [1.0, 1.75]})

    cmap = plt.cm.Blues
    cmap.set_bad(color=mcolors.CSS4_COLORS['whitesmoke'])
    masked = np.ma.masked_invalid(family_matrix)
    im = axes[0].imshow(masked, interpolation="nearest", aspect="equal", cmap=cmap, vmin=0.0, vmax=1.0)
    ticks = np.arange(family_matrix.shape[0])
    axes[0].set_xticks(ticks)
    axes[0].set_yticks(ticks)
    axes[0].set_xticklabels(ticks)
    axes[0].set_yticklabels(ticks)
    axes[0].set_xlabel("Retrieved family")
    axes[0].set_ylabel("Target family")
    axes[0].set_title("A. Family-level retrieval")
    for row in range(family_matrix.shape[0]):
        for col in range(family_matrix.shape[1]):
            val = family_matrix[row, col]
            if np.isnan(val):
                continue
            axes[0].text(col, row, f"{val:.2f}", ha="center", va="center", fontsize=9, color="white" if val > 0.55 else "black")
    cbar = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    cbar.set_label("Frequency")

    tasks = np.arange(alignment.shape[0])
    valid = ~np.isnan(alignment)
    axes[1].bar(tasks[valid], alignment[valid], color="#2f6fb0", width=0.8)
    axes[1].scatter(tasks[~valid], np.zeros((~valid).sum()), marker="x", color="black", s=24)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlim(-0.5, alignment.shape[0] - 0.5)
    axes[1].set_xlabel("Target task")
    axes[1].set_ylabel("Oracle-aligned selections")
    axes[1].set_title("B. Per-task oracle alignment")
    axes[1].set_xticks(tasks)
    axes[1].tick_params(axis="x", labelrotation=90)
    axes[1].grid(axis="y", color="#d9d9d9", linewidth=0.8)

    valid_total = totals[valid].sum()
    overall = np.nansum(alignment[valid] * totals[valid]) / valid_total if valid_total > 0 else np.nan
    if not np.isnan(overall):
        axes[1].axhline(overall, color="#111111", linestyle="--", linewidth=1.2, label=f"Overall {overall:.2f}")
        axes[1].legend(loc="lower right", frameon=False)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_heatmap(matrix, output_path, title, metric, annotate=False):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    masked = np.ma.masked_invalid(matrix)

    fig_size = max(8, matrix.shape[0] * 0.35)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    #cmap = plt.cm.viridis.copy()
    #cmap.set_bad(color="#f2f2f2")
    cmap = plt.cm.Blues
    cmap.set_bad(color=mcolors.CSS4_COLORS['whitesmoke'])

    vmax = 1.0 if metric == "frequency" else np.nanmax(matrix)
    im = ax.imshow(masked, interpolation="nearest", aspect="equal", cmap=cmap, vmin=0.0, vmax=vmax)

    ticks = np.arange(matrix.shape[0])
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticks)
    ax.set_yticklabels(ticks)
    ax.set_xlabel("Retrieved prior task")
    ax.set_ylabel("Target task being learned")
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=90)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar_label = "Selection frequency" if metric == "frequency" else "Selection count"
    cbar.set_label(cbar_label)

    if annotate:
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                val = matrix[row, col]
                if np.isnan(val):
                    continue
                label = f"{val:.2f}" if metric == "frequency" else f"{int(val)}"
                ax.text(col, row, label, ha="center", va="center", fontsize=6, color="white" if val > vmax * 0.55 else "black")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot a target-task x prior-task heatmap from task_similarities.csv files."
    )
    parser.add_argument("experiment_path", help="Experiment dir, either absolute/relative or relative to --log_root")
    parser.add_argument("--log_root", default="mask-lrl-cluster-optimization/log", help="Root log directory")
    parser.add_argument("--csv_name", default="task_similarities.csv", help="CSV filename to search for")
    parser.add_argument("--num_tasks", default=None, type=int, help="Number of tasks; inferred if omitted")
    parser.add_argument("--metric", choices=["frequency", "count"], default="frequency")
    parser.add_argument("--output", default=None, help="Output heatmap path")
    parser.add_argument("--matrix_csv", default=None, help="Optional output CSV for the plotted matrix")
    parser.add_argument("--compact_prefix", default=None, help="Optional prefix for compact family/alignment figures")
    parser.add_argument("--oracle_family_size", default=4, type=int, help="Oracle family size; CT28 uses same task_idx modulo this value")
    parser.add_argument("--skip_compact", action="store_true", help="Only write the full task x prior heatmap")
    parser.add_argument("--annotate", action="store_true", help="Write values inside heatmap cells")
    args = parser.parse_args()

    experiment_dir = resolve_experiment_path(args.experiment_path, args.log_root)
    csvs = discover_csvs(experiment_dir, args.csv_name)
    num_tasks = args.num_tasks or infer_num_tasks(csvs)

    matrix, selected_total, opportunity_total = aggregate(csvs, num_tasks, args.metric)

    default_stem = f"{experiment_dir.name}_task_selection_{args.metric}"
    output = Path(args.output) if args.output else experiment_dir / f"{default_stem}.pdf"
    matrix_csv = Path(args.matrix_csv) if args.matrix_csv else experiment_dir / f"{default_stem}.csv"

    title = f"{experiment_dir.name}: prior task selection ({len(csvs)} seeds)"
    plot_heatmap(matrix, output, title, args.metric, annotate=args.annotate)
    save_matrix_csv(matrix, matrix_csv)

    compact_paths = []
    if not args.skip_compact:
        compact_prefix = Path(args.compact_prefix) if args.compact_prefix else experiment_dir / f"{experiment_dir.name}_oracle_retrieval"
        family_matrix, family_counts = build_family_matrix(selected_total, args.oracle_family_size)
        alignment, correct, totals = build_oracle_alignment(selected_total, args.oracle_family_size)

        family_path = compact_prefix.with_name(f"{compact_prefix.name}_family_heatmap.pdf")
        alignment_path = compact_prefix.with_name(f"{compact_prefix.name}_alignment_by_task.pdf")
        combined_path = compact_prefix.with_name(f"{compact_prefix.name}_combined.pdf")
        alignment_csv = compact_prefix.with_name(f"{compact_prefix.name}_alignment_by_task.csv")
        family_csv = compact_prefix.with_name(f"{compact_prefix.name}_family_heatmap.csv")

        plot_family_heatmap(family_matrix, family_path, "Family-level prior retrieval")
        plot_oracle_alignment(alignment, totals, alignment_path, "Oracle alignment by target task")
        plot_combined(family_matrix, alignment, totals, combined_path, title)
        save_matrix_csv(family_matrix, family_csv)
        save_vector_csv(alignment, totals, alignment_csv)
        compact_paths = [family_path, alignment_path, combined_path, family_csv, alignment_csv]

    print(f"Found {len(csvs)} CSV files under {experiment_dir}")
    print(f"Saved heatmap: {output}")
    print(f"Saved matrix CSV: {matrix_csv}")
    for path in compact_paths:
        print(f"Saved compact analysis: {path}")


if __name__ == "__main__":
    main()
