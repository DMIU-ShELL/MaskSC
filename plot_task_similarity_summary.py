#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot task similarity summaries from task_similarities.csv."
    )
    parser.add_argument(
        "csv",
        type=Path,
        help="Path to task_similarities.csv.",
    )
    parser.add_argument(
        "--env-config",
        type=Path,
        default=None,
        help="Optional env_config.json. Defaults to a sibling env_config.json if present.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <csv parent>/similarity_plots.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--smooth-window", type=int, default=8)
    parser.add_argument(
        "--family-stride",
        type=int,
        default=2,
        help=(
            "Fallback family grouping when env_config labels are unavailable. "
            "With interleaved two-family tasks, task parity defines family."
        ),
    )
    return parser.parse_args()


def resolve_paths(args):
    csv_path = args.csv.expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    if args.env_config is None:
        env_config = csv_path.parent / "env_config.json"
        if not env_config.exists():
            env_config = None
    else:
        env_config = args.env_config.expanduser().resolve()
        if not env_config.exists():
            raise FileNotFoundError(f"env_config not found: {env_config}")

    out_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir is not None
        else csv_path.parent / "similarity_plots"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    return csv_path, env_config, out_dir


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    required = {
        "task_idx",
        "prev_idx",
        "iteration",
        "total_steps",
        "similarity",
        "selected",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")

    df = df.copy()
    df["similarity"] = pd.to_numeric(df["similarity"], errors="coerce")
    df.loc[~np.isfinite(df["similarity"]), "similarity"] = np.nan
    df["selected"] = pd.to_numeric(df["selected"], errors="coerce").fillna(0)
    return df


def short_label(task, idx):
    if "UpRight" in task:
        family = "UR"
    elif "DownLeft" in task:
        family = "DL"
    else:
        family = f"T{idx}"

    if "-R" in task:
        room = task.split("-R")[-1].split("-")[0]
        return f"{family}-R{room}"
    return family


def infer_labels_and_families(df, env_config, family_stride):
    max_task = int(max(df["task_idx"].max(), df["prev_idx"].max()))
    num_tasks = max_task + 1

    if env_config is not None:
        with env_config.open() as fh:
            tasks = json.load(fh).get("tasks", [])
        if len(tasks) >= num_tasks:
            labels = [short_label(task, idx) for idx, task in enumerate(tasks[:num_tasks])]
            families = [
                "UpRight"
                if "UpRight" in task
                else "DownLeft"
                if "DownLeft" in task
                else f"family_{idx % family_stride}"
                for idx, task in enumerate(tasks[:num_tasks])
            ]
            return labels, families

    labels = [f"T{idx}" for idx in range(num_tasks)]
    families = [f"family_{idx % family_stride}" for idx in range(num_tasks)]
    return labels, families


def build_mean_similarity(df, num_tasks):
    return (
        df.pivot_table(
            index="task_idx",
            columns="prev_idx",
            values="similarity",
            aggfunc="mean",
        )
        .reindex(index=range(num_tasks), columns=range(num_tasks))
    )


def build_latest_similarity(df, num_tasks):
    latest = (
        df.dropna(subset=["similarity"])
        .sort_values(["task_idx", "prev_idx", "iteration", "total_steps"])
        .groupby(["task_idx", "prev_idx"], as_index=False)
        .tail(1)
    )
    return (
        latest.pivot(index="task_idx", columns="prev_idx", values="similarity")
        .reindex(index=range(num_tasks), columns=range(num_tasks))
    )


def build_selection_rate(df, num_tasks):
    finite = df.dropna(subset=["similarity"])
    return (
        finite.pivot_table(
            index="task_idx",
            columns="prev_idx",
            values="selected",
            aggfunc="mean",
        )
        .reindex(index=range(num_tasks), columns=range(num_tasks))
    )


def quantile_limits(values, lower=0.02, upper=0.98):
    values = pd.Series(values).dropna()
    if values.empty:
        return 0.0, 1.0
    return (
        max(-1.0, float(values.quantile(lower))),
        min(1.0, float(values.quantile(upper))),
    )


def draw_heatmap(
    matrix,
    labels,
    title,
    output_path,
    dpi,
    cmap="Greens",
    vmin=None,
    vmax=None,
    fmt=".2f",
):
    num_tasks = len(labels)
    data = matrix.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(data)

    fig_width = max(9.0, 0.8 * num_tasks + 3.0)
    fig_height = max(7.0, 0.65 * num_tasks + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)

    colormap = plt.get_cmap(cmap).copy()
    colormap.set_bad(color="#f2f2f2")
    im = ax.imshow(masked, cmap=colormap, vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_title(title, fontsize=14, pad=14)
    ax.set_xlabel("Prior policy / previous task")
    ax.set_ylabel("Current learning task")
    ax.set_xticks(range(num_tasks), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(num_tasks), labels=labels)

    if vmin is None:
        vmin = np.nanmin(data) if np.isfinite(data).any() else 0.0
    if vmax is None:
        vmax = np.nanmax(data) if np.isfinite(data).any() else 1.0
    midpoint = vmin + 0.65 * (vmax - vmin)

    for row in range(num_tasks):
        for col in range(num_tasks):
            value = data[row, col]
            if np.isfinite(value):
                color = "black" if value > midpoint else "white"
                ax.text(
                    col,
                    row,
                    format(value, fmt),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=color,
                )

    for edge in range(1, num_tasks, 2):
        ax.axhline(edge + 0.5, color="white", linewidth=1.3)
        ax.axvline(edge + 0.5, color="white", linewidth=1.3)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.ax.set_ylabel(title, rotation=270, labelpad=18)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def add_family_columns(df, families):
    finite = df.dropna(subset=["similarity"]).copy()
    finite["current_family"] = finite["task_idx"].map(lambda idx: families[int(idx)])
    finite["prior_family"] = finite["prev_idx"].map(lambda idx: families[int(idx)])
    finite["relation"] = np.where(
        finite["current_family"] == finite["prior_family"],
        "same family",
        "opposite family",
    )
    return finite


def plot_same_vs_opposite(finite, labels, output_path, dpi):
    summary = (
        finite.groupby(["task_idx", "relation"])["similarity"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5.8), constrained_layout=True)
    specs = [
        ("same family", "#2b8cbe", "o"),
        ("opposite family", "#e34a33", "s"),
    ]
    for relation, color, marker in specs:
        sub = summary[summary["relation"] == relation]
        ax.plot(
            sub["task_idx"],
            sub["similarity"],
            marker=marker,
            linewidth=2.0,
            label=relation,
            color=color,
        )

    ax.set_title("Mean similarity to same-family vs opposite-family priors", fontsize=14)
    ax.set_xlabel("Current learning task")
    ax.set_ylabel("Mean cosine similarity")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def build_top_prior_table(latest_matrix, labels, families):
    records = []
    for task_idx in range(len(labels)):
        row = latest_matrix.loc[task_idx].dropna()
        if row.empty:
            continue
        top_prior = int(row.idxmax())
        records.append(
            {
                "task_idx": task_idx,
                "task": labels[task_idx],
                "top_prior_idx": top_prior,
                "top_prior": labels[top_prior],
                "similarity": float(row.loc[top_prior]),
                "same_family": families[task_idx] == families[top_prior],
            }
        )
    return pd.DataFrame(records)


def plot_top_prior(top_df, output_path, dpi):
    if top_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5.8), constrained_layout=True)
    colors = ["#2b8cbe" if same else "#e34a33" for same in top_df["same_family"]]
    bars = ax.bar(top_df["task"], top_df["similarity"], color=colors)

    ax.set_title("Most similar prior at latest snapshot", fontsize=14)
    ax.set_xlabel("Current learning task")
    ax.set_ylabel("Cosine similarity")
    lower = max(0.0, min(0.5, float(top_df["similarity"].min()) - 0.05))
    upper = min(1.0, float(top_df["similarity"].max()) + 0.05)
    ax.set_ylim(lower, upper)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.25)

    for bar, prior in zip(bars, top_df["top_prior"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            prior,
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )

    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor="#2b8cbe", label="same family"),
            Patch(facecolor="#e34a33", label="opposite family"),
        ],
        frameon=False,
    )
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_similarity_timeseries(finite, labels, output_path, dpi, smooth_window):
    relation_ts = (
        finite.groupby(["task_idx", "iteration", "relation"])["similarity"]
        .mean()
        .reset_index()
    )
    task_indices = sorted(relation_ts["task_idx"].unique())
    if not task_indices:
        return

    cols = 3
    rows = int(np.ceil(len(task_indices) / cols))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(14, max(4, 3.2 * rows)),
        sharey=True,
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(-1)

    specs = [("same family", "#2b8cbe"), ("opposite family", "#e34a33")]
    for ax, task_idx in zip(axes, task_indices):
        sub_task = relation_ts[relation_ts["task_idx"] == task_idx]
        for relation, color in specs:
            sub = sub_task[sub_task["relation"] == relation].sort_values("iteration")
            if sub.empty:
                continue
            y = sub["similarity"].rolling(smooth_window, min_periods=1).mean()
            ax.plot(sub["iteration"], y, color=color, linewidth=1.8, label=relation)
        ax.set_title(labels[int(task_idx)], fontsize=11)
        ax.grid(True, axis="y", alpha=0.2)

    for ax in axes[len(task_indices):]:
        ax.axis("off")

    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Similarity during learning: same vs opposite family", fontsize=15)
    fig.supxlabel("Iteration")
    fig.supylabel("Mean cosine similarity")
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main():
    args = parse_args()
    csv_path, env_config, out_dir = resolve_paths(args)
    df = load_data(csv_path)
    labels, families = infer_labels_and_families(df, env_config, args.family_stride)
    num_tasks = len(labels)

    mean_matrix = build_mean_similarity(df, num_tasks)
    latest_matrix = build_latest_similarity(df, num_tasks)
    selection_matrix = build_selection_rate(df, num_tasks)

    sim_vmin, sim_vmax = quantile_limits(df["similarity"])
    draw_heatmap(
        mean_matrix,
        labels,
        "Mean cosine similarity over learning",
        out_dir / "mean_similarity_heatmap.png",
        args.dpi,
        cmap="magma",
        vmin=sim_vmin,
        vmax=sim_vmax,
    )
    draw_heatmap(
        latest_matrix,
        labels,
        "Latest cosine similarity snapshot",
        out_dir / "latest_similarity_heatmap.png",
        args.dpi,
        cmap="magma",
        vmin=sim_vmin,
        vmax=sim_vmax,
    )
    draw_heatmap(
        selection_matrix,
        labels,
        "Policy selection rate",
        out_dir / "selection_rate_heatmap.png",
        args.dpi,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
    )

    finite = add_family_columns(df, families)
    plot_same_vs_opposite(
        finite,
        labels,
        out_dir / "same_vs_opposite_family_similarity.png",
        args.dpi,
    )

    top_df = build_top_prior_table(latest_matrix, labels, families)
    top_df.to_csv(out_dir / "latest_top_prior_by_task.csv", index=False)
    plot_top_prior(top_df, out_dir / "latest_top_prior_by_task.png", args.dpi)

    plot_similarity_timeseries(
        finite,
        labels,
        out_dir / "similarity_timeseries_same_vs_opposite.png",
        args.dpi,
        args.smooth_window,
    )

    print(f"Wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
