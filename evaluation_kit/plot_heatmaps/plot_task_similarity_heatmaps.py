#!/usr/bin/env python3
import argparse
import os
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Pdf")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def find_similarity_csvs(root: str) -> List[str]:
    csvs = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname == "task_similarities.csv":
                csvs.append(os.path.join(dirpath, fname))
    return sorted(csvs)


def extract_seed(path: str) -> str:
    patterns = [r"supermask-(\d+)", r"eval-run-(\d+)"]
    for pat in patterns:
        match = re.search(pat, path)
        if match:
            return match.group(1)
    return "unknown"


def grouped_order(n_tasks: int, group_stride: int) -> List[int]:
    return [i for offset in range(group_stride) for i in range(offset, n_tasks, group_stride)]


def build_similarity_matrix(df: pd.DataFrame, n_tasks: int, agg: str) -> np.ndarray:
    df = df.copy()
    df["similarity"] = pd.to_numeric(df["similarity"], errors="coerce")
    df.loc[~np.isfinite(df["similarity"]), "similarity"] = np.nan

    if agg == "last":
        df = df.sort_values("iteration")
        agg_df = df.groupby(["task_idx", "prev_idx"], as_index=False).last()
        vals = agg_df[["task_idx", "prev_idx", "similarity"]]
    elif agg == "max":
        vals = df.groupby(["task_idx", "prev_idx"], as_index=False)["similarity"].max()
    elif agg == "median":
        vals = df.groupby(["task_idx", "prev_idx"], as_index=False)["similarity"].median()
    else:
        vals = df.groupby(["task_idx", "prev_idx"], as_index=False)["similarity"].mean()

    mat = np.full((n_tasks, n_tasks), np.nan, dtype=float)
    for _, row in vals.iterrows():
        i = int(row["task_idx"])
        j = int(row["prev_idx"])
        mat[i, j] = float(row["similarity"])
    return mat


def build_selection_matrix(df: pd.DataFrame, n_tasks: int) -> np.ndarray:
    df = df.copy()
    df["selected"] = pd.to_numeric(df["selected"], errors="coerce").fillna(0)
    vals = df.groupby(["task_idx", "prev_idx"], as_index=False)["selected"].sum()

    mat = np.full((n_tasks, n_tasks), np.nan, dtype=float)
    for _, row in vals.iterrows():
        i = int(row["task_idx"])
        j = int(row["prev_idx"])
        mat[i, j] = float(row["selected"])
    return mat


def plot_heatmap(
    data: np.ndarray,
    labels: List[str],
    title: str,
    out_path: str,
    dpi: int,
    cmap_name: str,
    vmin: Optional[float],
    vmax: Optional[float],
    fmt: str,
    fontsize: int,
    gridlines: bool,
    grid_color: str,
    grid_width: float,
):
    n = data.shape[0]
    masked = np.ma.masked_invalid(data)

    fig = plt.figure(figsize=(9, 9))
    ax = fig.subplots()

    cmap = plt.get_cmap(cmap_name).copy()
    #cmap.set_bad(color=(216 / 255, 220 / 255, 213 / 255))
    cmap.set_bad(color=mcolors.CSS4_COLORS['whitesmoke'])

    if vmin is None:
        vmin = np.nanmin(data)
    if vmax is None:
        vmax = np.nanmax(data)

    ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax)
    if gridlines:
        # Draw only internal gridlines (skip outer border).
        internal = np.arange(0.5, n - 0.5, 1.0)
        ax.vlines(internal, -0.5, n - 0.5, colors=grid_color, linewidth=grid_width)
        ax.hlines(internal, -0.5, n - 0.5, colors=grid_color, linewidth=grid_width)
    ax.set_xticks(np.arange(n), labels=labels, fontsize=16)
    ax.set_yticks(np.arange(n), labels=labels, fontsize=16)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    midpoint = (vmin + vmax) / 2.0
    for i in range(n):
        for j in range(n):
            if np.isfinite(data[i, j]):
                text_color = "white" if data[i, j] > midpoint else "black"
                ax.text(
                    j,
                    i,
                    format(data[i, j], fmt),
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    color=text_color,
                )

    ax.set_title(title, fontsize=20)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def save_matrix(matrix: np.ndarray, labels: List[str], out_dir: str, stem: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{stem}.npy"), matrix)
    df = pd.DataFrame(matrix, index=labels, columns=labels)
    df.to_csv(os.path.join(out_dir, f"{stem}.csv"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot heatmaps from task_similarities.csv files")
    parser.add_argument(
        "--root",
        default="./log/ct28-interleaved-MaskSC-top-dense",
        help="Top-level folder containing seed runs",
    )
    parser.add_argument(
        "--sim-out",
        default="./log/sim_heatmaps",
        help="Output folder for similarity heatmaps",
    )
    parser.add_argument(
        "--sel-out",
        default="./log/selection_count_heatmaps",
        help="Output folder for selection count heatmaps",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--sim-agg",
        choices=["mean", "last", "max", "median"],
        default="mean",
        help="Aggregation for similarity over iterations",
    )
    parser.add_argument("--sim-vmin", type=float, default=-1.0)
    parser.add_argument("--sim-vmax", type=float, default=1.0)
    parser.add_argument("--sel-vmin", type=float, default=0.0)
    parser.add_argument("--sel-vmax", type=float, default=None)
    parser.add_argument(
        "--reorder-groups",
        action="store_true",
        help="Reorder tasks into interleaved groups (like beta heatmaps).",
    )
    parser.add_argument(
        "--group-stride",
        type=int,
        default=4,
        help="Stride for grouped task reordering (default: 4).",
    )
    parser.add_argument(
        "--gridlines",
        action="store_true",
        help="Draw white gridlines between cells (no outer border).",
    )
    parser.add_argument(
        "--grid-color",
        type=str,
        default="white",
        help="Gridline color.",
    )
    parser.add_argument(
        "--grid-width",
        type=float,
        default=0.5,
        help="Gridline width.",
    )
    args = parser.parse_args()

    csv_paths = find_similarity_csvs(args.root)
    if not csv_paths:
        raise SystemExit(f"No task_similarities.csv found under {args.root}")

    per_seed_sim: Dict[str, np.ndarray] = {}
    per_seed_sel: Dict[str, np.ndarray] = {}

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        max_task = int(max(df["task_idx"].max(), df["prev_idx"].max()))
        n_tasks = max_task + 1
        labels = [f"T{i}" for i in range(n_tasks)]

        seed = extract_seed(csv_path)
        seed_dir_sim = os.path.join(args.sim_out, f"seed{seed}")
        seed_dir_sel = os.path.join(args.sel_out, f"seed{seed}")
        os.makedirs(seed_dir_sim, exist_ok=True)
        os.makedirs(seed_dir_sel, exist_ok=True)

        sim_mat = build_similarity_matrix(df, n_tasks, args.sim_agg)
        sel_mat = build_selection_matrix(df, n_tasks)

        per_seed_sim[seed] = sim_mat
        per_seed_sel[seed] = sel_mat

        sim_plot = sim_mat
        sel_plot = sel_mat
        if args.reorder_groups:
            order = grouped_order(n_tasks, args.group_stride)
            labels = [f"T{i}" for i in order]
            sim_plot = sim_mat[np.ix_(order, order)]
            sel_plot = sel_mat[np.ix_(order, order)]

        sim_pdf = os.path.join(seed_dir_sim, "similarity_heatmap.pdf")
        sel_pdf = os.path.join(seed_dir_sel, "selection_counts_heatmap.pdf")

        plot_heatmap(
            sim_plot,
            labels,
            f"Cosine Similarity (seed {seed})",
            sim_pdf,
            args.dpi,
            "Blues",
            args.sim_vmin,
            args.sim_vmax,
            ".2f",
            5,
            args.gridlines,
            args.grid_color,
            args.grid_width,
        )
        save_matrix(sim_plot, labels, seed_dir_sim, "similarity_heatmap")

        plot_heatmap(
            sel_plot,
            labels,
            f"Selection Counts (seed {seed})",
            sel_pdf,
            args.dpi,
            "Oranges",
            args.sel_vmin,
            args.sel_vmax,
            ".0f",
            5,
            args.gridlines,
            args.grid_color,
            args.grid_width,
        )
        save_matrix(sel_plot, labels, seed_dir_sel, "selection_counts_heatmap")

    # Average across seeds
    all_seeds = sorted(per_seed_sim.keys())
    if all_seeds:
        sim_stack = np.stack([per_seed_sim[s] for s in all_seeds], axis=0)
        sel_stack = np.stack([per_seed_sel[s] for s in all_seeds], axis=0)

        sim_avg = np.nanmean(sim_stack, axis=0)
        sel_avg = np.nanmean(sel_stack, axis=0)

        n_tasks = sim_avg.shape[0]
        labels = [f"T{i}" for i in range(n_tasks)]

        sim_plot = sim_avg
        sel_plot = sel_avg
        if args.reorder_groups:
            order = grouped_order(n_tasks, args.group_stride)
            labels = [f"T{i}" for i in order]
            sim_plot = sim_avg[np.ix_(order, order)]
            sel_plot = sel_avg[np.ix_(order, order)]

        avg_sim_dir = os.path.join(args.sim_out, "average")
        avg_sel_dir = os.path.join(args.sel_out, "average")
        os.makedirs(avg_sim_dir, exist_ok=True)
        os.makedirs(avg_sel_dir, exist_ok=True)

        sim_pdf = os.path.join(avg_sim_dir, "similarity_heatmap_avg.pdf")
        sel_pdf = os.path.join(avg_sel_dir, "selection_counts_heatmap_avg.pdf")

        plot_heatmap(
            sim_plot,
            labels,
            "Cosine Similarity (average)",
            sim_pdf,
            args.dpi,
            "Blues",
            args.sim_vmin,
            args.sim_vmax,
            ".2f",
            5,
            args.gridlines,
            args.grid_color,
            args.grid_width,
        )
        save_matrix(sim_plot, labels, avg_sim_dir, "similarity_heatmap_avg")

        plot_heatmap(
            sel_plot,
            labels,
            "Selection Counts (average)",
            sel_pdf,
            args.dpi,
            "Oranges",
            args.sel_vmin,
            args.sel_vmax,
            ".1f",
            5,
            args.gridlines,
            args.grid_color,
            args.grid_width,
        )
        save_matrix(sel_plot, labels, avg_sel_dir, "selection_counts_heatmap_avg")

    print(f"Processed {len(csv_paths)} CSV files")


if __name__ == "__main__":
    main()
