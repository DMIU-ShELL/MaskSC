#!/usr/bin/env python3
import argparse
import os
import pickle
import re

import numpy as np

import matplotlib
matplotlib.use("Pdf")
import matplotlib.pyplot as plt


def grouped_order(n_tasks: int, group_stride: int = 4):
    return [i for offset in range(group_stride) for i in range(offset, n_tasks, group_stride)]


def softmax_triangular(data: np.ndarray) -> np.ndarray:
    out = data.copy()
    n = out.shape[0]
    for i in range(n):
        row = out[i, : i + 1]
        maxv = np.max(row)
        exps = np.exp(row - maxv)
        out[i, : i + 1] = exps / np.sum(exps)
    return out


def plot_heatmap(
    data: np.ndarray,
    labels,
    title: str,
    out_path: str,
    dpi: int = 300,
    vmin: float = 0.0,
    vmax: float = 0.5,
    gridlines: bool = False,
    grid_color: str = "white",
    grid_width: float = 0.5,
):
    n = data.shape[0]
    fig = plt.figure(figsize=(9, 9))
    ax = fig.subplots()
    ax.imshow(data, cmap="viridis_r", vmin=vmin, vmax=vmax)  #YlGn
    if gridlines:
        internal = np.arange(0.5, n - 0.5, 1.0)
        ax.vlines(internal, -0.5, n - 0.5, colors=grid_color, linewidth=grid_width)
        ax.hlines(internal, -0.5, n - 0.5, colors=grid_color, linewidth=grid_width)
    ax.set_xticks(np.arange(n), labels=labels, fontsize=16)
    ax.set_yticks(np.arange(n), labels=labels, fontsize=16)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    midpoint = (vmin + vmax) / 2.0
    for i in range(n):
        for j in range(n):
            text_color = "white" if data[i, j] > midpoint else "black"
            ax.text(
                j,
                i,
                f"{data[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=5,
                color=text_color,
            )
    ax.set_title(title, fontsize=20)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def extract_seed(path: str) -> str:
    match = re.search(r"eval-run-(\d+)", path)
    return match.group(1) if match else "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot clustered beta heatmaps from .bin files")
    parser.add_argument(
        "--eval-root",
        default="./log/ct28-interleaved-MaskLC/eval",
        help="Root directory containing eval outputs",
    )
    parser.add_argument(
        "--out-root",
        default="./log/beta_heatmaps_clustered",
        help="Output directory for clustered heatmaps",
    )
    parser.add_argument("--group-stride", type=int, default=4, help="Stride for interleaved task groups")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--vmin", type=float, default=0.0)
    parser.add_argument("--vmax", type=float, default=0.5)
    parser.add_argument(
        "--gridlines",
        action="store_true",
        help="Draw white gridlines between cells (no outer border).",
    )
    parser.add_argument("--grid-color", type=str, default="white", help="Gridline color.")
    parser.add_argument("--grid-width", type=float, default=0.5, help="Gridline width.")
    parser.add_argument(
        "--include-softmax",
        action="store_true",
        help="Also generate clustered softmax heatmaps",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.eval_root):
        raise SystemExit(f"Eval root not found: {args.eval_root}")

    bin_paths = []
    for root, _, files in os.walk(args.eval_root):
        for fname in files:
            if fname.startswith("betas_before_softmax_") and fname.endswith(".bin"):
                bin_paths.append(os.path.join(root, fname))

    if not bin_paths:
        raise SystemExit("No betas_before_softmax_*.bin found under eval root")

    for bin_path in sorted(bin_paths):
        seed = extract_seed(bin_path)
        out_dir = os.path.join(args.out_root, f"seed{seed}")
        os.makedirs(out_dir, exist_ok=True)

        with open(bin_path, "rb") as f:
            data = pickle.load(f)
        data = np.asarray(data)

        if data.ndim != 2 or data.shape[0] != data.shape[1]:
            print(f"Skipping non-square data: {bin_path} shape={data.shape}")
            continue

        n = data.shape[0]
        order = grouped_order(n, group_stride=args.group_stride)
        labels = [f"T{i}" for i in order]
        data_reordered = data[np.ix_(order, order)]

        base = os.path.splitext(os.path.basename(bin_path))[0]  # remove .bin
        title = base.replace("betas_before_softmax_", "")
        out_path = os.path.join(out_dir, f"{base}_clustered.pdf")
        plot_heatmap(
            data_reordered,
            labels,
            title,
            out_path,
            dpi=args.dpi,
            vmin=args.vmin,
            vmax=args.vmax,
            gridlines=args.gridlines,
            grid_color=args.grid_color,
            grid_width=args.grid_width,
        )

        if args.include_softmax:
            soft = softmax_triangular(data)
            soft_reordered = soft[np.ix_(order, order)]
            soft_base = base.replace("betas_before_softmax_", "betas_")
            soft_out = os.path.join(out_dir, f"{soft_base}_clustered.pdf")
            plot_heatmap(
                soft_reordered,
                labels,
                title,
                soft_out,
                dpi=args.dpi,
                vmin=args.vmin,
                vmax=args.vmax,
                gridlines=args.gridlines,
                grid_color=args.grid_color,
                grid_width=args.grid_width,
            )

    print(f"Processed {len(bin_paths)} files. Output under {args.out_root}")


if __name__ == "__main__":
    main()
