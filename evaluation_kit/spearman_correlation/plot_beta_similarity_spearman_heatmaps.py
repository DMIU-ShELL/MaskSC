#!/usr/bin/env python3
import argparse
import os
import pickle
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Pdf")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def grouped_order(n_tasks: int, group_stride: int = 4) -> List[int]:
    return [i for offset in range(group_stride) for i in range(offset, n_tasks, group_stride)]


def extract_seed(path: str) -> str:
    patterns = [r"supermask-(\d+)", r"eval-run-(\d+)", r"seed(\d+)"]
    for pat in patterns:
        match = re.search(pat, path)
        if match:
            return match.group(1)
    return "unknown"


def find_similarity_matrices(sim_root: str) -> Dict[str, str]:
    # Prefer .npy outputs from the similarity heatmap script.
    matches: Dict[str, List[str]] = {}
    for dirpath, _, filenames in os.walk(sim_root):
        for fname in filenames:
            if fname == "similarity_heatmap.npy" or fname == "similarity_heatmap.csv":
                path = os.path.join(dirpath, fname)
                if os.path.sep + "average" + os.path.sep in path:
                    continue
                seed = extract_seed(path)
                if seed == "unknown":
                    continue
                matches.setdefault(seed, []).append(path)
    out: Dict[str, str] = {}
    for seed, paths in matches.items():
        out[seed] = sorted(paths)[-1]
    return out


def find_beta_matrices(beta_root: str) -> Dict[Tuple[str, str], str]:
    # Map (seed, layer_name) -> path
    matches: Dict[Tuple[str, str], List[str]] = {}
    for dirpath, _, filenames in os.walk(beta_root):
        for fname in filenames:
            if fname.startswith("betas_before_softmax_") and fname.endswith(".bin"):
                path = os.path.join(dirpath, fname)
                seed = extract_seed(path)
                layer = fname.replace("betas_before_softmax_", "").replace(".betas.bin", "")
                matches.setdefault((seed, layer), []).append(path)
    out: Dict[Tuple[str, str], str] = {}
    for key, paths in matches.items():
        out[key] = sorted(paths)[-1]
    return out


def load_similarity_matrix(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path)
    df = pd.read_csv(path, index_col=0)
    return df.values


def load_beta_matrix(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return np.asarray(data)


def softmax_triangular(data: np.ndarray) -> np.ndarray:
    out = data.copy()
    n = out.shape[0]
    for i in range(n):
        row = out[i, : i + 1]
        maxv = np.max(row)
        exps = np.exp(row - maxv)
        out[i, : i + 1] = exps / np.sum(exps)
    return out


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return np.nan
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return np.nan
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def row_correlations(beta_stack: np.ndarray, sim_stack: np.ndarray) -> np.ndarray:
    # Correlate betas vs similarity per task (row), within each seed, then average across seeds.
    seeds, n, _ = beta_stack.shape
    out = np.full((n,), np.nan, dtype=float)
    for i in range(n):
        per_seed = []
        for s in range(seeds):
            b = beta_stack[s, i, :]
            sim = sim_stack[s, i, :]
            mask = np.isfinite(b) & np.isfinite(sim)
            if np.sum(mask) >= 2:
                per_seed.append(spearman_corr(b[mask], sim[mask]))
        if per_seed:
            out[i] = float(np.nanmean(per_seed))
    return out


def col_correlations(beta_stack: np.ndarray, sim_stack: np.ndarray) -> np.ndarray:
    # Correlate betas vs similarity per prior (column), within each seed, then average across seeds.
    seeds, n, _ = beta_stack.shape
    out = np.full((n,), np.nan, dtype=float)
    for j in range(n):
        per_seed = []
        for s in range(seeds):
            b = beta_stack[s, :, j]
            sim = sim_stack[s, :, j]
            mask = np.isfinite(b) & np.isfinite(sim)
            if np.sum(mask) >= 2:
                per_seed.append(spearman_corr(b[mask], sim[mask]))
        if per_seed:
            out[j] = float(np.nanmean(per_seed))
    return out


def plot_heatmap(
    data: np.ndarray,
    x_labels: List[str],
    y_labels: List[str],
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
    n_rows, n_cols = data.shape
    masked = np.ma.masked_invalid(data)

    fig = plt.figure(figsize=(9, 9))
    ax = fig.subplots()

    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color=mcolors.CSS4_COLORS['whitesmoke'])

    if vmin is None:
        vmin = np.nanmin(data)
    if vmax is None:
        vmax = np.nanmax(data)

    ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax)
    if gridlines:
        internal_x = np.arange(0.5, n_cols - 0.5, 1.0)
        internal_y = np.arange(0.5, n_rows - 0.5, 1.0)
        if internal_x.size:
            ax.vlines(internal_x, -0.5, n_rows - 0.5, colors=grid_color, linewidth=grid_width)
        if internal_y.size:
            ax.hlines(internal_y, -0.5, n_cols - 0.5, colors=grid_color, linewidth=grid_width)

    ax.set_xticks(np.arange(n_cols), labels=x_labels, fontsize=16)
    ax.set_yticks(np.arange(n_rows), labels=y_labels, fontsize=16)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    midpoint = (vmin + vmax) / 2.0
    for i in range(n_rows):
        for j in range(n_cols):
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


def save_matrix(matrix: np.ndarray, row_labels: List[str], col_labels: List[str], out_dir: str, stem: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{stem}.npy"), matrix)
    df = pd.DataFrame(matrix, index=row_labels, columns=col_labels)
    df.to_csv(os.path.join(out_dir, f"{stem}.csv"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Spearman correlation heatmaps between betas and similarities")
    parser.add_argument(
        "--beta-root",
        default="./log/ct28-interleaved-MaskLC/eval",
        help="Root directory containing beta .bin files",
    )
    parser.add_argument(
        "--sim-root",
        default="./log/sim_heatmaps",
        help="Root directory containing similarity_heatmap.npy/csv per seed",
    )
    parser.add_argument(
        "--out-root",
        default="./log/beta_similarity_spearman",
        help="Output directory for Spearman heatmaps",
    )
    parser.add_argument(
        "--beta-kind",
        choices=["before_softmax", "softmax"],
        default="softmax",
        help="Whether to use betas before softmax or apply softmax per row",
    )
    parser.add_argument(
        "--mode",
        choices=["cell", "row", "col"],
        default="cell",
        help="Correlation mode: per-cell across seeds, or per-row/per-column within seed then averaged.",
    )
    parser.add_argument("--reorder-groups", action="store_true", help="Reorder tasks into interleaved groups")
    parser.add_argument(
        "--sim-preordered",
        action="store_true",
        help="Similarity matrices are already reordered; skip similarity reordering",
    )
    parser.add_argument("--group-stride", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--vmin", type=float, default=-1.0)
    parser.add_argument("--vmax", type=float, default=1.0)
    parser.add_argument("--cmap", type=str, default="coolwarm")
    parser.add_argument("--gridlines", action="store_true", help="Draw internal gridlines")
    parser.add_argument("--grid-color", type=str, default="white")
    parser.add_argument("--grid-width", type=float, default=0.5)
    args = parser.parse_args()

    sim_paths = find_similarity_matrices(args.sim_root)
    if not sim_paths:
        raise SystemExit(f"No similarity_heatmap.npy/csv found under {args.sim_root}")

    beta_paths = find_beta_matrices(args.beta_root)
    if not beta_paths:
        raise SystemExit(f"No betas_before_softmax_*.bin found under {args.beta_root}")

    # Build a mapping: layer -> seed -> beta matrix
    beta_by_layer: Dict[str, Dict[str, np.ndarray]] = {}
    for (seed, layer), path in beta_paths.items():
        beta_by_layer.setdefault(layer, {})[seed] = load_beta_matrix(path)

    for layer, beta_by_seed in sorted(beta_by_layer.items()):
        # Find seeds available in both
        seeds = sorted(set(beta_by_seed.keys()) & set(sim_paths.keys()))
        if len(seeds) < 2:
            print(f"Skipping {layer}: need at least 2 seeds, have {len(seeds)}")
            continue

        beta_stack = []
        sim_stack = []
        for seed in seeds:
            beta = beta_by_seed[seed]
            sim = load_similarity_matrix(sim_paths[seed])

            if beta.shape != sim.shape:
                print(f"Shape mismatch for seed {seed} layer {layer}: {beta.shape} vs {sim.shape}")
                continue

            if args.beta_kind == "softmax":
                beta = softmax_triangular(beta)

            if args.reorder_groups:
                order = grouped_order(beta.shape[0], args.group_stride)
                beta = beta[np.ix_(order, order)]
                if not args.sim_preordered:
                    sim = sim[np.ix_(order, order)]

            beta_stack.append(beta)
            sim_stack.append(sim)

        if len(beta_stack) < 2:
            print(f"Skipping {layer}: insufficient matching seeds after filtering")
            continue

        beta_stack = np.stack(beta_stack, axis=0)
        sim_stack = np.stack(sim_stack, axis=0)
        n = beta_stack.shape[1]

        if args.mode == "cell":
            corr = np.full((n, n), np.nan, dtype=float)
            for i in range(n):
                for j in range(n):
                    x = beta_stack[:, i, j]
                    y = sim_stack[:, i, j]
                    mask = np.isfinite(x) & np.isfinite(y)
                    if np.sum(mask) >= 2:
                        corr[i, j] = spearman_corr(x[mask], y[mask])
            x_labels = [f"T{i}" for i in range(n)]
            y_labels = [f"T{i}" for i in range(n)]
        elif args.mode == "row":
            row_corr = row_correlations(beta_stack, sim_stack)
            corr = row_corr[:, None]
            x_labels = ["corr"]
            y_labels = [f"T{i}" for i in range(n)]
        else:
            col_corr = col_correlations(beta_stack, sim_stack)
            corr = col_corr[None, :]
            x_labels = [f"T{i}" for i in range(n)]
            y_labels = ["corr"]

        if args.reorder_groups:
            order = grouped_order(n, args.group_stride)
            if args.mode == "cell":
                x_labels = [f"T{i}" for i in order]
                y_labels = [f"T{i}" for i in order]
            elif args.mode == "row":
                y_labels = [f"T{i}" for i in order]
            else:
                x_labels = [f"T{i}" for i in order]
        out_dir = os.path.join(args.out_root, layer)
        os.makedirs(out_dir, exist_ok=True)

        out_pdf = os.path.join(out_dir, f"spearman_{layer}_{args.mode}.pdf")
        plot_heatmap(
            corr,
            x_labels,
            y_labels,
            f"Spearman ({args.mode}): betas vs similarity ({layer})",
            out_pdf,
            args.dpi,
            args.cmap,
            args.vmin,
            args.vmax,
            ".2f",
            5,
            args.gridlines,
            args.grid_color,
            args.grid_width,
        )
        save_matrix(corr, y_labels, x_labels, out_dir, f"spearman_{layer}_{args.mode}")

    print("Done.")


if __name__ == "__main__":
    main()
