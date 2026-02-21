#!/usr/bin/env python3
"""
Summarize beta-vs-similarity Spearman results for paper figures.

Produces:
1) Per-task mean ± bootstrap CI (with set coloring).
2) Per-layer distribution (boxplot) across tasks.
3) Within-set vs cross-set paired plot per layer.

Also saves tidy CSVs for downstream analysis.
"""

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


def grouped_order(n_tasks: int, group_stride: int) -> List[int]:
    return [i for offset in range(group_stride) for i in range(offset, n_tasks, group_stride)]


def extract_seed(path: str) -> str:
    patterns = [r"supermask-(\d+)", r"eval-run-(\d+)", r"seed(\d+)"]
    for pat in patterns:
        match = re.search(pat, path)
        if match:
            return match.group(1)
    return "unknown"


def find_similarity_matrices(sim_root: str) -> Dict[str, str]:
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


def bootstrap_ci(values: np.ndarray, n_boot: int, rng: np.random.Generator) -> Tuple[float, float]:
    vals = values[np.isfinite(values)]
    if vals.size < 2:
        return np.nan, np.nan
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(vals, size=vals.size, replace=True)
        boots.append(np.mean(sample))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def compute_row_correlations(
    beta: np.ndarray,
    sim: np.ndarray,
    task_ids: List[int],
    group_stride: int,
    exclude_self: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = beta.shape[0]
    corr_all = np.full((n,), np.nan, dtype=float)
    corr_within = np.full((n,), np.nan, dtype=float)
    corr_cross = np.full((n,), np.nan, dtype=float)

    task_sets = [tid % group_stride for tid in task_ids]

    for i in range(n):
        b = beta[i, :]
        s = sim[i, :]
        mask = np.isfinite(b) & np.isfinite(s)
        if exclude_self:
            mask &= np.array([task_ids[j] != task_ids[i] for j in range(n)])

        # All priors
        if np.sum(mask) >= 2:
            corr_all[i] = spearman_corr(b[mask], s[mask])

        # Within-set priors
        within_mask = mask & np.array([task_sets[j] == task_sets[i] for j in range(n)])
        if np.sum(within_mask) >= 2:
            corr_within[i] = spearman_corr(b[within_mask], s[within_mask])

        # Cross-set priors
        cross_mask = mask & np.array([task_sets[j] != task_sets[i] for j in range(n)])
        if np.sum(cross_mask) >= 2:
            corr_cross[i] = spearman_corr(b[cross_mask], s[cross_mask])

    return corr_all, corr_within, corr_cross


def plot_per_task(
    tasks: List[int],
    means: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    group_stride: int,
    title: str,
    out_path: str,
    dpi: int,
):
    x = np.arange(len(tasks))
    set_ids = [t % group_stride for t in tasks]
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
    c = [colors[s % len(colors)] for s in set_ids]

    fig, ax = plt.subplots(figsize=(12, 4.5))
    yerr = np.vstack((means - ci_low, ci_high - means))
    ax.errorbar(x, means, yerr=yerr, fmt="o", ecolor="gray", capsize=3, markersize=4, color="black")
    ax.scatter(x, means, c=c, s=30, zorder=3)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_xticks(x, labels=[f"T{t}" for t in tasks], rotation=45, ha="right")
    ax.set_ylabel("Spearman (row)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_layer_distribution(
    layer_values: Dict[str, np.ndarray],
    title: str,
    out_path: str,
    dpi: int,
):
    layers = list(layer_values.keys())
    data = [layer_values[layer] for layer in layers]
    means = [np.nanmean(vals) if len(vals) else np.nan for vals in data]

    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.boxplot(data, labels=layers, showfliers=False)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_ylabel("Spearman (row)")
    ax.set_title(title)

    # Draw mean lines and annotate just above them.
    finite_vals = np.concatenate([v[np.isfinite(v)] for v in data if len(v)])
    y_range = np.nanmax(finite_vals) - np.nanmin(finite_vals) if finite_vals.size else 1.0
    y_pad = 0.03 * y_range
    #for i, m in enumerate(means, start=1):
    #    if np.isfinite(m):
    #        ax.hlines(m, i - 0.25, i + 0.25, colors="#F58518", linewidth=2)
    #        ax.text(i, m + y_pad, f"{m:.2f}", ha="center", va="bottom", fontsize=9, color="#F58518")

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_within_cross(
    tasks: List[int],
    within_means: np.ndarray,
    cross_means: np.ndarray,
    title: str,
    out_path: str,
    dpi: int,
):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    x = np.array([0, 1])
    for i in range(len(tasks)):
        ax.plot(x, [within_means[i], cross_means[i]], color="gray", alpha=0.4, linewidth=1)
        ax.scatter(x, [within_means[i], cross_means[i]], color="black", s=12)

    ax.set_xticks([0, 1], labels=["Within", "Cross"])
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_ylabel("Spearman (row)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Spearman results for paper plots")
    parser.add_argument("--beta-root", default="./log/ct28-interleaved-MaskLC/eval")
    parser.add_argument("--sim-root", default="./log/sim_heatmaps")
    parser.add_argument("--out-root", default="./log/spearman_summary")
    parser.add_argument("--beta-kind", choices=["before_softmax", "softmax"], default="softmax")
    parser.add_argument("--reorder-groups", action="store_true")
    parser.add_argument("--sim-preordered", action="store_true")
    parser.add_argument("--group-stride", type=int, default=4)
    parser.add_argument("--omit-origin", action="store_true", help="Omit tasks 0..3 from summaries")
    parser.add_argument("--exclude-self", action="store_true", help="Exclude diagonal/self in per-row correlations")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_root, exist_ok=True)

    sim_paths = find_similarity_matrices(args.sim_root)
    if not sim_paths:
        raise SystemExit(f"No similarity_heatmap.npy/csv found under {args.sim_root}")

    beta_paths = find_beta_matrices(args.beta_root)
    if not beta_paths:
        raise SystemExit(f"No betas_before_softmax_*.bin found under {args.beta_root}")

    beta_by_layer: Dict[str, Dict[str, np.ndarray]] = {}
    for (seed, layer), path in beta_paths.items():
        beta_by_layer.setdefault(layer, {})[seed] = load_beta_matrix(path)

    all_records = []
    per_layer_values = {}

    for layer, beta_by_seed in sorted(beta_by_layer.items()):
        seeds = sorted(set(beta_by_seed.keys()) & set(sim_paths.keys()))
        if len(seeds) == 0:
            continue

        per_seed_all = []
        per_seed_within = []
        per_seed_cross = []
        task_ids = None

        for seed in seeds:
            beta = beta_by_seed[seed]
            sim = load_similarity_matrix(sim_paths[seed])
            if beta.shape != sim.shape:
                print(f"Shape mismatch seed {seed} layer {layer}: {beta.shape} vs {sim.shape}")
                continue

            if args.beta_kind == "softmax":
                beta = softmax_triangular(beta)

            n = beta.shape[0]
            order = grouped_order(n, args.group_stride) if args.reorder_groups else list(range(n))
            task_ids = order

            if args.reorder_groups:
                beta = beta[np.ix_(order, order)]
                if not args.sim_preordered:
                    sim = sim[np.ix_(order, order)]

            corr_all, corr_within, corr_cross = compute_row_correlations(
                beta,
                sim,
                task_ids=task_ids,
                group_stride=args.group_stride,
                exclude_self=args.exclude_self,
            )

            per_seed_all.append(corr_all)
            per_seed_within.append(corr_within)
            per_seed_cross.append(corr_cross)

            for i, tid in enumerate(task_ids):
                all_records.append(
                    {
                        "layer": layer,
                        "seed": seed,
                        "task_id": tid,
                        "set_id": tid % args.group_stride,
                        "corr_all": corr_all[i],
                        "corr_within": corr_within[i],
                        "corr_cross": corr_cross[i],
                    }
                )

        if not per_seed_all:
            continue

        per_seed_all = np.array(per_seed_all)  # seeds x tasks
        per_seed_within = np.array(per_seed_within)
        per_seed_cross = np.array(per_seed_cross)

        # Optionally omit origin tasks (0..3)
        if args.omit_origin and task_ids is not None:
            keep = [tid not in {0, 1, 2, 3} for tid in task_ids]
            keep_idx = np.where(keep)[0]
        else:
            keep_idx = np.arange(per_seed_all.shape[1])

        task_ids_kept = [task_ids[i] for i in keep_idx]

        means = np.nanmean(per_seed_all[:, keep_idx], axis=0)
        ci_lo = np.zeros_like(means)
        ci_hi = np.zeros_like(means)
        for i, col in enumerate(keep_idx):
            lo, hi = bootstrap_ci(per_seed_all[:, col], args.bootstrap, rng)
            ci_lo[i] = lo
            ci_hi[i] = hi

        out_layer_dir = os.path.join(args.out_root, layer)
        os.makedirs(out_layer_dir, exist_ok=True)

        plot_per_task(
            task_ids_kept,
            means,
            ci_lo,
            ci_hi,
            args.group_stride,
            f"Per-task Spearman (row) — {layer}",
            os.path.join(out_layer_dir, f"per_task_{layer}.pdf"),
            args.dpi,
        )

        # Within vs cross paired plot
        within_means = np.nanmean(per_seed_within[:, keep_idx], axis=0)
        cross_means = np.nanmean(per_seed_cross[:, keep_idx], axis=0)
        plot_within_cross(
            task_ids_kept,
            within_means,
            cross_means,
            f"Within vs Cross (row) — {layer}",
            os.path.join(out_layer_dir, f"within_cross_{layer}.pdf"),
            args.dpi,
        )

        # Collect for distribution plot
        vals = per_seed_all[:, keep_idx].reshape(-1)
        vals = vals[np.isfinite(vals)]
        per_layer_values[layer] = vals

        # Save per-layer summaries
        summary_df = pd.DataFrame(
            {
                "task_id": task_ids_kept,
                "mean": means,
                "ci_low": ci_lo,
                "ci_high": ci_hi,
            }
        )
        summary_df.to_csv(os.path.join(out_layer_dir, f"per_task_{layer}.csv"), index=False)

        within_df = pd.DataFrame(
            {
                "task_id": task_ids_kept,
                "within_mean": within_means,
                "cross_mean": cross_means,
            }
        )
        within_df.to_csv(os.path.join(out_layer_dir, f"within_cross_{layer}.csv"), index=False)

    # Global distribution plot across layers
    if per_layer_values:
        plot_layer_distribution(
            per_layer_values,
            "Per-task Spearman distribution by layer",
            os.path.join(args.out_root, "per_layer_distribution.pdf"),
            args.dpi,
        )

    # Save long-form data
    long_df = pd.DataFrame(all_records)
    long_df.to_csv(os.path.join(args.out_root, "per_task_long.csv"), index=False)


if __name__ == "__main__":
    main()
