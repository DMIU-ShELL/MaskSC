#!/usr/bin/env python3
"""
Generate per-layer mask similarity/distance heatmaps for task-specific masks.
Supports cosine similarity on raw or binarized masks, and Jaccard distance on
binarized masks (using the same binarization rule as GetSubnetDiscrete). Also
supports Pearson correlation on raw or binarized masks.
"""
from collections import OrderedDict
import argparse
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import autograd

DEFAULT_TASK_STATS = (
    Path(__file__).resolve().parent
    / "log/MetaCTgraph-ppo-supermask-86-mask-linear_comb-ct14_md/251212-123907/task_stats"
)


# Reference: GetSubnetDiscrete (scores >= a).float()
class GetSubnetDiscrete(autograd.Function):
    @staticmethod
    def forward(ctx, scores, a=0):
        return (scores >= a).float()

    @staticmethod
    def backward(ctx, g):
        return g


def _gate_mask_binary(scores: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    return (scores >= threshold).astype(np.float32)

def _gate_mask_continous(scores: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    return ((scores >= threshold)*scores).astype(np.float32)


def _find_checkpoints(task_stats_dir: Path):
    pattern = re.compile(r"task-(\d+)\.bin$")
    candidates = []
    for path in sorted(task_stats_dir.glob("*-model-*-task-*.bin")):
        match = pattern.search(path.name)
        if not match:
            continue
        task_id = int(match.group(1))
        candidates.append((task_id, path))
    if not candidates:
        raise FileNotFoundError(f"No model checkpoints found in {task_stats_dir}")
    candidates.sort(key=lambda item: item[0])
    return candidates


def _discover_layers(state_dict):
    layers = []
    for key in state_dict:
        if ".scores." not in key:
            continue
        prefix = key.split(".scores.")[0]
        if prefix not in layers:
            layers.append(prefix)
    if not layers:
        raise ValueError("No mask score tensors found in checkpoint")
    return layers


def _load_masks(checkpoints, use_final_checkpoint: bool):
    """
    If use_final_checkpoint is True, only load the last checkpoint (highest task id)
    and extract all task masks from its scores.* entries to reduce I/O.
    Otherwise, load per-task checkpoints and pick the matching scores entry.
    """
    if use_final_checkpoint:
        final_task_id, final_ckpt = sorted(checkpoints, key=lambda x: x[0])[-1]
        state = torch.load(final_ckpt, map_location="cpu")
        layer_names = _discover_layers(state)
        layer_masks = OrderedDict((name, {}) for name in layer_names)

        first_layer = layer_names[0]
        idxs = []
        for key in state:
            if key.startswith(f"{first_layer}.scores."):
                try:
                    idxs.append(int(key.split(".scores.")[-1]))
                except ValueError:
                    continue
        if not idxs:
            raise ValueError(f"No scores.* entries found in {final_ckpt}")
        idxs = sorted(set(idxs))
        task_ids = [i + 1 for i in idxs]

        for name in layer_names:
            for idx, task_id in zip(idxs, task_ids):
                score_key = f"{name}.scores.{idx}"
                if score_key not in state:
                    raise KeyError(f"Missing {score_key} in {final_ckpt}")
                scores = state[score_key].cpu().numpy()
                layer_masks[name][task_id] = scores
        return task_ids, layer_names, layer_masks

    layer_masks = OrderedDict()
    layer_names = None
    task_ids = []

    for task_id, ckpt in checkpoints:
        state = torch.load(ckpt, map_location="cpu")
        task_ids.append(task_id)
        if layer_names is None:
            layer_names = _discover_layers(state)
            for name in layer_names:
                layer_masks[name] = {}

        task_idx = task_id - 1  # internal mask index is zero-based
        for name in layer_names:
            score_key = f"{name}.scores.{task_idx}"
            if score_key not in state:
                raise KeyError(f"Missing {score_key} in {ckpt}")
            scores = state[score_key].cpu().numpy()
            # keep raw scores; binarize later depending on metric/flags
            layer_masks[name][task_id] = scores

    return task_ids, layer_names, layer_masks


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.ravel().astype(np.float64)
    b_flat = b.ravel().astype(np.float64)
    denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    if denom == 0:
        return 0.0
    return float(np.dot(a_flat, b_flat) / denom)


def _jaccard_distance(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.ravel().astype(np.float64)
    b_flat = b.ravel().astype(np.float64)
    intersection = np.logical_and(a_flat, b_flat).sum()
    union = np.logical_or(a_flat, b_flat).sum()
    if union == 0:
        return 0.0
    return 1.0 - float(intersection / union)


def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.ravel().astype(np.float64)
    b_flat = b.ravel().astype(np.float64)
    a_c = a_flat - a_flat.mean()
    b_c = b_flat - b_flat.mean()
    denom = np.linalg.norm(a_c) * np.linalg.norm(b_c)
    if denom == 0:
        return 0.0
    return float(np.dot(a_c, b_c) / denom)


def _similarity_matrix(task_ids, masks_per_task, metric: str, binarize: bool, threshold: float, zero_common: bool):
    # gate once for all tasks to reuse below
    gated_masks = []
    for tid in task_ids:
        m = masks_per_task[tid]
        if binarize or metric == "jaccard":
            m = _gate_mask_binary(m, threshold)
        else:
            m = _gate_mask_continous(m, threshold)
        gated_masks.append(m)

    if zero_common:
        common = np.logical_and.reduce([(m != 0) for m in gated_masks])
        gated_masks = [np.where(common, 0, m) for m in gated_masks]

    n = len(task_ids)
    sim = np.zeros((n, n), dtype=np.float32)
    for i, mask_a in enumerate(gated_masks):
        for j, mask_b in enumerate(gated_masks):
            if metric == "cosine":
                sim[i, j] = _cosine_similarity(mask_a, mask_b)
            elif metric == "jaccard":
                sim[i, j] = _jaccard_distance(mask_a, mask_b)
            elif metric == "pearson":
                sim[i, j] = _pearson_corr(mask_a, mask_b)
            else:
                raise ValueError(f"Unsupported metric: {metric}")
    return sim


def _plot_heatmaps(layer_sims, task_ids, output_path: Path, metric: str):
    n_layers = len(layer_sims)
    cols = math.ceil(math.sqrt(n_layers))
    rows = math.ceil(n_layers / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows), constrained_layout=True)
    axes = np.array(axes).reshape(rows, cols)

    cmap = plt.cm.Oranges
    color_ref = None
    for idx, (layer, matrix) in enumerate(layer_sims.items()):
        ax = axes[idx // cols, idx % cols]
        if metric == "pearson":
            cmap = plt.cm.viridis
            vmin, vmax = -1.0, 1.0
        else:
            vmin, vmax = 0.0, 1.0
        im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
        if color_ref is None:
            color_ref = im

        ax.set_xticks(range(len(task_ids)))
        ax.set_yticks(range(len(task_ids)))
        ax.set_xticklabels(task_ids, rotation=45, ha="right")
        ax.set_yticklabels(task_ids)

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)

        ax.set_title(layer, fontsize=12)

    for idx in range(n_layers, rows * cols):
        axes.flat[idx].axis("off")

    if color_ref is not None:
        if metric == "cosine":
            label = "cosine similarity"
        elif metric == "jaccard":
            label = "Jaccard distance"
        else:
            label = "Pearson correlation"
        fig.colorbar(color_ref, ax=axes.ravel().tolist(), shrink=0.8, label=label)
    fig.suptitle(
        (
            "Mask cosine similarity across tasks"
            if metric == "cosine"
            else "Mask Jaccard distance across tasks"
            if metric == "jaccard"
            else "Mask Pearson correlation across tasks"
        ),
        fontsize=14,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved heatmaps to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot per-layer mask similarity/distance heatmaps.")
    parser.add_argument(
        "--task-stats-dir",
        type=Path,
        default=DEFAULT_TASK_STATS,
        help="Directory containing per-task model checkpoints (task_stats folder)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path for the figure (PNG/PDF). If a directory is provided, a default filename is appended.",
    )
    parser.add_argument(
        "--metric",
        choices=["cosine", "jaccard", "pearson"],
        default="cosine",
        help="Metric to use. Jaccard operates on binarized masks and reports distance (0 = identical). Pearson uses correlation.",
    )
    parser.add_argument(
        "--binarize",
        action="store_true",
        help="Binarize masks before computing the metric (scores >= threshold). For Jaccard, binarization is always applied.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Threshold for binarization (matching GetSubnetDiscrete: scores >= threshold).",
    )
    parser.add_argument(
        "--zero-common",
        action="store_true",
        help="Zero out parameters that are active in all tasks (after gating) before computing metrics.",
    )
    parser.add_argument(
        "--use-final-checkpoint",
        action="store_true",
        help="Load only the final checkpoint and extract all task masks from its scores.* entries to reduce I/O.",
    )
    args = parser.parse_args()

    task_stats_dir = args.task_stats_dir.expanduser().resolve()
    if not task_stats_dir.exists():
        raise FileNotFoundError(f"Task stats directory not found: {task_stats_dir}")

    checkpoints = _find_checkpoints(task_stats_dir)

    task_ids, layer_names, layer_masks = _load_masks(checkpoints, args.use_final_checkpoint)

    metric = args.metric
    binarize = args.binarize or metric == "jaccard"
    threshold = args.threshold
    zero_common = args.zero_common

    layer_sims = OrderedDict()
    for name in layer_names:
        layer_sims[name] = _similarity_matrix(task_ids, layer_masks[name], metric, binarize, threshold, zero_common)

    output_path = args.output
    if output_path is None:
        suffix = "mask_similarity_heatmaps.pdf" if metric == "cosine" else "mask_jaccard_heatmaps.pdf"
        output_path = task_stats_dir / suffix
    output_path = output_path.expanduser().resolve()
    if output_path.is_dir():
        suffix = "mask_similarity_heatmaps.pdf" if metric == "cosine" else "mask_jaccard_heatmaps.pdf"
        output_path = output_path / suffix
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # save raw data for reproducibility
    npy_path = output_path.with_suffix(".npy")
    np.save(
        npy_path,
        {
            "task_ids": task_ids,
            "layer_names": layer_names,
            "metric": metric,
            "binarize": binarize,
            "threshold": threshold,
            "zero_common": zero_common,
            "layer_sims": layer_sims,
        },
    )
    print(f"Saved heatmap data to {npy_path}")

    _plot_heatmaps(layer_sims, task_ids, output_path, metric)


if __name__ == "__main__":
    main()
