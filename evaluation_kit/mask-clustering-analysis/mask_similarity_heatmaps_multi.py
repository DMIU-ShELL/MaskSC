#!/usr/bin/env python3
"""
Plot per-layer mask similarity/distance heatmaps across multiple experiments.
Handles overlapping task indices by tagging tasks with their experiment label.

Supports cosine similarity on raw/binarized masks, Jaccard distance on
binarized masks (GetSubnetDiscrete rule: scores >= threshold), and Pearson
correlation on raw/binarized masks.
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


def _binarize_mask(scores: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    return (scores >= threshold).astype(np.float32)


def _find_checkpoints(task_stats_dir: Path):
    pattern = re.compile(r"task-(\d+)\.bin$")
    candidates = []
    for path in sorted(task_stats_dir.glob("*-model-*-task-*.bin")):
        print(path)
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


def _load_masks_for_dir(checkpoints, use_final_checkpoint: bool):
    """
    Return (task_ids, layer_names, layer_masks) for one experiment directory.
    layer_masks: {layer_name: {task_id: scores}}
    """
    if use_final_checkpoint:
        _, final_ckpt = sorted(checkpoints, key=lambda x: x[0])[-1]
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


def _similarity_matrix(task_labels, masks_per_task, metric: str, binarize: bool, threshold: float):
    n = len(task_labels)
    sim = np.zeros((n, n), dtype=np.float32)
    for i, ti in enumerate(task_labels):
        for j, tj in enumerate(task_labels):
            mask_a = masks_per_task[ti]
            mask_b = masks_per_task[tj]
            if binarize:
                mask_a = _binarize_mask(mask_a, threshold)
                mask_b = _binarize_mask(mask_b, threshold)

            if metric == "cosine":
                sim[i, j] = _cosine_similarity(mask_a, mask_b)
            elif metric == "jaccard":
                sim[i, j] = _jaccard_distance(mask_a, mask_b)
            elif metric == "pearson":
                sim[i, j] = _pearson_corr(mask_a, mask_b)
            else:
                raise ValueError(f"Unsupported metric: {metric}")
    return sim


def _plot_heatmaps(layer_sims, task_labels, output_path: Path, metric: str):
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

        ax.set_xticks(range(len(task_labels)))
        ax.set_yticks(range(len(task_labels)))
        ax.set_xticklabels(task_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(task_labels, fontsize=8)

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7)

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
            "Mask cosine similarity across experiments"
            if metric == "cosine"
            else "Mask Jaccard distance across experiments"
            if metric == "jaccard"
            else "Mask Pearson correlation across experiments"
        ),
        fontsize=14,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved heatmaps to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot per-layer mask similarity/distance heatmaps across multiple experiments."
    )
    parser.add_argument(
        "task_stats_dirs",
        nargs="+",
        type=Path,
        help="One or more task_stats directories (each belonging to a different experiment).",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        help="Optional labels for experiments (same order as task_stats_dirs). Defaults to directory names.",
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
        help="Threshold for binarization (scores >= threshold).",
    )
    parser.add_argument(
        "--use-final-checkpoint",
        action="store_true",
        help="Load only the final checkpoint per experiment and extract all task masks from its scores.* entries to reduce I/O.",
    )
    args = parser.parse_args()

    task_stats_dirs = [p.expanduser().resolve() for p in args.task_stats_dirs]
    for d in task_stats_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Task stats directory not found: {d}")

    if args.labels:
        if len(args.labels) != len(task_stats_dirs):
            raise ValueError("--labels must match the number of task_stats_dirs")
        labels = args.labels
    else:
        labels = [d.name for d in task_stats_dirs]

    # Load each experiment
    per_exp_data = []
    reference_layers = None
    for d, label in zip(task_stats_dirs, labels):
        checkpoints = _find_checkpoints(d)
        task_ids, layer_names, layer_masks = _load_masks_for_dir(checkpoints, args.use_final_checkpoint)

        if reference_layers is None:
            reference_layers = layer_names
        else:
            if reference_layers != layer_names:
                raise ValueError(f"Layer mismatch in {d}: {layer_names} vs {reference_layers}")

        per_exp_data.append((label, task_ids, layer_masks))

    # Combine masks across experiments, with unique labels per task
    combined_task_labels = []
    combined_masks_per_layer = {name: {} for name in reference_layers}

    for label, task_ids, layer_masks in per_exp_data:
        for tid in task_ids:
            tag = f"{label}:T{tid}"
            combined_task_labels.append(tag)
            for lname in reference_layers:
                combined_masks_per_layer[lname][tag] = layer_masks[lname][tid]

    metric = args.metric
    binarize = args.binarize or metric == "jaccard"
    threshold = args.threshold

    layer_sims = OrderedDict()
    for name in reference_layers:
        layer_sims[name] = _similarity_matrix(
            combined_task_labels, combined_masks_per_layer[name], metric, binarize, threshold
        )

    output_path = args.output
    if output_path is None:
        suffix = "mask_similarity_multi.pdf" if metric == "cosine" else "mask_jaccard_multi.pdf"
        output_path = task_stats_dirs[0].parent / suffix
    output_path = output_path.expanduser().resolve()
    if output_path.is_dir():
        suffix = "mask_similarity_multi.pdf" if metric == "cosine" else "mask_jaccard_multi.pdf"
        output_path = output_path / suffix
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # save raw data for reproducibility
    npy_path = output_path.with_suffix(".npy")
    np.save(
        npy_path,
        {
            "task_labels": combined_task_labels,
            "layer_names": reference_layers,
            "metric": metric,
            "binarize": binarize,
            "threshold": threshold,
            "layer_sims": layer_sims,
        },
    )
    print(f"Saved heatmap data to {npy_path}")

    _plot_heatmaps(layer_sims, combined_task_labels, output_path, metric)


if __name__ == "__main__":
    main()
