#!/usr/bin/env python3
"""
Compute unison (intersection) masks across all tasks and report overlap stats.

For each layer:
- Load task-specific mask scores
- Binarize (scores >= threshold)
- Compute the unison mask as logical AND across tasks
- Count active parameters and densities; also compute full network totals

Outputs:
- Heatmap PNG summarizing per-layer unison density and counts
- NPY bundle containing unison masks and stats for reuse in Python
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

DEFAULT_TASK_STATS = (
    Path(__file__).resolve().parent
    / "log/MetaCTgraph-ppo-supermask-86-mask-linear_comb-ct14_md/251212-123907/task_stats"
)


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
    Return task_ids, layer_names, layer_masks (raw scores).
    If use_final_checkpoint is True, only load the final checkpoint and extract
    all scores.* entries to reduce I/O.
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
                layer_masks[name][task_id] = state[score_key].cpu().numpy()
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

        task_idx = task_id - 1
        for name in layer_names:
            score_key = f"{name}.scores.{task_idx}"
            if score_key not in state:
                raise KeyError(f"Missing {score_key} in {ckpt}")
            layer_masks[name][task_id] = state[score_key].cpu().numpy()

    return task_ids, layer_names, layer_masks


def _compute_unison(layer_masks, layer_names, task_ids, threshold: float):
    unison = {}
    per_layer_stats = {}
    total_active = 0
    total_params = 0

    for name in layer_names:
        # stack binarized masks for all tasks
        bin_masks = []
        for tid in task_ids:
            scores = layer_masks[name][tid]
            bin_masks.append((scores >= threshold))
        bin_stack = np.stack(bin_masks, axis=0)
        uni = np.logical_and.reduce(bin_stack, axis=0)
        unison[name] = uni

        active = int(uni.sum())
        size = uni.size
        per_layer_stats[name] = {"active": active, "total": size, "density": active / float(size)}
        total_active += active
        total_params += size

    summary = {"total_active": total_active, "total_params": total_params, "density": total_active / float(total_params)}
    return unison, per_layer_stats, summary


def _plot_unison(per_layer_stats, output_path: Path, summary):
    layer_names = list(per_layer_stats.keys())
    densities = [per_layer_stats[n]["density"] for n in layer_names]
    counts_m = [per_layer_stats[n]["active"] / 1e6 for n in layer_names]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    # density heatmap (layers x 1)
    dens_mat = np.array(densities)[:, None]
    im0 = axes[0].imshow(dens_mat, cmap=plt.cm.Oranges, vmin=0.0, vmax=1.0)
    axes[0].set_yticks(range(len(layer_names)))
    axes[0].set_yticklabels(layer_names, fontsize=8)
    axes[0].set_xticks([0])
    axes[0].set_xticklabels(["density"])
    for i, val in enumerate(densities):
        axes[0].text(0, i, f"{val:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im0, ax=axes[0], shrink=0.8, label="unison density")

    # counts heatmap (in millions)
    cnt_mat = np.array(counts_m)[:, None]
    im1 = axes[1].imshow(cnt_mat, cmap=plt.cm.Blues)
    axes[1].set_yticks(range(len(layer_names)))
    axes[1].set_yticklabels(layer_names, fontsize=8)
    axes[1].set_xticks([0])
    axes[1].set_xticklabels(["active (M)"])
    for i, val in enumerate(counts_m):
        axes[1].text(0, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im1, ax=axes[1], shrink=0.8, label="active params (millions)")

    fig.suptitle(
        f"Unison masks across tasks\nTotal active {summary['total_active']:,} of {summary['total_params']:,} (density {summary['density']:.3f})",
        fontsize=12,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved unison heatmaps to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute unison (intersection) masks across tasks.")
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
        help="Output file path for the heatmap figure (PNG/PDF). If a directory is provided, a default filename is appended.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Threshold for binarization (scores >= threshold considered active).",
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

    unison, per_layer_stats, summary = _compute_unison(layer_masks, layer_names, task_ids, args.threshold)

    output_path = args.output
    if output_path is None:
        output_path = task_stats_dir / "mask_unison_heatmap.png"
    output_path = output_path.expanduser().resolve()
    if output_path.is_dir():
        output_path = output_path / "mask_unison_heatmap.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save unison masks and stats for reuse
    npy_path = output_path.with_suffix(".npy")
    np.save(
        npy_path,
        {
            "task_ids": task_ids,
            "layer_names": layer_names,
            "threshold": args.threshold,
            "unison_masks": unison,
            "per_layer_stats": per_layer_stats,
            "summary": summary,
        },
    )
    print(f"Saved unison masks/data to {npy_path}")

    _plot_unison(per_layer_stats, output_path, summary)


if __name__ == "__main__":
    main()
