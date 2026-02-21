#!/usr/bin/env python3
"""
Analyze mask sparsity across tasks.

Loads task checkpoints, thresholds mask scores to active/inactive, and plots
how active parameter counts and densities evolve per layer as tasks accumulate.
Saves both a figure and a .npy bundle of the underlying numbers for reuse.
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
    Return task_ids, layer_names, layer_masks (raw scores) for the experiment.
    When use_final_checkpoint is True, only load the last checkpoint and pull
    all scores.* slices from it to reduce I/O.
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

        task_idx = task_id - 1  # zero-based internal index
        for name in layer_names:
            score_key = f"{name}.scores.{task_idx}"
            if score_key not in state:
                raise KeyError(f"Missing {score_key} in {ckpt}")
            layer_masks[name][task_id] = state[score_key].cpu().numpy()

    return task_ids, layer_names, layer_masks


def _compute_sparsity(task_ids, layer_names, layer_masks, threshold: float):
    counts = {name: [] for name in layer_names}
    densities = {name: [] for name in layer_names}
    for tid in task_ids:
        for name in layer_names:
            scores = layer_masks[name][tid]
            active = (scores >= threshold)
            active_count = int(active.sum())
            total = active.size
            counts[name].append(active_count)
            densities[name].append(active_count / float(total))
    return counts, densities


def _plot_sparsity(task_ids, layer_names, counts, densities, output_path: Path, threshold: float):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)

    # Counts
    for name in layer_names:
        axes[0].plot(task_ids, counts[name], marker="o", label=name)
    axes[0].set_ylabel("Active params (count)")
    axes[0].set_title(f"Mask sparsity vs task (threshold={threshold})")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(fontsize=8)

    # Density
    for name in layer_names:
        axes[1].plot(task_ids, densities[name], marker="o", label=name)
    axes[1].set_ylabel("Active density")
    axes[1].set_xlabel("Task index")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved sparsity plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze mask sparsity across tasks.")
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
        help="Output file path for the plot (PNG/PDF). If a directory is provided, a default filename is appended.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Threshold for counting active parameters (scores >= threshold).",
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

    threshold = args.threshold
    counts, densities = _compute_sparsity(task_ids, layer_names, layer_masks, threshold)

    output_path = args.output
    if output_path is None:
        output_path = task_stats_dir / "mask_sparsity.png"
    output_path = output_path.expanduser().resolve()
    if output_path.is_dir():
        output_path = output_path / "mask_sparsity.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save underlying data for reuse
    npy_path = output_path.with_suffix(".npy")
    np.save(
        npy_path,
        {
            "task_ids": task_ids,
            "layer_names": layer_names,
            "threshold": threshold,
            "counts": counts,
            "densities": densities,
        },
    )
    print(f"Saved sparsity data to {npy_path}")

    _plot_sparsity(task_ids, layer_names, counts, densities, output_path, threshold)


if __name__ == "__main__":
    main()
