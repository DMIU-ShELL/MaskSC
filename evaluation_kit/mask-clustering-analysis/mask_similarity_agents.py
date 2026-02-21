#!/usr/bin/env python3
"""
Compute per-layer mask similarity heatmaps across multiple agent checkpoints
(single-task agents). Each agent contributes one mask per layer (scores.0).

Default directory layout (can override with --root-dir):
/home/lunet/cosn5/c3l/aaai_experiments/deeprl-shell/log/FINAL/mctgraph/fullcomm/seed1/
Each subfolder under seed1 is an agent directory containing a .bin checkpoint.
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
#DEFAULT_ROOT = Path("./c3l/aaai_experiments/deeprl-shell/log/FINAL/mctgraph/fullcomm/seed1/")
DEFAULT_ROOT = Path("./c3l/aaai_experiments/deeprl-shell/log/FINAL/minigrid/fullcomm/seed1/")
TYPE='MINIGRID' #CTGRAPH


def _binarize(scores: np.ndarray, threshold: float) -> np.ndarray:
    return (scores >= threshold).astype(np.float32)


def _gate_continuous(scores: np.ndarray, threshold: float) -> np.ndarray:
    return ((scores >= threshold) * scores).astype(np.float32)


def _list_agent_bins(root: Path):
    if TYPE=='MINIGRID':
        def _agent_order(name: str):
            # natural order: use mg_agent<number> if present, else fall back to name
            m = re.search(r"mg_agent(\d+)", name)
            if m:
                return (int(m.group(1)), name)
            return (10**9, name)
            
        agents = []
        for sub in sorted(root.iterdir(), key=lambda p: _agent_order(p.name)):
            if not sub.is_dir():
                continue
            # search recursively for checkpoints
            bins = sorted(sub.rglob("*.bin"))
            if not bins:
                continue
            bins.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            agents.append((sub.name, bins[0]))
        if not agents:
            raise FileNotFoundError(f"No agent .bin files found under {root}")
        return agents

    elif TYPE=='CTGRAPH':
        agents = []
        for sub in sorted(root.iterdir()):
            if not sub.is_dir():
                continue
            # search recursively for checkpoints
            bins = sorted(sub.rglob("*.bin"))
            if not bins:
                continue
            bins.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            agents.append((sub.name, bins[0]))
        if not agents:
            raise FileNotFoundError(f"No agent .bin files found under {root}")
        return agents


def _short_label_ctgraph(agent_name: str) -> str:
    """
    Extract a concise label like a<id>_d<id> from names such as
    'MetaCTgraph-shell-dist-ct28_a0_d1-seed-9157'. Falls back to full name.
    """
    match = re.search(r"(a\d+_d\d+)", agent_name)
    return match.group(1) if match else agent_name

def _short_label_minigrid(agent_name: str) -> str:
    """
    Extract a concise label like a<id>_d<id> from names such as
    'MetaCTgraph-shell-dist-ct28_a0_d1-seed-9157'. Falls back to full name.
    """
    match = re.search(r"(agent\d+)", agent_name)
    return match.group(1) if match else agent_name

def _short_label(agent_name: str) -> str:
    if TYPE=='CTGRAPH':
        return _short_label_ctgraph(agent_name)
    elif TYPE=='MINIGRID':
        return _short_label_minigrid(agent_name)


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


def _load_agent_masks(agent_bins):
    """
    Load scores.0 for each agent, returning agent_labels, layer_names, layer_masks.
    layer_masks: {layer_name: {agent_label: scores_array}}
    """
    layer_masks = OrderedDict()
    layer_names = None
    agent_labels = []

    for label, ckpt in sorted(agent_bins):
        state = torch.load(ckpt, map_location="cpu")
        agent_labels.append(label)
        if layer_names is None:
            layer_names = _discover_layers(state)
            for name in layer_names:
                layer_masks[name] = {}
        else:
            # ensure layers match
            cur_layers = _discover_layers(state)
            if cur_layers != layer_names:
                raise ValueError(f"Layer mismatch for {label}: {cur_layers} vs {layer_names}")

        for name in layer_names:
            key = f"{name}.scores.0"
            if key not in state:
                raise KeyError(f"Missing {key} in {ckpt}")
            layer_masks[name][label] = state[key].cpu().numpy()

    return agent_labels, layer_names, layer_masks


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


def _similarity_matrix(labels, masks_per_agent, metric: str, binarize: bool, threshold: float):
    n = len(labels)
    sim = np.zeros((n, n), dtype=np.float32)
    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            mask_a = masks_per_agent[li]
            mask_b = masks_per_agent[lj]
            if binarize or metric == "jaccard":
                mask_a = _binarize(mask_a, threshold)
                mask_b = _binarize(mask_b, threshold)
            else:
                mask_a = _gate_continuous(mask_a, threshold)
                mask_b = _gate_continuous(mask_b, threshold)

            if metric == "cosine":
                sim[i, j] = _cosine_similarity(mask_a, mask_b)
            elif metric == "jaccard":
                sim[i, j] = _jaccard_distance(mask_a, mask_b)
            elif metric == "pearson":
                sim[i, j] = _pearson_corr(mask_a, mask_b)
            else:
                raise ValueError(f"Unsupported metric: {metric}")
    return sim


def _plot_heatmaps(layer_sims, agent_labels, output_path: Path, metric: str):
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
            vmin, vmax = -1.0, 1.0
        else:
            vmin, vmax = 0.0, 1.0
        im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
        if color_ref is None:
            color_ref = im

        ax.set_xticks(range(len(agent_labels)))
        ax.set_yticks(range(len(agent_labels)))
        ax.set_xticklabels(agent_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(agent_labels, fontsize=8)

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7)

        ax.set_title(layer, fontsize=11)

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
        "Mask similarity across agents" if metric == "cosine" else f"Mask {metric} across agents",
        fontsize=14,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved heatmaps to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Mask similarity heatmaps across single-task agents.")
    parser.add_argument("--root-dir", type=Path, default=DEFAULT_ROOT, help="Root directory containing agent subfolders")
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
        help="Similarity metric.",
    )
    parser.add_argument(
        "--binarize",
        action="store_true",
        help="Binarize masks (scores >= threshold) before computing metric (always on for Jaccard).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Threshold for binarization or gating.",
    )
    args = parser.parse_args()

    root = args.root_dir.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")

    agent_bins = _list_agent_bins(root)
    # short labels for tick marks
    short_labels = [_short_label(lbl) for lbl, _ in agent_bins]
    agent_labels, layer_names, layer_masks_full = _load_agent_masks(agent_bins)

    # remap masks to short labels so keys align with tick labels
    layer_masks = {}
    for lname in layer_names:
        masks_for_layer = {}
        for full_label, short_label in zip(agent_labels, short_labels):
            masks_for_layer[short_label] = layer_masks_full[lname][full_label]
        layer_masks[lname] = masks_for_layer

    metric = args.metric
    binarize = args.binarize or metric == "jaccard"
    threshold = args.threshold

    layer_sims = OrderedDict()
    for name in layer_names:
        layer_sims[name] = _similarity_matrix(short_labels, layer_masks[name], metric, binarize, threshold)

    output_path = args.output
    if output_path is None:
        suffix = f"mask_similarity_agents_{metric}.png"
        output_path = root / suffix
    output_path = output_path.expanduser().resolve()
    if output_path.is_dir():
        suffix = f"mask_similarity_agents_{metric}.png"
        output_path = output_path / suffix
    output_path.parent.mkdir(parents=True, exist_ok=True)

    npy_path = output_path.with_suffix(".npy")
    np.save(
        npy_path,
        {
            "agent_labels": short_labels,
            "agent_labels_full": agent_labels,
            "layer_names": layer_names,
            "metric": metric,
            "binarize": binarize,
            "threshold": threshold,
            "layer_sims": layer_sims,
        },
    )
    print(f"Saved heatmap data to {npy_path}")

    print(short_labels)
    _plot_heatmaps(layer_sims, short_labels, output_path, metric)


if __name__ == "__main__":
    main()
