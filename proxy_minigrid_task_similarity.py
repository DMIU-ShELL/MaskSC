#!/usr/bin/env python3
"""
Estimate MiniGrid task similarities without running PPO.

The script collects cheap state/action/reward buffers from fixed proxy policies,
then feeds them through the same Detect LWE embedding used by MaskSC.

Typical commands:

    # Online MaskSC-style embeddings plus full-dataset embeddings.
    python proxy_minigrid_task_similarity.py \
        --env-config env_configs/minigrid_object_remap_seed841.json \
        --out-dir proxy_similarity_mg_object_remap_seed841 \
        --policies random oracle \
        --random-episodes 24 \
        --oracle-episodes 8 \
        --online-batches 12 \
        --online-batch-size 512 \
        --plot \
        --family-stride 4

    # Very fast visual/state separability probe.
    python proxy_minigrid_task_similarity.py \
        --env-config env_configs/minigrid_object_remap_seed841.json \
        --out-dir proxy_similarity_static \
        --policies static \
        --static-samples 2048 \
        --online-batches 8 \
        --plot \
        --family-stride 4

Outputs:
    proxy_task_similarities_online.csv
        Training-style snapshots while each task embedding is updated online.

    proxy_task_similarities_full.csv
        Training-compatible prior-only similarities from one full-dataset
        embedding per task.

    proxy_pairwise_similarity_full.csv
        Full pairwise matrix records, including future-task comparisons.

    proxy_embeddings_online.npy / proxy_embeddings_full.npy
        Final normalized embedding arrays.

    proxy_online_update_comparison.csv
        Pair-level cosine-similarity trajectories for the Mask-SC EMA,
        unit-input EMA, and no-EMA updates. Written when
        --compare-online-updates is enabled.

    proxy_online_update_comparison.pdf / .png
        Same-family, cross-family, and separation summaries over online
        batches for the three update methods.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import deque
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import gym
import numpy as np
import ot
import torch
import torch.nn.functional as F
from gym_minigrid.minigrid import Door
from gym_minigrid.wrappers import ImgObsWrapper, ReseedWrapper

from deep_rl.detect_modules.detect import Detect


REDUCED_TO_MINIGRID_ACTION = {
    0: 0,  # turn left
    1: 1,  # turn right
    2: 2,  # move forward
    3: 5,  # toggle
}

DIR_TO_VEC = {
    0: (1, 0),   # east
    1: (0, 1),   # south
    2: (-1, 0),  # west
    3: (0, -1),  # north
}
VEC_TO_DIR = {vec: direction for direction, vec in DIR_TO_VEC.items()}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Estimate MiniGrid Detect/MaskSC task similarities from cheap proxy "
            "state-action-reward datasets."
        )
    )
    parser.add_argument("--env-config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["random", "oracle"],
        choices=["random", "oracle", "static"],
        help="Proxy data sources to concatenate for each task.",
    )
    parser.add_argument("--random-episodes", type=int, default=24)
    parser.add_argument("--oracle-episodes", type=int, default=8)
    parser.add_argument("--static-samples", type=int, default=2048)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Rollout cap per episode. Defaults to env.max_steps.",
    )
    parser.add_argument("--state-scale", type=float, default=0.1)
    parser.add_argument("--action-dim", type=int, default=None)
    parser.add_argument("--reference-num", type=int, default=50)
    parser.add_argument("--detect-num-samples", type=int, default=512)
    parser.add_argument("--online-batches", type=int, default=12)
    parser.add_argument("--online-batch-size", type=int, default=512)
    parser.add_argument("--online-ema", type=float, default=0.5)
    parser.add_argument(
        "--compare-online-updates",
        action="store_true",
        help=(
            "Compare the implemented Mask-SC EMA update against an EMA that "
            "normalizes each new embedding before averaging and a no-EMA "
            "baseline. Writes a detailed CSV and summary figure."
        ),
    )
    parser.add_argument(
        "--online-sampling",
        default="iid",
        choices=["iid", "sequential", "replay"],
        help=(
            "How online MaskSC-style batches are formed. `iid` samples from the "
            "entire proxy dataset, `sequential` reads chronological chunks, and "
            "`replay` appends chronological chunks to a task-local buffer before "
            "sampling from that buffer."
        ),
    )
    parser.add_argument(
        "--full-num-samples",
        type=int,
        default=0,
        help=(
            "Samples for full-dataset embeddings. Use 0 to use all collected "
            "proxy samples for each task."
        ),
    )
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument(
        "--similarity-metrics",
        nargs="+",
        default=["cosine"],
        choices=["cosine", "euclidean"],
        help=(
            "Embedding comparison metrics to compute. Cosine keeps the current "
            "MaskSC-style files. Euclidean writes additional *_euclidean.csv "
            "files with `similarity = -distance` plus a raw `distance` column."
        ),
    )
    parser.add_argument(
        "--euclidean-threshold",
        type=float,
        default=None,
        help=(
            "Optional maximum Euclidean distance for selected=1 in Euclidean "
            "diagnostic files. If omitted, Euclidean selected=1 marks the top-k "
            "nearest priors."
        ),
    )
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for Detect embedding computation.",
    )
    parser.add_argument(
        "--detect-normalized",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Detect's global state normalization.",
    )
    parser.add_argument(
        "--detect-one-hot-actions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="One-hot encode actions inside Detect, matching MaskSC defaults.",
    )
    parser.add_argument(
        "--save-sar",
        action="store_true",
        help="Save each collected SAR array to <out-dir>/sar/taskXXX.npy.",
    )
    parser.add_argument(
        "--separate-embeddings",
        action="store_true",
        help=(
            "Also compute separate state-only, action-only, and reward-only "
            "LWE embeddings and similarity CSVs."
        ),
    )
    parser.add_argument(
        "--component-average-similarity",
        action="store_true",
        help=(
            "When --separate-embeddings is enabled, also average selected "
            "component similarities and write/plot component-average CSVs."
        ),
    )
    parser.add_argument(
        "--component-average-components",
        nargs="+",
        default=None,
        choices=["state", "action", "reward"],
        help=(
            "Subset of separate components to average when "
            "--component-average-similarity is enabled. Defaults to all "
            "--separate-components."
        ),
    )
    parser.add_argument(
        "--separate-components",
        nargs="+",
        default=["state", "action", "reward"],
        choices=["state", "action", "reward"],
        help="Components to compute when --separate-embeddings is enabled.",
    )
    parser.add_argument(
        "--sliced-wasserstein-embeddings",
        action="store_true",
        help=(
            "Also compute Sliced Wasserstein Embedding (SWE) similarities. "
            "SWE projects task samples onto random directions, stores sorted "
            "projected quantiles, and compares the resulting embedding vectors."
        ),
    )
    parser.add_argument("--swe-num-projections", type=int, default=128)
    parser.add_argument("--swe-num-quantiles", type=int, default=128)
    parser.add_argument(
        "--swe-seed",
        type=int,
        default=98,
        help="Random projection seed for Sliced Wasserstein embeddings.",
    )
    parser.add_argument(
        "--swe-normalize-embedding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "L2-normalize SWE vectors before similarity computation. Disable "
            "this if you want Euclidean distance to be closer to raw sliced "
            "Wasserstein distance."
        ),
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Run plot_task_similarity_summary.py for online and full CSVs.",
    )
    parser.add_argument(
        "--family-stride",
        type=int,
        default=4,
        help="Passed through to plot_task_similarity_summary.py when --plot is set.",
    )
    return parser.parse_args()


def load_env_config(path):
    with path.open() as fh:
        config = json.load(fh)
    tasks = config["tasks"]
    seeds = config.get("seeds", 0)
    if isinstance(seeds, int):
        seeds = [seeds] * len(tasks)
    if len(seeds) != len(tasks):
        raise ValueError("env config `seeds` length must match `tasks` length")
    action_dim = int(config.get("action_dim", 4))
    return config, tasks, seeds, action_dim


def maybe_import_curriculum_envs():
    try:
        import CurriculumMinigrid  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Could not import CurriculumMinigrid; run this script from the "
            "mask-lrl-cluster-optimization repo root or set PYTHONPATH."
        ) from exc


def reset_env(env):
    ret = env.reset()
    if isinstance(ret, tuple) and len(ret) == 2:
        return ret[0]
    return ret


def step_env(env, reduced_action):
    env_action = REDUCED_TO_MINIGRID_ACTION[int(reduced_action)]
    ret = env.step(env_action)
    if isinstance(ret, tuple) and len(ret) == 5:
        obs, reward, terminated, truncated, info = ret
        return obs, reward, bool(terminated or truncated), info
    return ret


def make_env(env_id, seed):
    env = ImgObsWrapper(gym.make(env_id))
    env = ReseedWrapper(env, seeds=[int(seed)])
    return env


def flatten_state(obs, state_scale):
    return np.asarray(obs, dtype=np.float32).reshape(-1) * float(state_scale)


def append_sar(rows, obs, action, reward, state_scale):
    rows.append(
        np.concatenate(
            [
                flatten_state(obs, state_scale),
                np.asarray([action], dtype=np.float32),
                np.asarray([reward], dtype=np.float32),
            ]
        )
    )


def max_steps_for(env, override):
    if override is not None:
        return int(override)
    base = env.unwrapped
    return int(getattr(base, "max_steps", 200))


def collect_random_sar(env, episodes, max_steps, action_dim, state_scale, rng):
    rows = []
    for _ in range(int(episodes)):
        obs = reset_env(env)
        for _step in range(max_steps):
            action = int(rng.integers(action_dim))
            next_obs, reward, done, _info = step_env(env, action)
            append_sar(rows, obs, action, reward, state_scale)
            obs = reset_env(env) if done else next_obs
            if done:
                break
    return rows


def is_door(obj):
    return isinstance(obj, Door) or getattr(obj, "type", None) == "door"


def traversable_for_plan(obj):
    if obj is None:
        return True
    if is_door(obj):
        return True
    return bool(obj.can_overlap())


def shortest_position_path(base, start, goal):
    start = tuple(start)
    goal = tuple(goal)
    queue = deque([start])
    parent = {start: None}

    while queue:
        pos = queue.popleft()
        if pos == goal:
            break
        for dx, dy in DIR_TO_VEC.values():
            nxt = (pos[0] + dx, pos[1] + dy)
            if nxt in parent:
                continue
            x, y = nxt
            if x < 0 or y < 0 or x >= base.width or y >= base.height:
                continue
            if not traversable_for_plan(base.grid.get(x, y)):
                continue
            parent[nxt] = pos
            queue.append(nxt)

    if goal not in parent:
        return None

    path = []
    pos = goal
    while pos is not None:
        path.append(pos)
        pos = parent[pos]
    return list(reversed(path))


def turn_actions(current_dir, target_dir):
    delta = (target_dir - current_dir) % 4
    if delta == 0:
        return []
    if delta == 1:
        return [1]
    if delta == 2:
        return [1, 1]
    return [0]


def plan_actions_to_goal(base):
    path = shortest_position_path(base, base.agent_pos, base.goal_pos)
    if path is None or len(path) < 2:
        return []

    actions = []
    sim_dir = int(base.agent_dir)

    for cur, nxt in zip(path[:-1], path[1:]):
        step_vec = (nxt[0] - cur[0], nxt[1] - cur[1])
        target_dir = VEC_TO_DIR[step_vec]
        turns = turn_actions(sim_dir, target_dir)
        actions.extend(turns)
        if turns:
            sim_dir = target_dir

        obj = base.grid.get(*nxt)
        if is_door(obj) and not getattr(obj, "is_open", False):
            actions.append(3)
        actions.append(2)

    return actions


def collect_oracle_sar(env, episodes, max_steps, action_dim, state_scale, rng):
    rows = []
    for _ in range(int(episodes)):
        obs = reset_env(env)
        actions = plan_actions_to_goal(env.unwrapped)
        if not actions:
            actions = [int(rng.integers(action_dim)) for _ in range(max_steps)]

        steps = 0
        for action in actions:
            if steps >= max_steps:
                break
            next_obs, reward, done, _info = step_env(env, action)
            append_sar(rows, obs, action, reward, state_scale)
            steps += 1
            obs = next_obs
            if done:
                break
    return rows


def passable_room_positions(base):
    positions = []
    rooms = getattr(base, "rooms", [])
    for room in rooms:
        top_x, top_y = room.top
        size_x, size_y = room.size
        for x in range(top_x + 1, top_x + size_x - 1):
            for y in range(top_y + 1, top_y + size_y - 1):
                obj = base.grid.get(x, y)
                if obj is None or obj.can_overlap():
                    positions.append((x, y))
    if positions:
        return positions

    for x in range(base.width):
        for y in range(base.height):
            obj = base.grid.get(x, y)
            if obj is None or obj.can_overlap():
                positions.append((x, y))
    return positions


def current_symbolic_obs(base):
    obs = base.gen_obs()
    if isinstance(obs, dict):
        return obs["image"]
    return obs


def collect_static_sar(env, samples, action_dim, state_scale, rng):
    rows = []
    reset_env(env)
    base = env.unwrapped
    positions = passable_room_positions(base)
    if not positions:
        raise RuntimeError("No passable cells found for static observation sampling")

    for _ in range(int(samples)):
        pos = positions[int(rng.integers(len(positions)))]
        base.agent_pos = pos
        base.agent_dir = int(rng.integers(4))
        action = int(rng.integers(action_dim))
        obs = current_symbolic_obs(base)
        append_sar(rows, obs, action, 0.0, state_scale)

    return rows


def collect_task_sar(env_id, seed, args, action_dim, rng):
    env = make_env(env_id, seed)
    max_steps = max_steps_for(env, args.max_steps)

    rows = []
    if "random" in args.policies:
        rows.extend(
            collect_random_sar(
                env,
                args.random_episodes,
                max_steps,
                action_dim,
                args.state_scale,
                rng,
            )
        )
    if "oracle" in args.policies:
        rows.extend(
            collect_oracle_sar(
                env,
                args.oracle_episodes,
                max_steps,
                action_dim,
                args.state_scale,
                rng,
            )
        )
    if "static" in args.policies:
        rows.extend(
            collect_static_sar(
                env,
                args.static_samples,
                action_dim,
                args.state_scale,
                rng,
            )
        )

    env.close()
    if not rows:
        raise RuntimeError(f"No proxy SAR rows collected for {env_id}")
    return np.stack(rows).astype(np.float32)


def make_detect(input_dim, action_dim, args, num_samples):
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    detect = Detect(
        args.reference_num,
        input_dim,
        action_dim,
        num_samples,
        one_hot=args.detect_one_hot_actions,
        normalized=args.detect_normalized,
        device=device,
    )
    detect.set_reference(input_dim, args.reference_num, action_dim)
    return detect


def resolve_device(args):
    if args.device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return args.device


def compute_embedding(detect, sar, action_dim, num_samples, normalize=True):
    detect.set_num_samples(num_samples)
    tensor = torch.as_tensor(sar, dtype=torch.float32)
    with torch.no_grad():
        emb = detect.lwe(tensor, action_dim)
        if normalize:
            emb = F.normalize(emb, dim=0, eps=1e-8)
    return emb.detach().cpu()


class RawLWEEmbedder:
    """LWE embedding for an already-built feature matrix."""

    def __init__(self, feature_dim, reference_num, device):
        self.feature_dim = int(feature_dim)
        self.reference_num = int(reference_num)
        self.device = device

        torch.manual_seed(98)
        self.ref = torch.rand(
            self.reference_num,
            self.feature_dim,
            device=self.device,
            dtype=torch.float32,
        )

    @torch.no_grad()
    def embed(self, features, num_samples=None, normalized=False):
        if not isinstance(features, torch.Tensor):
            features = torch.as_tensor(features)

        if num_samples is not None and features.shape[0] > num_samples:
            idx = torch.randperm(features.shape[0])[:num_samples]
            features = features.index_select(0, idx)

        features = features.to(self.device, dtype=torch.float32)

        if normalized:
            mean = features.mean()
            std = features.std().clamp_min(1e-8)
            features = (features - mean) / std

        ref_size = self.ref.shape[0]
        cost = ot.dist(features, self.ref, p=2)
        a = torch.full(
            (features.shape[0],),
            1.0 / features.shape[0],
            device=features.device,
            dtype=features.dtype,
        )
        b = torch.full(
            (ref_size,),
            1.0 / ref_size,
            device=self.ref.device,
            dtype=self.ref.dtype,
        )
        gamma = ot.emd(a, b, cost, numItermax=700_000)
        emb = (ref_size * gamma).T @ features
        emb = (emb - self.ref) / (ref_size ** 0.5)
        emb = F.normalize(emb.reshape(-1), dim=0, eps=1e-8)
        return emb.detach().cpu()


class SlicedWassersteinEmbedder:
    """Random-projection quantile embedding for empirical distributions."""

    def __init__(
        self,
        feature_dim,
        num_projections,
        num_quantiles,
        seed,
        device,
        normalize_embedding=True,
    ):
        self.feature_dim = int(feature_dim)
        self.num_projections = int(num_projections)
        self.num_quantiles = int(num_quantiles)
        self.device = device
        self.normalize_embedding = bool(normalize_embedding)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        directions = torch.randn(
            self.num_projections,
            self.feature_dim,
            generator=generator,
            dtype=torch.float32,
        )
        directions = F.normalize(directions, dim=1, eps=1e-8)
        self.directions = directions.to(self.device)

    @torch.no_grad()
    def embed(self, features, num_samples=None, normalized=False):
        if not isinstance(features, torch.Tensor):
            features = torch.as_tensor(features)

        if num_samples is not None and features.shape[0] > num_samples:
            idx = torch.randperm(features.shape[0])[:num_samples]
            features = features.index_select(0, idx)

        features = features.to(self.device, dtype=torch.float32)
        if normalized:
            mean = features.mean(dim=0, keepdim=True)
            std = features.std(dim=0, keepdim=True).clamp_min(1e-8)
            features = (features - mean) / std

        projections = features @ self.directions.T
        projections = torch.sort(projections, dim=0).values

        if projections.shape[0] != self.num_quantiles:
            projection_channels = projections.T.unsqueeze(0)
            projections = F.interpolate(
                projection_channels,
                size=self.num_quantiles,
                mode="linear",
                align_corners=True,
            ).squeeze(0).T

        emb = projections.T.reshape(-1)
        emb = emb / max(self.num_projections * self.num_quantiles, 1) ** 0.5
        if self.normalize_embedding:
            emb = F.normalize(emb, dim=0, eps=1e-8)
        return emb.detach().cpu()


def combined_sar_features(sar, input_dim, action_dim, one_hot_actions=True):
    state = sar[:, :input_dim].astype(np.float32)
    action_feature, _ = component_features(
        sar,
        input_dim,
        action_dim,
        "action",
        one_hot_actions=one_hot_actions,
    )
    reward = sar[:, input_dim + 1 : input_dim + 2].astype(np.float32)
    return np.concatenate([state, action_feature, reward], axis=1)


def component_features(sar, input_dim, action_dim, component, one_hot_actions=True):
    if component == "state":
        return sar[:, :input_dim], True

    if component == "action":
        actions = sar[:, input_dim].astype(np.int64)
        if one_hot_actions:
            return np.eye(action_dim, dtype=np.float32)[actions], False
        return actions.reshape(-1, 1).astype(np.float32), False

    if component == "reward":
        return sar[:, input_dim + 1 : input_dim + 2].astype(np.float32), False

    raise ValueError(f"unknown component: {component}")


def sample_batch(sar, batch_size, rng):
    replace = sar.shape[0] < batch_size
    idx = rng.choice(sar.shape[0], size=batch_size, replace=replace)
    return sar[idx]


def sequential_batch(sar, cursor, batch_size):
    if sar.shape[0] == 0:
        raise ValueError("cannot sample from an empty SAR array")

    idx = (np.arange(batch_size) + cursor) % sar.shape[0]
    cursor = int((cursor + batch_size) % sar.shape[0])
    return sar[idx], cursor


def online_batch(sar, batch_size, rng, sampling, state):
    if sampling == "iid":
        return sample_batch(sar, batch_size, rng), state

    if sampling == "sequential":
        cursor = int(state.get("cursor", 0))
        batch, cursor = sequential_batch(sar, cursor, batch_size)
        state["cursor"] = cursor
        return batch, state

    if sampling == "replay":
        cursor = int(state.get("cursor", 0))
        replay = state.get("replay")
        if replay is None:
            replay = []
            state["replay"] = replay
        chunk, cursor = sequential_batch(sar, cursor, batch_size)
        replay.append(chunk)
        state["cursor"] = cursor
        replay_array = np.concatenate(replay, axis=0)
        return sample_batch(replay_array, batch_size, rng), state

    raise ValueError(f"unknown online sampling mode: {sampling}")


def cosine(a, b):
    return float(torch.dot(a, b).item())


def metric_record(idx, a, b, metric):
    if metric == "cosine":
        return {
            "idx": idx,
            "similarity": cosine(a, b),
            "distance": np.nan,
            "metric": metric,
        }
    if metric == "euclidean":
        distance = float(torch.linalg.vector_norm(a - b).item())
        return {
            "idx": idx,
            "similarity": -distance,
            "distance": distance,
            "metric": metric,
        }
    raise ValueError(f"unknown similarity metric: {metric}")


def select_from_metric_records(records, args, metric):
    if metric == "cosine":
        passing = [
            record
            for record in records
            if record["similarity"] > args.threshold
        ]
        passing.sort(key=lambda record: record["similarity"], reverse=True)
    elif metric == "euclidean":
        passing = [
            record
            for record in records
            if (
                args.euclidean_threshold is None
                or record["distance"] <= args.euclidean_threshold
            )
        ]
        passing.sort(key=lambda record: record["distance"])
    else:
        raise ValueError(f"unknown similarity metric: {metric}")

    if args.topk is not None and args.topk > 0:
        passing = passing[: args.topk]
    return {record["idx"] for record in passing}


def write_similarity_rows(path, rows):
    fieldnames = [
        "learn_block",
        "task_idx",
        "iteration",
        "total_steps",
        "prev_idx",
        "similarity",
        "selected",
        "prior_perf",
        "current_perf",
        "eligible",
        "mode",
        "metric",
        "distance",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


ONLINE_UPDATE_LABELS = {
    "masksc": "Mask-SC EMA",
    "unit_input_ema": "Unit-input EMA",
    "no_ema": "No EMA",
}


def update_online_embedding(current, new_emb, ema, update_mode):
    """Apply one online embedding update and return a unit-length vector."""
    if update_mode not in ONLINE_UPDATE_LABELS:
        raise ValueError(f"unknown online embedding update mode: {update_mode}")

    new_emb = new_emb.detach().float().cpu()
    if update_mode == "no_ema" or current is None:
        updated = new_emb
    elif update_mode == "masksc":
        # Matches PPOAgent._update_embedding: average the raw WTE with the
        # previously normalized stored embedding, then normalize the result.
        updated = ema * current + (1.0 - ema) * new_emb
    else:
        # Direction-only alternative: remove the new WTE magnitude before
        # applying the EMA.
        new_unit = F.normalize(new_emb, dim=0, eps=1e-8)
        updated = ema * current + (1.0 - ema) * new_unit

    return F.normalize(updated, dim=0, eps=1e-8)


def online_embeddings_and_rows(
    task_sars,
    embed_batch_fn,
    args,
    rng,
    mode="online",
    metric="cosine",
    update_mode="masksc",
):
    embeddings = [None] * len(task_sars)
    rows = []
    iteration = 0
    total_steps = 0

    for task_idx, sar in enumerate(task_sars):
        current = None
        online_state = {}
        for _batch_idx in range(args.online_batches):
            iteration += 1
            batch, online_state = online_batch(
                sar,
                args.online_batch_size,
                rng,
                args.online_sampling,
                online_state,
            )
            total_steps += batch.shape[0]

            new_emb = embed_batch_fn(batch)
            current = update_online_embedding(
                current,
                new_emb,
                args.online_ema,
                update_mode,
            )
            embeddings[task_idx] = current

            if task_idx == 0:
                continue

            sim_records = []
            for prev_idx in range(task_idx):
                if embeddings[prev_idx] is None:
                    continue
                sim_records.append(
                    metric_record(prev_idx, current, embeddings[prev_idx], metric)
                )
            selected = select_from_metric_records(sim_records, args, metric)

            for record in sim_records:
                rows.append(
                    {
                        "learn_block": 0,
                        "task_idx": task_idx,
                        "iteration": iteration,
                        "total_steps": total_steps,
                        "prev_idx": record["idx"],
                        "similarity": f"{record['similarity']:.6f}",
                        "selected": int(record["idx"] in selected),
                        "prior_perf": "nan",
                        "current_perf": "nan",
                        "eligible": 1,
                        "mode": mode,
                        "metric": metric,
                        "distance": (
                            "nan"
                            if not np.isfinite(record["distance"])
                            else f"{record['distance']:.6f}"
                        ),
                    }
                )

    return embeddings, rows


def compare_online_embedding_updates(task_sars, embed_batch_fn, args, rng):
    """Evaluate all update rules on identical online batches and raw WTEs."""
    update_modes = tuple(ONLINE_UPDATE_LABELS)
    embeddings = {
        update_mode: [None] * len(task_sars)
        for update_mode in update_modes
    }
    rows = []
    iteration = 0
    total_steps = 0

    for task_idx, sar in enumerate(task_sars):
        current = {update_mode: None for update_mode in update_modes}
        online_state = {}

        for task_batch in range(1, args.online_batches + 1):
            iteration += 1
            batch, online_state = online_batch(
                sar,
                args.online_batch_size,
                rng,
                args.online_sampling,
                online_state,
            )
            total_steps += batch.shape[0]
            raw_emb = embed_batch_fn(batch)

            for update_mode in update_modes:
                current[update_mode] = update_online_embedding(
                    current[update_mode],
                    raw_emb,
                    args.online_ema,
                    update_mode,
                )
                embeddings[update_mode][task_idx] = current[update_mode]

                for prev_idx in range(task_idx):
                    prior = embeddings[update_mode][prev_idx]
                    if prior is None:
                        continue
                    rows.append(
                        {
                            "task_idx": task_idx,
                            "task_batch": task_batch,
                            "iteration": iteration,
                            "total_steps": total_steps,
                            "prev_idx": prev_idx,
                            "similarity": cosine(current[update_mode], prior),
                            "update_mode": update_mode,
                            "update_label": ONLINE_UPDATE_LABELS[update_mode],
                            "same_family": int(
                                task_idx % args.family_stride
                                == prev_idx % args.family_stride
                            ),
                        }
                    )

    return embeddings, rows


def write_online_update_comparison(path, rows):
    fieldnames = [
        "task_idx",
        "task_batch",
        "iteration",
        "total_steps",
        "prev_idx",
        "similarity",
        "update_mode",
        "update_label",
        "same_family",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def comparison_rows_to_similarity_rows(rows, args, update_mode="masksc"):
    """Convert one comparison trajectory to the standard online CSV schema."""
    grouped = {}
    for row in rows:
        if row["update_mode"] != update_mode:
            continue
        key = (
            int(row["task_idx"]),
            int(row["iteration"]),
            int(row["total_steps"]),
        )
        grouped.setdefault(key, []).append(row)

    standard_rows = []
    for (task_idx, iteration, total_steps), snapshot in sorted(grouped.items()):
        records = [
            {
                "idx": int(row["prev_idx"]),
                "similarity": float(row["similarity"]),
                "distance": np.nan,
                "metric": "cosine",
            }
            for row in snapshot
        ]
        selected = select_from_metric_records(records, args, "cosine")
        for record in records:
            standard_rows.append(
                {
                    "learn_block": 0,
                    "task_idx": task_idx,
                    "iteration": iteration,
                    "total_steps": total_steps,
                    "prev_idx": record["idx"],
                    "similarity": f"{record['similarity']:.6f}",
                    "selected": int(record["idx"] in selected),
                    "prior_perf": "nan",
                    "current_perf": "nan",
                    "eligible": 1,
                    "mode": "online",
                    "metric": "cosine",
                    "distance": "nan",
                }
            )
    return standard_rows


def _mean_and_sem(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.nan, np.nan
    mean = float(values.mean())
    if values.size == 1:
        return mean, 0.0
    return mean, float(values.std(ddof=1) / np.sqrt(values.size))


def plot_online_update_comparison(rows, out_dir, args):
    import matplotlib.pyplot as plt

    if not rows:
        print("Skipping online-update comparison plot: no prior-task pairs.")
        return

    grouped = {}
    for row in rows:
        key = (
            row["update_mode"],
            int(row["task_batch"]),
            int(row["same_family"]),
        )
        grouped.setdefault(key, []).append(float(row["similarity"]))

    batches = np.arange(1, args.online_batches + 1)
    colors = {
        "masksc": "#1f77b4",
        "unit_input_ema": "#d62728",
        "no_ema": "#2ca02c",
    }
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), sharex=True)

    for update_mode, label in ONLINE_UPDATE_LABELS.items():
        relation_stats = {}
        for same_family in (1, 0):
            means = []
            sems = []
            for task_batch in batches:
                mean, sem = _mean_and_sem(
                    grouped.get((update_mode, int(task_batch), same_family), [])
                )
                means.append(mean)
                sems.append(sem)
            relation_stats[same_family] = (
                np.asarray(means, dtype=float),
                np.asarray(sems, dtype=float),
            )

        for axis, same_family in zip(axes[:2], (1, 0)):
            means, sems = relation_stats[same_family]
            axis.plot(
                batches,
                means,
                marker="o",
                markersize=3,
                linewidth=1.8,
                color=colors[update_mode],
                label=label,
            )
            axis.fill_between(
                batches,
                means - sems,
                means + sems,
                color=colors[update_mode],
                alpha=0.15,
                linewidth=0,
            )

        same_means, _ = relation_stats[1]
        cross_means, _ = relation_stats[0]
        axes[2].plot(
            batches,
            same_means - cross_means,
            marker="o",
            markersize=3,
            linewidth=1.8,
            color=colors[update_mode],
            label=label,
        )

    axes[0].set_title("Same-family priors")
    axes[1].set_title("Cross-family priors")
    axes[2].set_title("Similarity separation")
    axes[0].set_ylabel("Mean cosine similarity")
    axes[2].set_ylabel("Same-family minus cross-family")
    for axis in axes:
        axis.set_xlabel("Online embedding batch within task")
        axis.grid(alpha=0.25)
        axis.set_xticks(batches)
    axes[0].axhline(
        args.threshold,
        color="black",
        linestyle="--",
        linewidth=1.0,
        alpha=0.6,
        label=rf"Threshold $\theta={args.threshold:g}$",
    )
    axes[1].axhline(
        args.threshold,
        color="black",
        linestyle="--",
        linewidth=1.0,
        alpha=0.6,
    )
    axes[2].axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "Online task-similarity trajectories by embedding update "
        rf"($\gamma_{{\mathrm{{ema}}}}={args.online_ema:g}$)"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "proxy_online_update_comparison.pdf", bbox_inches="tight")
    fig.savefig(
        out_dir / "proxy_online_update_comparison.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def full_embeddings_and_rows(task_sars, embed_full_fn, args, mode="full", metric="cosine"):
    embeddings = [embed_full_fn(sar) for sar in task_sars]

    prior_rows = []
    pairwise_rows = []
    for task_idx, current in enumerate(embeddings):
        sim_records = [
            metric_record(prev_idx, current, embeddings[prev_idx], metric)
            for prev_idx in range(task_idx)
        ]
        selected = select_from_metric_records(sim_records, args, metric)

        for record in sim_records:
            prior_rows.append(
                {
                    "learn_block": 0,
                    "task_idx": task_idx,
                    "iteration": 1,
                    "total_steps": int(sum(sar.shape[0] for sar in task_sars)),
                    "prev_idx": record["idx"],
                    "similarity": f"{record['similarity']:.6f}",
                    "selected": int(record["idx"] in selected),
                    "prior_perf": "nan",
                    "current_perf": "nan",
                    "eligible": 1,
                    "mode": mode,
                    "metric": metric,
                    "distance": (
                        "nan"
                        if not np.isfinite(record["distance"])
                        else f"{record['distance']:.6f}"
                    ),
                }
            )

        for other_idx, other in enumerate(embeddings):
            if other_idx == task_idx:
                continue
            record = metric_record(other_idx, current, other, metric)
            pairwise_rows.append(
                {
                    "task_idx": task_idx,
                    "prev_idx": other_idx,
                    "similarity": f"{record['similarity']:.6f}",
                    "mode": f"{mode}_pairwise",
                    "metric": metric,
                    "distance": (
                        "nan"
                        if not np.isfinite(record["distance"])
                        else f"{record['distance']:.6f}"
                    ),
                }
            )

    return embeddings, prior_rows, pairwise_rows


def _row_float(row, key):
    value = row.get(key, "nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def average_component_similarity_rows(component_rows, args, metric, mode):
    if not component_rows:
        return []

    values_by_key = {}
    for rows in component_rows:
        for row in rows:
            key = (
                int(row["task_idx"]),
                int(row["iteration"]),
                int(row["total_steps"]),
                int(row["prev_idx"]),
            )
            values_by_key.setdefault(key, []).append(
                (_row_float(row, "similarity"), _row_float(row, "distance"))
            )

    grouped_records = {}
    for key, values in values_by_key.items():
        task_idx, iteration, total_steps, prev_idx = key
        sims = np.asarray([value[0] for value in values], dtype=float)
        dists = np.asarray([value[1] for value in values], dtype=float)
        avg_similarity = float(np.nanmean(sims))
        avg_distance = float(np.nanmean(dists)) if np.isfinite(dists).any() else np.nan
        grouped_records.setdefault((task_idx, iteration, total_steps), []).append(
            {
                "idx": prev_idx,
                "similarity": avg_similarity,
                "distance": avg_distance,
                "metric": metric,
            }
        )

    out_rows = []
    for group_key in sorted(grouped_records):
        task_idx, iteration, total_steps = group_key
        records = sorted(grouped_records[group_key], key=lambda record: record["idx"])
        selected = select_from_metric_records(records, args, metric)
        for record in records:
            out_rows.append(
                {
                    "learn_block": 0,
                    "task_idx": task_idx,
                    "iteration": iteration,
                    "total_steps": total_steps,
                    "prev_idx": record["idx"],
                    "similarity": f"{record['similarity']:.6f}",
                    "selected": int(record["idx"] in selected),
                    "prior_perf": "nan",
                    "current_perf": "nan",
                    "eligible": 1,
                    "mode": mode,
                    "metric": metric,
                    "distance": (
                        "nan"
                        if not np.isfinite(record["distance"])
                        else f"{record['distance']:.6f}"
                    ),
                }
            )
    return out_rows


def average_component_pairwise_rows(component_pairwise_rows, metric, mode):
    if not component_pairwise_rows:
        return []

    values_by_key = {}
    for rows in component_pairwise_rows:
        for row in rows:
            key = (int(row["task_idx"]), int(row["prev_idx"]))
            values_by_key.setdefault(key, []).append(
                (_row_float(row, "similarity"), _row_float(row, "distance"))
            )

    out_rows = []
    for task_idx, prev_idx in sorted(values_by_key):
        values = values_by_key[(task_idx, prev_idx)]
        sims = np.asarray([value[0] for value in values], dtype=float)
        dists = np.asarray([value[1] for value in values], dtype=float)
        avg_similarity = float(np.nanmean(sims))
        avg_distance = float(np.nanmean(dists)) if np.isfinite(dists).any() else np.nan
        out_rows.append(
            {
                "task_idx": task_idx,
                "prev_idx": prev_idx,
                "similarity": f"{avg_similarity:.6f}",
                "mode": mode,
                "metric": metric,
                "distance": (
                    "nan" if not np.isfinite(avg_distance) else f"{avg_distance:.6f}"
                ),
            }
        )
    return out_rows


def write_pairwise_rows(path, rows):
    fieldnames = ["task_idx", "prev_idx", "similarity", "mode", "metric", "distance"]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_embeddings(path, embeddings):
    array = np.stack([emb.numpy() for emb in embeddings]).astype(np.float32)
    np.save(path, array)


def metric_suffix(metric):
    return "" if metric == "cosine" else f"_{metric}"


def run_plotter(csv_path, env_config, out_dir, family_stride):
    cmd = [
        sys.executable,
        "plot_task_similarity_summary.py",
        str(csv_path),
        "--env-config",
        str(env_config),
        "--out-dir",
        str(out_dir),
        "--family-stride",
        str(family_stride),
    ]
    subprocess.run(cmd, check=True)


def component_embedding_fns(component, input_dim, action_dim, args):
    sample_sar = np.zeros((1, input_dim + 2), dtype=np.float32)
    sample_features, normalize_component = component_features(
        sample_sar,
        input_dim,
        action_dim,
        component,
        one_hot_actions=args.detect_one_hot_actions,
    )
    embedder = RawLWEEmbedder(
        feature_dim=sample_features.shape[1],
        reference_num=args.reference_num,
        device=resolve_device(args),
    )
    full_samples = None if args.full_num_samples == 0 else int(args.full_num_samples)

    def embed_batch(sar):
        features, _ = component_features(
            sar,
            input_dim,
            action_dim,
            component,
            one_hot_actions=args.detect_one_hot_actions,
        )
        return embedder.embed(
            features,
            num_samples=args.online_batch_size,
            normalized=args.detect_normalized and normalize_component,
        )

    def embed_full(sar):
        features, _ = component_features(
            sar,
            input_dim,
            action_dim,
            component,
            one_hot_actions=args.detect_one_hot_actions,
        )
        return embedder.embed(
            features,
            num_samples=full_samples,
            normalized=args.detect_normalized and normalize_component,
        )

    return embed_batch, embed_full


def sliced_wasserstein_embedding_fns(input_dim, action_dim, args, component=None):
    sample_sar = np.zeros((1, input_dim + 2), dtype=np.float32)
    if component is None:
        sample_features = combined_sar_features(
            sample_sar,
            input_dim,
            action_dim,
            one_hot_actions=args.detect_one_hot_actions,
        )
        normalize_features = True
    else:
        sample_features, normalize_features = component_features(
            sample_sar,
            input_dim,
            action_dim,
            component,
            one_hot_actions=args.detect_one_hot_actions,
        )

    embedder = SlicedWassersteinEmbedder(
        feature_dim=sample_features.shape[1],
        num_projections=args.swe_num_projections,
        num_quantiles=args.swe_num_quantiles,
        seed=args.swe_seed,
        device=resolve_device(args),
        normalize_embedding=args.swe_normalize_embedding,
    )
    full_samples = None if args.full_num_samples == 0 else int(args.full_num_samples)

    def features_from_sar(sar):
        if component is None:
            return combined_sar_features(
                sar,
                input_dim,
                action_dim,
                one_hot_actions=args.detect_one_hot_actions,
            )
        features, _ = component_features(
            sar,
            input_dim,
            action_dim,
            component,
            one_hot_actions=args.detect_one_hot_actions,
        )
        return features

    def embed_batch(sar):
        return embedder.embed(
            features_from_sar(sar),
            num_samples=args.online_batch_size,
            normalized=args.detect_normalized and normalize_features,
        )

    def embed_full(sar):
        return embedder.embed(
            features_from_sar(sar),
            num_samples=full_samples,
            normalized=args.detect_normalized and normalize_features,
        )

    return embed_batch, embed_full


def main():
    args = parse_args()
    if not 0.0 <= args.online_ema <= 1.0:
        raise ValueError("--online-ema must be in [0, 1]")
    if args.family_stride <= 0:
        raise ValueError("--family-stride must be positive")
    if args.component_average_similarity and not args.separate_embeddings:
        raise ValueError(
            "--component-average-similarity requires --separate-embeddings"
        )
    component_average_components = (
        list(args.component_average_components)
        if args.component_average_components is not None
        else list(args.separate_components)
    )
    missing_average_components = sorted(
        set(component_average_components) - set(args.separate_components)
    )
    if missing_average_components:
        raise ValueError(
            "--component-average-components must be a subset of "
            "--separate-components. Missing from --separate-components: "
            f"{missing_average_components}"
        )
    maybe_import_curriculum_envs()

    env_config, tasks, seeds, config_action_dim = load_env_config(args.env_config)
    action_dim = int(args.action_dim or config_action_dim)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    copied_config_path = out_dir / "env_config.json"
    with copied_config_path.open("w") as fh:
        json.dump(env_config, fh, indent=4)
        fh.write("\n")
    with (out_dir / "proxy_args.json").open("w") as fh:
        args_dict = vars(args).copy()
        args_dict["env_config"] = str(args.env_config)
        args_dict["out_dir"] = str(args.out_dir)
        json.dump(args_dict, fh, indent=4)
        fh.write("\n")

    task_sars = []
    summary_rows = []
    sar_dir = out_dir / "sar"
    if args.save_sar:
        sar_dir.mkdir(exist_ok=True)

    print("Collecting proxy SAR datasets...")
    for task_idx, (env_id, env_seed) in enumerate(zip(tasks, seeds)):
        task_rng = np.random.default_rng(args.seed + 1009 * task_idx)
        sar = collect_task_sar(env_id, env_seed, args, action_dim, task_rng)
        task_sars.append(sar)
        summary_rows.append(
            {
                "task_idx": task_idx,
                "task": env_id,
                "seed": env_seed,
                "samples": sar.shape[0],
                "state_dim": sar.shape[1] - 2,
                "reward_mean": float(sar[:, -1].mean()),
                "reward_nonzero": int(np.count_nonzero(sar[:, -1])),
            }
        )
        if args.save_sar:
            np.save(sar_dir / f"task{task_idx:03d}.npy", sar)
        print(
            f"  task{task_idx:02d}: samples={sar.shape[0]} "
            f"reward_nonzero={np.count_nonzero(sar[:, -1])} {env_id}"
        )

    with (out_dir / "proxy_sar_summary.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "task_idx",
                "task",
                "seed",
                "samples",
                "state_dim",
                "reward_mean",
                "reward_nonzero",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    input_dim = task_sars[0].shape[1] - 2
    detect = make_detect(input_dim, action_dim, args, args.detect_num_samples)
    full_samples = None if args.full_num_samples == 0 else int(args.full_num_samples)
    combined_embed_batch = lambda batch: compute_embedding(
        detect,
        batch,
        action_dim,
        num_samples=args.online_batch_size,
        normalize=False,
    )
    combined_embed_full = lambda sar: compute_embedding(
        detect,
        sar,
        action_dim,
        num_samples=full_samples,
        normalize=True,
    )

    online_update_embeddings = None
    online_update_rows = None
    if args.compare_online_updates:
        print("Comparing online embedding update methods on shared batches...")
        online_update_embeddings, online_update_rows = compare_online_embedding_updates(
            task_sars,
            combined_embed_batch,
            args,
            np.random.default_rng(args.seed),
        )
        write_online_update_comparison(
            out_dir / "proxy_online_update_comparison.csv",
            online_update_rows,
        )
        for update_mode, embeddings in online_update_embeddings.items():
            save_embeddings(
                out_dir / f"proxy_embeddings_online_{update_mode}.npy",
                embeddings,
            )
        plot_online_update_comparison(online_update_rows, out_dir, args)

    for metric in args.similarity_metrics:
        suffix = metric_suffix(metric)
        if (
            metric == "cosine"
            and online_update_embeddings is not None
            and online_update_rows is not None
        ):
            print("Reusing Mask-SC trajectory from online-update comparison (cosine)...")
            online_embeddings = online_update_embeddings["masksc"]
            online_rows = comparison_rows_to_similarity_rows(
                online_update_rows,
                args,
                update_mode="masksc",
            )
        else:
            print(f"Computing online MaskSC-style embeddings ({metric})...")
            online_embeddings, online_rows = online_embeddings_and_rows(
                task_sars,
                combined_embed_batch,
                args,
                np.random.default_rng(args.seed),
                mode="online",
                metric=metric,
                update_mode="masksc",
            )
        online_csv = out_dir / f"proxy_task_similarities_online{suffix}.csv"
        write_similarity_rows(online_csv, online_rows)
        save_embeddings(
            out_dir / f"proxy_embeddings_online{suffix}.npy", online_embeddings
        )

        print(f"Computing full-dataset embeddings ({metric})...")
        full_embeddings, full_rows, pairwise_rows = full_embeddings_and_rows(
            task_sars, combined_embed_full, args, mode="full", metric=metric
        )
        full_csv = out_dir / f"proxy_task_similarities_full{suffix}.csv"
        pairwise_csv = out_dir / f"proxy_pairwise_similarity_full{suffix}.csv"
        write_similarity_rows(full_csv, full_rows)
        write_pairwise_rows(pairwise_csv, pairwise_rows)
        save_embeddings(out_dir / f"proxy_embeddings_full{suffix}.npy", full_embeddings)

        if args.plot:
            print(f"Writing summary plots ({metric})...")
            run_plotter(
                online_csv,
                copied_config_path,
                out_dir / f"similarity_plots_online{suffix}",
                args.family_stride,
            )
            run_plotter(
                full_csv,
                copied_config_path,
                out_dir / f"similarity_plots_full{suffix}",
                args.family_stride,
            )

    if args.sliced_wasserstein_embeddings:
        swe_embed_batch, swe_embed_full = sliced_wasserstein_embedding_fns(
            input_dim, action_dim, args
        )
        for metric in args.similarity_metrics:
            suffix = metric_suffix(metric)
            print(f"Computing online sliced-Wasserstein embeddings ({metric})...")
            swe_online_embeddings, swe_online_rows = online_embeddings_and_rows(
                task_sars,
                swe_embed_batch,
                args,
                np.random.default_rng(args.seed),
                mode="online_swe",
                metric=metric,
            )
            swe_online_csv = out_dir / f"proxy_task_similarities_online_swe{suffix}.csv"
            write_similarity_rows(swe_online_csv, swe_online_rows)
            save_embeddings(
                out_dir / f"proxy_embeddings_online_swe{suffix}.npy",
                swe_online_embeddings,
            )

            print(f"Computing full-dataset sliced-Wasserstein embeddings ({metric})...")
            swe_full_embeddings, swe_full_rows, swe_pairwise_rows = (
                full_embeddings_and_rows(
                    task_sars,
                    swe_embed_full,
                    args,
                    mode="full_swe",
                    metric=metric,
                )
            )
            swe_full_csv = out_dir / f"proxy_task_similarities_full_swe{suffix}.csv"
            swe_pairwise_csv = out_dir / f"proxy_pairwise_similarity_full_swe{suffix}.csv"
            write_similarity_rows(swe_full_csv, swe_full_rows)
            write_pairwise_rows(swe_pairwise_csv, swe_pairwise_rows)
            save_embeddings(
                out_dir / f"proxy_embeddings_full_swe{suffix}.npy",
                swe_full_embeddings,
            )

            if args.plot:
                run_plotter(
                    swe_online_csv,
                    copied_config_path,
                    out_dir / f"similarity_plots_online_swe{suffix}",
                    args.family_stride,
                )
                run_plotter(
                    swe_full_csv,
                    copied_config_path,
                    out_dir / f"similarity_plots_full_swe{suffix}",
                    args.family_stride,
                )

    if args.separate_embeddings:
        component_online_rows_by_metric = {
            metric: {} for metric in args.similarity_metrics
        }
        component_full_rows_by_metric = {
            metric: {} for metric in args.similarity_metrics
        }
        component_pairwise_rows_by_metric = {
            metric: {} for metric in args.similarity_metrics
        }
        for component in args.separate_components:
            print(f"Computing {component}-only embeddings...")
            embed_batch, embed_full = component_embedding_fns(
                component, input_dim, action_dim, args
            )
            for metric in args.similarity_metrics:
                suffix = metric_suffix(metric)
                comp_online_embeddings, comp_online_rows = online_embeddings_and_rows(
                    task_sars,
                    embed_batch,
                    args,
                    np.random.default_rng(args.seed),
                    mode=f"online_{component}",
                    metric=metric,
                )
                comp_online_csv = (
                    out_dir
                    / f"proxy_task_similarities_online_{component}{suffix}.csv"
                )
                write_similarity_rows(comp_online_csv, comp_online_rows)
                component_online_rows_by_metric[metric][component] = comp_online_rows
                save_embeddings(
                    out_dir / f"proxy_embeddings_online_{component}{suffix}.npy",
                    comp_online_embeddings,
                )

                comp_full_embeddings, comp_full_rows, comp_pairwise_rows = (
                    full_embeddings_and_rows(
                        task_sars,
                        embed_full,
                        args,
                        mode=f"full_{component}",
                        metric=metric,
                    )
                )
                comp_full_csv = (
                    out_dir / f"proxy_task_similarities_full_{component}{suffix}.csv"
                )
                comp_pairwise_csv = (
                    out_dir / f"proxy_pairwise_similarity_full_{component}{suffix}.csv"
                )
                write_similarity_rows(comp_full_csv, comp_full_rows)
                write_pairwise_rows(comp_pairwise_csv, comp_pairwise_rows)
                component_full_rows_by_metric[metric][component] = comp_full_rows
                component_pairwise_rows_by_metric[metric][component] = (
                    comp_pairwise_rows
                )
                save_embeddings(
                    out_dir / f"proxy_embeddings_full_{component}{suffix}.npy",
                    comp_full_embeddings,
                )

                if args.plot:
                    run_plotter(
                        comp_online_csv,
                        copied_config_path,
                        out_dir / f"similarity_plots_online_{component}{suffix}",
                        args.family_stride,
                    )
                    run_plotter(
                        comp_full_csv,
                        copied_config_path,
                        out_dir / f"similarity_plots_full_{component}{suffix}",
                        args.family_stride,
                    )

        if args.component_average_similarity:
            component_label = "_".join(component_average_components)
            component_file_tag = f"component_avg_{component_label}"
            for metric in args.similarity_metrics:
                suffix = metric_suffix(metric)
                print(
                    "Computing averaged component similarities "
                    f"({component_label}, {metric})..."
                )
                selected_online_rows = [
                    component_online_rows_by_metric[metric][component]
                    for component in component_average_components
                ]
                selected_full_rows = [
                    component_full_rows_by_metric[metric][component]
                    for component in component_average_components
                ]
                selected_pairwise_rows = [
                    component_pairwise_rows_by_metric[metric][component]
                    for component in component_average_components
                ]

                avg_online_rows = average_component_similarity_rows(
                    selected_online_rows,
                    args,
                    metric,
                    mode=f"online_component_avg_{component_label}",
                )
                avg_online_csv = (
                    out_dir
                    / f"proxy_task_similarities_online_{component_file_tag}{suffix}.csv"
                )
                write_similarity_rows(avg_online_csv, avg_online_rows)

                avg_full_rows = average_component_similarity_rows(
                    selected_full_rows,
                    args,
                    metric,
                    mode=f"full_component_avg_{component_label}",
                )
                avg_full_csv = (
                    out_dir
                    / f"proxy_task_similarities_full_{component_file_tag}{suffix}.csv"
                )
                write_similarity_rows(avg_full_csv, avg_full_rows)

                avg_pairwise_rows = average_component_pairwise_rows(
                    selected_pairwise_rows,
                    metric,
                    mode=f"full_component_avg_pairwise_{component_label}",
                )
                avg_pairwise_csv = (
                    out_dir
                    / f"proxy_pairwise_similarity_full_{component_file_tag}{suffix}.csv"
                )
                write_pairwise_rows(avg_pairwise_csv, avg_pairwise_rows)

                if args.plot:
                    run_plotter(
                        avg_online_csv,
                        copied_config_path,
                        out_dir / f"similarity_plots_online_{component_file_tag}{suffix}",
                        args.family_stride,
                    )
                    run_plotter(
                        avg_full_csv,
                        copied_config_path,
                        out_dir / f"similarity_plots_full_{component_file_tag}{suffix}",
                        args.family_stride,
                    )

    print(f"Wrote proxy similarity outputs to {out_dir}")


if __name__ == "__main__":
    main()
