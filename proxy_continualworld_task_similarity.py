#!/usr/bin/env python3
"""
Estimate ContinualWorld task similarities without running SAC.

The script collects cheap state/action/reward buffers from fixed proxy
policies, then feeds them through the same Detect embedding path used by
MaskSC. It is intended for CW10 configs such as:

    env_configs/continualworld_10.json

Typical command:

    python proxy_continualworld_task_similarity.py \
        --env-config env_configs/continualworld_10.json \
        --out-dir proxy_similarity_cw10 \
        --policies random zero \
        --random-episodes 8 \
        --zero-episodes 2 \
        --online-batches 12 \
        --online-batch-size 1000 \
        --plot

For SWE diagnostics:

    python proxy_continualworld_task_similarity.py \
        --env-config env_configs/continualworld_10.json \
        --out-dir proxy_similarity_cw10_swe \
        --detect-embedding-method swe \
        --swe-num-workers 4
"""

import argparse
import csv
import inspect
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import numpy as np
import ot
import torch
import torch.nn.functional as F

from deep_rl.component.task import ContinualWorld
from deep_rl.detect_modules.detect import Detect
from deep_rl.utils.config import Config
from deep_rl.utils.normalizer import RunningStatsNormalizer, RewardRunningStatsNormalizer


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Estimate ContinualWorld Detect/MaskSC task similarities from cheap "
            "state-action-reward proxy datasets."
        )
    )
    parser.add_argument(
        "--env-config",
        type=Path,
        default=Path("env_configs/continualworld_10.json"),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["random", "zero"],
        choices=["random", "zero", "reset"],
        help=(
            "Proxy data sources to concatenate for each task. `random` uses "
            "uniform random actions, `zero` uses zero actions, and `reset` "
            "samples reset observations with random actions and zero reward."
        ),
    )
    parser.add_argument("--random-episodes", type=int, default=8)
    parser.add_argument("--zero-episodes", type=int, default=2)
    parser.add_argument("--reset-samples", type=int, default=512)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Rollout cap per episode. Defaults to the active CW task horizon.",
    )
    parser.add_argument(
        "--normalize-buffer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Apply RunningStatsNormalizer and RewardRunningStatsNormalizer while "
            "collecting proxy SAR, matching the SAC buffer more closely."
        ),
    )
    parser.add_argument(
        "--state-scale",
        type=float,
        default=1.0,
        help="Only used when --no-normalize-buffer is set.",
    )
    parser.add_argument(
        "--reward-scale",
        type=float,
        default=1.0,
        help="Only used when --no-normalize-buffer is set.",
    )
    parser.add_argument("--reference-num", type=int, default=50)
    parser.add_argument(
        "--detect-num-samples",
        type=int,
        default=1000,
        help="Matches train_continualworld.py default.",
    )
    parser.add_argument("--online-batches", type=int, default=12)
    parser.add_argument("--online-batch-size", type=int, default=1000)
    parser.add_argument("--online-ema", type=float, default=0.5)
    parser.add_argument(
        "--online-sampling",
        default="iid",
        choices=["iid", "sequential", "replay"],
        help=(
            "`iid` samples from the full proxy dataset, `sequential` reads "
            "chronological chunks, and `replay` appends chunks to a task-local "
            "buffer before sampling from that buffer."
        ),
    )
    parser.add_argument(
        "--full-num-samples",
        type=int,
        default=0,
        help="Samples for full-dataset embeddings. Use 0 for all proxy samples.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--similarity-metrics",
        nargs="+",
        default=["cosine"],
        choices=["cosine", "euclidean"],
    )
    parser.add_argument(
        "--euclidean-threshold",
        type=float,
        default=None,
        help=(
            "Optional maximum Euclidean distance for selected=1 in Euclidean "
            "diagnostic files. If omitted, selected=1 marks top-k nearest priors."
        ),
    )
    parser.add_argument("--topk", type=int, default=None)
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
        "--detect-embedding-method",
        choices=["lwe", "swe"],
        default="lwe",
        help="Embedding method used by the combined SAR Detect path.",
    )
    parser.add_argument("--swe-num-projections", type=int, default=128)
    parser.add_argument("--swe-num-quantiles", type=int, default=128)
    parser.add_argument("--swe-num-workers", type=int, default=1)
    parser.add_argument("--swe-seed", type=int, default=98)
    parser.add_argument(
        "--swe-normalize-embedding",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save-sar",
        action="store_true",
        help="Save each collected SAR array to <out-dir>/sar/taskXXX.npy.",
    )
    parser.add_argument(
        "--separate-embeddings",
        action="store_true",
        help="Also compute separate state-only, action-only, and reward-only LWE embeddings.",
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
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Run plot_task_similarity_summary.py for online and full CSVs.",
    )
    parser.add_argument(
        "--family-stride",
        type=int,
        default=10,
        help=(
            "Passed to plot_task_similarity_summary.py. CW10 has no real "
            "families, so the default treats each task as its own group."
        ),
    )
    return parser.parse_args()


def load_env_config(path):
    with path.open() as fh:
        config = json.load(fh)
    tasks = list(config["tasks"])
    return config, tasks


def reset_env(env):
    ret = env.reset()
    if isinstance(ret, tuple) and len(ret) == 2:
        return ret[0]
    return ret


def step_env(env, action):
    ret = env.step(action)
    if isinstance(ret, tuple) and len(ret) == 5:
        obs, reward, terminated, truncated, info = ret
        return obs, reward, bool(terminated or truncated), info
    return ret


def seed_envs(task_env, seed):
    for offset, env in enumerate(task_env.envs.values()):
        env_seed = int(seed) + offset
        if hasattr(env, "seed"):
            try:
                env.seed(env_seed)
            except Exception:
                pass
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(env_seed)
        if hasattr(env.observation_space, "seed"):
            env.observation_space.seed(env_seed)


def max_steps_for(env, override):
    if override is not None:
        return int(override)
    active_env = getattr(env, "env", env)
    return int(
        getattr(
            active_env,
            "_max_episode_steps",
            getattr(getattr(active_env, "env", active_env), "_max_episode_steps", 500),
        )
    )


def normalize_state(obs, state_normalizer, args):
    obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)
    if args.normalize_buffer:
        return np.asarray(state_normalizer(obs)[0], dtype=np.float32)
    return (obs[0] * float(args.state_scale)).astype(np.float32)


def normalize_reward(reward, reward_normalizer, args):
    if args.normalize_buffer:
        value = reward_normalizer(np.asarray([reward], dtype=np.float32))
        return float(np.asarray(value).reshape(-1)[0])
    return float(reward) * float(args.reward_scale)


def append_sar(rows, obs, action, reward):
    rows.append(
        np.concatenate(
            [
                np.asarray(obs, dtype=np.float32).reshape(-1),
                np.asarray(action, dtype=np.float32).reshape(-1),
                np.asarray([reward], dtype=np.float32),
            ]
        )
    )


def random_action(action_space, rng):
    low = np.asarray(action_space.low, dtype=np.float32)
    high = np.asarray(action_space.high, dtype=np.float32)
    return rng.uniform(low=low, high=high).astype(np.float32)


def zero_action(action_space):
    low = np.asarray(action_space.low, dtype=np.float32)
    high = np.asarray(action_space.high, dtype=np.float32)
    return np.clip(np.zeros_like(low), low, high).astype(np.float32)


def collect_rollout_sar(
    task_env,
    task_info,
    episodes,
    max_steps,
    policy,
    state_normalizer,
    reward_normalizer,
    args,
    rng,
):
    rows = []
    for _episode in range(int(episodes)):
        raw_obs = task_env.reset_task(task_info)
        obs = normalize_state(raw_obs, state_normalizer, args)
        for _step in range(max_steps):
            if policy == "random":
                action = random_action(task_env.action_space, rng)
            elif policy == "zero":
                action = zero_action(task_env.action_space)
            else:
                raise ValueError(f"unknown rollout policy: {policy}")

            next_obs, reward, done, _info = step_env(task_env, action)
            reward = normalize_reward(reward, reward_normalizer, args)
            append_sar(rows, obs, action, reward)
            obs = normalize_state(next_obs, state_normalizer, args)
            if done:
                break
    return rows


def collect_reset_sar(task_env, task_info, samples, state_normalizer, args, rng):
    rows = []
    for _ in range(int(samples)):
        raw_obs = task_env.reset_task(task_info)
        obs = normalize_state(raw_obs, state_normalizer, args)
        action = random_action(task_env.action_space, rng)
        append_sar(rows, obs, action, 0.0)
    return rows


def collect_task_sar(task_env, task_info, args, state_normalizer, reward_normalizer, rng):
    max_steps = max_steps_for(task_env, args.max_steps)
    rows = []

    if "random" in args.policies:
        rows.extend(
            collect_rollout_sar(
                task_env,
                task_info,
                args.random_episodes,
                max_steps,
                "random",
                state_normalizer,
                reward_normalizer,
                args,
                rng,
            )
        )
    if "zero" in args.policies:
        rows.extend(
            collect_rollout_sar(
                task_env,
                task_info,
                args.zero_episodes,
                max_steps,
                "zero",
                state_normalizer,
                reward_normalizer,
                args,
                rng,
            )
        )
    if "reset" in args.policies:
        rows.extend(
            collect_reset_sar(
                task_env,
                task_info,
                args.reset_samples,
                state_normalizer,
                args,
                rng,
            )
        )

    if not rows:
        raise RuntimeError(f"No proxy SAR rows collected for {task_info['name']}")
    return np.stack(rows).astype(np.float32)


def resolve_device(args):
    if args.device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return args.device


def _detect_supports_embedding_method():
    return "embedding_method" in inspect.signature(Detect.__init__).parameters


def make_detect(input_dim, action_dim, args, num_samples):
    device = resolve_device(args)
    if args.detect_embedding_method == "swe" and not _detect_supports_embedding_method():
        detect = SlicedWassersteinDetect(
            input_dim=input_dim,
            action_dim=action_dim,
            num_samples=num_samples,
            normalized=args.detect_normalized,
            device=device,
            swe_num_projections=args.swe_num_projections,
            swe_num_quantiles=args.swe_num_quantiles,
            swe_num_workers=args.swe_num_workers,
            swe_seed=args.swe_seed,
            swe_normalize_embedding=args.swe_normalize_embedding,
        )
        detect.set_reference(input_dim, args.reference_num, action_dim)
        return detect

    detect_kwargs = {
        "one_hot": False,
        "normalized": args.detect_normalized,
        "device": device,
    }
    if _detect_supports_embedding_method():
        detect_kwargs.update(
            {
                "embedding_method": args.detect_embedding_method,
                "swe_num_projections": args.swe_num_projections,
                "swe_num_quantiles": args.swe_num_quantiles,
                "swe_num_workers": args.swe_num_workers,
                "swe_seed": args.swe_seed,
                "swe_normalize_embedding": args.swe_normalize_embedding,
            }
        )

    detect = Detect(
        args.reference_num,
        input_dim,
        action_dim,
        num_samples,
        **detect_kwargs,
    )
    detect.set_reference(input_dim, args.reference_num, action_dim)
    return detect


def compute_embedding(detect, sar, action_dim, num_samples):
    detect.set_num_samples(num_samples)
    tensor = torch.as_tensor(sar, dtype=torch.float32)
    with torch.no_grad():
        emb = detect.lwe(tensor, action_dim)
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


class SlicedWassersteinDetect:
    """SWE-compatible Detect wrapper for repos with the original LWE-only Detect."""

    def __init__(
        self,
        input_dim,
        action_dim,
        num_samples,
        normalized,
        device,
        swe_num_projections,
        swe_num_quantiles,
        swe_num_workers,
        swe_seed,
        swe_normalize_embedding,
    ):
        self.input_dim = int(input_dim)
        self.action_dim = int(action_dim)
        self.num_samples = num_samples
        self.normalized = bool(normalized)
        self.device = device
        self.swe_num_projections = int(swe_num_projections)
        self.swe_num_quantiles = int(swe_num_quantiles)
        self.swe_num_workers = max(int(swe_num_workers), 1)
        self.swe_seed = int(swe_seed)
        self.swe_normalize_embedding = bool(swe_normalize_embedding)
        self.swe_directions = None
        self._set_swe_directions(self.input_dim + self.action_dim + 1)

    def set_reference(self, task_observation_dim, _reference_num, action_dim):
        feature_dim = int(task_observation_dim) + int(action_dim) + 1
        if self.swe_directions is None or self.swe_directions.shape[1] != feature_dim:
            self._set_swe_directions(feature_dim)

    def set_num_samples(self, num_samples):
        self.num_samples = num_samples

    def _set_swe_directions(self, feature_dim):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.swe_seed)
        directions = torch.randn(
            self.swe_num_projections,
            int(feature_dim),
            generator=generator,
            dtype=torch.float32,
        )
        directions = F.normalize(directions, dim=1, eps=1e-8)
        self.swe_directions = directions.to(self.device)

    def preprocess_dataset(self, X, _action_space_size):
        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X)

        if self.num_samples is not None and X.shape[0] > self.num_samples:
            idx = torch.randperm(X.shape[0], device=X.device)[: self.num_samples]
            X = X.index_select(0, idx)

        X = X.to(self.device, dtype=torch.float32)
        state = X[:, : self.input_dim]
        action = X[:, self.input_dim : self.input_dim + self.action_dim]
        reward = X[:, self.input_dim + self.action_dim : self.input_dim + self.action_dim + 1]

        if self.normalized:
            mean = state.mean()
            std = state.std().clamp_min(1e-8)
            state = (state - mean) / std

        return torch.cat((state, action, reward), dim=1)

    @torch.no_grad()
    def _swe_chunk_embedding(self, X, directions):
        projections = X @ directions.T
        projections = torch.sort(projections, dim=0).values

        if projections.shape[0] != self.swe_num_quantiles:
            projection_channels = projections.T.unsqueeze(0)
            projections = F.interpolate(
                projection_channels,
                size=self.swe_num_quantiles,
                mode="linear",
                align_corners=True,
            ).squeeze(0).T

        return projections.T.reshape(-1)

    @torch.no_grad()
    def lwe(self, X, action_space_size):
        X = self.preprocess_dataset(X, action_space_size)
        if self.swe_directions is None or self.swe_directions.shape[1] != X.shape[1]:
            self._set_swe_directions(X.shape[1])

        directions = self.swe_directions
        num_workers = min(self.swe_num_workers, directions.shape[0])
        if num_workers <= 1:
            emb = self._swe_chunk_embedding(X, directions)
        else:
            chunks = [
                chunk
                for chunk in torch.chunk(directions, num_workers, dim=0)
                if chunk.numel() > 0
            ]
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                parts = list(
                    executor.map(
                        lambda chunk: self._swe_chunk_embedding(X, chunk),
                        chunks,
                    )
                )
            emb = torch.cat(parts, dim=0)

        emb = emb / max(self.swe_num_projections * self.swe_num_quantiles, 1) ** 0.5
        if self.swe_normalize_embedding:
            emb = F.normalize(emb, dim=0, eps=1e-8)
        return emb


def component_features(sar, input_dim, action_dim, component):
    if component == "state":
        return sar[:, :input_dim], True
    if component == "action":
        return sar[:, input_dim : input_dim + action_dim], False
    if component == "reward":
        return sar[:, input_dim + action_dim : input_dim + action_dim + 1], False
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
        passing = [record for record in records if record["similarity"] > args.threshold]
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


def online_embeddings_and_rows(
    task_sars, embed_batch_fn, args, rng, mode="online", metric="cosine"
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
            if current is None:
                current = new_emb
            else:
                current = args.online_ema * current + (1.0 - args.online_ema) * new_emb
                current = F.normalize(current, dim=0, eps=1e-8)
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


def full_embeddings_and_rows(task_sars, embed_full_fn, args, mode="full", metric="cosine"):
    embeddings = [embed_full_fn(sar) for sar in task_sars]
    total_steps = int(sum(sar.shape[0] for sar in task_sars))

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
    plotter = Path(__file__).with_name("plot_task_similarity_summary.py")
    cmd = [
        sys.executable,
        str(plotter),
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
    sample_sar = np.zeros((1, input_dim + action_dim + 1), dtype=np.float32)
    sample_features, normalize_component = component_features(
        sample_sar,
        input_dim,
        action_dim,
        component,
    )
    embedder = RawLWEEmbedder(
        feature_dim=sample_features.shape[1],
        reference_num=args.reference_num,
        device=resolve_device(args),
    )
    full_samples = None if args.full_num_samples == 0 else int(args.full_num_samples)

    def embed_batch(sar):
        features, _ = component_features(sar, input_dim, action_dim, component)
        return embedder.embed(
            features,
            num_samples=args.online_batch_size,
            normalized=args.detect_normalized and normalize_component,
        )

    def embed_full(sar):
        features, _ = component_features(sar, input_dim, action_dim, component)
        return embedder.embed(
            features,
            num_samples=full_samples,
            normalized=args.detect_normalized and normalize_component,
        )

    return embed_batch, embed_full


def write_task_metadata(path, rows):
    fieldnames = ["task_idx", "label", "name"]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def close_task_env(task_env):
    for env in getattr(task_env, "envs", {}).values():
        close = getattr(env, "close", None)
        if callable(close):
            close()


def main():
    args = parse_args()
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

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    env_config_path = args.env_config.expanduser().resolve()
    env_config, config_tasks = load_env_config(env_config_path)

    copied_config_path = out_dir / "env_config.json"
    with copied_config_path.open("w") as fh:
        json.dump(env_config, fh, indent=4)
        fh.write("\n")
    with (out_dir / "proxy_args.json").open("w") as fh:
        args_dict = vars(args).copy()
        args_dict["env_config"] = str(env_config_path)
        args_dict["out_dir"] = str(out_dir)
        json.dump(args_dict, fh, indent=4)
        fh.write("\n")

    try:
        task_env = ContinualWorld(
            Config.ENV_CONTINUALWORLD,
            str(env_config_path),
            log_dir=None,
            seed=args.seed,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "continualworld":
            raise RuntimeError(
                "Could not import the `continualworld` package. Run this "
                "script in the same environment used for CW10 training."
            ) from exc
        raise
    seed_envs(task_env, args.seed)
    tasks = task_env.get_all_tasks(requires_task_label=True)
    if len(tasks) != len(config_tasks):
        raise RuntimeError(
            f"ContinualWorld returned {len(tasks)} tasks, but config lists "
            f"{len(config_tasks)} tasks."
        )

    action_dim = int(task_env.action_dim)
    metadata_rows = [
        {"task_idx": idx, "label": task_info["name"], "name": task_info["name"]}
        for idx, task_info in enumerate(tasks)
    ]
    write_task_metadata(out_dir / "proxy_task_metadata.csv", metadata_rows)

    task_sars = []
    summary_rows = []
    sar_dir = out_dir / "sar"
    if args.save_sar:
        sar_dir.mkdir(exist_ok=True)

    state_normalizer = RunningStatsNormalizer()
    reward_normalizer = RewardRunningStatsNormalizer()

    print("Collecting ContinualWorld proxy SAR datasets...")
    for task_idx, task_info in enumerate(tasks):
        task_rng = np.random.default_rng(args.seed + 1009 * task_idx)
        sar = collect_task_sar(
            task_env,
            task_info,
            args,
            state_normalizer,
            reward_normalizer,
            task_rng,
        )
        task_sars.append(sar)
        summary_rows.append(
            {
                "task_idx": task_idx,
                "task": task_info["name"],
                "samples": sar.shape[0],
                "state_dim": sar.shape[1] - action_dim - 1,
                "action_dim": action_dim,
                "reward_mean": float(sar[:, -1].mean()),
                "reward_nonzero": int(np.count_nonzero(sar[:, -1])),
            }
        )
        if args.save_sar:
            np.save(sar_dir / f"task{task_idx:03d}.npy", sar)
        print(
            f"  task{task_idx:02d}: {task_info['name']} samples={sar.shape[0]} "
            f"reward_nonzero={np.count_nonzero(sar[:, -1])}"
        )

    with (out_dir / "proxy_sar_summary.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "task_idx",
                "task",
                "samples",
                "state_dim",
                "action_dim",
                "reward_mean",
                "reward_nonzero",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    input_dim = task_sars[0].shape[1] - action_dim - 1
    detect = make_detect(input_dim, action_dim, args, args.detect_num_samples)
    full_samples = None if args.full_num_samples == 0 else int(args.full_num_samples)
    combined_embed_batch = lambda batch: compute_embedding(
        detect,
        batch,
        action_dim,
        num_samples=args.online_batch_size,
    )
    combined_embed_full = lambda sar: compute_embedding(
        detect,
        sar,
        action_dim,
        num_samples=full_samples,
    )

    for metric in args.similarity_metrics:
        suffix = metric_suffix(metric)
        print(f"Computing online Detect embeddings ({metric})...")
        online_embeddings, online_rows = online_embeddings_and_rows(
            task_sars,
            combined_embed_batch,
            args,
            np.random.default_rng(args.seed),
            mode=f"online_{args.detect_embedding_method}",
            metric=metric,
        )
        online_csv = out_dir / f"proxy_task_similarities_online{suffix}.csv"
        write_similarity_rows(online_csv, online_rows)
        save_embeddings(out_dir / f"proxy_embeddings_online{suffix}.npy", online_embeddings)

        print(f"Computing full-dataset Detect embeddings ({metric})...")
        full_embeddings, full_rows, pairwise_rows = full_embeddings_and_rows(
            task_sars,
            combined_embed_full,
            args,
            mode=f"full_{args.detect_embedding_method}",
            metric=metric,
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
                component,
                input_dim,
                action_dim,
                args,
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
                    out_dir / f"proxy_task_similarities_online_{component}{suffix}.csv"
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

    close_task_env(task_env)
    print(f"Wrote proxy similarity outputs to {out_dir}")


if __name__ == "__main__":
    main()
