#!/usr/bin/env python3
"""
Unified significance testing for continual RL metrics.

By default, TPOT/AUC/eval/forgetting/BWT are computed on tasks that have been
exposed by the current point in the curriculum. This avoids inflating main
continual-learning metrics with zero-shot performance on future/unseen tasks.
Use --include-unseen-eval-tasks to reproduce the legacy all-task eval behavior.
Use --metric fgt_cw for the Continual World Section 4.1 forgetting
definition, F_i = p_i(i * Delta) - p_i(T). The legacy --metric forgetting
keeps the max_t p_i(t) - p_i(T) definition.

Recommended command templates:

CT-graph, where task performance is already in [0, 1]:

  TPOT/AUC:
    python significance_test_metrics.py --ref Mask-SC-2 \\
      --methods Mask-SC-2=<masksc_log_root> Mask-LC=<masklc_log_root> \\
      --task-order-config env_configs/ct28/seed1/meta_ctgraph_ct28_interleaved.json \\
      --metric tpot --tests all --alternative greater

  FWT:
    python significance_test_metrics.py --ref Mask-SC-2 \\
      --methods Mask-SC-2=<masksc_log_root> Mask-LC=<masklc_log_root> \\
      --expert-root log/ct28/ct28-interleaved-single-task-experts/ \\
      --task-order-config env_configs/ct28/seed1/meta_ctgraph_ct28_interleaved.json \\
      --metric fwt --tests all --alternative greater

  Continual World forgetting:
    python significance_test_metrics.py --ref Mask-SC-2 \\
      --methods Mask-SC-2=<masksc_log_root> Mask-LC=<masklc_log_root> \\
      --task-order-config env_configs/ct28/seed1/meta_ctgraph_ct28_interleaved.json \\
      --metric fgt_cw --tests all --alternative less

  BWT:
    python significance_test_metrics.py --ref Mask-SC-2 \\
      --methods Mask-SC-2=<masksc_log_root> Mask-LC=<masklc_log_root> \\
      --task-order-config env_configs/ct28/seed1/meta_ctgraph_ct28_interleaved.json \\
      --metric bwt --tests all --alternative greater

MiniGrid, where raw returns can have task-dependent optimal values:

  Add --performance-normalization minigrid_shortest_path to normalize each
  task by the theoretical shortest-path return before computing metrics.
  --performance-normalization auto does this automatically when the task config
  contains MiniGrid task ids.

  TPOT/AUC:
    python significance_test_metrics.py --ref Mask-SC-2 \\
      --methods Mask-SC-2=<masksc_log_root> Mask-LC=<masklc_log_root> \\
      --task-order-config env_configs/minigrid_object_remap_seed841.json \\
      --performance-normalization minigrid_shortest_path \\
      --metric tpot --tests all --alternative greater

  FWT:
    python significance_test_metrics.py --ref Mask-SC-2 \\
      --methods Mask-SC-2=<masksc_log_root> Mask-LC=<masklc_log_root> \\
      --expert-root log/mg16-interleaved-single-task-experts-object_remap_random_v2/ \\
      --task-order-config env_configs/minigrid_object_remap_seed841.json \\
      --performance-normalization minigrid_shortest_path \\
      --metric fwt --tests all --alternative greater

    For hierarchical MiniGrid curricula, R2 tasks are cold-start tasks rather
    than transfer tasks. To focus FWT on R3-R5 in a 4-family x R2-R5 setup:
      add --exclude-fwt-task-ids 0 1 2 3

  Continual World forgetting:
    python significance_test_metrics.py --ref Mask-SC-2 \\
      --methods Mask-SC-2=<masksc_log_root> Mask-LC=<masklc_log_root> \\
      --task-order-config env_configs/minigrid_object_remap_seed841.json \\
      --performance-normalization minigrid_shortest_path \\
      --metric fgt_cw --tests all --alternative less

  BWT:
    python significance_test_metrics.py --ref Mask-SC-2 \\
      --methods Mask-SC-2=<masksc_log_root> Mask-LC=<masklc_log_root> \\
      --task-order-config env_configs/minigrid_object_remap_seed841.json \\
      --performance-normalization minigrid_shortest_path \\
      --metric bwt --tests all --alternative greater
"""
import argparse
import collections
import json
import os
import re
import numpy as np
from scipy import stats


def find_runs(root):
    run_dirs = set()
    for dirpath, _, files in os.walk(root):
        for fname in files:
            if fname in ("eval_metrics.npy", "eval_metrics.csv"):
                run_dirs.add(dirpath)
                break
    return sorted(run_dirs)


def load_eval_matrix(run_dir):
    npy = os.path.join(run_dir, "eval_metrics.npy")
    csv = os.path.join(run_dir, "eval_metrics.csv")
    is_csv = False
    if os.path.isfile(npy):
        mat = np.load(npy)
    elif os.path.isfile(csv):
        mat = np.loadtxt(csv, delimiter=",")
        is_csv = True
    else:
        raise FileNotFoundError(f"Missing eval_metrics.(npy|csv) in {run_dir}")
    if mat.ndim != 2:
        raise ValueError(f"Expected 2D eval matrix in {run_dir}, got {mat.shape}")
    # Drop trailing timestamp only for CSVs
    if is_csv and mat.shape[1] > 1 and np.all(np.diff(mat[:, -1]) >= 0):
        mat = mat[:, :-1]
    return mat  # shape [T, num_tasks]


def parse_methods(arglist):
    methods = {}
    for item in arglist:
        if "=" not in item:
            raise ValueError(f"Method spec should be NAME=PATH, got {item}")
        name, path = item.split("=", 1)
        methods[name] = path
    return methods


def extract_seed_generic(path):
    m = re.search(r"seed(\d+)", path)
    if m:
        return m.group(1)

    # Most run directories encode the RL seed as a dash-separated token in the
    # experiment name, e.g. `...-supermask-86-mask-...`,
    # `...-baseline-92-ct8`, or `...-ser-86-surprise`.  Match only path
    # components that contain alphabetic experiment text so timestamp folders
    # such as `260401-175521` are not mistaken for seeds.
    path_parts = re.split(r"[\\/]+", str(path))
    for part in path_parts:
        if not re.search(r"[A-Za-z]", part):
            continue
        m = re.search(r"(?:^|-)([0-9]{2,5})(?=-[A-Za-z_])", part)
        if m:
            return m.group(1)
        # Some baseline runs put the seed at the end of the experiment
        # component, e.g. `Minigrid-ppo-ewc_multi_head-86`.
        m = re.search(r"(?:^|-)([0-9]{2,5})$", part)
        if m:
            return m.group(1)

    m = re.search(r"-([0-9]{2,5})-(?:mask|ppo|supermask|ewc|si|linear|ct)", path)
    if m:
        return m.group(1)
    m = re.search(r"/([0-9]{2,5})/", path)
    if m:
        return m.group(1)
    return None


def bootstrap_mean_ci(x, iters=10000, alpha=0.05, rng=None):
    rng = rng or np.random.default_rng()
    x = np.asarray(x)
    boots = []
    for _ in range(iters):
        xb = rng.choice(x, size=len(x), replace=True)
        boots.append(np.nanmean(xb))
    boots = np.sort(boots)
    lo = boots[int((alpha / 2) * iters)]
    hi = boots[int((1 - alpha / 2) * iters)]
    return lo, hi


def bootstrap_diff_ci(x, y, iters=10000, alpha=0.05, rng=None):
    rng = rng or np.random.default_rng()
    x = np.asarray(x)
    y = np.asarray(y)
    diffs = []
    for _ in range(iters):
        xb = rng.choice(x, size=len(x), replace=True)
        yb = rng.choice(y, size=len(y), replace=True)
        diffs.append(np.nanmean(xb) - np.nanmean(yb))
    diffs = np.sort(diffs)
    lo = diffs[int((alpha / 2) * iters)]
    hi = diffs[int((1 - alpha / 2) * iters)]
    return lo, hi


def bootstrap_paired_diff_ci(diffs, iters=10000, alpha=0.05, rng=None):
    rng = rng or np.random.default_rng()
    diffs = np.asarray(diffs, float)
    diffs = diffs[np.isfinite(diffs)]
    boots = []
    for _ in range(iters):
        db = rng.choice(diffs, size=len(diffs), replace=True)
        boots.append(np.nanmean(db))
    boots = np.sort(boots)
    lo = boots[int((alpha / 2) * iters)]
    hi = boots[int((1 - alpha / 2) * iters)]
    return lo, hi


def welch_ttest(x, y):
    t, p = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
    return p


def paired_ttest_from_diffs(diffs, alternative="two-sided"):
    diffs = np.asarray(diffs, float)
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) < 2:
        return np.nan
    if np.allclose(diffs, 0.0):
        return 1.0
    return stats.ttest_1samp(diffs, 0.0, alternative=alternative, nan_policy="omit").pvalue


def wilcoxon_signed_rank_from_diffs(diffs, alternative="two-sided"):
    diffs = np.asarray(diffs, float)
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) < 1:
        return np.nan
    if np.allclose(diffs, 0.0):
        return 1.0
    return stats.wilcoxon(diffs, zero_method="wilcox", alternative=alternative).pvalue


def paired_sign_flip_test(diffs, alternative="two-sided", exact_max_pairs=20, rng=None, iters=10000):
    rng = rng or np.random.default_rng()
    diffs = np.asarray(diffs, float)
    diffs = diffs[np.isfinite(diffs)]
    n = len(diffs)
    if n < 1:
        return np.nan, False
    obs = float(np.mean(diffs))
    if np.isclose(obs, 0.0):
        return 1.0, n <= exact_max_pairs

    def extreme(null_means):
        if alternative == "greater":
            return null_means >= obs
        if alternative == "less":
            return null_means <= obs
        return np.abs(null_means) >= abs(obs)

    if n <= exact_max_pairs:
        null_means = []
        for mask in range(1 << n):
            signs = np.ones(n)
            for i in range(n):
                if mask & (1 << i):
                    signs[i] = -1.0
            null_means.append(float(np.mean(signs * diffs)))
        null_means = np.asarray(null_means)
        return float(np.mean(extreme(null_means))), True

    signs = rng.choice([-1.0, 1.0], size=(iters, n))
    null_means = np.mean(signs * diffs[None, :], axis=1)
    return float(np.mean(extreme(null_means))), False


def expand_tests(tests):
    if "all" in tests:
        return ["welch", "paired_t", "wilcoxon", "sign_flip"]
    out = []
    for test in tests:
        if test not in out:
            out.append(test)
    return out


def seed_sort_key(seed):
    return (0, int(seed)) if str(seed).isdigit() else (1, str(seed))


def paired_diffs_by_seed(ref_seed_vals, other_seed_vals):
    common = sorted(set(ref_seed_vals) & set(other_seed_vals), key=seed_sort_key)
    seeds = []
    diffs = []
    skipped = []
    for seed in common:
        ref_vals = ref_seed_vals[seed]
        other_vals = other_seed_vals[seed]
        if len(ref_vals) != 1 or len(other_vals) != 1:
            skipped.append(seed)
            continue
        diff = ref_vals[0] - other_vals[0]
        if np.isfinite(diff):
            seeds.append(seed)
            diffs.append(diff)
    return seeds, np.asarray(diffs, float), skipped


def final_total_perf(run_dir):
    mat = load_eval_matrix(run_dir)
    return float(np.nansum(mat[-1, :]))


def task_training_slices(num_rows, num_tasks):
    if num_tasks <= 0:
        raise ValueError("num_tasks must be positive")
    if num_rows % num_tasks != 0:
        raise ValueError(
            f"Cannot infer per-task evaluation blocks: {num_rows} rows, "
            f"{num_tasks} tasks"
        )
    evals_per_task = num_rows // num_tasks
    return [
        slice(task_idx * evals_per_task, (task_idx + 1) * evals_per_task)
        for task_idx in range(num_tasks)
    ]


def task_end_indices(num_rows, num_tasks):
    return np.asarray(
        [row_slice.stop - 1 for row_slice in task_training_slices(num_rows, num_tasks)],
        dtype=int,
    )


def seen_task_mask(num_rows, num_tasks):
    """Mask of task columns that should count at each eval row.

    eval_metrics rows are logged periodically during each task's training block.
    The first block corresponds to task 0, the second to task 1, and so on. Main
    continual-learning metrics should not count future task columns before those
    tasks have been exposed.
    """
    slices = task_training_slices(num_rows, num_tasks)
    mask = np.zeros((num_rows, num_tasks), dtype=bool)
    for task_idx, row_slice in enumerate(slices):
        mask[row_slice, : task_idx + 1] = True
    return mask


def mask_unseen_eval_tasks(mat):
    mat = np.asarray(mat, float)
    mask = seen_task_mask(mat.shape[0], mat.shape[1])
    out = mat.copy()
    out[~mask] = np.nan
    return out


def auc01(curve):
    """Average AUC for a performance curve already normalized to [0, 1]."""
    curve = np.asarray(curve, float)
    curve = curve[np.isfinite(curve)]
    if curve.size == 0:
        return np.nan
    if curve.size == 1:
        return float(curve[0])
    return float(np.trapz(curve, dx=1.0) / (curve.size - 1))


def raw_per_task_auc(mat):
    return np.array([np.trapz(mat[:, j], dx=1.0) for j in range(mat.shape[1])])


def per_task_auc(mat):
    return raw_per_task_auc(mat)


def normalize_performance_matrix(mat, max_returns, clip=True):
    mat = np.asarray(mat, float)
    max_returns = np.asarray(max_returns, float)
    if max_returns.ndim == 0:
        max_returns = np.full(mat.shape[1], float(max_returns))
    if len(max_returns) != mat.shape[1]:
        raise ValueError(
            f"Expected {mat.shape[1]} return normalization values, got "
            f"{len(max_returns)}"
        )
    denom = max_returns.reshape(1, -1)
    denom = np.where(np.abs(denom) < 1e-12, np.nan, denom)
    norm = mat / denom
    if clip:
        norm = np.clip(norm, 0.0, 1.0)
    return norm


def normalized_per_task_auc(mat, max_returns, clip=True):
    norm = normalize_performance_matrix(mat, max_returns, clip=clip)
    return np.array([auc01(norm[:, j]) for j in range(norm.shape[1])])


def per_task_auc01_from_matrix(norm_mat):
    return np.array([auc01(norm_mat[:, j]) for j in range(norm_mat.shape[1])])


def normalized_own_task_auc(mat, max_returns, clip=True):
    norm = normalize_performance_matrix(mat, max_returns, clip=clip)
    n_tasks = norm.shape[1]
    slices = task_training_slices(norm.shape[0], n_tasks)
    return np.array([auc01(norm[slices[j], j]) for j in range(n_tasks)])


def normalized_single_task_auc(mat, max_return, clip=True):
    mat = np.asarray(mat, float)
    if mat.shape[1] != 1:
        raise ValueError(f"Expected single-task expert matrix, got {mat.shape}")
    norm = normalize_performance_matrix(mat, [max_return], clip=clip)
    return auc01(norm[:, 0])


def normalize_auc(arr, max_return):
    return np.clip(arr / max_return, 0.0, 1.0)


def forward_transfer(auc_ll, auc_expert, eps=1e-8, min_den=1e-3):
    denom = 1.0 - auc_expert
    fwt = (auc_ll - auc_expert) / (denom + eps)
    fwt[denom < min_den] = np.nan
    return fwt


def resolve_metric_task_ids(num_tasks, include_ids=None, exclude_ids=None):
    ids = list(range(num_tasks)) if include_ids is None else list(include_ids)
    invalid = [idx for idx in ids if idx < 0 or idx >= num_tasks]
    if invalid:
        raise ValueError(f"Task ids outside 0..{num_tasks - 1}: {invalid}")
    if exclude_ids:
        invalid = [idx for idx in exclude_ids if idx < 0 or idx >= num_tasks]
        if invalid:
            raise ValueError(f"Excluded task ids outside 0..{num_tasks - 1}: {invalid}")
        excluded = set(exclude_ids)
        ids = [idx for idx in ids if idx not in excluded]
    if not ids:
        raise ValueError("No task ids left after FWT include/exclude filtering")
    return ids


def load_task_order_config(path):
    if path is None:
        return None, None
    with open(path, "r") as f:
        cfg = json.load(f)

    task_names = cfg.get("tasks")
    if task_names is not None and not isinstance(task_names, list):
        raise ValueError(f"`tasks` in {path} must be a list")

    task_order_value = cfg.get("task_order")
    if task_order_value is not None and task_order_value != "default":
        if not isinstance(task_order_value, list):
            raise ValueError(
                f"`task_order` in {path} must be a list or the string "
                f"`default`; got {task_order_value!r}"
            )
        return list(task_order_value), task_names
    if "filter_tasks" in cfg:
        return list(cfg["filter_tasks"]), task_names
    if task_names is not None:
        return list(range(len(task_names))), task_names
    n = cfg.get("num_tasks", None)
    return (list(range(n)) if n is not None else None), task_names


def load_task_order(path):
    task_order, _ = load_task_order_config(path)
    return task_order


def extract_expert_task_id(run_dir, task_names=None):
    if task_names is not None:
        env_config = os.path.join(run_dir, "env_config.json")
        if os.path.isfile(env_config):
            with open(env_config, "r") as f:
                cfg = json.load(f)
            tasks = cfg.get("tasks")
            if isinstance(tasks, list) and len(tasks) == 1:
                task_name = tasks[0]
                if task_name in task_names:
                    return task_names.index(task_name)
                raise ValueError(
                    f"Expert {run_dir} has task {task_name!r}, which is not in "
                    "--task-order-config"
                )

    m = re.findall(r"(?:task)(\d+)", run_dir)
    if m:
        return int(m[-1])
    return None


def looks_like_minigrid_tasks(task_names):
    if not task_names:
        return False
    return any(
        "MiniGrid" in task_name or "CurriculumMultiRoom" in task_name
        for task_name in task_names
    )


def reset_env_with_optional_seed(env, seed):
    if seed is None:
        return env.reset()
    try:
        return env.reset(seed=int(seed))
    except TypeError:
        if hasattr(env, "seed"):
            env.seed(int(seed))
        return env.reset()


def step_env(env, action):
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, reward, bool(terminated or truncated), info
    obs, reward, done, info = result
    return obs, reward, bool(done), info


def env_unwrapped(env):
    return getattr(env, "unwrapped", env)


def grid_cell(env, x, y):
    grid = env_unwrapped(env).grid
    if x < 0 or y < 0 or x >= grid.width or y >= grid.height:
        return None
    return grid.get(x, y)


def cell_type(cell):
    return getattr(cell, "type", None)


def is_door_cell(cell):
    return hasattr(cell, "is_open") and hasattr(cell, "is_locked")


def build_door_index(env):
    grid = env_unwrapped(env).grid
    door_index = {}
    for x in range(grid.width):
        for y in range(grid.height):
            cell = grid.get(x, y)
            if is_door_cell(cell):
                door_index[(x, y)] = len(door_index)
    return door_index


def door_is_open(cell, pos, opened_mask, door_index):
    if not is_door_cell(cell):
        return False
    bit = door_index[pos]
    return bool(opened_mask & (1 << bit))


def cell_can_overlap(cell, pos, opened_mask, door_index):
    if cell is None:
        return True
    if is_door_cell(cell):
        return door_is_open(cell, pos, opened_mask, door_index)
    if cell_type(cell) == "lava":
        return False
    if hasattr(cell, "can_overlap"):
        try:
            return bool(cell.can_overlap())
        except TypeError:
            return False
    return False


def shortest_minigrid_action_sequence(env):
    base_env = env_unwrapped(env)
    if not hasattr(base_env, "grid"):
        raise ValueError("MiniGrid shortest-path normalization requires an env with a grid")
    if getattr(base_env, "agent_pos", None) is None:
        raise ValueError("MiniGrid env has no agent_pos after reset")

    goal_pos = getattr(base_env, "goal_pos", None)
    if goal_pos is None:
        # Fallback for environments that do not store goal_pos explicitly.
        grid = base_env.grid
        for x in range(grid.width):
            for y in range(grid.height):
                if cell_type(grid.get(x, y)) == "goal":
                    goal_pos = (x, y)
                    break
            if goal_pos is not None:
                break
    if goal_pos is None:
        raise ValueError("Could not locate MiniGrid goal position")
    goal_pos = tuple(goal_pos)

    door_index = build_door_index(env)
    initial_opened = 0
    for pos, bit in door_index.items():
        cell = grid_cell(env, *pos)
        if getattr(cell, "is_open", False):
            initial_opened |= 1 << bit

    start_pos = tuple(int(v) for v in base_env.agent_pos)
    start_dir = int(base_env.agent_dir)
    start = (start_pos[0], start_pos[1], start_dir, initial_opened)
    queue = collections.deque([(start, [])])
    seen = {start}
    dir_vecs = ((1, 0), (0, 1), (-1, 0), (0, -1))

    while queue:
        (x, y, direction, opened_mask), actions = queue.popleft()
        if (x, y) == goal_pos:
            return actions

        transitions = [
            ((x, y, (direction - 1) % 4, opened_mask), 0),
            ((x, y, (direction + 1) % 4, opened_mask), 1),
        ]

        dx, dy = dir_vecs[direction]
        nx, ny = x + dx, y + dy
        target = grid_cell(env, nx, ny)
        if cell_can_overlap(target, (nx, ny), opened_mask, door_index):
            transitions.append(((nx, ny, direction, opened_mask), 2))

        if is_door_cell(target) and not getattr(target, "is_locked", False):
            bit = door_index[(nx, ny)]
            transitions.append(((x, y, direction, opened_mask ^ (1 << bit)), 5))

        for next_state, action in transitions:
            if next_state not in seen:
                seen.add(next_state)
                queue.append((next_state, actions + [action]))

    raise RuntimeError("Could not find a shortest successful MiniGrid path")


def shortest_path_return_for_minigrid_task(task_name, seed=None):
    try:
        import gym
        import CurriculumMinigrid  # noqa: F401  # Registers custom envs.
    except Exception as exc:
        raise RuntimeError(
            "MiniGrid shortest-path normalization requires gym, gym_minigrid, "
            "and CurriculumMinigrid to be importable"
        ) from exc

    env = gym.make(task_name)
    try:
        reset_env_with_optional_seed(env, seed)
        actions = shortest_minigrid_action_sequence(env)
        reset_env_with_optional_seed(env, seed)
        total_reward = 0.0
        done = False
        for action in actions:
            _, reward, done, _ = step_env(env, action)
            total_reward += float(reward)
            if done:
                break
        if not done:
            raise RuntimeError(
                f"Shortest path for {task_name} did not terminate after replay"
            )
        if total_reward <= 0:
            raise RuntimeError(
                f"Shortest path for {task_name} produced non-positive return "
                f"{total_reward}"
            )
        return total_reward
    finally:
        env.close()


def task_names_in_eval_order(task_names, task_order, num_tasks):
    if task_names is None:
        return None
    if task_order is None:
        ordered = task_names[:num_tasks]
    else:
        if len(task_order) < num_tasks:
            raise ValueError(
                f"Need at least {num_tasks} task ids in --task-order-config; "
                f"got {len(task_order)}"
            )
        ordered = []
        for tid in task_order[:num_tasks]:
            if not isinstance(tid, int):
                raise ValueError(
                    "MiniGrid task_order entries must be integer indices into "
                    "the `tasks` list"
                )
            if tid < 0 or tid >= len(task_names):
                raise ValueError(
                    f"MiniGrid task_order id {tid} is outside the `tasks` list "
                    f"of length {len(task_names)}"
                )
            ordered.append(task_names[tid])
    if len(ordered) != num_tasks:
        raise ValueError(
            f"Need {num_tasks} task names for normalization; got {len(ordered)}"
        )
    return ordered


def resolve_return_scales(args, task_names, task_order, num_tasks):
    mode = args.performance_normalization
    if mode == "auto":
        mode = "minigrid_shortest_path" if looks_like_minigrid_tasks(task_names) else "scalar"

    if mode == "none":
        return np.ones(num_tasks, dtype=float), mode

    if mode == "scalar":
        return np.full(num_tasks, float(args.max_return), dtype=float), mode

    if mode != "minigrid_shortest_path":
        raise ValueError(f"Unknown performance normalization mode: {mode}")

    ordered_task_names = task_names_in_eval_order(task_names, task_order, num_tasks)
    if not ordered_task_names:
        raise ValueError(
            "--performance-normalization minigrid_shortest_path requires a "
            "--task-order-config with a `tasks` list"
        )

    seeds = args.minigrid_shortest_path_seeds
    if seeds is None and args.task_order_config is not None:
        with open(args.task_order_config, "r") as f:
            cfg = json.load(f)
        seeds = cfg.get("seeds")

    scales = []
    for idx, task_name in enumerate(ordered_task_names):
        seed = None
        if isinstance(seeds, list) and idx < len(seeds):
            seed = seeds[idx]
        scales.append(shortest_path_return_for_minigrid_task(task_name, seed=seed))

    return np.asarray(scales, dtype=float), mode


def maybe_print_return_scales(scales, mode, task_names):
    if mode != "minigrid_shortest_path":
        return
    print("MiniGrid shortest-path return normalization:")
    for idx, scale in enumerate(scales):
        label = task_names[idx] if task_names and idx < len(task_names) else f"task{idx}"
        print(f"  task{idx}: max_return={scale:.6f} {label}")


def compute_forgetting(mat):
    mat = np.asarray(mat, float)
    start_vals = np.nanmax(mat, axis=0)
    final_vals = mat[-1, :]
    per_task = start_vals - final_vals
    return float(np.nanmean(per_task))


def compute_rel_forgetting(mat):
    mat = np.asarray(mat, float)
    start_vals = np.nanmax(mat, axis=0)
    final_vals = mat[-1, :]
    denom = np.where(np.isclose(start_vals, 0.0), np.nan, start_vals)
    per_task = (start_vals - final_vals) / denom
    return float(np.nanmean(per_task))


def compute_forgetting_cw(mat):
    """Continual World Section 4.1 forgetting: p_i(i*Delta) - p_i(T)."""
    mat = np.asarray(mat, float)
    n_tasks = mat.shape[1]
    idx = task_end_indices(mat.shape[0], n_tasks)
    task_end_vals = mat[idx, np.arange(n_tasks)]
    final_vals = mat[-1, :]
    per_task = task_end_vals - final_vals
    return float(np.nanmean(per_task))


def compute_bwt(mat):
    t_steps, n_tasks = mat.shape
    if t_steps % n_tasks != 0:
        raise ValueError(f"Cannot infer per-task evaluation blocks: {t_steps} rows, {n_tasks} tasks")
    evals_per_task = t_steps // n_tasks
    idx = np.arange(n_tasks) * evals_per_task + (evals_per_task - 1)
    r_ii = mat[idx, np.arange(n_tasks)]
    r_t = mat[-1, :]
    if n_tasks <= 1:
        return 0.0
    return float(np.nanmean(r_t[:-1] - r_ii[:-1]))


def main():
    ap = argparse.ArgumentParser(
        description="Unified significance testing for eval/fwt/forgetting/BWT metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--ref", required=True)
    ap.add_argument("--methods", nargs="+", required=True, help="NAME=PATH")
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument(
        "--metric",
        choices=[
            "eval",
            "auc",
            "auc_global",
            "tpot",
            "fwt",
            "forgetting",
            "rel_forgetting",
            "fgt_cw",
            "bwt",
        ],
        default="eval",
    )
    # Used in the paper: tpot, fwt, forgetting, bwt
    ap.add_argument("--expert-root", type=str, default=None, help="Required for --metric fwt")
    ap.add_argument("--task-order-config", type=str, default=None, help="Task order config for experts")
    ap.add_argument(
        "--max-return",
        type=float,
        default=1.0,
        help=(
            "Scalar max return used when --performance-normalization=scalar. "
            "CT-graph normally uses 1.0."
        ),
    )
    ap.add_argument(
        "--performance-normalization",
        choices=["auto", "scalar", "minigrid_shortest_path", "none"],
        default="auto",
        help=(
            "How to map raw eval returns to p_i(t) in [0, 1]. `auto` uses "
            "MiniGrid shortest-path returns when the task config has MiniGrid "
            "task names, otherwise scalar --max-return."
        ),
    )
    ap.add_argument(
        "--minigrid-shortest-path-seeds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional per-task seeds for MiniGrid shortest-path normalization. "
            "Defaults to the `seeds` list in --task-order-config when present."
        ),
    )
    ap.add_argument(
        "--print-return-scales",
        action="store_true",
        help="Print the per-task return normalization constants.",
    )
    ap.add_argument(
        "--include-unseen-eval-tasks",
        action="store_true",
        help=(
            "Legacy behavior: include future/unseen task columns when computing "
            "eval/AUC/TPOT/forgetting/BWT from eval_metrics. By default these "
            "metrics use only task columns exposed at each eval row. FWT is "
            "unchanged by this flag."
        ),
    )
    ap.add_argument("--min-denominator", type=float, default=1e-3, help="Min (1 - AUC_expert) before FWT is NaN")
    ap.add_argument(
        "--fwt-task-ids",
        type=int,
        nargs="+",
        default=None,
        help=(
            "For --metric fwt, average only these task column ids. Useful for "
            "excluding cold-start tasks from a hierarchical curriculum."
        ),
    )
    ap.add_argument(
        "--exclude-fwt-task-ids",
        type=int,
        nargs="+",
        default=None,
        help="For --metric fwt, exclude these task column ids from the average.",
    )
    ap.add_argument(
        "--print-fwt-details",
        action="store_true",
        help="Print per-task mean lifelong AUC, expert AUC, denominator, and FWT.",
    )
    ap.add_argument(
        "--tests",
        nargs="+",
        choices=["welch", "paired_t", "wilcoxon", "sign_flip", "all"],
        default=["welch"],
        help="Significance tests to report. Use `all` to compare Welch and paired tests.",
    )
    ap.add_argument(
        "--alternative",
        choices=["two-sided", "greater", "less"],
        default="two-sided",
        help="Alternative hypothesis for paired tests on ref - method. 'greater' for AUC and FWT (higher is better) and 'less' for FGT and BWT",
    )
    ap.add_argument(
        "--exact-max-pairs",
        type=int,
        default=20,
        help="Use exact sign-flip enumeration up to this many paired runs; above it, sample --iters flips.",
    )
    ap.add_argument("--print-per-run", action="store_true")
    args = ap.parse_args()

    methods = parse_methods(args.methods)
    if args.ref not in methods:
        raise ValueError(f"Reference {args.ref} not provided")

    # Load experts if needed
    task_order, task_names = load_task_order_config(args.task_order_config)
    return_scales_cache = None
    normalization_mode = None

    def get_return_scales(num_tasks):
        nonlocal return_scales_cache, normalization_mode
        if return_scales_cache is None or len(return_scales_cache) != num_tasks:
            return_scales_cache, normalization_mode = resolve_return_scales(
                args, task_names, task_order, num_tasks
            )
            if args.print_return_scales:
                maybe_print_return_scales(
                    return_scales_cache,
                    normalization_mode,
                    task_names_in_eval_order(task_names, task_order, num_tasks),
                )
        return return_scales_cache

    configured_num_tasks = None
    if task_names is not None:
        configured_num_tasks = len(task_names)
    elif task_order is not None:
        configured_num_tasks = len(task_order)
    if configured_num_tasks is not None:
        get_return_scales(configured_num_tasks)

    expert_by_seed = None
    default_expert_auc = None
    if args.metric == "fwt":
        if args.expert_root is None:
            raise ValueError("--expert-root required for --metric fwt")
        eruns = find_runs(args.expert_root)
        if not eruns:
            raise ValueError(f"No expert eval_metrics under {args.expert_root}")
        if task_order is None:
            task_ids = []
            for p in eruns:
                task_id = extract_expert_task_id(p, task_names)
                if task_id is not None:
                    task_ids.append(task_id)
            if task_ids:
                task_ids_sorted = sorted(set(task_ids))
                consecutive = (task_ids_sorted[-1] - task_ids_sorted[0] + 1) == len(task_ids_sorted)
                small_ids = task_ids_sorted[0] in (0, 1) and task_ids_sorted[-1] < 1000
                if not (consecutive and small_ids):
                    print(
                        "WARNING: --task-order-config not provided and expert task ids look non-consecutive. "
                        "FWT may be misaligned. Consider passing --task-order-config.",
                    )

        def load_expert_set(paths):
            id_to_path = {}
            for p in paths:
                task_id = extract_expert_task_id(p, task_names)
                if task_id is not None:
                    if task_id in id_to_path:
                        raise ValueError(
                            f"Duplicate expert run for task{task_id}: "
                            f"{id_to_path[task_id]} and {p}"
                        )
                    id_to_path[task_id] = p
            if task_order:
                expected_ids = set(task_order)
                observed_ids = set(id_to_path)
                missing_ids = sorted(expected_ids - observed_ids)
                extra_ids = sorted(observed_ids - expected_ids)
                if missing_ids or extra_ids:
                    raise ValueError(
                        "Expert task ids do not match --task-order-config. "
                        f"Missing expected ids: {missing_ids}; "
                        f"unexpected extra ids: {extra_ids}. "
                        "For MiniGrid curriculum configs with a `tasks` list, "
                        "expected ids are 0..len(tasks)-1 unless you provide an "
                        "explicit `task_order` field."
                    )
                ordered = [id_to_path[tid] for tid in task_order]
            else:
                ordered = sorted(paths)
            return_scales = get_return_scales(len(ordered))
            aucs = []
            for pos, p in enumerate(ordered):
                emat = load_eval_matrix(p)
                if emat.shape[1] != 1:
                    raise ValueError(f"Expert {p} should have 1 task column")
                aucs.append(
                    normalized_single_task_auc(
                        emat,
                        return_scales[pos],
                        clip=normalization_mode != "none",
                    )
                )
            return np.asarray(aucs)

        tmp = {}
        for er in eruns:
            sd = re.search(r"seed(\d+)", er)
            if sd:
                tmp.setdefault(sd.group(1), []).append(er)
        if tmp:
            expert_by_seed = {s: load_expert_set(ps) for s, ps in tmp.items()}
        else:
            default_expert_auc = load_expert_set(eruns)

    tests = expand_tests(args.tests)
    samples = {}
    seed_samples = {}
    for name, root in methods.items():
        runs = find_runs(root)
        if not runs:
            raise ValueError(f"No eval_metrics under {root}")
        vals = []
        vals_by_seed = {}
        fwt_details = []
        for rd in runs:
            seed = extract_seed_generic(rd)
            mat = load_eval_matrix(rd)
            return_scales = get_return_scales(mat.shape[1])
            clip_norm = normalization_mode != "none"
            norm_mat = normalize_performance_matrix(mat, return_scales, clip=clip_norm)
            if args.metric != "fwt" and not args.include_unseen_eval_tasks:
                try:
                    metric_mat = mask_unseen_eval_tasks(norm_mat)
                except ValueError as exc:
                    raise ValueError(
                        f"Could not infer seen-task exposure mask for {rd}: {exc}. "
                        "Pass --include-unseen-eval-tasks to use the legacy "
                        "all-task eval behavior."
                    ) from exc
            else:
                metric_mat = norm_mat
            if args.metric == "eval":
                val = float(np.nansum(metric_mat[-1, :]))
            elif args.metric in ("auc", "auc_global", "tpot"):
                if args.metric == "auc":
                    val = float(np.nanmean(per_task_auc01_from_matrix(metric_mat)))
                else:
                    val = float(np.trapz(np.nansum(metric_mat, axis=1), dx=1.0))
            elif args.metric == "fwt":
                ll_auc = normalized_own_task_auc(
                    mat, return_scales, clip=clip_norm
                )
                exp_auc = None
                if expert_by_seed and seed in expert_by_seed:
                    exp_auc = expert_by_seed[seed]
                elif default_expert_auc is not None:
                    exp_auc = default_expert_auc
                else:
                    raise ValueError(f"No experts for seed {seed}")
                fwt_vec = forward_transfer(ll_auc, exp_auc, min_den=args.min_denominator)
                fwt_task_ids = resolve_metric_task_ids(
                    len(fwt_vec),
                    include_ids=args.fwt_task_ids,
                    exclude_ids=args.exclude_fwt_task_ids,
                )
                val = float(np.nanmean(fwt_vec[fwt_task_ids]))
                if args.print_fwt_details:
                    fwt_details.append((ll_auc.copy(), exp_auc.copy(), fwt_vec.copy()))
            elif args.metric == "forgetting":
                val = compute_forgetting(metric_mat)
            elif args.metric == "rel_forgetting":
                val = compute_rel_forgetting(metric_mat)
            elif args.metric == "fgt_cw":
                val = compute_forgetting_cw(metric_mat)
            else:
                val = compute_bwt(metric_mat)
            vals.append(val)
            if seed is not None:
                vals_by_seed.setdefault(seed, []).append(val)
            if args.print_per_run:
                seed_str = f"seed{seed}" if seed is not None else "seed?"
                print(f"{name} {seed_str}: {args.metric} = {val:.4f}")

        vals = np.asarray(vals, float)
        samples[name] = vals
        seed_samples[name] = vals_by_seed
        if args.metric == "fwt" and args.print_fwt_details and fwt_details:
            ll_stack = np.asarray([row[0] for row in fwt_details])
            exp_stack = np.asarray([row[1] for row in fwt_details])
            fwt_stack = np.asarray([row[2] for row in fwt_details])
            fwt_task_ids = resolve_metric_task_ids(
                fwt_stack.shape[1],
                include_ids=args.fwt_task_ids,
                exclude_ids=args.exclude_fwt_task_ids,
            )
            ordered_names = task_names_in_eval_order(
                task_names, task_order, fwt_stack.shape[1]
            )
            print(f"{name} FWT per-task means:")
            for tid in fwt_task_ids:
                label = ordered_names[tid] if ordered_names else f"task{tid}"
                label = label.replace("CurriculumMultiRoomObjectRemapEnv-", "")
                label = label.replace("-v0", "")
                exp_mean = float(np.nanmean(exp_stack[:, tid]))
                denom_mean = float(np.nanmean(1.0 - exp_stack[:, tid]))
                print(
                    f"  task{tid} {label}: "
                    f"ll_auc={np.nanmean(ll_stack[:, tid]):.4f}, "
                    f"expert_auc={exp_mean:.4f}, "
                    f"denom={denom_mean:.4f}, "
                    f"fwt={np.nanmean(fwt_stack[:, tid]):.4f}"
                )
        mean = float(np.nanmean(vals))
        lo, hi = bootstrap_mean_ci(vals, iters=args.iters)
        print(f"{name}: {len(vals)} runs, mean={mean:.4f}, 95% CI=[{lo:.4f}, {hi:.4f}]")

    ref_vals = samples[args.ref]
    for name, vals in samples.items():
        if name == args.ref:
            continue
        show_only_welch = tests == ["welch"]
        if show_only_welch:
            p = welch_ttest(ref_vals, vals)
            lo, hi = bootstrap_diff_ci(ref_vals, vals, iters=args.iters)
            print(f"Compare {args.ref} vs {name}: p={p:.3e}, BCI=[{lo:.4f}, {hi:.4f}] (μ_ref - μ_{name})")
            continue

        mean_diff = float(np.nanmean(ref_vals) - np.nanmean(vals))
        print(f"Compare {args.ref} vs {name}: mean_diff={mean_diff:.4f} (μ_ref - μ_{name})")

        if "welch" in tests:
            p = welch_ttest(ref_vals, vals)
            lo, hi = bootstrap_diff_ci(ref_vals, vals, iters=args.iters)
            print(f"  welch_unpaired: p={p:.3e}, BCI=[{lo:.4f}, {hi:.4f}]")

        paired_tests = {"paired_t", "wilcoxon", "sign_flip"} & set(tests)
        if paired_tests:
            seeds, diffs, skipped = paired_diffs_by_seed(seed_samples[args.ref], seed_samples[name])
            if len(diffs) == 0:
                print("  paired: skipped, no uniquely matched seeds")
                continue
            lo, hi = bootstrap_paired_diff_ci(diffs, iters=args.iters)
            seed_list = ",".join(seeds)
            print(
                f"  paired: n={len(diffs)}, mean_delta={np.mean(diffs):.4f}, "
                f"BCI=[{lo:.4f}, {hi:.4f}], seeds=[{seed_list}]"
            )
            if skipped:
                print(f"  paired: skipped duplicate seeds=[{','.join(skipped)}]")
            if "paired_t" in tests:
                p = paired_ttest_from_diffs(diffs, alternative=args.alternative)
                print(f"  paired_t ({args.alternative}): p={p:.3e}")
            if "wilcoxon" in tests:
                p = wilcoxon_signed_rank_from_diffs(diffs, alternative=args.alternative)
                print(f"  wilcoxon_signed_rank ({args.alternative}): p={p:.3e}")
            if "sign_flip" in tests:
                p, exact = paired_sign_flip_test(
                    diffs,
                    alternative=args.alternative,
                    exact_max_pairs=args.exact_max_pairs,
                    iters=args.iters,
                )
                label = "exact" if exact else f"sampled_{args.iters}"
                print(f"  paired_sign_flip_{label} ({args.alternative}): p={p:.3e}")


if __name__ == "__main__":
    main()
