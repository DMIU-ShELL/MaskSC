#!/usr/bin/env python3
"""
Run a single transfer probe for a given (current_task, prior_task) pair by
loading a trained Mask-SC checkpoint, mixing the prior/new heads (0.5/0.5),
training the new task for a short budget, and reporting the resulting utility.

Example:
python probe_transfer_pair.py \
  --env_config env_configs/ct28/seed1/meta_ctgraph_ct28_interleaved.json \
  --model_path log/ct28-interleaved-MaskSC-top-dense/runs/MetaCTgraph-ppo-supermask-86-mask-linear_comb-ct14_md/260112-205346/DetectLLAgent-<tag>-model-ctgraph.bin \
  --curr_idx 7 --prior_idx 3 --steps_per_task 51200 \
  --out_csv /tmp/probe_results.csv
"""

import argparse
import copy
import csv
import json
import os
import re
import sys
import time

import numpy as np
import torch

# Import project utils
from deep_rl import *
from deep_rl.agent.PPO_agent import DetectLLAgent
from deep_rl.mask_modules import (
    set_model_task,
    set_num_tasks_learned,
    consolidate_mask,
    set_selected_task_indices,
)


def trapz_over_checkpoints(curve):
    # Uniform spacing over checkpoints; integrates “area under learning curve”
    return float(np.trapz(np.asarray(curve, dtype=np.float32), dx=1.0))


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
    # Only drop a trailing time column for CSVs
    if is_csv and mat.shape[1] > 1 and np.all(np.diff(mat[:, -1]) >= 0):
        mat = mat[:, :-1]
    return mat  # shape [num_checkpoints, num_tasks]


def find_expert_run(expert_root, task_id, seed=None):
    if not expert_root:
        return None
    candidates = []
    task_pat = re.compile(rf"task{task_id}(?:\\b|_|-|/)")
    for dirpath, _, filenames in os.walk(expert_root):
        if not any(fname in ("eval_metrics.npy", "eval_metrics.csv") for fname in filenames):
            continue
        if task_pat.search(dirpath) is None:
            continue
        candidates.append(dirpath)
    if not candidates:
        return None
    if seed is not None:
        seed_pat = re.compile(rf"(?:seed{seed}|supermask-{seed}|-{seed}-)")
        seeded = [p for p in candidates if seed_pat.search(p)]
        if seeded:
            candidates = seeded
    return sorted(candidates)[-1]


def build_config(env_name, env_config_path, seed, max_steps, exp_id, new_task_mask="linear_comb"):
    """Mirror the Mask-SC eval/training config for CT-graph."""
    config = Config()
    config.env_name = env_name
    config.env_config_path = env_config_path
    config.lr = 0.00015
    config.cl_preservation = "supermask"
    config.seed = seed
    random_seed(config.seed)
    exp_id = '-{0}-mask-{1}-{2}'.format(config.seed, new_task_mask, exp_id)
    log_name = f'probe-experiments{config.seed}/' + env_name + '-ppo' + '-' + config.cl_preservation + exp_id
    config.log_dir = get_default_log_dir(log_name)
    config.num_workers = 4

    with open(env_config_path, "r") as f:
        env_cfg = json.load(f)
    num_tasks = env_cfg["num_tasks"]

    task_fn = lambda log_dir: MetaCTgraphFlatObs(env_name, env_config_path, log_dir)
    config.task_fn = lambda: ParallelizedTask(task_fn, config.num_workers, log_dir=config.log_dir)
    eval_task_fn = lambda log_dir: MetaCTgraphFlatObs(env_name, env_config_path, log_dir)
    config.eval_task_fn = eval_task_fn
    config.optimizer_fn = lambda params, lr: torch.optim.RMSprop(params, lr=lr)
    config.network_fn = lambda state_dim, action_dim, label_dim: CategoricalActorCriticNet_SS(
        state_dim,
        action_dim,
        label_dim,
        phi_body=FCBody_SS(
            state_dim,
            task_label_dim=label_dim,
            hidden_units=(200, 200, 200),
            num_tasks=num_tasks,
            new_task_mask=new_task_mask,
        ),
        actor_body=DummyBody_CL(200),
        critic_body=DummyBody_CL(200),
        num_tasks=num_tasks,
        new_task_mask=new_task_mask,
    )
    config.policy_fn = SamplePolicy
    config.state_normalizer = ImageNormalizer()
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.99
    config.entropy_weight = 0.1
    config.rollout_length = 128
    config.optimization_epochs = 8
    config.num_mini_batches = 64
    config.ppo_ratio_clip = 0.1
    config.iteration_log_interval = 1
    config.gradient_clip = 5
    config.max_steps = max_steps
    config.evaluation_episodes = 10
    config.logger = get_logger(log_dir=config.log_dir, file_name="probe-log")
    config.cl_requires_task_label = True
    config.eval_interval = 10
    config.task_ids = np.arange(num_tasks).tolist()

    # Detect module settings (mirror training defaults)
    config.detect_reference_num = 50
    config.detect_num_samples = 128
    config.detect_emb_dist_threshold = 24
    config.detect_frequency = 1
    config.detect_fn = lambda input_dim, action_dim: Detect(
        config.detect_reference_num,
        input_dim,
        action_dim,
        config.detect_num_samples,
        one_hot=True,
        normalized=True,
    )
    config.detect_topk = 3
    config.select_frequency = 5
    config.warmup_steps = 10000
    config.wte_momentum = 0.5

    config.tag = exp_id
    return config, num_tasks


def load_agent(agent, path, num_tasks_learnt, tasks):
    agent.load(path)
    for idx in range(num_tasks_learnt):
        agent.seen_tasks[idx] = tasks[idx]["task_label"]
    set_num_tasks_learned(agent.network, num_tasks_learnt - 1)
    set_model_task(agent.network, num_tasks_learnt - 1)
    consolidate_mask(agent.network)
    set_num_tasks_learned(agent.network, num_tasks_learnt)
    return agent


def _resolve_model_path(path):
    """If `path` is a directory, pick the highest-task *model* bin from task_stats; otherwise return path."""
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        raise ValueError(f"Model path {path} is neither file nor directory.")

    best_model = None
    best_model_task = -1
    best_any = None
    best_any_task = -1
    for dirpath, _, files in os.walk(path):
        if "task_stats" not in dirpath:
            continue
        for fname in files:
            m = re.search(r"task-(\d+)\.bin", fname)
            if not m:
                continue
            t = int(m.group(1))
            full = os.path.join(dirpath, fname)
            if "model" in fname and t > best_model_task:
                best_model_task = t
                best_model = full
            if t > best_any_task:
                best_any_task = t
                best_any = full
    best = best_model or best_any
    if best is None:
        raise ValueError(f"No task_* checkpoint found under {path}")
    return best


def run_probe(args):
    config, num_tasks = build_config(
        env_name=args.env_name,
        env_config_path=args.env_config,
        seed=args.seed,
        max_steps=args.steps_per_task,
        exp_id=args.exp_id,
        new_task_mask=args.new_task_mask,
    )
    if args.eval_interval is not None:
        config.eval_interval = args.eval_interval
    if args.eval_episodes is not None:
        config.evaluation_episodes = args.eval_episodes
    config.log_dir = args.tmp_log
    mkdir(config.log_dir)

    agent = DetectLLAgent(config)
    tasks = agent.config.cl_tasks_info

    curr_idx = args.curr_idx
    prior_idx = args.prior_idx
    if not (0 <= prior_idx < curr_idx < num_tasks):
        raise ValueError(f"Invalid indices: prior {prior_idx}, curr {curr_idx}, num_tasks {num_tasks}")

    # resolve/load checkpoint
    model_path = _resolve_model_path(args.model_path)
    # load checkpoint corresponding to tasks seen up to (curr_idx - 1)
    agent = load_agent(agent, model_path, curr_idx, tasks)

    # snapshot base state
    base_net = copy.deepcopy(agent.network.state_dict())
    base_opt = copy.deepcopy(agent.opt.state_dict())

    # activate new task head/mask
    set_model_task(agent.network, curr_idx, new_task=True)
    agent.curr_train_task_label = tasks[curr_idx]["task_label"]
    # set current task for training
    agent.task.reset_task(tasks[curr_idx])
    agent.states = config.state_normalizer(agent.task.reset_task(tasks[curr_idx]))
    agent.task_train_start(tasks[curr_idx]["task_label"])

    # restore clean state
    agent.network.load_state_dict(base_net)
    agent.opt.load_state_dict(base_opt)

    # restrict LC to a single prior + current task mask
    set_selected_task_indices(agent.network, [prior_idx])

    # set betas so softmax over [prior_idx, curr_idx] is 0.5/0.5
    for module in agent.network.modules():
        if hasattr(module, "betas") and module.betas is not None:
            module.betas.data[curr_idx].fill_(-1e9)
            module.betas.data[curr_idx, prior_idx] = 0.0
            module.betas.data[curr_idx, curr_idx] = 0.0

    def _eval_current(iteration):
        agent.task_eval_start(tasks[curr_idx]["task_label"])
        agent.evaluation_env.reset_task(tasks[curr_idx])
        perf, _ = agent.evaluate_cl(num_iterations=config.evaluation_episodes)
        agent.task_eval_end()
        mean_perf = float(np.mean(perf))
        eval_curve.append(mean_perf)
        eval_steps.append(iteration)

    rollout_len = config.rollout_length
    # iters = max(1, args.steps_per_task // rollout_len)
    iters = max(1, args.steps_per_task // (rollout_len * config.num_workers))
    start_ts = time.time()
    eval_curve = []
    eval_steps = []
    for i in range(iters):
        agent.iteration()
        iter_no = i + 1
        if config.eval_interval and (iter_no % config.eval_interval == 0):
            _eval_current(iter_no)
        if args.print_every > 0:
            if i == 0 or iter_no % args.print_every == 0 or iter_no == iters:
                elapsed = time.time() - start_ts
                print(f"[probe] iter {iter_no}/{iters} elapsed={elapsed:.1f}s", flush=True)

    agent.task_train_end()

    # final evaluation (also acts as utility metric)
    agent.task_eval_start(tasks[curr_idx]["task_label"])
    agent.evaluation_env.reset_task(tasks[curr_idx])
    perf, _ = agent.evaluate_cl(num_iterations=config.evaluation_episodes)
    utility = float(np.mean(perf))
    agent.task_eval_end()

    # persist eval curve for AUC
    eval_mat = None
    if eval_curve:
        eval_mat = np.asarray(eval_curve, dtype=np.float32).reshape(-1, 1)
        np.save(os.path.join(config.log_dir, "eval_metrics.npy"), eval_mat)
        eval_csv = os.path.join(config.log_dir, "eval_metrics.csv")
        times = np.asarray(eval_steps, dtype=np.float32).reshape(-1, 1)
        np.savetxt(eval_csv, np.hstack([eval_mat, times]), delimiter=",", fmt="%.6f")

    agent.close()

    # AUC + optional FWT
    auc = trapz_over_checkpoints(eval_curve) if eval_curve else float("nan")
    max_return = args.max_return
    auc_norm = float(np.clip(auc / max_return, 0.0, 1.0)) if max_return else auc
    expert_auc = float("nan")
    expert_auc_norm = float("nan")
    fwt = float("nan")

    if args.expert_root:
        # map task index to environment task id (filter_tasks if present)
        with open(args.env_config, "r") as f:
            env_cfg = json.load(f)
        filter_tasks = env_cfg.get("filter_tasks", None)
        task_id = filter_tasks[curr_idx] if filter_tasks else curr_idx

        expert_run = find_expert_run(args.expert_root, task_id, seed=args.seed)
        if expert_run:
            try:
                emat = load_eval_matrix(expert_run)
                if emat.shape[1] != 1:
                    raise ValueError(f"Expert run {expert_run} should have exactly one task column.")
                expert_auc = trapz_over_checkpoints(emat[:, 0])
                expert_auc_norm = float(np.clip(expert_auc / max_return, 0.0, 1.0)) if max_return else expert_auc
                denom = 1.0 - expert_auc_norm
                if denom >= args.min_denominator:
                    fwt = (auc_norm - expert_auc_norm) / denom
            except Exception as e:
                print(f"[WARN] Failed to load expert AUC from {expert_run}: {e}", flush=True)
        else:
            print(f"[WARN] No expert run found for task_id={task_id} under {args.expert_root}", flush=True)

    # write/append result
    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_exists = os.path.isfile(args.out_csv)
    with open(args.out_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if not out_exists:
            writer.writerow([
                "seed",
                "run_path",
                "current_task",
                "prior_task",
                "utility",
                "auc",
                "auc_norm",
                "expert_auc",
                "expert_auc_norm",
                "fwt",
                "n_eval",
            ])
        writer.writerow([
            args.seed,
            model_path,
            curr_idx,
            prior_idx,
            utility,
            auc,
            auc_norm,
            expert_auc,
            expert_auc_norm,
            fwt,
            len(eval_curve),
        ])
    return utility


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_name", default="ctgraph")
    ap.add_argument("--env_config", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--curr_idx", type=int, required=True)
    ap.add_argument("--prior_idx", type=int, required=True)
    ap.add_argument("--steps_per_task", type=int, default=51200)
    ap.add_argument("--exp_id", type=str, default="probe")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tmp_log", type=str, default="/tmp/probe_transfer")
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--new_task_mask", type=str, default="linear_comb")
    ap.add_argument("--print_every", type=int, default=50, help="Progress print interval (iterations). 0 disables.")
    ap.add_argument("--eval_interval", type=int, default=None, help="Override eval interval (iterations).")
    ap.add_argument("--eval_episodes", type=int, default=None, help="Override evaluation episodes.")
    ap.add_argument("--expert_root", type=str, default=None, help="Root for single-task expert runs (eval_metrics.*).")
    ap.add_argument("--max_return", type=float, default=1.0, help="Max return for AUC normalization.")
    ap.add_argument("--min_denominator", type=float, default=1e-3, help="Min (1 - AUC_expert) before FWT is NaN.")
    args = ap.parse_args()

    select_device(0)  # change if needed
    utility = run_probe(args)
    print(f"Probe utility curr={args.curr_idx} prior={args.prior_idx}: {utility:.4f}")


if __name__ == "__main__":
    main()
