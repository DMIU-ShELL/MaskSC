#!/usr/bin/env python3
"""
Enumerate (current_task, prior_task) pairs and launch transfer probes in parallel.

Each probe calls probe_transfer_pair.py to:
  - load a trained Mask-SC checkpoint
  - mix prior/new betas 0.5/0.5
  - train the current task for a budget
  - log the resulting utility to a shared CSV

Example:
python probe_transfer_jobs.py \
  --env_config env_configs/ct28/seed1/meta_ctgraph_ct28_interleaved.json \
  --run_root log/ct28-interleaved-MaskSC-top-dense/runs \
  --out_csv log/ct28-interleaved-MaskSC-top-dense/probe_utils_parallel.csv \
  --steps_per_task 51200 \
  --concurrency 4
"""

import argparse
import csv
import json
import os
import re
import subprocess
import time
from multiprocessing.pool import ThreadPool


def find_task_bins(root):
    """
    Return mapping: run_key -> {task_id: checkpoint_path}.
    Prefer files containing 'model' in the name; fall back to any task-XX.bin.
    Group by the path prefix before /task_stats.
    """
    best_model = {}
    best_any = {}
    for dirpath, _, filenames in os.walk(root):
        if "task_stats" not in dirpath:
            continue
        for fname in filenames:
            m = re.search(r"task-(\d+)\.bin", fname)
            if not m:
                continue
            t = int(m.group(1))
            run_key = dirpath.split("/task_stats")[0]
            full = os.path.join(dirpath, fname)
            # track model-only and any per task id
            if "model" in fname:
                best_model.setdefault(run_key, {})
                prev = best_model[run_key].get(t)
                if prev is None:
                    best_model[run_key][t] = full
            best_any.setdefault(run_key, {})
            prev_any = best_any[run_key].get(t)
            if prev_any is None:
                best_any[run_key][t] = full

    chosen = {}
    for rk in set(list(best_any.keys()) + list(best_model.keys())):
        chosen[rk] = {}
        all_tasks = set(best_any.get(rk, {}).keys()) | set(best_model.get(rk, {}).keys())
        for t in all_tasks:
            if rk in best_model and t in best_model[rk]:
                chosen[rk][t] = best_model[rk][t]
            else:
                chosen[rk][t] = best_any[rk][t]
    return chosen


def task_pairs(num_tasks, max_curr=None):
    pairs = []
    limit = num_tasks if max_curr is None else min(num_tasks, max_curr)
    for curr in range(1, limit):
        for prior in range(curr):
            pairs.append((curr, prior))
    return pairs


def run_probe(cmd_delay_env):
    """Wrapper for pool; supports optional delay before launch."""
    cmd, delay, env = cmd_delay_env
    if delay > 0:
        time.sleep(delay)
    ret = subprocess.call(cmd, env=env)
    return cmd, ret


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_config", required=True, help="Path to meta env config (json).")
    ap.add_argument("--run_root", required=True, help="Root containing trained runs (per-seed).")
    ap.add_argument("--out_csv", required=True, help="Aggregate CSV to append probe results.")
    ap.add_argument("--steps_per_task", type=int, default=51200)
    ap.add_argument("--concurrency", type=int, default=14)
    ap.add_argument("--max_curr", type=int, default=None, help="Optional cap on current task index.")
    ap.add_argument("--tmp_root", type=str, default="/tmp/probe_transfer_runs", help="Per-job tmp log root.")
    ap.add_argument("--job_out_root", type=str, default=None, help="Per-job CSV root (defaults to <tmp_root>/job_out).")
    ap.add_argument("--new_task_mask", type=str, default="linear_comb")
    ap.add_argument("--seed", type=int, default=86)
    ap.add_argument("--stagger", type=float, default=0.0, help="Seconds to stagger launches to avoid log collisions.")
    ap.add_argument("--env_name", type=str, default="ctgraph", help="Environment name passed to probe_transfer_pair.py.")
    ap.add_argument("--eval_interval", type=int, default=None, help="Override eval interval (iterations).")
    ap.add_argument("--eval_episodes", type=int, default=None, help="Override evaluation episodes.")
    ap.add_argument("--expert_root", type=str, default=None, help="Root for single-task expert runs (eval_metrics.*).")
    ap.add_argument("--max_return", type=float, default=1.0, help="Max return for AUC normalization.")
    ap.add_argument("--min_denominator", type=float, default=1e-3, help="Min (1 - AUC_expert) before FWT is NaN.")
    ap.add_argument(
        "--cuda_devices",
        type=str,
        default="MIG-c3ce33ce-ced8-5961-bb87-2b40eb100277,MIG-280489c4-1d98-5b07-b4f6-2fc85fc874fa,"
        "MIG-c432df19-0894-5232-ac1c-9a3440fc267e,MIG-e8f61a95-352a-56cc-b95d-0c35fc14e8bf,"
        "MIG-35ecef79-db2e-590b-9e8c-2c07c787008e,MIG-76cd8dd7-7703-5581-8ac5-a7ee81a402a0,"
        "MIG-b35e1a68-f7a4-5ef9-b34a-1abf6d1f8c2e,MIG-2d5b6364-fc42-587b-97c6-ee316a82e2f3,"
        "MIG-4590f80d-be70-58e4-af75-eeb950255d4a,MIG-e76a2a9b-9867-5f8a-b145-d857cd5ed8e2,"
        "MIG-2593b912-5975-58e9-bc3d-495311cee807,MIG-51069529-f343-59c6-bac7-a75648296e7b,"
        "MIG-187573d8-7df7-5e5f-87b2-c8b8f73c54e7,MIG-3045e3dd-28b6-5ee8-96b5-60a085c9fcf1",
        help="Comma-separated CUDA/MIG device UUIDs for round-robin assignment. Empty disables override.",
    )
    args = ap.parse_args()

    with open(args.env_config, "r") as f:
        env_cfg = json.load(f)
    num_tasks = env_cfg["num_tasks"]
    pairs = task_pairs(num_tasks, args.max_curr)

    bins_by_run = find_task_bins(args.run_root)
    if len(bins_by_run) == 0:
        raise ValueError(f"No task-* checkpoints found under {args.run_root}")

    devices = [d.strip() for d in args.cuda_devices.split(",") if d.strip()] if args.cuda_devices else []

    job_out_root = args.job_out_root or os.path.join(args.tmp_root, "job_out")
    cmds = []
    job_outs = []
    for run_key, task_bins in bins_by_run.items():
        # try to parse seed from path
        m = re.search(r"supermask-(\d+)", run_key)
        seed = int(m.group(1)) if m else args.seed
        run_tmp = os.path.join(args.tmp_root, f"seed{seed}")
        for curr, prior in pairs:
            model_path = task_bins.get(curr)
            if model_path is None:
                print(f"[WARN] Missing checkpoint for task-{curr} under {run_key}; skipping pair ({curr}, {prior})")
                continue
            job_tmp = os.path.join(run_tmp, f"curr{curr}_prior{prior}")
            job_out = os.path.join(job_out_root, f"seed{seed}", f"curr{curr}_prior{prior}.csv")
            cmd = [
                "python",
                os.path.join(os.path.dirname(__file__), "probe_transfer_pair.py"),
                "--env_name",
                args.env_name,
                "--env_config",
                args.env_config,
                "--model_path",
                model_path,
                "--curr_idx",
                str(curr),
                "--prior_idx",
                str(prior),
                "--steps_per_task",
                str(args.steps_per_task),
                "--tmp_log",
                job_tmp,
                "--out_csv",
                job_out,
                "--seed",
                str(seed),
                "--new_task_mask",
                args.new_task_mask,
            ]
            if args.eval_interval is not None:
                cmd += ["--eval_interval", str(args.eval_interval)]
            if args.eval_episodes is not None:
                cmd += ["--eval_episodes", str(args.eval_episodes)]
            if args.expert_root:
                cmd += ["--expert_root", args.expert_root]
            if args.max_return is not None:
                cmd += ["--max_return", str(args.max_return)]
            if args.min_denominator is not None:
                cmd += ["--min_denominator", str(args.min_denominator)]
            delay = args.stagger * len(cmds)
            env = os.environ.copy()
            if devices:
                env["CUDA_VISIBLE_DEVICES"] = devices[len(cmds) % len(devices)]
            cmds.append((cmd, delay, env))
            job_outs.append(job_out)

    pool = ThreadPool(processes=args.concurrency)
    for cmd, ret in pool.imap_unordered(run_probe, cmds):
        if ret != 0:
            print(f"[WARN] Command failed (rc={ret}): {' '.join(cmd)}")

    # merge per-job CSVs into aggregate output
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    seen = set()
    with open(args.out_csv, "w", newline="") as out_f:
        writer = csv.writer(out_f)
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
        for path in job_outs:
            if path in seen:
                continue
            seen.add(path)
            if not os.path.isfile(path):
                print(f"[WARN] Missing job output: {path}")
                continue
            with open(path, "r", newline="") as f:
                reader = csv.reader(f)
                for row_idx, row in enumerate(reader):
                    if row_idx == 0 and row and row[0] == "seed":
                        continue
                    if row:
                        writer.writerow(row)


if __name__ == "__main__":
    main()
