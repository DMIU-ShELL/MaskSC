#!/usr/bin/env python3
import argparse
import os
import re
import numpy as np

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
    return mat

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
    m = re.search(r"-([0-9]{2,5})-(?:mask|ppo|supermask|ewc|si|linear|ct)", path)
    if m:
        return m.group(1)
    m = re.search(r"/([0-9]{2,5})/", path)
    if m:
        return m.group(1)
    return None

def compute_bwt(mat):
    # Backward transfer: mean over tasks of (final performance - performance right after learning the task).
    # Assumes eval_metrics rows are evenly spaced per task.
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", required=True, help="NAME=PATH")
    ap.add_argument("--threshold", type=float, default=0.9,
                    help="Fraction of max_return required to count task as learned")
    ap.add_argument("--max-return", type=float, default=1.0,
                    help="Max return used to scale the threshold")
    ap.add_argument(
        "--metric",
        choices=["final", "auc", "forgetting", "rel_forgetting", "bwt"],
        default="final",
        help="Metric to compute: final/auc -> count tasks learned; forgetting -> avg(max-final); rel_forgetting -> avg((max-final)/max); bwt -> avg(final - post-task)",
    )
    ap.add_argument("--print-per-run", action="store_true",
                    help="Print task counts per run")
    args = ap.parse_args()

    methods = parse_methods(args.methods)
    threshold_value = args.threshold * args.max_return

    for name, root in methods.items():
        runs = find_runs(root)
        if not runs:
            raise ValueError(f"No eval_metrics under {root}")
        values = []
        for rd in runs:
            mat = load_eval_matrix(rd)
            if args.metric == "final":
                per_task = mat[-1, :]
                metric_value = int(np.sum(per_task >= threshold_value))
                metric_label = "learned"
            elif args.metric == "auc":
                per_task = np.trapz(mat, axis=0, dx=1.0)
                metric_value = int(np.sum(per_task >= threshold_value))
                metric_label = "learned"
            elif args.metric == "forgetting":
                per_task = np.nanmax(mat, axis=0) - mat[-1, :]
                metric_value = float(np.nanmean(per_task))
                metric_label = "avg_forgetting"
            elif args.metric == "rel_forgetting":
                max_vals = np.nanmax(mat, axis=0)
                denom = np.where(max_vals == 0, np.nan, max_vals)
                per_task = (max_vals - mat[-1, :]) / denom
                metric_value = float(np.nanmean(per_task))
                metric_label = "avg_rel_forgetting"
            else:
                metric_value = compute_bwt(mat)
                metric_label = "bwt"

            values.append(metric_value)
            if args.print_per_run:
                seed = extract_seed_generic(rd)
                seed_str = f"seed{seed}" if seed is not None else "seed?"
                if metric_label == "learned":
                    print(f"{name} {seed_str}: {metric_value}/{per_task.shape[0]} learned")
                else:
                    print(f"{name} {seed_str}: {metric_label} = {metric_value:.4f}")

        values = np.asarray(values, dtype=float)
        mean = np.mean(values)
        std = np.std(values, ddof=1) if len(values) > 1 else 0.0
        if metric_label == "learned":
            print(f"{name}: {len(values)} runs, avg tasks learned = {mean:.2f} ± {std:.2f}")
        else:
            print(f"{name}: {len(values)} runs, {metric_label} = {mean:.4f} ± {std:.4f}")

if __name__ == "__main__":
    main()
