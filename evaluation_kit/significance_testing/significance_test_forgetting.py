#!/usr/bin/env python3
import argparse
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


def welch_ttest(x, y):
    t, p = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
    return p


def bootstrap_ci(x, y, iters=10000, alpha=0.05, rng=None):
    rng = rng or np.random.default_rng()
    x = np.asarray(x)
    y = np.asarray(y)
    diffs = []
    for _ in range(iters):
        xb = rng.choice(x, size=len(x), replace=True)
        yb = rng.choice(y, size=len(y), replace=True)
        diffs.append(xb.mean() - yb.mean())
    diffs = np.sort(diffs)
    lo = diffs[int((alpha / 2) * iters)]
    hi = diffs[int((1 - alpha / 2) * iters)]
    return lo, hi


def compute_forgetting(mat):
    max_vals = np.nanmax(mat, axis=0)
    final_vals = mat[-1, :]
    per_task = max_vals - final_vals
    return float(np.nanmean(per_task))


def compute_rel_forgetting(mat):
    max_vals = np.nanmax(mat, axis=0)
    final_vals = mat[-1, :]
    denom = np.where(max_vals == 0, np.nan, max_vals)
    per_task = (max_vals - final_vals) / denom
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--methods", nargs="+", required=True, help="NAME=PATH")
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--metric", choices=["forgetting", "rel_forgetting", "bwt"], default="forgetting")
    ap.add_argument("--print-per-run", action="store_true")
    args = ap.parse_args()

    methods = parse_methods(args.methods)
    if args.ref not in methods:
        raise ValueError(f"Reference {args.ref} not provided")

    samples = {}
    for name, root in methods.items():
        runs = find_runs(root)
        if not runs:
            raise ValueError(f"No eval_metrics under {root}")
        vals = []
        for rd in runs:
            mat = load_eval_matrix(rd)
            if args.metric == "forgetting":
                val = compute_forgetting(mat)
            elif args.metric == "rel_forgetting":
                val = compute_rel_forgetting(mat)
            else:
                val = compute_bwt(mat)
            vals.append(val)
            if args.print_per_run:
                seed = extract_seed_generic(rd)
                seed_str = f"seed{seed}" if seed is not None else "seed?"
                print(f"{name} {seed_str}: {args.metric} = {val:.4f}")
        vals = np.asarray(vals, float)
        samples[name] = vals
        mean = float(np.nanmean(vals))
        lo, hi = bootstrap_ci(vals, vals, iters=args.iters)
        print(f"{name}: {len(vals)} runs, mean={mean:.4f}, 95% CI=[{lo:.4f}, {hi:.4f}]")

    ref_vals = samples[args.ref]
    for name, vals in samples.items():
        if name == args.ref:
            continue
        p = welch_ttest(ref_vals, vals)
        lo, hi = bootstrap_ci(ref_vals, vals, iters=args.iters)
        print(f"Compare {args.ref} vs {name}: p={p:.3e}, BCI=[{lo:.4f}, {hi:.4f}] (μ_ref - μ_{name})")


if __name__ == "__main__":
    main()
