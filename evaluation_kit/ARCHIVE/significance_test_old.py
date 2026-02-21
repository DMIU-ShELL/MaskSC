#!/usr/bin/env python3
import argparse, os, json, re, numpy as np
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

def final_total_perf(run_dir):
    mat = load_eval_matrix(run_dir)
    return float(mat[-1, :].sum())  # sum over tasks at final checkpoint

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
    lo = diffs[int((alpha/2)*iters)]
    hi = diffs[int((1 - alpha/2)*iters)]
    return lo, hi

def parse_methods(arglist):
    methods = {}
    for item in arglist:
        if "=" not in item:
            raise ValueError(f"Method spec should be NAME=PATH, got {item}")
        name, path = item.split("=", 1)
        methods[name] = path
    return methods

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="Reference method name (e.g., MaskLC)")
    ap.add_argument("--methods", nargs="+", required=True,
                    help="Method specs NAME=PATH where PATH is top-level dir containing seed runs")
    ap.add_argument("--iters", type=int, default=10000, help="Bootstrap iterations")
    args = ap.parse_args()

    methods = parse_methods(args.methods)
    if args.ref not in methods:
        raise ValueError(f"Reference {args.ref} not provided in --methods")

    samples = {}
    for name, root in methods.items():
        runs = find_runs(root)
        if not runs:
            raise ValueError(f"No eval_metrics found under {root}")
        vals = [final_total_perf(rd) for rd in runs]
        samples[name] = np.asarray(vals, dtype=float)
        print(f"{name}: {len(vals)} runs, mean={np.mean(vals):.2f}, std={np.std(vals, ddof=1):.2f}")

    ref_vals = samples[args.ref]
    for name, vals in samples.items():
        if name == args.ref:
            continue
        p = welch_ttest(ref_vals, vals)
        lo, hi = bootstrap_ci(ref_vals, vals, iters=args.iters)
        print(f"Compare {args.ref} vs {name}: p={p:.3e}, BCI=[{lo:.2f}, {hi:.2f}] (μ_ref - μ_{name})")

if __name__ == "__main__":
    main()
