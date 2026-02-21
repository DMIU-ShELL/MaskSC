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

def per_task_auc(mat):
    return np.array([np.trapz(mat[:, j], dx=1.0) for j in range(mat.shape[1])])

def normalize_auc(arr, max_return):
    return np.clip(arr / max_return, 0.0, 1.0)

def forward_transfer(auc_ll, auc_expert, eps=1e-8, min_den=1e-3):
    denom = 1.0 - auc_expert
    fwt = (auc_ll - auc_expert) / (denom + eps)
    fwt[denom < min_den] = np.nan
    return fwt

def load_task_order(path):
    if path is None:
        return None
    with open(path, "r") as f:
        cfg = json.load(f)
    if "filter_tasks" in cfg:
        return cfg["filter_tasks"]
    n = cfg.get("num_tasks", None)
    return list(range(n)) if n is not None else None

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--methods", nargs="+", required=True,
                    help="NAME=PATH")
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--metric", choices=["eval", "fwt"], default="eval",
                    help="Use final total eval (eval) or mean FWT per run (fwt)")
    ap.add_argument("--expert-root", type=str, default=None,
                    help="Root with single-task expert runs (required for --metric fwt)")
    ap.add_argument("--task-order-config", type=str, default=None,
                    help="Meta config to order tasks/experts")
    ap.add_argument("--max-return", type=float, default=1.0,
                    help="Max return to normalize AUCs for FWT")
    ap.add_argument("--min-denominator", type=float, default=1e-3,
                    help="Min (1 - AUC_expert) before FWT is set to NaN")
    args = ap.parse_args()

    methods = parse_methods(args.methods)
    if args.ref not in methods:
        raise ValueError(f"Reference {args.ref} not provided")

    # Load experts if needed
    task_order = load_task_order(args.task_order_config)
    expert_by_seed = None
    default_expert_auc = None
    if args.metric == "fwt":
        if args.expert_root is None:
            raise ValueError("--expert-root required for --metric fwt")
        eruns = find_runs(args.expert_root)
        if not eruns:
            raise ValueError(f"No expert eval_metrics under {args.expert_root}")
        if task_order is None:
            # Heuristic warning: large, non-consecutive task ids usually imply custom ordering.
            task_ids = []
            for p in eruns:
                m = re.findall(r"(?:task)(\d+)", p)
                if m:
                    task_ids.append(int(m[-1]))
            if task_ids:
                task_ids_sorted = sorted(set(task_ids))
                consecutive = (task_ids_sorted[-1] - task_ids_sorted[0] + 1) == len(task_ids_sorted)
                small_ids = task_ids_sorted[0] in (0, 1) and task_ids_sorted[-1] < 1000
                if not (consecutive and small_ids):
                    print(
                        "WARNING: --task-order-config not provided and expert task ids look non-consecutive. "
                        "FWT may be misaligned. Consider passing --task-order-config.",
                    )
        # group experts by seed if present
        tmp = {}
        for er in eruns:
            sd = re.search(r"seed(\d+)", er)
            if sd:
                tmp.setdefault(sd.group(1), []).append(er)
        def load_expert_set(paths):
            id_to_path = {}
            for p in paths:
                m = re.findall(r"(?:task)(\d+)", p)
                if m:
                    id_to_path[int(m[-1])] = p
            ordered = []
            if task_order:
                for tid in task_order:
                    if tid not in id_to_path:
                        raise ValueError(f"Missing expert for task {tid}")
                    ordered.append(id_to_path[tid])
            else:
                ordered = sorted(paths)
            aucs = []
            for p in ordered:
                emat = load_eval_matrix(p)
                if emat.shape[1] != 1:
                    raise ValueError(f"Expert {p} should have 1 task column")
                aucs.append(np.trapz(emat[:, 0], dx=1.0))
            return np.asarray(aucs)
        if tmp:
            expert_by_seed = {s: load_expert_set(ps) for s, ps in tmp.items()}
        # Always keep a default expert set to fall back on
        default_expert_auc = load_expert_set(eruns)

    samples = {}
    for name, root in methods.items():
        runs = find_runs(root)
        if not runs:
            raise ValueError(f"No eval_metrics under {root}")
        vals = []
        for rd in runs:
            if args.metric == "eval":
                vals.append(float(load_eval_matrix(rd)[-1, :].sum()))
            else:
                ll_auc = per_task_auc(load_eval_matrix(rd))
                seed = extract_seed_generic(rd)
                exp_auc = None
                if expert_by_seed and seed in expert_by_seed:
                    exp_auc = expert_by_seed[seed]
                elif default_expert_auc is not None:
                    exp_auc = default_expert_auc
                else:
                    raise ValueError(f"No experts for seed {seed}")
                ll_norm = normalize_auc(ll_auc, args.max_return)
                exp_norm = normalize_auc(exp_auc, args.max_return)
                fwt_vec = forward_transfer(ll_norm, exp_norm, min_den=args.min_denominator)
                vals.append(np.nanmean(fwt_vec))
        samples[name] = np.asarray(vals, float)
        mean = float(np.nanmean(samples[name]))
        lo, hi = bootstrap_ci(samples[name], samples[name], iters=args.iters)
        print(f"{name}: {len(vals)} runs, mean={mean:.4f}, 95% CI=[{lo:.4f}, {hi:.4f}]")

    ref_vals = samples[args.ref]
    for name, vals in samples.items():
        if name == args.ref:
            continue
        p = stats.ttest_ind(ref_vals, vals, equal_var=False, nan_policy="omit")[1]
        lo, hi = bootstrap_ci(ref_vals, vals, iters=args.iters)
        print(f"Compare {args.ref} vs {name}: p={p:.3e}, BCI=[{lo:.2f}, {hi:.2f}] (μ_ref - μ_{name})")

if __name__ == "__main__":
    main()
