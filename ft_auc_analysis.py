#!/usr/bin/env python3
import argparse, os, json, csv, re, numpy as np

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

def trapz_over_checkpoints(curve):
    # Uniform spacing over checkpoints; integrates “area under learning curve”
    return np.trapz(curve, dx=1.0)

def per_task_auc(mat):
    # mat: [T checkpoints, N tasks] -> per-task AUC over checkpoints
    return np.array([trapz_over_checkpoints(mat[:, j]) for j in range(mat.shape[1])])

def global_auc(mat):
    # ICR curve = sum over tasks at each checkpoint; integrate to get TPOT/AUC
    icr = mat.sum(axis=1)
    return trapz_over_checkpoints(icr)

def normalize_auc(arr, max_return):
    """Normalize AUCs into [0, 1] using a provided max_return (scalar)."""
    if max_return is None:
        return arr
    return np.clip(arr / max_return, 0.0, 1.0)

def forward_transfer(auc_ll, auc_expert, eps=1e-8, min_den=1e-3):
    # auc_* expected in [0, 1]
    denom = 1.0 - auc_expert
    fwt = (auc_ll - auc_expert) / (denom + eps)
    fwt[denom < min_den] = np.nan  # avoid blowing up when the expert is near-perfect
    return fwt


def summarize(values):
    values = np.asarray(values)
    mean = values.mean()
    ci95 = 1.96 * values.std(ddof=1) / np.sqrt(len(values))
    return mean, ci95

def summarize_nan(values):
    values = np.asarray(values)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan, 0
    mean = values.mean()
    ci95 = 1.96 * values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
    return mean, ci95, len(values)
    return mean, ci95

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=[],
                    help="Paths to lifelong runs (one per seed).")
    ap.add_argument("--expert-runs", nargs="+", default=None,
                    help="Paths to single-task expert runs, one per task, or leave empty to skip FWT.")
    ap.add_argument("--lifelong-root", type=str, default=None,
                    help="Root directory to auto-discover lifelong runs (finds eval_metrics.*).")
    ap.add_argument("--expert-root", type=str, default=None,
                    help="Root directory to auto-discover single-task expert runs (finds eval_metrics.*).")
    ap.add_argument("--max-return", type=float, default=1.0,
                    help="Scalar max return to normalize AUCs into [0,1] for FWT (default 1.0).")
    ap.add_argument("--min-denominator", type=float, default=1e-3,
                    help="Minimum allowed (1 - AUC_expert) before FWT is set to NaN.")
    ap.add_argument("--task-order-config", type=str, default=None,
                    help="Path to meta env config (e.g., meta_ctgraph_ct28_interleaved.json) to derive task ordering. If omitted, experts are ordered by numeric task id ascending.")
    args = ap.parse_args()

    def find_run_dirs(root):
        run_dirs = set()
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                if fname in ("eval_metrics.npy", "eval_metrics.csv"):
                    run_dirs.add(dirpath)
                    break
        return sorted(run_dirs)

    def extract_seed(path):
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

    def group_runs_by_seed(run_dirs):
        grouped = {}
        for rd in run_dirs:
            seed = extract_seed(rd) or os.path.basename(os.path.dirname(rd))
            grouped.setdefault(seed, []).append(rd)
        # pick one run per seed (latest path)
        return {seed: sorted(runs)[-1] for seed, runs in grouped.items()}

    def natural_key(path):
        nums = re.findall(r"\d+", os.path.basename(path))
        return int(nums[-1]) if nums else os.path.basename(path)

    def load_task_order():
        if args.task_order_config is None:
            return None
        with open(args.task_order_config, "r") as f:
            cfg = json.load(f)
        if "filter_tasks" in cfg:
            return cfg["filter_tasks"]
        num_tasks = cfg.get("num_tasks", None)
        if num_tasks is None:
            return None
        return list(range(num_tasks))

    # Load lifelong runs
    runs = list(args.runs)
    if args.lifelong_root:
        runs.extend(find_run_dirs(args.lifelong_root))
    if len(runs) == 0:
        raise ValueError("No lifelong runs provided.")
    lifelong_by_seed = group_runs_by_seed(runs)

    ll_per_task_aucs = []
    ll_global_aucs = []
    seeds = sorted(lifelong_by_seed.keys())
    for seed in seeds:
        run = lifelong_by_seed[seed]
        mat = load_eval_matrix(run)
        ll_per_task_aucs.append(per_task_auc(mat))
        ll_global_aucs.append(global_auc(mat))

    # Stack per-task AUCs to [num_runs, num_tasks]
    ll_per_task_aucs = np.stack(ll_per_task_aucs, axis=0)
    ll_global_aucs = np.asarray(ll_global_aucs)

    # Expert AUCs (optional)
    fwt_all = None
    expert_runs = list(args.expert_runs) if args.expert_runs else []
    if args.expert_root:
        expert_runs.extend(find_run_dirs(args.expert_root))

    expert_by_seed = None
    if expert_runs:
        expert_by_seed = {}
        # group experts by seed if present
        tmp = {}
        for er in expert_runs:
            seed = extract_seed(er)
            if seed:
                tmp.setdefault(seed, []).append(er)
        if tmp:
            expert_by_seed = {s: sorted(rs, key=natural_key) for s, rs in tmp.items()}
        else:
            # single shared expert set
            expert_by_seed = {"default": sorted(expert_runs, key=natural_key)}

        expert_aucs = []
        # choose expert set per seed or fallback to default
        task_order = load_task_order()

        def load_expert_set(eruns):
            # map task id -> path
            id_to_path = {}
            for erun in eruns:
                nums = re.findall(r"(?:task)(\d+)", erun)
                if nums:
                    tid = int(nums[-1])
                    id_to_path[tid] = erun
            ordered_paths = []
            if task_order is not None:
                for tid in task_order:
                    if tid not in id_to_path:
                        raise ValueError(f"Missing expert run for task id {tid}")
                    ordered_paths.append(id_to_path[tid])
            else:
                ordered_paths = sorted(eruns, key=natural_key)

            aucs = []
            for erun in ordered_paths:
                emat = load_eval_matrix(erun)
                if emat.shape[1] != 1:
                    raise ValueError(f"Expert run {erun} should have exactly one task column.")
                aucs.append(trapz_over_checkpoints(emat[:, 0]))
            return np.asarray(aucs)

        default_experts = None
        if "default" in (expert_by_seed or {}):
            default_experts = load_expert_set(expert_by_seed["default"])

        fwt_runs = []
        expert_masks = []
        for i, seed in enumerate(seeds):
            eruns = None
            if expert_by_seed:
                eruns = expert_by_seed.get(seed, None)
            if eruns is None:
                eruns = default_experts
                if eruns is None:
                    raise ValueError(f"No experts found for seed {seed} and no default experts provided.")
            
            
            aucs = load_expert_set(eruns) if isinstance(eruns, list) else eruns
            if aucs.shape[0] != ll_per_task_aucs.shape[1]:
                raise ValueError(f"Number of expert runs ({aucs.shape[0]}) must match num_tasks ({ll_per_task_aucs.shape[1]})")

            ll_norm = normalize_auc(ll_per_task_aucs[i], args.max_return)
            aucs_norm = normalize_auc(aucs, args.max_return)

            fwt = forward_transfer(ll_norm, aucs_norm, min_den=args.min_denominator)
            expert_masks.append(~np.isnan(fwt))
            fwt_runs.append(fwt)

            
        fwt_all = np.stack(fwt_runs, axis=0)  # [num_runs, num_tasks]
        expert_masks = np.stack(expert_masks, axis=0)

    # Report
    mean_global, ci_global = summarize(ll_global_aucs)
    print(f"Global AUC/TPOT: {mean_global:.4f} ± {ci_global:.4f} (95% CI) over {len(seeds)} runs")

    per_task_mean = ll_per_task_aucs.mean(axis=0)
    per_task_ci = 1.96 * ll_per_task_aucs.std(axis=0, ddof=1) / np.sqrt(len(seeds))
    for t, (m, c) in enumerate(zip(per_task_mean, per_task_ci)):
        print(f"Task {t+1} AUC: {m:.4f} ± {c:.4f}")

    if fwt_all is not None:
        flat = fwt_all.reshape(-1)
        fwt_mean, fwt_ci, n_flat = summarize_nan(flat)
        skipped = np.sum(np.isnan(flat))
        print(f"Forward Transfer (avg over tasks & runs): {fwt_mean:.4f} ± {fwt_ci:.4f} (valid={n_flat}, skipped={skipped})")
        ft_task_mean = []
        ft_task_ci = []
        ft_task_n = []
        for j in range(fwt_all.shape[1]):
            m, c, n = summarize_nan(fwt_all[:, j])
            ft_task_mean.append(m); ft_task_ci.append(c); ft_task_n.append(n)
            print(f"Task {j+1} FWT: {m:.4f} ± {c:.4f} (n={n})")

if __name__ == "__main__":
    main()
