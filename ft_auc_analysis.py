#!/usr/bin/env python3
import argparse, os, json, numpy as np

def load_eval_matrix(run_dir):
    npy = os.path.join(run_dir, "eval_metrics.npy")
    csv = os.path.join(run_dir, "eval_metrics.csv")
    if os.path.isfile(npy):
        mat = np.load(npy)
    elif os.path.isfile(csv):
        mat = np.loadtxt(csv, delimiter=",")
    else:
        raise FileNotFoundError(f"Missing eval_metrics.(npy|csv) in {run_dir}")
    # If the last column is a timestamp (float, increasing), drop it
    if mat.ndim != 2:
        raise ValueError(f"Expected 2D eval matrix in {run_dir}, got {mat.shape}")
    if mat.shape[1] > 1 and np.all(np.diff(mat[:, -1]) >= 0):
        # Heuristic: treat last column as time if it monotonically increases
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

def forward_transfer(auc_ll, auc_expert, eps=1e-8):
    # auc_ll, auc_expert: arrays of shape [num_tasks]
    return (auc_ll - auc_expert) / (np.abs(auc_expert) + eps)

def summarize(values):
    values = np.asarray(values)
    mean = values.mean()
    ci95 = 1.96 * values.std(ddof=1) / np.sqrt(len(values))
    return mean, ci95

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="Paths to lifelong runs (one per seed).")
    ap.add_argument("--expert-runs", nargs="+", default=None,
                    help="Paths to single-task expert runs, one per task, or leave empty to skip FWT.")
    args = ap.parse_args()

    # Load lifelong runs
    ll_per_task_aucs = []
    ll_global_aucs = []
    for run in args.runs:
        mat = load_eval_matrix(run)
        ll_per_task_aucs.append(per_task_auc(mat))
        ll_global_aucs.append(global_auc(mat))

    # Stack per-task AUCs to [num_runs, num_tasks]
    ll_per_task_aucs = np.stack(ll_per_task_aucs, axis=0)
    ll_global_aucs = np.asarray(ll_global_aucs)

    # Expert AUCs (optional)
    fwt_all = None
    if args.expert_runs:
        expert_aucs = []
        for erun in args.expert_runs:
            emat = load_eval_matrix(erun)
            # For a single-task expert, take the only column
            if emat.shape[1] != 1:
                raise ValueError(f"Expert run {erun} should have exactly one task column.")
            expert_aucs.append(trapz_over_checkpoints(emat[:, 0]))
        expert_aucs = np.asarray(expert_aucs)
        if expert_aucs.shape[0] != ll_per_task_aucs.shape[1]:
            raise ValueError("Number of expert runs must match num_tasks.")
        fwt_runs = [forward_transfer(ll_per_task_aucs[i], expert_aucs)
                    for i in range(ll_per_task_aucs.shape[0])]
        fwt_all = np.stack(fwt_runs, axis=0)  # [num_runs, num_tasks]

    # Report
    mean_global, ci_global = summarize(ll_global_aucs)
    print(f"Global AUC/TPOT: {mean_global:.4f} ± {ci_global:.4f} (95% CI) over {len(args.runs)} runs")

    per_task_mean = ll_per_task_aucs.mean(axis=0)
    per_task_ci = 1.96 * ll_per_task_aucs.std(axis=0, ddof=1) / np.sqrt(len(args.runs))
    for t, (m, c) in enumerate(zip(per_task_mean, per_task_ci)):
        print(f"Task {t+1} AUC: {m:.4f} ± {c:.4f}")

    if fwt_all is not None:
        fwt_mean = fwt_all.mean()
        fwt_ci = 1.96 * fwt_all.std(ddof=1) / np.sqrt(fwt_all.size / fwt_all.shape[1])
        print(f"Forward Transfer (avg over tasks & runs): {fwt_mean:.4f} ± {fwt_ci:.4f}")
        ft_task_mean = fwt_all.mean(axis=0)
        ft_task_ci = 1.96 * fwt_all.std(axis=0, ddof=1) / np.sqrt(fwt_all.shape[0])
        for t, (m, c) in enumerate(zip(ft_task_mean, ft_task_ci)):
            print(f"Task {t+1} FWT: {m:.4f} ± {c:.4f}")

if __name__ == "__main__":
    main()
