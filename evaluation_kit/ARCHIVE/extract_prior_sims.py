#!/usr/bin/env python3
import argparse, os, re, csv
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

SIM_LINE_RE = re.compile(r"Prior sims:\s*\[(.*)\]")
# Logs look like: Prior sims: [(0.3157, 0), (0.12, 5), ...]
# i.e., (similarity_float, prior_task_idx_int)
PAIR_RE = re.compile(r"\(\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*(\d+)\s*\)")

def find_logs(root):
    logs = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.startswith("train-log") and f.endswith(".txt"):
                logs.append(os.path.join(dirpath, f))
    return sorted(logs)

def extract_sims_from_log(log_path, sims_per_task=20, task_offset=0):
    """
    Returns list of tuples: (current_task_idx, prior_task_idx, sim, seed, run_path)
    Heuristic: prior sims are logged every few iterations; after `sims_per_task`
    occurrences we advance to the next task. If the first task has no sims logged,
    use task_offset=1 so the first occurrence maps to task 1 instead of 0.
    """
    rows = []
    occ = 0
    with open(log_path, "r") as f:
        for line in f:
            m = SIM_LINE_RE.search(line)
            if not m:
                continue
            current_task_idx = task_offset + (occ // sims_per_task)
            occ += 1
            payload = m.group(1)
            for pm in PAIR_RE.finditer(payload):
                sim = float(pm.group(1))
                prior_idx = int(pm.group(2))
                rows.append((current_task_idx, prior_idx, sim))
    return rows

def aggregate_pairs(rows):
    # rows: list of (current_task, prior_task, sim)
    # Build mean matrix
    tasks = set()
    for c, p, _ in rows:
        tasks.add(c); tasks.add(p)
    max_task = max(tasks)
    sims = defaultdict(list)
    for c, p, s in rows:
        sims[(c, p)].append(s)
    mat = np.full((max_task+1, max_task+1), np.nan, dtype=float)
    for (c, p), vals in sims.items():
        mat[c, p] = float(np.mean(vals))
    return mat

def save_long_csv(rows, seeds, run_paths, out_csv):
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "run_path", "current_task", "prior_task", "similarity"])
        for (seed, run_path, c, p, s) in rows:
            w.writerow([seed, run_path, c, p, s])

def plot_heatmap(mat, out_png, title="Prior cosine similarities"):
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(mat, annot=False, cmap="viridis", mask=np.isnan(mat))
    ax.set_xlabel("Prior task")
    ax.set_ylabel("Current task")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def extract_seed(path):
    # Try to infer seed from path
    m = re.search(r"seed(\d+)", path)
    if m:
        return m.group(1)
    m = re.search(r"-([0-9]{2,5})-", path)
    if m:
        return m.group(1)
    return os.path.basename(os.path.dirname(path))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="Experiment root containing seed run folders with train-log-*.txt files")
    ap.add_argument("--out-csv", default="prior_sims.csv",
                    help="Output CSV path for long-form similarities")
    ap.add_argument("--out-heatmap", default="prior_sims_heatmap.png",
                    help="Output PNG path for heatmap (mean over runs)")
    ap.add_argument("--sims-per-task", type=int, default=20,        # This is given by (total steps per task / rollout length) / embedding computation frequency
                    help="Number of Prior sims occurrences per task (default 20 for 100 iters logged every 5).")
    ap.add_argument("--task-offset", type=int, default=0,
                    help="Offset for task indexing when first task has no sims (set to 1 if task0 logs none).")
    args = ap.parse_args()

    logs = find_logs(args.root)
    if not logs:
        raise SystemExit(f"No train-log-*.txt files found under {args.root}")

    long_rows = []
    all_pairs = []
    for log in logs:
        seed = extract_seed(log)
        run_path = os.path.dirname(log)
        sims = extract_sims_from_log(log, sims_per_task=args.sims_per_task, task_offset=args.task_offset)
        for c, p, s in sims:
            long_rows.append((seed, run_path, c, p, s))
            all_pairs.append((c, p, s))

    # Save long-form CSV
    save_long_csv(long_rows, None, None, args.out_csv)
    print(f"Wrote {len(long_rows)} rows to {args.out_csv}")

    # Aggregate across all runs and plot heatmap
    mat = aggregate_pairs(all_pairs)
    plot_heatmap(mat, args.out_heatmap, title="Mean prior cosine similarity")
    print(f"Saved heatmap to {args.out_heatmap}")

if __name__ == "__main__":
    main()
