#!/usr/bin/env python3
import argparse, pandas as pd, numpy as np
from scipy.stats import spearmanr, pearsonr

def ndcg_at_k(relevances, k):
    rel = np.asfarray(relevances)[:k]
    if rel.size == 0:
        return np.nan
    dcg = (2**rel - 1) / np.log2(np.arange(2, rel.size + 2))
    ideal = np.sort(rel)[::-1]
    idcg = (2**ideal - 1) / np.log2(np.arange(2, ideal.size + 2))
    denom = idcg.sum()
    return dcg.sum() / denom if denom > 0 else np.nan

def precision_recall_at_k(sorted_labels, k, total_relevant=None):
    rel = np.asarray(sorted_labels)[:k]
    tp = rel.sum()
    precision = tp / k
    if total_relevant is None:
        total_relevant = np.asarray(sorted_labels).sum()
    recall = tp / total_relevant if total_relevant > 0 else np.nan
    return precision, recall


def safe_spearman(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) < 2 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return np.nan
    return spearmanr(x, y).correlation


def safe_pearson(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) < 2 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return np.nan
    return pearsonr(x, y)[0]


def load_prior_sims(path, probe_utils, seed=None):
    sims = pd.read_csv(path)
    sims = sims.rename(columns={"task_idx": "current_task", "prev_idx": "prior_task"})
    required = {"current_task", "prior_task", "similarity"}
    missing = required - set(sims.columns)
    if missing:
        raise SystemExit(f"--prior-sims is missing required columns: {sorted(missing)}")

    if "seed" not in sims.columns:
        if seed is not None:
            sims["seed"] = seed
        elif "seed" in probe_utils.columns and probe_utils["seed"].nunique() == 1:
            sims["seed"] = int(probe_utils["seed"].iloc[0])
        else:
            raise SystemExit(
                "--prior-sims has no seed column. Pass --seed when matching one seed, "
                "or provide a prior-sims CSV with a seed column."
            )

    sims = sims.replace([np.inf, -np.inf], np.nan).dropna(subset=["similarity"])

    group_cols = ["seed", "current_task", "prior_task"]
    if sims.duplicated(group_cols).any():
        sort_cols = group_cols + [c for c in ("iteration", "total_steps") if c in sims.columns]
        sims = sims.sort_values(sort_cols).groupby(group_cols, as_index=False).tail(1)

    return sims

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--prior-sims",
        required=True,
        help="CSV with similarity rows. Accepts either current_task/prior_task or task_idx/prev_idx columns.",
    )
    ap.add_argument("--probe-utils", required=True, help="CSV with seed,run_path,current_task,prior_task,utility")
    ap.add_argument("--seed", type=int, default=None,
                    help="Seed to attach when --prior-sims has no seed column.")
    ap.add_argument("--k", type=int, default=3, help="k for @k metrics")
    ap.add_argument("--utility-threshold", type=float, default=None,
                    help="Threshold to binarize utilities for precision/recall; if None, uses top-k by utility as relevant")
    ap.add_argument("--out", default="sim_transfer_stats.csv", help="Output CSV for summary stats")
    ap.add_argument("--relevant-pairs", default=None,
                    help="Optional CSV of current_task,prior_task pairs to keep before analysis.")
    args = ap.parse_args()

    utils = pd.read_csv(args.probe_utils)
    sims = load_prior_sims(args.prior_sims, utils, seed=args.seed)
    # keep common rows
    merged = pd.merge(sims, utils, on=["seed","current_task","prior_task"], suffixes=("_sim","_util"))
    if merged.empty:
        raise SystemExit("No overlapping (seed,current_task,prior_task) between sims and utils")

    records = []
    # group by seed for correlations and per-task rankings
    '''for seed, df_seed in merged.groupby("seed"):
        # Correlations across all pairs for this seed
        svals = df_seed["similarity"].to_numpy()
        uvals = df_seed["utility"].to_numpy()
        spear = spearmanr(svals, uvals).correlation
        pear = pearsonr(svals, uvals)[0]
        records.append({"seed": seed, "metric": "spearman_all", "value": spear})
        records.append({"seed": seed, "metric": "pearson_all", "value": pear})'''

    print(f"Merged rows: {len(merged)}")

    # optional: filter to relevant pairs
    if args.relevant_pairs:
        rel = pd.read_csv(args.relevant_pairs)  # cols: current_task, prior_task
        before = len(merged)
        merged = merged.merge(rel, on=["current_task","prior_task"], how="inner")
        print(f"Filtered to relevant pairs: {before} -> {len(merged)} rows")
        if merged.empty:
            raise SystemExit("No rows remain after --relevant-pairs filtering")

    for seed, df_seed in merged.groupby("seed"):
        # pooled over all pairs
        records.append({
            "seed": seed, "metric": "spearman_all",
            "value": safe_spearman(df_seed["similarity"], df_seed["utility"])
        })
        records.append({
            "seed": seed, "metric": "pearson_all",
            "value": safe_pearson(df_seed["similarity"], df_seed["utility"])
        })
        # per-task correlations
        for t, df_task in df_seed.groupby("current_task"):
            if len(df_task) < 2:
                continue
            sp = safe_spearman(df_task["similarity"], df_task["utility"])
            pr = safe_pearson(df_task["similarity"], df_task["utility"])
            records.append({"seed": seed, "metric": "spearman_task", "task": t, "value": sp})
            records.append({"seed": seed, "metric": "pearson_task", "task": t, "value": pr})


        # Ranking metrics per task
        for t, df_task in df_seed.groupby("current_task"):
            df_task = df_task.sort_values("similarity", ascending=False)
            util_sorted = df_task["utility"].to_numpy()
            sim_sorted = df_task["similarity"].to_numpy()
            k = min(args.k, len(df_task))
            # NDCG: use utility as relevance
            ndcg = ndcg_at_k(util_sorted, k)
            # Precision/Recall: either threshold utilities or use top-k-by-utility as relevant
            if args.utility_threshold is not None:
                labels = (util_sorted >= args.utility_threshold).astype(int)
                total_relevant = labels.sum()
            else:
                # If every prior has identical utility, top-k relevance is
                # otherwise arbitrary. Treat all-positive ties as all relevant
                # and all-zero ties as no identifiable useful prior.
                labels = np.zeros_like(util_sorted, dtype=int)
                if np.allclose(util_sorted, util_sorted[0]):
                    if util_sorted[0] > 0:
                        labels[:] = 1
                else:
                    topk_idx = np.argsort(df_task["utility"].to_numpy())[::-1][:k]
                    labels[topk_idx] = 1
                total_relevant = labels.sum()
            prec, rec = precision_recall_at_k(labels, k, total_relevant=total_relevant)
            records.append({"seed": seed, "metric": "ndcg", "task": t, "value": ndcg})
            records.append({"seed": seed, "metric": "precision", "task": t, "value": prec})
            records.append({"seed": seed, "metric": "recall", "task": t, "value": rec})

    out_df = pd.DataFrame(records)
    out_df.to_csv(args.out, index=False)
    # Print quick summary
    summary = out_df.groupby("metric")["value"].agg(["mean","std","count"])
    print(summary)

if __name__ == "__main__":
    main()
