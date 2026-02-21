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

def precision_recall_at_k(sorted_labels, k):
    rel = np.asarray(sorted_labels)[:k]
    tp = rel.sum()
    precision = tp / k
    recall = tp / rel.sum() if rel.sum() > 0 else np.nan  # only if any relevant exists
    return precision, recall

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior-sims", required=True, help="CSV from extract_prior_sims.py")
    ap.add_argument("--probe-utils", required=True, help="CSV with seed,run_path,current_task,prior_task,utility")
    ap.add_argument("--k", type=int, default=3, help="k for @k metrics")
    ap.add_argument("--utility-threshold", type=float, default=None,
                    help="Threshold to binarize utilities for precision/recall; if None, uses top-k by utility as relevant")
    ap.add_argument("--out", default="sim_transfer_stats.csv", help="Output CSV for summary stats")
    ap.add_argument("--relevant-pairs", default="log/ct28-interleaved-MaskSC-top-dense/relevant_pairs.csv", help="Task hierarchy ground truth")
    args = ap.parse_args()

    sims = pd.read_csv(args.prior_sims)
    utils = pd.read_csv(args.probe_utils)
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

    # optional: filter to relevant pairs
    if args.relevant_pairs:
        rel = pd.read_csv(args.relevant_pairs)  # cols: current_task, prior_task
        merged = merged.merge(rel, on=["current_task","prior_task"], how="inner")

    for seed, df_seed in merged.groupby("seed"):
        # pooled over all pairs
        records.append({
            "seed": seed, "metric": "spearman_all",
            "value": spearmanr(df_seed["similarity"], df_seed["utility"]).correlation
        })
        records.append({
            "seed": seed, "metric": "pearson_all",
            "value": pearsonr(df_seed["similarity"], df_seed["utility"])[0]
        })
        # per-task correlations
        for t, df_task in df_seed.groupby("current_task"):
            if len(df_task) < 2:
                continue
            sp = spearmanr(df_task["similarity"], df_task["utility"]).correlation
            pr = pearsonr(df_task["similarity"], df_task["utility"])[0]
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
            else:
                # mark relevant those in top-k by utility
                topk_idx = np.argsort(df_task["utility"].to_numpy())[::-1][:k]
                labels = np.zeros_like(util_sorted, dtype=int)
                labels[topk_idx] = 1
            prec, rec = precision_recall_at_k(labels, k)
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
