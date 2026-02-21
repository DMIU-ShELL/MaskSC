# Run this once sim_transfer_stats_seed*.csv are produced for all seed runs using sim_transfer_analysis.py
import pandas as pd, glob
paths = glob.glob("log/ct28-interleaved-MaskSC-top-dense/sim_transfer_stats_seed*.csv")
df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
summary = df.groupby("metric")["value"].agg(["mean","std","count"])
summary["ci95"] = 1.96 * summary["std"] / summary["count"]**0.5
print("Aggregated over", len(paths), "seeds")
print(summary)