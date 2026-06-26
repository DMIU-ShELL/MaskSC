import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load_curve(path):
    df = pd.read_csv(path)
    if "Iteration" not in df.columns or "SummedAverageReturn" not in df.columns:
        raise ValueError(f"Unexpected columns in {path}")
    df["Iteration"] = pd.to_numeric(df["Iteration"], errors="coerce")
    df["SummedAverageReturn"] = pd.to_numeric(df["SummedAverageReturn"], errors="coerce")
    df = df.dropna(subset=["Iteration", "SummedAverageReturn"])
    df = df.sort_values("Iteration")
    df["CumulativeReturn"] = df["SummedAverageReturn"].cumsum()
    return df


def label_from_path(path):
    base = os.path.basename(path)
    label = base.replace("_summed.csv", "")
    label = label.replace("_", " ")
    return label


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_input_dir = os.path.join(base_dir, "log", "plots", "summed_returns")
    default_output = os.path.join(default_input_dir, "summed_returns_all.pdf")
    ap = argparse.ArgumentParser(
        description="Plot all summed return curves from a directory into a single PDF.",
    )
    ap.add_argument(
        "--input-dir",
        default=default_input_dir,
        help="Directory containing *_summed.csv files.",
    )
    ap.add_argument(
        "--output",
        default=default_output,
        help="Output PDF path.",
    )
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--figsize", default="12,6", help="Figure size as W,H in inches.")
    ap.add_argument("--legend-cols", type=int, default=3)
    args = ap.parse_args()

    fig_w, fig_h = (float(v) for v in args.figsize.split(","))
    paths = sorted(glob.glob(os.path.join(args.input_dir, "*_summed.csv")))
    if not paths:
        raise SystemExit(f"No *_summed.csv files found in {args.input_dir}")

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    for path in paths:
        df = load_curve(path)
        ax.plot(
            df["Iteration"],
            df["CumulativeReturn"],
            linewidth=2,
            label=label_from_path(path),
        )

    ax.set_xlabel("Iterations")
    ax.set_ylabel("Summed return")
    ax.grid(True, alpha=0.3)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=args.legend_cols,
        frameon=False,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.subplots_adjust(bottom=0.25)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")


if __name__ == "__main__":
    main()
