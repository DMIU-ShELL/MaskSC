#!/usr/bin/env python3
import argparse
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt


def plot_master_pkl(pkl_path, out_path, title=None, y_label="Summed Return"):
    with open(pkl_path, "rb") as f:
        master = pickle.load(f)

    fig = plt.figure(figsize=(14, 6))
    ax = fig.subplots()
    ax.set_xlabel("Iteration")
    ax.set_ylabel(y_label)
    ax.grid(True, which="both")

    for method_name, result_dict in master.items():
        xdata = np.asarray(result_dict["xdata"])
        ydata = np.asarray(result_dict["ydata"])
        cfi = np.asarray(result_dict.get("ydata_cfi", np.zeros_like(ydata)))
        ax.plot(xdata, ydata, linewidth=2.5, label=method_name, alpha=0.8)
        ax.fill_between(xdata, ydata - cfi, ydata + cfi, alpha=0.2)

    if title:
        ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9, ncol=3)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True, help="Path to master*.pkl file.")
    ap.add_argument("--out", required=True, help="Output PDF path.")
    ap.add_argument("--title", default=None, help="Optional plot title.")
    ap.add_argument("--y_label", default="Summed Return", help="Y-axis label.")
    args = ap.parse_args()

    plot_master_pkl(args.pkl, args.out, title=args.title, y_label=args.y_label)


if __name__ == "__main__":
    main()
