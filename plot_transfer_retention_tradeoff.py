#!/usr/bin/env python3
"""Plot the forward-transfer and retention trade-off across methods.

The script reuses significance_test_metrics_latex.py for metric loading,
normalization, and computation. The resulting figure places FWT on the x-axis,
mean task AUC (MT-AUC) on the y-axis, and encodes Continual World forgetting
(FGT) by marker colour. Bootstrap 95% confidence intervals are shown on both
axes.

Example for CT28:

    python plot_transfer_retention_tradeoff.py \
      --methods \
        PPO=log/ct28/ct28-interleaved-PPO/ \
        Mask-RI=log/ct28/ct28-interleaved-MaskRI/ \
        CLEAR=log/ct28/ct28-interleaved-CLEAR/ \
        SER=log/ct28/ct28-interleaved-SER/ \
        Mask-LC=log/ct28/ct28-interleaved-MaskLC/ \
        Mask-SC-3=log/checklist_runs/ct28-interleaved-MaskSC-3-thesis/ \
      --expert-root log/ct28/ct28-interleaved-single-task-experts-PPO/ \
      --task-order-config env_configs/ct28/seed1/meta_ctgraph_ct28_interleaved.json \
      --highlight Mask-SC-3 \
      --title "CT28" \
      --out-dir transfer_retention_ct28

For MG16, add:

    --performance-normalization minigrid_shortest_path

Use ``--exclude-fwt-task-ids 0 1 2 3`` when FWT is intended to cover only
transfer tasks at depth 3 or greater.

Existing benchmark summaries can be overlaid in one figure:

    python plot_transfer_retention_tradeoff.py \
      --summary-csv \
        CT28=transfer_retention_ct28_full \
        MG16=transfer_retention_mg16_full \
      --benchmark-marker CT28=o \
      --benchmark-marker MG16=^ \
      --combined-labels all \
      --highlight Mask-SC-3 Mask-SC-4 \
      --title "Transfer and retention across benchmarks" \
      --out-dir transfer_retention_combined
"""

import argparse
import csv
import os
import tempfile
from types import SimpleNamespace

# Cluster home directories may not expose writable font or Matplotlib caches.
os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "lara-matplotlib-cache")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", os.path.join(tempfile.gettempdir(), "lara-xdg-cache")
)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import significance_test_metrics_latex as metric_lib


METRICS = ("auc", "fwt", "fgt_cw")


def parse_methods(items):
    methods = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected NAME=PATH, got {item}")
        name, path = item.split("=", 1)
        if not name:
            raise ValueError(f"Missing method name in {item}")
        if name in methods:
            raise ValueError(f"Duplicate method name: {name}")
        methods[name] = path
    return methods


def parse_key_value_items(items, description):
    parsed = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected {description}=VALUE, got {item}")
        key, value = item.split("=", 1)
        if not key or not value:
            raise ValueError(f"Expected {description}=VALUE, got {item}")
        if key in parsed:
            raise ValueError(f"Duplicate {description}: {key}")
        parsed[key] = value
    return parsed


def parse_label_offsets(items):
    offsets = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected METHOD=DX,DY, got {item}")
        name, value = item.split("=", 1)
        parts = value.split(",")
        if len(parts) != 2:
            raise ValueError(f"Expected METHOD=DX,DY, got {item}")
        offsets[name] = (float(parts[0]), float(parts[1]))
    return offsets


def resolve_summary_csv(path):
    if os.path.isdir(path):
        path = os.path.join(path, "transfer_retention_tradeoff.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Summary CSV not found: {path}")
    return path


def load_summary_csv(path):
    path = resolve_summary_csv(path)
    summary = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"method"}
        for metric in METRICS:
            required.update(
                {
                    f"{metric}_mean",
                    f"{metric}_ci_low",
                    f"{metric}_ci_high",
                    f"{metric}_n",
                }
            )
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path} is missing summary columns: {sorted(missing)}"
            )

        for row in reader:
            method = row["method"]
            if method in summary:
                raise ValueError(f"Duplicate method {method!r} in {path}")
            summary[method] = {}
            for metric in METRICS:
                summary[method][metric] = {
                    "mean": float(row[f"{metric}_mean"]),
                    "low": float(row[f"{metric}_ci_low"]),
                    "high": float(row[f"{metric}_ci_high"]),
                    "n": int(row[f"{metric}_n"]),
                }
    if not summary:
        raise ValueError(f"No method rows found in {path}")
    return summary


def write_combined_summary_csv(path, benchmark_summaries):
    fields = ["benchmark", "method"]
    for metric in METRICS:
        fields.extend(
            [
                f"{metric}_mean",
                f"{metric}_ci_low",
                f"{metric}_ci_high",
                f"{metric}_n",
            ]
        )

    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for benchmark, summary in benchmark_summaries.items():
            for method, method_summary in summary.items():
                row = {"benchmark": benchmark, "method": method}
                for metric in METRICS:
                    stats = method_summary[metric]
                    row[f"{metric}_mean"] = f"{stats['mean']:.8f}"
                    row[f"{metric}_ci_low"] = f"{stats['low']:.8f}"
                    row[f"{metric}_ci_high"] = f"{stats['high']:.8f}"
                    row[f"{metric}_n"] = stats["n"]
                writer.writerow(row)


def bootstrap_summary(values, iters, rng_seed, metric, method):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean": np.nan, "low": np.nan, "high": np.nan, "n": 0}
    rng = metric_lib.stable_rng(rng_seed, "tradeoff", metric, method)
    low, high = metric_lib.bootstrap_mean_ci(values, iters=iters, rng=rng)
    return {
        "mean": float(np.mean(values)),
        "low": float(low),
        "high": float(high),
        "n": int(values.size),
    }


def build_metric_args(args):
    return SimpleNamespace(
        task_order_config=args.task_order_config,
        expert_root=args.expert_root,
        performance_normalization=args.performance_normalization,
        max_return=args.max_return,
        minigrid_shortest_path_seeds=args.minigrid_shortest_path_seeds,
        print_return_scales=args.print_return_scales,
        min_denominator=args.min_denominator,
        fwt_task_ids=args.fwt_task_ids,
        exclude_fwt_task_ids=args.exclude_fwt_task_ids,
        include_unseen_eval_tasks=args.include_unseen_eval_tasks,
    )


def summarize_methods(samples, methods, args):
    summary = {}
    for method in methods:
        summary[method] = {
            metric: bootstrap_summary(
                samples[metric][method],
                args.iters,
                args.rng_seed,
                metric,
                method,
            )
            for metric in METRICS
        }
    return summary


def write_summary_csv(path, methods, summary):
    fields = ["method"]
    for metric in METRICS:
        fields.extend(
            [
                f"{metric}_mean",
                f"{metric}_ci_low",
                f"{metric}_ci_high",
                f"{metric}_n",
            ]
        )

    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method in methods:
            row = {"method": method}
            for metric in METRICS:
                stats = summary[method][metric]
                row[f"{metric}_mean"] = f"{stats['mean']:.8f}"
                row[f"{metric}_ci_low"] = f"{stats['low']:.8f}"
                row[f"{metric}_ci_high"] = f"{stats['high']:.8f}"
                row[f"{metric}_n"] = stats["n"]
            writer.writerow(row)


def finite_metric_values(summary, metric):
    return np.asarray(
        [
            method_stats[metric]["mean"]
            for method_stats in summary.values()
            if np.isfinite(method_stats[metric]["mean"])
        ],
        dtype=float,
    )


def padded_limits(values, error_lows, error_highs, minimum_padding=0.04):
    low = np.nanmin(np.asarray(values) - np.asarray(error_lows))
    high = np.nanmax(np.asarray(values) + np.asarray(error_highs))
    span = high - low
    padding = max(span * 0.12, minimum_padding)
    return low - padding, high + padding


def repel_annotations(fig, annotations, max_iterations=200):
    """Move overlapping annotation labels without drawing leader lines."""
    if len(annotations) < 2:
        return

    pixels_to_points = 72.0 / fig.dpi
    for iteration in range(max_iterations):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        boxes = [
            annotation.get_window_extent(renderer).expanded(1.04, 1.14)
            for annotation in annotations
        ]
        shifts = np.zeros((len(annotations), 2), dtype=float)
        collisions = 0

        for left in range(len(boxes)):
            box_left = boxes[left]
            for right in range(left + 1, len(boxes)):
                box_right = boxes[right]
                if not box_left.overlaps(box_right):
                    continue

                collisions += 1
                overlap_x = min(box_left.x1, box_right.x1) - max(
                    box_left.x0, box_right.x0
                )
                overlap_y = min(box_left.y1, box_right.y1) - max(
                    box_left.y0, box_right.y0
                )
                center_left = np.asarray(
                    [
                        (box_left.x0 + box_left.x1) / 2.0,
                        (box_left.y0 + box_left.y1) / 2.0,
                    ]
                )
                center_right = np.asarray(
                    [
                        (box_right.x0 + box_right.x1) / 2.0,
                        (box_right.y0 + box_right.y1) / 2.0,
                    ]
                )

                # Move along the axis requiring the smaller displacement.
                if overlap_y <= overlap_x:
                    direction = np.sign(center_left[1] - center_right[1])
                    if direction == 0:
                        direction = -1.0 if (left + iteration) % 2 == 0 else 1.0
                    displacement = min(overlap_y / 2.0 + 1.5, 8.0)
                    shifts[left, 1] += direction * displacement
                    shifts[right, 1] -= direction * displacement
                else:
                    direction = np.sign(center_left[0] - center_right[0])
                    if direction == 0:
                        direction = -1.0 if (left + iteration) % 2 == 0 else 1.0
                    displacement = min(overlap_x / 2.0 + 1.5, 8.0)
                    shifts[left, 0] += direction * displacement
                    shifts[right, 0] -= direction * displacement

        if collisions == 0:
            break

        for annotation, shift in zip(annotations, shifts):
            shift = np.clip(shift, -10.0, 10.0) * pixels_to_points
            old_position = np.asarray(annotation.get_position(), dtype=float)
            annotation.set_position(tuple(old_position + shift))


def configure_plot_style(args):
    mpl.rcParams.update(
        {
            "font.family": args.font_family,
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def resolve_color_scale(summaries, args):
    fgt_values = []
    for summary in summaries:
        fgt_values.extend(finite_metric_values(summary, "fgt_cw"))
    fgt_values = np.asarray(fgt_values, dtype=float)
    if fgt_values.size == 0:
        raise ValueError("No finite FGT values were computed")
    color_min = (
        args.fgt_min if args.fgt_min is not None else min(0.0, np.min(fgt_values))
    )
    color_max = args.fgt_max if args.fgt_max is not None else np.max(fgt_values)
    if np.isclose(color_min, color_max):
        color_max = color_min + 1e-3
    return mpl.colors.Normalize(vmin=color_min, vmax=color_max)


def make_plot(methods, summary, args):
    configure_plot_style(args)

    fig, ax = plt.subplots(
        figsize=(args.width, args.height),
        constrained_layout=True,
    )

    color_norm = resolve_color_scale([summary], args)
    cmap = mpl.colormaps[args.colormap]

    label_offsets = parse_label_offsets(args.label_offset)
    highlighted = set(args.highlight or [])

    xs = []
    ys = []
    xerr_low = []
    xerr_high = []
    yerr_low = []
    yerr_high = []

    for index, method in enumerate(methods):
        fwt = summary[method]["fwt"]
        auc = summary[method]["auc"]
        fgt = summary[method]["fgt_cw"]
        if not all(np.isfinite(item["mean"]) for item in (fwt, auc, fgt)):
            print(f"Skipping {method}: at least one plotted metric is non-finite")
            continue

        x = fwt["mean"]
        y = auc["mean"]
        x_low = max(0.0, x - fwt["low"])
        x_high = max(0.0, fwt["high"] - x)
        y_low = max(0.0, y - auc["low"])
        y_high = max(0.0, auc["high"] - y)

        xs.append(x)
        ys.append(y)
        xerr_low.append(x_low)
        xerr_high.append(x_high)
        yerr_low.append(y_low)
        yerr_high.append(y_high)

        is_highlighted = method in highlighted
        marker = "*" if is_highlighted else "o"
        marker_size = 150 if is_highlighted else 78
        edge_width = 1.4 if is_highlighted else 0.8

        ax.errorbar(
            x,
            y,
            xerr=np.asarray([[x_low], [x_high]]),
            yerr=np.asarray([[y_low], [y_high]]),
            fmt="none",
            ecolor="#8a8a8a",
            elinewidth=0.8,
            capsize=2.0,
            alpha=0.75,
            zorder=1,
        )
        ax.scatter(
            x,
            y,
            s=marker_size,
            marker=marker,
            c=[cmap(color_norm(fgt["mean"]))],
            edgecolors="black",
            linewidths=edge_width,
            zorder=3,
        )

        offset = label_offsets.get(method, default_offsets[index % len(default_offsets)])
        horizontal_alignment = "left" if offset[0] >= 0 else "right"
        ax.annotate(
            method,
            xy=(x, y),
            xytext=offset,
            textcoords="offset points",
            ha=horizontal_alignment,
            va="bottom" if offset[1] >= 0 else "top",
            fontsize=8,
            fontweight="bold" if is_highlighted else "normal",
            zorder=4,
        )

    if not xs:
        raise ValueError("No methods had finite FWT, MT-AUC, and FGT values")

    ax.set_xlim(
        *padded_limits(xs, xerr_low, xerr_high, minimum_padding=args.x_padding)
    )
    ax.set_ylim(
        *padded_limits(ys, yerr_low, yerr_high, minimum_padding=args.y_padding)
    )
    ax.set_xlabel("Forward transfer (FWT)")
    ax.set_ylabel("Mean task AUC (MT-AUC)")
    if args.title:
        ax.set_title(args.title)
    ax.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)

    scalar_mappable = mpl.cm.ScalarMappable(norm=color_norm, cmap=cmap)
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(scalar_mappable, ax=ax, pad=0.02)
    colorbar.set_label("Forgetting (FGT; lower is better)")

    return fig


def make_combined_plot(benchmark_summaries, args):
    configure_plot_style(args)

    fig, ax = plt.subplots(
        figsize=(args.width, args.height),
        constrained_layout=True,
    )

    color_norm = resolve_color_scale(list(benchmark_summaries.values()), args)
    cmap = mpl.colormaps[args.colormap]
    marker_overrides = parse_key_value_items(
        args.benchmark_marker, "benchmark marker"
    )
    default_markers = ["o", "^", "s", "D", "P", "X"]
    markers = {
        benchmark: marker_overrides.get(
            benchmark, default_markers[index % len(default_markers)]
        )
        for index, benchmark in enumerate(benchmark_summaries)
    }
    unknown_markers = sorted(set(marker_overrides) - set(benchmark_summaries))
    if unknown_markers:
        raise ValueError(
            f"Marker overrides reference unknown benchmarks: {unknown_markers}"
        )

    default_offsets = [
        (5, 5),
        (5, -11),
        (-5, 5),
        (-5, -11),
        (7, 0),
        (-7, 0),
    ]
    label_offsets = parse_label_offsets(args.label_offset)
    highlighted = set(args.highlight or [])
    label_benchmarks = set()
    if args.combined_labels == "all":
        label_benchmarks = set(benchmark_summaries)
    elif args.combined_labels == "first":
        label_benchmarks = {next(iter(benchmark_summaries))}
    elif args.combined_labels != "none":
        if args.combined_labels not in benchmark_summaries:
            raise ValueError(
                f"--combined-labels={args.combined_labels!r} does not match "
                f"a benchmark in {list(benchmark_summaries)}"
            )
        label_benchmarks = {args.combined_labels}

    method_occurrences = {}
    for benchmark, summary in benchmark_summaries.items():
        for method in summary:
            method_occurrences.setdefault(method, []).append(benchmark)

    if args.connect_methods:
        for method, benchmarks in method_occurrences.items():
            if len(benchmarks) < 2:
                continue
            points = []
            for benchmark in benchmarks:
                method_summary = benchmark_summaries[benchmark][method]
                x = method_summary["fwt"]["mean"]
                y = method_summary["auc"]["mean"]
                if np.isfinite(x) and np.isfinite(y):
                    points.append((x, y))
            if len(points) >= 2:
                ax.plot(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    color="#b8b8b8",
                    linewidth=0.7,
                    alpha=0.45,
                    zorder=0,
                )

    xs = []
    ys = []
    xerr_low = []
    xerr_high = []
    yerr_low = []
    yerr_high = []
    point_index = 0
    annotations = []
    benchmark_indices = {
        benchmark: index for index, benchmark in enumerate(benchmark_summaries)
    }

    for benchmark, summary in benchmark_summaries.items():
        for method, method_summary in summary.items():
            fwt = method_summary["fwt"]
            auc = method_summary["auc"]
            fgt = method_summary["fgt_cw"]
            if not all(np.isfinite(item["mean"]) for item in (fwt, auc, fgt)):
                print(
                    f"Skipping {benchmark}/{method}: at least one plotted "
                    "metric is non-finite"
                )
                continue

            x = fwt["mean"]
            y = auc["mean"]
            x_low = max(0.0, x - fwt["low"])
            x_high = max(0.0, fwt["high"] - x)
            y_low = max(0.0, y - auc["low"])
            y_high = max(0.0, auc["high"] - y)

            xs.append(x)
            ys.append(y)
            xerr_low.append(x_low)
            xerr_high.append(x_high)
            yerr_low.append(y_low)
            yerr_high.append(y_high)

            is_highlighted = method in highlighted
            marker_size = 120 if is_highlighted else 72
            edge_width = 1.8 if is_highlighted else 0.8

            ax.errorbar(
                x,
                y,
                xerr=np.asarray([[x_low], [x_high]]),
                yerr=np.asarray([[y_low], [y_high]]),
                fmt="none",
                ecolor="#999999",
                elinewidth=0.7,
                capsize=1.8,
                alpha=0.65,
                zorder=1,
            )
            ax.scatter(
                x,
                y,
                s=marker_size,
                marker=markers[benchmark],
                c=[cmap(color_norm(fgt["mean"]))],
                edgecolors="black",
                linewidths=edge_width,
                zorder=3,
            )

            # Label shared methods once. Methods present in only one summary
            # remain labelled regardless of the selected benchmark.
            if args.combined_labels != "none" and (
                benchmark in label_benchmarks
                or len(method_occurrences[method]) == 1
            ):
                benchmark_index = benchmark_indices[benchmark]
                benchmark_default = (
                    (5, 6 + 3 * (point_index % 2))
                    if benchmark_index % 2 == 0
                    else (5, -12 - 3 * (point_index % 2))
                )
                offset = label_offsets.get(
                    f"{benchmark}:{method}",
                    label_offsets.get(
                        method,
                        benchmark_default,
                    ),
                )
                annotations.append(
                    ax.annotate(
                        method,
                        xy=(x, y),
                        xytext=offset,
                        textcoords="offset points",
                        ha="left" if offset[0] >= 0 else "right",
                        va="bottom" if offset[1] >= 0 else "top",
                        fontsize=7.5,
                        fontweight="bold" if is_highlighted else "normal",
                        zorder=4,
                    )
                )
            point_index += 1

    if not xs:
        raise ValueError("No benchmark methods had finite plotted metrics")

    ax.set_xlim(
        *padded_limits(xs, xerr_low, xerr_high, minimum_padding=args.x_padding)
    )
    ax.set_ylim(
        *padded_limits(ys, yerr_low, yerr_high, minimum_padding=args.y_padding)
    )
    ax.set_xlabel("Forward transfer (FWT)")
    ax.set_ylabel("Mean task AUC (MT-AUC)")
    if args.title:
        ax.set_title(args.title)
    ax.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)

    benchmark_handles = [
        Line2D(
            [0],
            [0],
            marker=markers[benchmark],
            linestyle="none",
            markerfacecolor="#bdbdbd",
            markeredgecolor="black",
            markersize=7,
            label=benchmark,
        )
        for benchmark in benchmark_summaries
    ]
    ax.legend(
        handles=benchmark_handles,
        title="Benchmark",
        loc=args.legend_location,
        frameon=True,
        fontsize=8,
        title_fontsize=8,
    )

    scalar_mappable = mpl.cm.ScalarMappable(norm=color_norm, cmap=cmap)
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(scalar_mappable, ax=ax, pad=0.02)
    colorbar.set_label("Forgetting (FGT; lower is better)")
    if args.repel_labels:
        repel_annotations(fig, annotations)

    return fig


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Plot FWT against MT-AUC, with Continual World forgetting encoded "
            "by marker colour."
        )
    )
    parser.add_argument("--methods", nargs="+", help="NAME=PATH")
    parser.add_argument("--expert-root")
    parser.add_argument("--task-order-config")
    parser.add_argument(
        "--summary-csv",
        nargs="+",
        metavar="BENCHMARK=PATH",
        help=(
            "Merge existing benchmark summaries. PATH may be a generated CSV "
            "or its output directory. This mode does not require --methods, "
            "--expert-root, or --task-order-config."
        ),
    )
    parser.add_argument(
        "--benchmark-marker",
        action="append",
        default=[],
        metavar="BENCHMARK=MARKER",
        help="Matplotlib marker for a merged benchmark; may be repeated.",
    )
    parser.add_argument(
        "--combined-labels",
        default="all",
        help=(
            "Labels to draw in merge mode: `all`, `none`, `first`, or one "
            "benchmark name. Methods unique to one benchmark are always "
            "labelled unless this value is `none`."
        ),
    )
    parser.add_argument(
        "--connect-methods",
        action="store_true",
        help=(
            "Connect points with the same method name across merged "
            "benchmarks."
        ),
    )
    parser.add_argument(
        "--no-repel-labels",
        dest="repel_labels",
        action="store_false",
        help="Disable automatic collision resolution for merged labels.",
    )
    parser.set_defaults(repel_labels=True)
    parser.add_argument(
        "--legend-location",
        default="lower right",
        help="Benchmark legend location in merge mode.",
    )
    parser.add_argument(
        "--performance-normalization",
        choices=["auto", "scalar", "minigrid_shortest_path", "none"],
        default="auto",
    )
    parser.add_argument("--max-return", type=float, default=1.0)
    parser.add_argument("--minigrid-shortest-path-seeds", type=int, nargs="+")
    parser.add_argument("--print-return-scales", action="store_true")
    parser.add_argument("--min-denominator", type=float, default=1e-3)
    parser.add_argument("--fwt-task-ids", type=int, nargs="+")
    parser.add_argument("--exclude-fwt-task-ids", type=int, nargs="+")
    parser.add_argument("--include-unseen-eval-tasks", action="store_true")
    parser.add_argument("--iters", type=int, default=10000)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--out-dir", default="transfer_retention_tradeoff")
    parser.add_argument("--filename", default="transfer_retention_tradeoff")
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--title", default=None)
    parser.add_argument("--width", type=float, default=6.4)
    parser.add_argument("--height", type=float, default=4.6)
    parser.add_argument("--font-family", default="DejaVu Sans")
    parser.add_argument("--colormap", default="plasma")
    parser.add_argument("--fgt-min", type=float, default=None)
    parser.add_argument("--fgt-max", type=float, default=None)
    parser.add_argument("--x-padding", type=float, default=0.04)
    parser.add_argument("--y-padding", type=float, default=0.04)
    parser.add_argument(
        "--highlight",
        nargs="*",
        default=[],
        help=(
            "Methods emphasized with a star in single mode or a larger, "
            "heavier-edged benchmark marker in merge mode."
        ),
    )
    parser.add_argument(
        "--label-offset",
        action="append",
        default=[],
        metavar="METHOD=DX,DY",
        help="Override a label offset in points; may be repeated.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    csv_path = os.path.join(args.out_dir, f"{args.filename}.csv")
    if args.summary_csv:
        if args.methods:
            raise ValueError("--summary-csv cannot be combined with --methods")
        summary_paths = parse_key_value_items(args.summary_csv, "benchmark")
        benchmark_summaries = {
            benchmark: load_summary_csv(path)
            for benchmark, path in summary_paths.items()
        }
        available_methods = {
            method
            for summary in benchmark_summaries.values()
            for method in summary
        }
        unknown_highlights = sorted(set(args.highlight) - available_methods)
        if unknown_highlights:
            raise ValueError(
                f"Highlighted methods not present in any summary: "
                f"{unknown_highlights}"
            )
        write_combined_summary_csv(csv_path, benchmark_summaries)
        fig = make_combined_plot(benchmark_summaries, args)
    else:
        if not args.methods:
            raise ValueError("--methods is required unless --summary-csv is used")
        if not args.expert_root:
            raise ValueError(
                "--expert-root is required unless --summary-csv is used"
            )
        if not args.task_order_config:
            raise ValueError(
                "--task-order-config is required unless --summary-csv is used"
            )
        methods = parse_methods(args.methods)
        unknown_highlights = sorted(set(args.highlight) - set(methods))
        if unknown_highlights:
            raise ValueError(
                f"Highlighted methods not in --methods: {unknown_highlights}"
            )
        metric_args = build_metric_args(args)
        samples, _ = metric_lib.compute_all_metrics(metric_args, methods, METRICS)
        summary = summarize_methods(samples, methods, args)
        write_summary_csv(csv_path, methods, summary)
        fig = make_plot(methods, summary, args)

    output_paths = []
    for output_format in args.formats:
        output_format = output_format.lower().lstrip(".")
        output_path = os.path.join(
            args.out_dir,
            f"{args.filename}.{output_format}",
        )
        fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
        output_paths.append(output_path)
    plt.close(fig)

    print(f"Wrote metric summary to {os.path.abspath(csv_path)}")
    for output_path in output_paths:
        print(f"Wrote figure to {os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
