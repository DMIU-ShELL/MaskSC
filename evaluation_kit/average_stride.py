import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd


def average_point_stride(values, stride):
    if stride <= 0:
        raise ValueError("stride must be > 0")
    n = len(values)
    usable = n - (n % stride)
    if usable == 0:
        return np.array([], dtype=float)
    arr = np.asarray(values[:usable], dtype=float)
    return arr.reshape(-1, stride).mean(axis=1)


def average_task_groups(values, task_len, tasks_per_group):
    if task_len <= 0:
        raise ValueError("task_len must be > 0")
    if tasks_per_group <= 0:
        raise ValueError("tasks_per_group must be > 0")
    n = len(values)
    n_tasks = n // task_len
    usable_tasks = (n_tasks // tasks_per_group) * tasks_per_group
    usable_points = usable_tasks * task_len
    if usable_points == 0:
        return np.array([], dtype=float)
    arr = np.asarray(values[:usable_points], dtype=float)
    arr = arr.reshape(usable_tasks, task_len)
    n_groups = usable_tasks // tasks_per_group
    grouped = arr.reshape(n_groups, tasks_per_group, task_len).mean(axis=1)
    return grouped.reshape(-1)


def process_file(
    filepath,
    mode,
    stride,
    task_len,
    tasks_per_group,
    value_col,
):
    df = pd.read_csv(filepath)
    if value_col not in df.columns:
        raise ValueError(f"Missing value column '{value_col}' in {filepath}")

    if mode == "point-stride":
        value_out = average_point_stride(df[value_col].values, stride)
        wall_out = None
        if "Wall Time" in df.columns:
            wall_out = average_point_stride(df["Wall Time"].values, stride)
    else:
        value_out = average_task_groups(df[value_col].values, task_len, tasks_per_group)
        wall_out = None
        if "Wall Time" in df.columns:
            wall_out = average_task_groups(df["Wall Time"].values, task_len, tasks_per_group)

    step_out = np.arange(len(value_out), dtype=int)

    out = {
        "Step": step_out,
        value_col: value_out,
    }
    if wall_out is not None:
        if len(wall_out) != len(value_out):
            min_len = min(len(wall_out), len(value_out))
            wall_out = wall_out[:min_len]
            out["Step"] = step_out[:min_len]
            out[value_col] = value_out[:min_len]
        out["Wall Time"] = wall_out
        # Keep column order similar to TensorBoard CSVs
        out_df = pd.DataFrame(out, columns=["Wall Time", "Step", value_col])
    else:
        out_df = pd.DataFrame(out, columns=["Step", value_col])

    return out_df


def build_default_output_dir(method_dir, mode, stride, task_len, tasks_per_group):
    if mode == "point-stride":
        name = f"avg_stride{stride}"
    else:
        name = f"avg_tasks{tasks_per_group}_len{task_len}"
    return os.path.join(method_dir, name)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Average per-seed training curves either by averaging every N points "
            "(point-stride) or by grouping tasks into chunks and averaging aligned iterations "
            "(task-group)."
        )
    )
    parser.add_argument(
        "method_dir",
        help="Path to method directory containing per-seed .csv files",
    )
    parser.add_argument(
        "--mode",
        choices=["task-group", "point-stride"],
        default="task-group",
        help="Averaging mode (default: task-group)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=4,
        help="Stride size for point-stride mode (default: 4)",
    )
    parser.add_argument(
        "--task-len",
        type=int,
        default=100,
        help="Iterations per task for task-group mode (default: 100)",
    )
    parser.add_argument(
        "--tasks-per-group",
        type=int,
        default=4,
        help="Tasks per group for task-group mode (default: 4)",
    )
    parser.add_argument(
        "--value-col",
        default="Value",
        help="Column name to average (default: Value)",
    )
    parser.add_argument(
        "--pattern",
        default="*.csv",
        help="Glob pattern for input CSVs (default: *.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for averaged CSVs (default: <method_dir>/avg_...)",
    )

    args = parser.parse_args()

    method_dir = os.path.abspath(args.method_dir)
    if not os.path.isdir(method_dir):
        print(f"Not a directory: {method_dir}", file=sys.stderr)
        return 1

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = build_default_output_dir(
            method_dir,
            args.mode,
            args.stride,
            args.task_len,
            args.tasks_per_group,
        )

    os.makedirs(output_dir, exist_ok=True)

    pattern = os.path.join(method_dir, args.pattern)
    files = sorted([p for p in glob.glob(pattern) if os.path.isfile(p)])
    if not files:
        print(f"No CSV files found at: {pattern}", file=sys.stderr)
        return 1

    for path in files:
        try:
            out_df = process_file(
                path,
                args.mode,
                args.stride,
                args.task_len,
                args.tasks_per_group,
                args.value_col,
            )
        except Exception as exc:
            print(f"Failed to process {path}: {exc}", file=sys.stderr)
            continue

        out_path = os.path.join(output_dir, os.path.basename(path))
        out_df.to_csv(out_path, index=False)
        print(f"Wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
