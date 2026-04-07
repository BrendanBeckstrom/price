#!/usr/bin/env python3
"""Load per-run constraint_*.npy under results/suboptimal, IoU vs wall ground truth, heatmaps."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def binary_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    p = pred.astype(bool)
    g = gt.astype(bool)
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    if union == 0:
        return float("nan")
    return float(inter / union)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare suboptimal-experiment constraint grids vs UMaze wall mask."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory containing one subfolder per run (default: ./results/suboptimal next to this script).",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=None,
        help="Path to ground_truth_maze.npy (default: next to this script).",
    )
    parser.add_argument(
        "--constraint-epoch",
        type=int,
        default=4,
        help="Which constraint_{epoch}.npy to load (e.g. 4 for final epoch when --epochs 5).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Binarize learned grid with grid > threshold (matches planner_icl plot_grid thresh).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Figure output path (default: suboptimal_comparison.pdf next to this script).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional path to write IoU table as CSV.",
    )
    parser.add_argument(
        "--ncols",
        type=int,
        default=4,
        help="Max columns in subplot grid before wrapping.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    results_dir = args.results_dir or (script_dir / "results" / "suboptimal")
    gt_path = args.ground_truth or (script_dir / "ground_truth_maze.npy")
    out_path = args.output or (script_dir / "suboptimal_comparison.pdf")

    gt = np.load(gt_path)
    if gt.shape != (10, 10):
        raise ValueError(f"Expected ground truth (10, 10), got {gt.shape} from {gt_path}")

    subdirs = sorted(p for p in results_dir.iterdir() if p.is_dir())
    rows: list[tuple[str, float, Path]] = []
    grids: list[np.ndarray] = []

    for sd in subdirs:
        npy_path = sd / f"constraint_{args.constraint_epoch}.npy"
        if not npy_path.is_file():
            continue
        grid = np.load(npy_path)
        if grid.shape != (10, 10):
            raise ValueError(f"{npy_path}: expected (10, 10), got {grid.shape}")
        pred_bin = grid > args.threshold
        iou = binary_iou(pred_bin, gt)
        rows.append((sd.name, iou, npy_path))
        grids.append(grid)

    if not grids:
        raise SystemExit(
            f"No constraint_{args.constraint_epoch}.npy found under {results_dir}"
        )

    n = len(grids)
    ncols = min(n, max(1, args.ncols))
    nrows = (n + ncols - 1) // ncols
    fig_w = 3.2 * ncols
    fig_h = 3.0 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    for idx, (grid, (name, iou, _)) in enumerate(zip(grids, rows)):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        sns.heatmap(
            grid > args.threshold,
            ax=ax,
            annot=False,
            linewidth=0.25,
            cmap="Blues",
            cbar=False,
        )
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title(f"{name}\nIoU={iou:.4f}", fontsize=11)

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote figure to {out_path}")
    for name, iou, _ in rows:
        print(f"  {name}: IoU = {iou:.6f}")

    if args.csv is not None:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run", "iou"])
            for name, iou, _ in rows:
                w.writerow([name, f"{iou:.8f}"])
        print(f"Wrote CSV to {args.csv}")


if __name__ == "__main__":
    main()
