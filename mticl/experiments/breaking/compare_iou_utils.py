"""Shared IoU computation and matplotlib helpers for MT-ICL vs PRICE plots."""

from __future__ import annotations

import os
import os.path as osp
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def umaze_ground_truth_grid() -> np.ndarray:
    """Binary wall mask (10x10) matching gen_icl_plots / gen_maze_plots."""
    gt = np.zeros((10, 10), dtype=np.float64)
    gt[0:6, 2:5] = 1.0
    gt[4:10, 6:9] = 1.0
    return gt


def constraint_iou(pred: np.ndarray, gt: np.ndarray, thresh: float = 0.5) -> float:
    p = pred > thresh
    g = gt > 0.5
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    return float(inter / max(union, 1e-8))


def latest_constraint_path(run_dir: str) -> Optional[str]:
    paths = sorted(Path(run_dir).glob("constraint_*.npy"))
    if not paths:
        return None
    return str(paths[-1])


def iou_for_run_dir(run_dir: str, epoch_index: int = -1) -> Optional[float]:
    paths = sorted(Path(run_dir).glob("constraint_*.npy"))
    if not paths:
        return None
    path = str(paths[epoch_index])
    grid = np.load(path)
    return constraint_iou(grid, umaze_ground_truth_grid())


def collect_run_metrics(
    parent: str,
    compare_parent: Optional[str] = None,
) -> Tuple[List[str], Dict[str, float], Dict[str, Optional[float]]]:
    """
    One subdirectory of `parent` per condition (e.g. violate_05, clean_baseline).
    Returns sorted names, primary IoUs, optional compare IoUs aligned by name.
    """
    names = sorted([p.name for p in Path(parent).iterdir() if p.is_dir()])
    primary: Dict[str, float] = {}
    secondary: Dict[str, Optional[float]] = {}
    gt = umaze_ground_truth_grid()
    for name in names:
        pdir = osp.join(parent, name)
        lp = latest_constraint_path(pdir)
        if lp is None:
            primary[name] = float("nan")
        else:
            primary[name] = constraint_iou(np.load(lp), gt)
        if compare_parent:
            cdir = osp.join(compare_parent, name)
            clp = latest_constraint_path(cdir) if osp.isdir(cdir) else None
            secondary[name] = (
                constraint_iou(np.load(clp), gt) if clp is not None else None
            )
        else:
            secondary[name] = None
    return names, primary, secondary


# Explicit x order for suboptimal / violation-rate curves (never sort folder names).
DEFAULT_CONDITION_ORDER = ["clean", "vr5", "vr10", "vr20", "vr50"]

# Human-readable x tick labels aligned with DEFAULT_CONDITION_ORDER
DEFAULT_VIOLATION_XTICKLABELS = ["Clean", "5%", "10%", "20%", "50%"]


def iou_series_ordered(
    root: str,
    condition_order: List[str],
) -> np.ndarray:
    """IoU values for each condition subdirectory, in ``condition_order`` only (no sorting)."""
    gt = umaze_ground_truth_grid()
    ys = []
    for name in condition_order:
        pdir = osp.join(root, name)
        lp = latest_constraint_path(pdir) if osp.isdir(pdir) else None
        if lp is None:
            ys.append(float("nan"))
        else:
            ys.append(constraint_iou(np.load(lp), gt))
    return np.asarray(ys, dtype=np.float64)


def collect_run_metrics_ordered(
    parent: str,
    condition_order: List[str],
    compare_parent: Optional[str] = None,
) -> Tuple[List[str], Dict[str, float], Dict[str, Optional[float]]]:
    """
    Like collect_run_metrics but uses ``condition_order`` for bar x positions (not sorted()).
    """
    names = list(condition_order)
    primary: Dict[str, float] = {}
    secondary: Dict[str, Optional[float]] = {}
    gt = umaze_ground_truth_grid()
    for name in names:
        pdir = osp.join(parent, name)
        lp = latest_constraint_path(pdir) if osp.isdir(pdir) else None
        if lp is None:
            primary[name] = float("nan")
        else:
            primary[name] = constraint_iou(np.load(lp), gt)
        if compare_parent:
            cdir = osp.join(compare_parent, name)
            clp = latest_constraint_path(cdir) if osp.isdir(cdir) else None
            secondary[name] = (
                constraint_iou(np.load(clp), gt) if clp is not None else None
            )
        else:
            secondary[name] = None
    return names, primary, secondary


def plot_iou_vs_violation_curves(
    primary_root: str,
    out_path: str,
    compare_root: Optional[str] = None,
    condition_order: Optional[List[str]] = None,
    xticklabels: Optional[List[str]] = None,
    primary_label: str = "MT-ICL",
    secondary_label: str = "PRICE",
) -> None:
    """
    Line plot: IoU vs violation condition with explicit x order (see DEFAULT_CONDITION_ORDER).
    """
    order = list(condition_order or DEFAULT_CONDITION_ORDER)
    y1 = iou_series_ordered(primary_root, order)
    x = np.arange(len(order))

    if xticklabels is not None and len(xticklabels) != len(order):
        raise ValueError("xticklabels length must match condition_order length")
    xlabels = (
        xticklabels
        if xticklabels is not None
        else (
            DEFAULT_VIOLATION_XTICKLABELS
            if len(order) == len(DEFAULT_VIOLATION_XTICKLABELS)
            else order
        )
    )

    fig, ax = plt.subplots(figsize=(max(7, len(order) * 1.2), 4.5))
    ax.plot(x, y1, "o-", label=primary_label, linewidth=2, markersize=9, color="C0")
    if compare_root:
        y2 = iou_series_ordered(compare_root, order)
        ax.plot(
            x,
            y2,
            "s-",
            label=secondary_label,
            linewidth=2,
            markersize=9,
            color="C1",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("IoU vs ground-truth maze walls")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    os.makedirs(osp.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_iou_comparison(
    names: List[str],
    primary: Dict[str, float],
    secondary: Dict[str, Optional[float]],
    out_path: str,
    primary_label: str = "MT-ICL",
    secondary_label: str = "PRICE",
) -> None:
    x = np.arange(len(names))
    w = 0.35
    y1 = [primary[n] for n in names]
    has_sec = any(secondary[n] is not None for n in names)
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.6), 4))
    ax.bar(x - w / 2, y1, width=w, label=primary_label)
    if has_sec:
        y2 = [secondary[n] if secondary[n] is not None else float("nan") for n in names]
        ax.bar(x + w / 2, y2, width=w, label=secondary_label)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("IoU vs ground-truth maze walls")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    os.makedirs(osp.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
