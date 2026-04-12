"""
Suboptimal / corrupted-demo IoU plots: grouped bars and/or IoU vs violation-rate curves.

Examples:
  python plot_suboptimal.py --primary_root path/to/mticl/results/suboptimal \\
      --compare_root path/to/price/results/price_suboptimal --out comparison_suboptimal.pdf

  python plot_suboptimal.py --primary_root ... --compare_root ... \\
      --tertiary_root path/to/price_forward_suboptimal \\
      --tertiary_label PRICE-forward --plot_style both --curves_out iou_three.pdf
"""

from __future__ import annotations

import argparse
import math
import os.path as osp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_iou_utils import (
    DEFAULT_CONDITION_ORDER,
    collect_run_metrics_ordered,
    plot_iou_comparison,
    plot_iou_vs_violation_curves,
)


def _parse_conditions(s: str) -> list[str]:
    return [p.strip() for p in s.split(",") if p.strip()]


def main():
    p = argparse.ArgumentParser(description="MT-ICL vs PRICE IoU (suboptimal)")
    p.add_argument(
        "--primary_root",
        required=True,
        help="Directory with one subfolder per condition (e.g. clean, vr5).",
    )
    p.add_argument(
        "--compare_root",
        default=None,
        help="Second results tree (same condition folder names), e.g. PRICE-conservative.",
    )
    p.add_argument(
        "--tertiary_root",
        default=None,
        help="Optional third results tree, e.g. PRICE-forward (same condition folder names).",
    )
    p.add_argument(
        "--out",
        default="suboptimal_iou_compare.pdf",
        help="Output path for bar chart (when plot_style includes bars).",
    )
    p.add_argument(
        "--curves_out",
        default=None,
        help="Output path for line plot. Default: <primary_root>/suboptimal_iou_curves.pdf",
    )
    p.add_argument(
        "--plot_style",
        choices=("bars", "curves", "both"),
        default="both",
        help="bars: grouped bar chart; curves: IoU vs violation rate; both: emit both.",
    )
    p.add_argument(
        "--conditions",
        type=str,
        default=",".join(DEFAULT_CONDITION_ORDER),
        help=(
            "Comma-separated condition folder names in x-axis order "
            '(default: "clean,vr5,vr10,vr20,vr50"). Never use sorted(glob).'
        ),
    )
    p.add_argument("--primary_label", default="MT-ICL")
    p.add_argument("--secondary_label", default="PRICE")
    p.add_argument("--tertiary_label", default="PRICE-forward")
    args = p.parse_args()

    condition_order = _parse_conditions(args.conditions)
    if not condition_order:
        raise SystemExit("--conditions must list at least one name")

    curves_out = args.curves_out
    if curves_out is None:
        curves_out = osp.join(args.primary_root, "suboptimal_iou_curves.pdf")
    elif not osp.isabs(curves_out):
        curves_out = osp.join(args.primary_root, curves_out)

    bar_out = args.out
    if not osp.isabs(bar_out):
        bar_out = osp.join(args.primary_root, bar_out)

    if args.plot_style in ("bars", "both"):
        names, primary, secondary, tertiary = collect_run_metrics_ordered(
            args.primary_root,
            condition_order,
            args.compare_root,
            args.tertiary_root,
        )
        if all(isinstance(primary[n], float) and math.isnan(primary[n]) for n in names):
            raise SystemExit(
                f"No constraint_*.npy under {args.primary_root}/<condition>/"
            )
        plot_iou_comparison(
            names,
            primary,
            secondary,
            bar_out,
            primary_label=args.primary_label,
            secondary_label=args.secondary_label,
            tertiary=tertiary if args.tertiary_root else None,
            tertiary_label=args.tertiary_label,
        )
        print(f"Wrote {bar_out}")

    if args.plot_style in ("curves", "both"):
        plot_iou_vs_violation_curves(
            args.primary_root,
            curves_out,
            compare_root=args.compare_root,
            tertiary_root=args.tertiary_root,
            condition_order=condition_order,
            primary_label=args.primary_label,
            secondary_label=args.secondary_label,
            tertiary_label=args.tertiary_label,
        )
        print(f"Wrote {curves_out}")


if __name__ == "__main__":
    main()
