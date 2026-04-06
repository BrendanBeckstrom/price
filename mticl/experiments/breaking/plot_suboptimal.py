"""
Overlay IoU across suboptimal / corrupted-demo runs (same subdirectory names in two trees).

Example:
  python plot_suboptimal.py --primary_root path/to/mticl/results/suboptimal \\
      --compare_root path/to/price/results/suboptimal --out comparison_suboptimal.pdf
"""

from __future__ import annotations

import argparse
import os.path as osp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_iou_utils import collect_run_metrics, plot_iou_comparison


def main():
    p = argparse.ArgumentParser(description="MT-ICL vs PRICE IoU (suboptimal matrix)")
    p.add_argument(
        "--primary_root",
        required=True,
        help="Directory containing one subfolder per condition (e.g. violate_05).",
    )
    p.add_argument(
        "--compare_root",
        default=None,
        help="Second results tree with the same subfolder names (e.g. PRICE runs).",
    )
    p.add_argument(
        "--out",
        default="suboptimal_iou_compare.pdf",
        help="Output figure path.",
    )
    p.add_argument("--primary_label", default="MT-ICL")
    p.add_argument("--secondary_label", default="PRICE")
    args = p.parse_args()

    names, primary, secondary = collect_run_metrics(
        args.primary_root, args.compare_root
    )
    if not names:
        raise SystemExit(f"No subdirectories under {args.primary_root}")
    out = args.out
    if not osp.isabs(out):
        out = osp.join(args.primary_root, out)
    plot_iou_comparison(
        names,
        primary,
        secondary,
        out,
        primary_label=args.primary_label,
        secondary_label=args.secondary_label,
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
