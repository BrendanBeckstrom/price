"""
Behaviorally corrupted maze demos (Breaking Experiment 2).

For each rollout, with probability ``violation_rate`` the planner uses a **zeroed**
constraint grid so the ant crosses wall regions; otherwise the true U-maze mask is used.
Outputs ``demos/corrupted/maze_goal_{g}_demos_{tag}.npz`` matching clean demo keys.
"""

from __future__ import annotations

import argparse
import itertools
import os
import os.path as osp
import sys
from pathlib import Path

import numpy as np

_MTICL_ROOT = Path(__file__).resolve().parents[2]
if str(_MTICL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MTICL_ROOT))

from utils import MazePlanner  # noqa: E402

TRUE_MAZE = np.zeros((10, 10))
TRUE_MAZE[0:6, 2:5] = 1
TRUE_MAZE[4:10, 6:9] = 1


def generate_goal_demos(
    goal_row: int,
    violation_rate: float,
    rng: np.random.Generator,
    out_path: str,
) -> None:
    task_trajs, task_acts, task_constraint_input, task_dirs = [], [], [], []
    task_rewards = []

    for start_row, _ in itertools.product([0, 9], range(10)):
        start, goal = (start_row, 0), (goal_row, 9)
        grid = (
            np.zeros((10, 10))
            if rng.random() < violation_rate
            else TRUE_MAZE.copy()
        )
        planner = MazePlanner(start, goal, grid)
        traj, acts, ci, dirs, rewards = planner.gen_valid_demo()
        task_trajs.extend(traj)
        task_acts.extend(acts)
        task_constraint_input.extend(ci)
        task_dirs.extend(dirs)
        task_rewards.append(rewards)

    os.makedirs(osp.dirname(out_path) or ".", exist_ok=True)
    np.savez(
        out_path,
        trajs=np.array(task_trajs),
        acts=np.array(task_acts),
        constraint_input=np.array(task_constraint_input),
        dirs=np.array(task_dirs),
        rewards=np.asarray(task_rewards, dtype=object),
    )
    print(f"Wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate behaviorally corrupted maze demos")
    p.add_argument(
        "--out_dir",
        default="demos/corrupted",
        help="Output directory (default: demos/corrupted)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--goals",
        type=int,
        nargs="*",
        default=None,
        help="Goal row indices to generate (default: 0..9).",
    )
    p.add_argument(
        "--tag_rate",
        nargs=2,
        action="append",
        metavar=("TAG", "RATE"),
        default=[
            ("vr5", "0.05"),
            ("vr10", "0.10"),
            ("vr20", "0.20"),
            ("vr50", "0.50"),
        ],
        help="Tag and rate (vr5 0.05 -> maze_goal_*_demos_vr5.npz). Repeatable.",
    )
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    goals = args.goals if args.goals is not None else list(range(10))
    for tag, rate_s in args.tag_rate:
        rate = float(rate_s)
        for g in goals:
            out = osp.join(args.out_dir, f"maze_goal_{g}_demos_{tag}.npz")
            generate_goal_demos(g, rate, rng, out)


if __name__ == "__main__":
    main()
