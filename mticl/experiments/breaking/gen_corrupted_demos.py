"""
Generate maze expert demos with a mix of constraint-respecting and blind planners.

For each trajectory, with probability ``violation_rate`` the discrete planner uses a
**blind** grid (all cells free) so shortest paths may cut through logical wall cells.
The default MuJoCo ``maze_map`` in ``MazePlanner.create_maze_env`` (when render_mode is
None) uses an **open interior** map, not the full UMaze wall layout—so corruption is
primarily via **expert labels** in ``constraint_input`` (positions vs ground-truth
walls), not necessarily blocked physics. See ``utils/maze_utils.MazePlanner``.

Run from the ``mticl`` package root with ``PYTHONPATH`` set to that directory, e.g.::

    python experiments/breaking/gen_corrupted_demos.py --violation_rate 0.05 --output_dir demos/corrupted
"""

from __future__ import annotations

import argparse
import itertools
import os
import os.path as osp
import random
import sys
from pathlib import Path

import numpy as np

_MTICL_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_MTICL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MTICL_ROOT))

from utils import MazePlanner


def _wall_grid() -> np.ndarray:
    maze = np.zeros((10, 10))
    maze[0:6, 2:5] = 1
    maze[4:10, 6:9] = 1
    return maze


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate corrupted maze demos with optional blind-planning violations."
    )
    parser.add_argument(
        "--violation_rate",
        type=float,
        required=True,
        help="Probability [0,1] of using blind (all-free) constraint grid per trajectory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="demos/corrupted",
        help="Directory to write maze_goal_*_demos_vr*.npz files.",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    args = parser.parse_args()

    if not 0.0 <= args.violation_rate <= 1.0:
        raise SystemExit("--violation_rate must be between 0 and 1.")

    random.seed(args.seed)
    np.random.seed(args.seed)

    vr_id = int(args.violation_rate * 100)
    maze = _wall_grid()
    blind_maze = np.zeros((10, 10))

    os.makedirs(args.output_dir, exist_ok=True)

    for goal_row in range(10):
        print(f"Generating demos for goal row {goal_row}")

        task_trajs, task_acts, task_constraint_input, task_dirs = [], [], [], []
        task_rewards = []
        n_corrupted = 0
        n_clean = 0

        for _start_pair in itertools.product([0, 9], range(10)):
            start_row, _ = _start_pair
            start, goal = (start_row, 0), (goal_row, 9)
            use_blind = random.random() < args.violation_rate
            grid = blind_maze if use_blind else maze
            if use_blind:
                n_corrupted += 1
            else:
                n_clean += 1

            planner = MazePlanner(start, goal, grid)
            traj, acts, ci, dirs, rewards = planner.gen_valid_demo()

            task_trajs.extend(traj)
            task_acts.extend(acts)
            task_constraint_input.extend(ci)
            task_dirs.extend(dirs)
            task_rewards.append(rewards)

        total = n_corrupted + n_clean
        print(
            f"  goal {goal_row}: total_trajectories={total}, "
            f"corrupted={n_corrupted}, clean={n_clean}"
        )

        base = osp.join(
            args.output_dir, f"maze_goal_{goal_row}_demos_vr{vr_id}"
        )
        # One entry per trajectory; each is a variable-length list of step rewards.
        np.savez(
            base,
            trajs=np.array(task_trajs),
            acts=np.array(task_acts),
            constraint_input=np.array(task_constraint_input),
            dirs=np.array(task_dirs),
            rewards=np.array(task_rewards, dtype=object),
        )


if __name__ == "__main__":
    main()
