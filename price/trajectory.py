from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrajectoryRecord:
    """One rollout used for PRICE preference sampling (maze planner or RL episodes)."""

    positions: np.ndarray  # (T, D) — rows match constraint learner inputs (maze: D=2 xy)
    cum_return: float  # scalar return (e.g. sum of step rewards)
    wall_violations: int  # count of steps in wall cells under ground-truth maze
    goal_id: int  # task index, or -1 if single-task / N/A
