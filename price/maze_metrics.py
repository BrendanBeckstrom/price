"""Maze-specific helpers for PRICE (depends only on numpy + a planner with compute_constraint)."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def wall_violation_count(planner: Any, constraint_rows: Sequence[np.ndarray]) -> int:
    """Count steps where xy lies in a ground-truth wall cell (MazePlanner.compute_constraint == 1)."""
    n = 0
    for row in constraint_rows:
        pos = np.asarray(row).reshape(-1)[:2]
        if int(planner.compute_constraint(pos)) == 1:
            n += 1
    return n
