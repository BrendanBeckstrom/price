from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from price.config import PRICEConfig
from price.trajectory import TrajectoryRecord


@dataclass
class RoutingResult:
    """Outputs of one constraint-phase PRICE routing pass."""

    safety_positions: np.ndarray  # (P, 2) points to treat as learner-class (+1) in MSE loss
    low_reward_pairs: List[Tuple[TrajectoryRecord, TrajectoryRecord, int]]
    """(traj_a, traj_b, dispreferred_idx) for RM training; dispreferred has low reward vs batch median."""
    dispreferred_returns: np.ndarray
    high_reward_mask: np.ndarray  # bool per pair


def _median_high_reward_mask(dispreferred_returns: np.ndarray) -> np.ndarray:
    """G^- >= median(batch) => high reward (inclusive upper half)."""
    if dispreferred_returns.size == 0:
        return np.array([], dtype=bool)
    med = float(np.median(dispreferred_returns))
    return dispreferred_returns >= med


def label_dispreferred_gt_wall(a: TrajectoryRecord, b: TrajectoryRecord) -> int:
    """Oracle: disprefer the trajectory with more wall violations; tie -> lower cum_return."""
    if a.wall_violations > b.wall_violations:
        return 0
    if b.wall_violations > a.wall_violations:
        return 1
    return 0 if a.cum_return < b.cum_return else 1


def label_dispreferred_random(
    a: TrajectoryRecord, b: TrajectoryRecord, rng: np.random.Generator
) -> int:
    return int(rng.integers(0, 2))


def sample_trajectory_pairs(
    trajs: Sequence[TrajectoryRecord],
    num_pairs: int,
    rng: np.random.Generator,
) -> List[Tuple[TrajectoryRecord, TrajectoryRecord]]:
    """Sample with replacement if fewer than 2 trajectories."""
    n = len(trajs)
    if n < 2:
        return []
    pairs = []
    for _ in range(num_pairs):
        i, j = rng.choice(n, size=2, replace=False)
        pairs.append((trajs[i], trajs[j]))
    return pairs


def route_preference_batch(
    pairs: Sequence[Tuple[TrajectoryRecord, TrajectoryRecord]],
    label_fn: Callable[[TrajectoryRecord, TrajectoryRecord], int],
    cfg: PRICEConfig,
    rng: Optional[np.random.Generator] = None,
) -> RoutingResult:
    """
    For each pair, label dispreferred index, then split by high-vs-low reward
    (median over dispreferred returns in this batch).
    """
    if not pairs or cfg.oracle == "none":
        return RoutingResult(
            safety_positions=np.zeros((0, 2), dtype=np.float64),
            low_reward_pairs=[],
            dispreferred_returns=np.array([]),
            high_reward_mask=np.array([], dtype=bool),
        )

    dis_idx: List[int] = []
    dis_returns: List[float] = []
    for a, b in pairs:
        d = label_fn(a, b)
        dis_idx.append(d)
        dis = a if d == 0 else b
        dis_returns.append(dis.cum_return)

    dispreferred_returns = np.asarray(dis_returns, dtype=np.float64)
    high_mask = _median_high_reward_mask(dispreferred_returns)

    safety_chunks: List[np.ndarray] = []
    low_pairs: List[Tuple[TrajectoryRecord, TrajectoryRecord, int]] = []

    for (a, b), d, is_high in zip(pairs, dis_idx, high_mask):
        dis = a if d == 0 else b
        pref = b if d == 0 else a
        if is_high:
            safety_chunks.append(np.asarray(dis.positions, dtype=np.float64))
        else:
            low_pairs.append((a, b, d))

    rng = rng or np.random.default_rng()
    if safety_chunks:
        safety_positions = np.concatenate(safety_chunks, axis=0)
        if safety_positions.shape[0] > cfg.routed_positions_cap:
            idx = rng.choice(
                safety_positions.shape[0],
                size=cfg.routed_positions_cap,
                replace=False,
            )
            safety_positions = safety_positions[idx]
    else:
        safety_positions = np.zeros((0, 2), dtype=np.float64)

    return RoutingResult(
        safety_positions=safety_positions,
        low_reward_pairs=low_pairs,
        dispreferred_returns=dispreferred_returns,
        high_reward_mask=high_mask,
    )
