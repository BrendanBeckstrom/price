"""Lightweight trajectory-level reward model for PRICE low-reward routed pairs (Phase 2 / optional)."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from price.trajectory import TrajectoryRecord


def _traj_features(traj: TrajectoryRecord) -> np.ndarray:
    """Fixed-size summary: mean xy (2,) + log1p(length)."""
    p = np.asarray(traj.positions, dtype=np.float32)
    mean_xy = p.mean(axis=0) if p.size else np.zeros(2, dtype=np.float32)
    length = np.log1p(float(len(p))).astype(np.float32)
    return np.concatenate([mean_xy, np.array([length], dtype=np.float32)], axis=0)


class TrajectoryRewardMLP(nn.Module):
    def __init__(self, in_dim: int = 3, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_reward_model_on_pairs(
    model: Optional[TrajectoryRewardMLP],
    low_reward_pairs: List[Tuple[TrajectoryRecord, TrajectoryRecord, int]],
    steps: int,
    lr: float,
    device: str = "cpu",
) -> Tuple[Optional[TrajectoryRewardMLP], float]:
    """
    Bradley-Terry style: maximize log sigmoid(R(pref) - R(dis)) for each pair.
    dispreferred index is 0 or 1.
    """
    if not low_reward_pairs or steps <= 0:
        return model, 0.0

    fdim = int(_traj_features(low_reward_pairs[0][0]).shape[0])
    if model is None:
        model = TrajectoryRewardMLP(in_dim=fdim)
    model = model.to(device)
    opt = Adam(model.parameters(), lr=lr)
    feats = []
    labels = []
    for a, b, d in low_reward_pairs:
        fa = _traj_features(a)
        fb = _traj_features(b)
        feats.append(np.stack([fa, fb], axis=0))
        # preferred index is 1-d
        pref = 1 - d
        labels.append((pref, d))

    stack = torch.tensor(np.stack(feats, axis=0), device=device)  # (B, 2, 3)
    total_loss = 0.0
    for _ in range(steps):
        idx = np.random.randint(0, len(low_reward_pairs))
        x = stack[idx]  # (2, 3)
        pref_i, dis_i = labels[idx]
        r0, r1 = model(x[0]), model(x[1])
        # want r_pref > r_dis
        if pref_i == 0:
            diff = r0 - r1
        else:
            diff = r1 - r0
        loss = -torch.nn.functional.logsigmoid(diff).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += float(loss.detach().cpu())

    return model, total_loss / steps

#  (9, 0)] 91
# ===== vr20 finished in 21001s =====

# =================================================================
#   Diagnostic v3 Summary
# =================================================================
# Final IoU: 0.5574
# gen_valid_demo exhaustions: 150

# Wall violations per epoch:
# [price] epoch 0: 198/200 epoch trajectories have wall_violations > 0
# [price] epoch 1: 18/200 epoch trajectories have wall_violations > 0
# [price] epoch 2: 168/200 epoch trajectories have wall_violations > 0

# Routed pool size per epoch:
# [price] epoch 0: routed pool size = 16384 (added 16384 this epoch, growing)
# [price] epoch 1: routed pool size = 32768 (added 16384 this epoch, growing)
# [price] epoch 2: routed pool size = 49152 (added 16384 this epoch, growing)

# Elapsed time: 21001s
# Results saved under experiments\breaking\results\price_stable_diagnostic_v3\violate_20
# =================================================================

# (mticl) C:\Users\super\OneDrive\Desktop\AI final proj\price\mticl\experiments\breaking>