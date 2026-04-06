"""Build TrajectoryRecords from Tianshou buffers after Collector.collect."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from price.trajectory import TrajectoryRecord


def trajectory_records_from_collect_stats(
    buffer: Any,
    stats: Dict[str, Any],
) -> List[TrajectoryRecord]:
    """
    One TrajectoryRecord per completed episode in the last collect call.
    Uses stats['idxs'] and stats['lens'] when present (Tianshou convention).
    """
    idxs = stats.get("idxs")
    lens = stats.get("lens")
    if idxs is None or lens is None:
        return []

    idxs = np.asarray(idxs).reshape(-1)
    lens = np.asarray(lens).reshape(-1)
    buf_len = len(buffer)
    if buf_len == 0:
        return []

    out: List[TrajectoryRecord] = []
    for st, ln in zip(idxs, lens):
        ln = int(ln)
        if ln <= 0:
            continue
        st = int(st)
        inds = (np.arange(st, st + ln, dtype=np.int64) % buf_len).astype(np.int64)
        sub = buffer[inds]
        obs = np.asarray(sub.obs)
        rew = float(np.sum(np.asarray(sub.rew)))
        ci = np.asarray(sub.info.constraint_input)
        if ci.ndim == 1:
            ci = ci.reshape(ln, -1)
        obs_trim = obs[:, :-1]
        rows = np.concatenate([obs_trim, ci.reshape(ln, -1)], axis=1).astype(np.float32)

        constr = getattr(sub.info, "constraint", None)
        if constr is not None:
            c = np.asarray(constr).reshape(-1)
            wv = int(np.sum(c > 0))
        else:
            wv = 0

        out.append(
            TrajectoryRecord(
                positions=rows,
                cum_return=rew,
                wall_violations=wv,
                goal_id=-1,
            )
        )
    return out
