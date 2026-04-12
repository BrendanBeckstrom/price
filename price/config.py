from dataclasses import dataclass, field


@dataclass
class PRICEConfig:
    """Hyperparameters for PRICE; nested in ExpConfig / ICLConfig."""

    enabled: bool = False
    """Master switch; when False, behavior matches baseline MT-ICL."""

    pairs_per_constraint_phase: int = 32
    """Number of trajectory pairs sampled at the start of each constraint update phase."""

    oracle: str = "gt_wall"
    """Preference oracle: 'gt_wall' (disprefer more wall violations), 'random', or 'none'."""

    routed_positions_cap: int = 4096
    """Max (x,y) points pooled from routed safety trajectories per phase (memory cap)."""

    reward_model_steps: int = 0
    """SGD steps on the pairwise reward model per constraint phase (0 = skip RM training)."""

    reward_model_lr: float = 1e-3

    routed_batch_fraction: float = 0.25
    """Fraction of each constraint minibatch filled from routed safety positions (rest from learner buffer)."""

    route_all_dispreferred: bool = False
    """If True (gt_wall oracle), pool safety from dispreferred trajs with wall violations without median gating."""

    rng_seed: int = 0

    high_reward_rule_doc: str = field(
        default=(
            "Within the current batch of N pairs, let G_i^- be the return of the "
            "dispreferred trajectory in pair i. Dispreferred is 'high reward' iff "
            "G_i^- >= median({G_1^-, ..., G_N^-}) (inclusive upper half)."
        ),
        repr=False,
    )
