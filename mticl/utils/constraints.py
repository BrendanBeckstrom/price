import sys
from pathlib import Path

import numpy as np
import torch
from tianshou.data import ReplayBuffer
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from price.replay_episodes import trajectory_records_from_collect_stats
from price.reward_model import train_reward_model_on_pairs
from price.router import (
    label_dispreferred_gt_wall,
    label_dispreferred_random,
    route_preference_batch,
    sample_trajectory_pairs,
)
from price.trajectory import TrajectoryRecord


def to_torch(x: np.ndarray):
    return torch.tensor(x, dtype=torch.float32)


class Constraint(nn.Module):
    def __init__(self, in_dim=1):
        super(Constraint, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1),
        )
        self.in_dim = in_dim

    def raw_forward(self, inputs):
        inputs = inputs.view(inputs.shape[0], -1)
        net_inputs = inputs[:, -self.in_dim :]
        output = self.net(net_inputs)
        return output.view(-1)

    def forward(self, inputs):
        return torch.log(nn.functional.relu(self.raw_forward(inputs)) + 1.0)

    def eval_trajs(self, trajs: np.ndarray, act=True):
        inputs = to_torch(trajs)
        with torch.no_grad():
            output = self.forward(inputs) if act else self.raw_forward(inputs)
        return torch.Tensor.numpy(output, force=True)


class MazeConstraint(Constraint):
    def __init__(self, in_dim=2, dim=128, out_dim=1):
        super(MazeConstraint, self).__init__(in_dim=in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, out_dim),
        )


class ReducedVelocityConstraint(Constraint):
    def __init__(self, in_dim=1):
        super(ReducedVelocityConstraint, self).__init__(in_dim=in_dim)
        self.net = nn.Parameter(torch.randn(1))

    # inputs = (batch_size x [x, y])
    def raw_forward(self, inputs):
        inputs = inputs.view(inputs.shape[0], -1)
        net_inputs = inputs[:, -self.in_dim :]
        output = net_inputs - self.net
        return output.view(-1)


class ReducedPositionConstraint(Constraint):
    def __init__(self, in_dim=1):
        super(ReducedPositionConstraint, self).__init__(in_dim=in_dim)
        self.net = nn.Parameter(torch.randn(1))

    # inputs = (batch_size x [x, y])
    def raw_forward(self, inputs):
        inputs = inputs.view(inputs.shape[0], -1)
        net_inputs = inputs[:, -self.in_dim :]
        output = net_inputs[:, 0] * self.net - net_inputs[:, 1]
        return output.view(-1)


class ConstraintLearner:
    def __init__(self, args):
        self.args = args

        match (args.constraint_type, args.method):
            case ("Velocity", "cpo"):
                self.constraint = Constraint(in_dim=args.dim)
                constraint_weight = torch.tensor([[1.0]], dtype=torch.float32)
                constraint_bias = torch.tensor(
                    [-args.constraint_limit], dtype=torch.float32
                )
                state = {
                    "net.0.weight": constraint_weight,
                    "net.0.bias": constraint_bias,
                }
                self.constraint.load_state_dict(state)

            case ("Velocity", "icl"):
                if args.dim == 1:
                    self.constraint = ReducedVelocityConstraint(in_dim=args.dim)
                else:
                    self.constraint = Constraint(in_dim=args.dim)

            case ("Position", "cpo"):
                self.constraint = Constraint(in_dim=2)
                constraint_weight = torch.tensor(
                    [[args.constraint_limit, -1.0]], dtype=torch.float32
                )
                constraint_bias = torch.tensor([0.0], dtype=torch.float32)
                state = {
                    "net.0.weight": constraint_weight,
                    "net.0.bias": constraint_bias,
                }
                self.constraint.load_state_dict(state)

            case ("Position", "icl"):
                if args.dim == 2:
                    self.constraint = ReducedPositionConstraint(in_dim=args.dim)
                else:
                    self.constraint = Constraint(in_dim=args.dim)

            case _:
                raise NotImplementedError

        if args.method == "icl":
            self.demos = np.load(args.expert_traj_path, allow_pickle=True)
            self.expert_steps = self.demos["trajs"].shape[0]
            self.expert_trajs = np.concatenate(
                [
                    self.demos["trajs"].reshape(self.expert_steps, -1)[:, :-1],
                    self.demos["constraint_input"].reshape(self.expert_steps, -1),
                ],
                axis=1,
            )
            self.c_opt = Adam(self.constraint.parameters(), lr=args.constraint_lr)
            self.buffer = ReplayBuffer(size=(args.outer_epochs * self.expert_steps))

        self.epoch_trajs: list[TrajectoryRecord] = []
        self.routed_safety_features = np.zeros((0, 1), dtype=np.float32)
        self._price_rm_pairs: list = []
        self._reward_model = None

    def expert_cost(self):
        expert_costs = self.norm_constraint.eval_trajs(self.expert_trajs)
        return expert_costs.sum() / self.args.num_expert_trajs

    def collect_trajs(self, test_collector):
        stats = test_collector.collect(n_episode=self.args.episode_per_collect)
        self.epoch_trajs = trajectory_records_from_collect_stats(
            test_collector.buffer, stats
        )
        learner_sample, _ = test_collector.buffer.sample(self.expert_steps)
        for batch in learner_sample.split(1):
            self.buffer.add(batch)
        test_collector.reset()

    def prepare_price_constraint_phase(self, outer_epoch: int) -> None:
        fdim = int(self.expert_trajs.shape[1]) if self.expert_trajs.size else 1
        self.routed_safety_features = np.zeros((0, fdim), dtype=np.float32)
        self._price_rm_pairs = []
        if self.args.method != "icl" or not self.args.price.enabled:
            return
        if self.args.price.oracle == "none":
            return
        if len(self.epoch_trajs) < 2:
            return

        rng = np.random.default_rng(int(self.args.price.rng_seed) + int(outer_epoch))
        n_pairs = int(self.args.price.pairs_per_constraint_phase)
        pairs = sample_trajectory_pairs(self.epoch_trajs, n_pairs, rng)
        if not pairs:
            return

        if self.args.price.oracle == "gt_wall":
            label_fn = label_dispreferred_gt_wall
        elif self.args.price.oracle == "random":
            label_fn = lambda a, b: label_dispreferred_random(a, b, rng)
        else:
            raise ValueError(f"Unknown price.oracle: {self.args.price.oracle}")

        res = route_preference_batch(pairs, label_fn, self.args.price, rng)
        self.routed_safety_features = np.asarray(res.safety_positions, dtype=np.float32)
        self._price_rm_pairs = res.low_reward_pairs

    def train_price_reward_model_pass(self) -> None:
        if not self.args.price.enabled or self.args.price.reward_model_steps <= 0:
            return
        if not self._price_rm_pairs:
            return
        self._reward_model, loss = train_reward_model_on_pairs(
            self._reward_model,
            self._price_rm_pairs,
            self.args.price.reward_model_steps,
            self.args.price.reward_model_lr,
            device=str(self.args.device),
        )
        print(f"PRICE reward model mean loss: {loss:.6f}")

    def sample_batch(self):
        batch_size = self.args.constraint_batch_size
        expert_indices = np.random.choice(self.expert_steps, batch_size)
        expert_batch = self.expert_trajs[expert_indices]

        if (
            self.args.price.enabled
            and self.routed_safety_features.size > 0
            and self.routed_safety_features.shape[0] > 0
            and self.routed_safety_features.shape[1] == expert_batch.shape[1]
        ):
            n_route = min(
                int(batch_size * self.args.price.routed_batch_fraction),
                self.routed_safety_features.shape[0],
                batch_size,
            )
            n_rest = batch_size - n_route
            route_idx = np.random.choice(
                self.routed_safety_features.shape[0], size=n_route, replace=True
            )
            route_part = self.routed_safety_features[route_idx]
            if n_rest > 0:
                learner_indices = self.buffer.sample_indices(n_rest)
                learner_obs = self.buffer.get(learner_indices, key="obs").reshape(
                    n_rest, -1
                )[:, :-1]
                learner_ci = self.buffer.get(
                    learner_indices, key="info"
                ).constraint_input.reshape(n_rest, -1)
                rest_part = np.concatenate([learner_obs, learner_ci], axis=1)
                learner_batch = np.concatenate([route_part, rest_part], axis=0)
            else:
                learner_batch = route_part
        else:
            learner_indices = self.buffer.sample_indices(batch_size)
            learner_obs = self.buffer.get(learner_indices, key="obs").reshape(
                batch_size, -1
            )[:, :-1]
            learner_ci = self.buffer.get(
                learner_indices, key="info"
            ).constraint_input.reshape(batch_size, -1)
            learner_batch = np.concatenate([learner_obs, learner_ci], axis=1)

        expert_data = to_torch(expert_batch)
        learner_data = to_torch(learner_batch)
        assert expert_data.shape == learner_data.shape
        return expert_data, learner_data

    def set_norm_constraint(self):
        self.norm_constraint = self.constraint
        if self.args.full_state:
            self.norm_constraint = Constraint(in_dim=self.args.dim)
            with torch.no_grad():
                constraint_weight, constraint_bias = list(self.constraint.parameters())
                l2_norm = torch.linalg.vector_norm(constraint_weight, ord=2)
                assert l2_norm != 0
                state = {
                    "net.0.weight": constraint_weight / l2_norm.item(),
                    "net.0.bias": constraint_bias / l2_norm.item(),
                }
                self.norm_constraint.load_state_dict(state)

    def update_constraint(self):
        self.constraint.train()
        self.c_opt = Adam(self.constraint.parameters(), lr=self.args.constraint_lr)
        for idx in (pbar := tqdm(range(self.args.constraint_steps))):
            self.c_opt.zero_grad()
            expert_batch, learner_batch = self.sample_batch()

            c_learner = self.constraint.raw_forward(learner_batch.float())
            c_expert = self.constraint.raw_forward(expert_batch.float())
            c_output = torch.concat([c_expert, c_learner])
            c_labels = torch.concat(
                [-1 * torch.ones(c_expert.shape), torch.ones(c_learner.shape)]
            )
            c_loss = torch.mean((c_output - c_labels) ** 2)
            pbar.set_description(f"Constraint Loss {c_loss.item()}")

            c_loss.backward()
            self.c_opt.step()

        self.constraint.eval()
        self.set_norm_constraint()
