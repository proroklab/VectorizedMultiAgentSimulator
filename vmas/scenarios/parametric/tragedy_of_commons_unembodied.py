#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
from typing import Dict, List, Sequence

import torch
from genagg import GenAgg
from torch import Tensor
from torch_geometric.nn import SoftmaxAggregation

from vmas import render_interactively
from vmas.simulator.core import Agent, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import AGENT_INFO_TYPE, TorchUtils


def agg_max(x, dim):
    return x.max(dim=dim, keepdim=True)[0]


def agg_min(x, dim):
    return x.min(dim=dim, keepdim=True)[0]


def agg_mean(x, dim):
    return x.mean(dim=dim, keepdim=True)


class Square:
    def forward(self, x):
        return (x + 1e-7) ** 2

    def inverse(self, x):
        return (x.abs() + 1e-7).sqrt()


class LAF(torch.nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

        self.a = torch.nn.Parameter(torch.ones(1))
        self.b = torch.nn.Parameter(torch.ones(1))
        self.c = torch.nn.Parameter(torch.ones(1))
        self.d = torch.nn.Parameter(torch.ones(1))
        self.e = torch.nn.Parameter(torch.ones(1))
        self.f = torch.nn.Parameter(torch.ones(1))
        self.g = torch.nn.Parameter(torch.ones(1))
        self.h = torch.nn.Parameter(torch.ones(1))

        self.alpha = torch.nn.Parameter(torch.ones(1))
        self.beta = torch.nn.Parameter(torch.ones(1))
        self.gamma = torch.nn.Parameter(torch.ones(1))
        self.delta = torch.nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor, dim=-1):
        x = x.clamp(self.eps, 1 - self.eps)
        not_x = 1 - x

        inputs = torch.stack([x, not_x, x, not_x], dim=0)
        inner_exps = torch.stack(
            [self.b.abs(), self.d.abs(), self.f.abs(), self.h.abs()], dim=0
        )
        outer_exps = torch.stack(
            [self.a.abs(), self.c.abs(), self.e.abs(), self.g.abs()], dim=0
        )
        nominator_1, nominator_2, denominator_1, denominator_2 = self.lpnorm(
            inputs, inner_exp=inner_exps, outer_exp=outer_exps, dim=dim
        ).unbind(dim=0)

        nominator = self.alpha * nominator_1 + self.beta * nominator_2
        denominator = self.gamma * denominator_1 + self.delta * denominator_2

        return nominator / denominator

    @staticmethod
    def lpnorm(x, inner_exp, outer_exp, dim):
        x_summed = x.pow(expand_right(inner_exp, x.shape)).sum(dim=dim)
        return x_summed.pow(expand_right(outer_exp, x_summed.shape)).unsqueeze(dim)


def expand_right(tensor: Tensor, shape: Sequence[int]) -> Tensor:
    """Expand a tensor on the right to match a desired shape.

    Args:
        tensor: tensor to be expanded
        shape: target shape

    Returns:
         a tensor with shape matching the target shape.

    Examples:
        >>> tensor = torch.zeros(3,4)
        >>> shape = (3,4,5)
        >>> print(expand_right(tensor, shape).shape)
        torch.Size([3,4,5])

    """
    tensor_expand = tensor
    while tensor_expand.ndimension() < len(shape):
        tensor_expand = tensor_expand.unsqueeze(-1)
    tensor_expand = tensor_expand.expand(shape)
    return tensor_expand


def tanh_squash(loc, low, high):
    tanh_loc = torch.nn.functional.tanh(loc)
    scale = (high - low) / 2
    add = (high + low) / 2
    return tanh_loc * scale + add


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.n_agents = kwargs.get("n_agents", 2)

        self.resource_growth_rate = torch.nn.Parameter(
            torch.tensor(
                [kwargs.get("resource_growth_rate", 0.0)],
                device=device,
                dtype=torch.float,
            )
        )

        self.max_resources = torch.nn.Parameter(
            torch.tensor(
                [kwargs.get("max_resources", 20.0)],
                device=device,
                dtype=torch.float,
            )
        )
        self.gen_agg_type = kwargs.get("gen_agg_type", "max")
        self.gen_agg_learn_type = kwargs.get("gen_agg_learn_type", "laf")
        if self.gen_agg_type == "learn":
            if self.gen_agg_learn_type == "softmax":
                self.gen_agg = SoftmaxAggregation(t=1, learn=True).to(device)
            elif self.gen_agg_learn_type == "genagg":
                self.gen_agg = GenAgg(a=0.0, b=0.0).to(device)
            elif self.gen_agg_learn_type == "laf":
                self.gen_agg = LAF().to(device)
            elif self.gen_agg_learn_type == "genagg_square":
                self.gen_agg = GenAgg(f=Square()).to(device)

        elif self.gen_agg_type == "max":
            self.gen_agg = agg_max
        elif self.gen_agg_type == "mean":
            self.gen_agg = agg_mean
        elif self.gen_agg_type == "min":
            self.gen_agg = agg_min
        else:
            raise AssertionError

        self.initial_resources_range_min = kwargs.get("initial_resources_range_min", 10)
        self.initial_resources_range_max = kwargs.get("initial_resources_range_max", 10)

        self.selfishness = kwargs.get("selfishness", 0.0)  # [0,1] 1 is selfish
        self.eating_reward_coeff = kwargs.get("eating_reward_coeff", 1.0)
        self.energy_reward_coeff = kwargs.get("energy_reward_coeff", 0.0)

        # Make world
        world = World(
            batch_dim,
            device,
        )
        # Add agents
        for i in range(self.n_agents):
            # Constraint: all agents have same action range and multiplier
            agent = Agent(
                name=f"agent_{i}",
                collide=False,
                action_size=1,
                u_range=0.5,
            )
            world.add_agent(agent)
            agent.eating_reward = torch.zeros(world.batch_dim, device=world.device)
            agent.energy_reward = torch.zeros(world.batch_dim, device=world.device)
            agent.consumed_resources = torch.zeros(world.batch_dim, device=world.device)

        self.t = 0
        self.consumed_resources = torch.zeros(world.batch_dim, device=world.device)
        self.eating_reward = torch.zeros(world.batch_dim, device=world.device)

        return world

    def parameters(self) -> List:
        if self.gen_agg_type == "learn":
            return self.gen_agg.parameters()
        else:
            return []

    def to_log(self) -> Dict:
        return {}

    def reset_world_at(self, env_index: int = None):
        if env_index is None:
            self.initial_resources = torch.empty(
                (self.world.batch_dim,),
                device=self.world.device,
                dtype=torch.float32,
            ).uniform_(
                self.initial_resources_range_min, self.initial_resources_range_max
            )
            self.current_resources = self.initial_resources.clone()
        else:
            self.initial_resources = TorchUtils.where_from_index(
                env_index,
                torch.empty(
                    (1,),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    self.initial_resources_range_min, self.initial_resources_range_max
                ),
                self.initial_resources,
            )
            self.current_resources = TorchUtils.where_from_index(
                env_index, self.initial_resources, self.current_resources
            )

    def process_action(self, agent: Agent):
        # Actions are integers
        agent.consume_action = (agent.action.u + agent.action.u_range).squeeze(-1)
        agent.action.u = torch.zeros(
            (self.world.batch_dim, agent.dynamics.needed_action_size),
            device=self.world.device,
            dtype=torch.float,
        )

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]
        is_last = agent == self.world.agents[-1]

        if is_first:
            if self.world.batch_dim == 1:
                import time

                self.t += 1
                print("Time", self.t)
                time.sleep(0.1)

            self.consumed_resources = torch.zeros(
                self.world.batch_dim, device=self.world.device
            )

            for a in self.world.agents:
                # You cannot take more than available
                a.consumed_resources = torch.minimum(
                    a.consume_action, self.current_resources
                )
                # If someone already ate a resource then you cannot take it
                # a.consumed_resources = torch.where(
                #     self.consumed_resources > 0, 0, a.consumed_resources
                # )

                self.current_resources = self.current_resources - a.consumed_resources
                self.consumed_resources = self.consumed_resources + a.consumed_resources

                a.energy_reward = (
                    -a.consume_action
                    * self.energy_reward_coeff
                    # * (
                    #     1 - a.consumed_resources
                    # )  # No energy reward for successfully eating
                )
                a.eating_reward = a.consumed_resources * self.eating_reward_coeff

            # Reward for eating
            # self.eating_reward = self.consumed_resources * self.eating_reward_coeff

        if is_last:
            self.evolve_state()

        agent_rewards = torch.stack(
            [a.eating_reward for a in self.world.agents], dim=-1
        )
        if self.world.batch_dim == 1:
            agent_rewards = agent_rewards.repeat(2, 1)
        self.global_reward = self.gen_agg(agent_rewards, dim=-1).squeeze(-1)
        # self.global_reward = tanh_squash(self.global_reward, 0, 1)
        if self.world.batch_dim == 1:
            self.global_reward = self.global_reward[:1]
        return (
            self.global_reward * (1 - self.selfishness)
            + agent.eating_reward * self.selfishness
            + agent.energy_reward
        )

    def evolve_state(self):
        # Resources population growth --> Logistic growth model
        self.current_resources = self.current_resources + (
            self.resource_growth_rate
            * self.current_resources
            * (1 - self.current_resources / self.max_resources)
        )

    def observation(self, agent: Agent):
        return torch.cat(
            [
                # self.current_resources.unsqueeze(-1),
                torch.zeros_like(self.current_resources.unsqueeze(-1))
            ],
            dim=-1,
        )

    def info(self, agent: Agent) -> AGENT_INFO_TYPE:
        return {
            "current_resources": self.current_resources,
            "total_consumed_resources": self.consumed_resources,
            "agent_consumed_resources": agent.consumed_resources,
            "agent_eating_reward": agent.eating_reward,
            "agent_energy_reward": agent.energy_reward,
        }


if __name__ == "__main__":
    render_interactively(__file__, control_two_agents=True)
