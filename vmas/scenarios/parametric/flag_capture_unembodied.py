#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
from typing import Dict, List

import torch

from torch_geometric.nn import SoftmaxAggregation

from vmas import render_interactively
from vmas.simulator.core import Agent, World
from vmas.simulator.dynamics.static import Static
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import ScenarioUtils


def agg_max(x, dim):
    return x.max(dim=dim, keepdim=True)[0]


def agg_min(x, dim):
    return x.min(dim=dim, keepdim=True)[0]


def agg_mean(x, dim):
    return x.mean(dim=dim, keepdim=True)


def agg_sum(x, dim):
    return x.sum(dim=dim, keepdim=True)


class Square:
    def forward(self, x):
        return (x + 1e-7) ** 2

    def inverse(self, x):
        return (x.abs() + 1e-7).sqrt()


def get_aggregation_function(name, device):
    if name == "softmax":
        return SoftmaxAggregation(t=0, learn=True).to(device)
    elif name == "max":
        return agg_max
    elif name == "mean":
        return agg_mean
    elif name == "min":
        return agg_min
    elif name == "sum":
        return agg_sum
    else:
        raise AssertionError


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.n_agents = kwargs.pop("n_agents", 2)
        self.task_rewards = kwargs.pop("task_rewards", [1.0, 1.0])
        self.n_tasks = len(self.task_rewards)

        self.gen_agg_type_task = kwargs.pop("gen_agg_type_task", "softmax")
        self.gen_agg_type_agent = kwargs.pop("gen_agg_type_agent", "softmax")

        self.task_agg = get_aggregation_function(self.gen_agg_type_task, device)
        self.agent_agg = get_aggregation_function(self.gen_agg_type_agent, device)

        ScenarioUtils.check_kwargs_consumed(kwargs)

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
                discrete_action_nvec=[self.n_tasks + 1],
                u_range=self.n_tasks / 2,
                dynamics=Static(),
            )
            world.add_agent(agent)

        return world

    def parameters(self) -> List:
        params = []
        if hasattr(self.task_agg, "parameters"):
            params += self.task_agg.parameters()
        if hasattr(self.agent_agg, "parameters"):
            params += self.agent_agg.parameters()
        return params

    def to_log(self) -> Dict:

        result = {}

        if hasattr(self.task_agg, "t"):
            result["task_agg_t"] = self.task_agg.t.item()
        if hasattr(self.agent_agg, "t"):
            result["agent_agg_t"] = self.agent_agg.t.item()

        return result

    def reset_world_at(self, env_index: int = None):
        pass

    def process_action(self, agent: Agent):
        # Actions are integers
        agent.discrete_action = (
            (agent.action.u + agent.action.u_range).squeeze(-1).to(torch.int)
        )
        agent.action.u = torch.zeros(
            (self.world.batch_dim, agent.dynamics.needed_action_size),
            device=self.world.device,
            dtype=torch.float,
        )

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            reward_matrix = torch.zeros(
                self.world.batch_dim,
                self.n_agents,
                self.n_tasks,
                device=self.world.device,
            )
            for i, a in enumerate(self.world.agents):
                for j in range(self.n_tasks):
                    reward_matrix[:, i, j] = self.task_rewards[j] * (
                        a.discrete_action == j
                    )
            task_matrix = self.agent_agg(reward_matrix, dim=-2).squeeze(-2)
            self.rew = self.task_agg(task_matrix, dim=-1).squeeze(-1)

        return self.rew

    def observation(self, agent: Agent):
        return torch.zeros(
            self.world.batch_dim, 1, dtype=torch.float, device=self.world.device
        )


if __name__ == "__main__":
    render_interactively(__file__, control_two_agents=True)
