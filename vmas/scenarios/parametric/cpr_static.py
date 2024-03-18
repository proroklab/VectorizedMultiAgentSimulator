#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
from typing import Dict, List

import torch

from vmas import render_interactively
from vmas.simulator.core import Agent, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import AGENT_INFO_TYPE, TorchUtils


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.n_agents = kwargs.get("n_agents", 2)

        self.resource_growth_rate = torch.nn.Parameter(
            torch.tensor(
                [kwargs.get("resource_growth_rate", 0.01)],
                device=device,
                dtype=torch.float,
            )
        )

        self.max_resources = torch.nn.Parameter(
            torch.tensor(
                [kwargs.get("max_resources", 10.0)],
                device=device,
                dtype=torch.float,
            )
        )

        self.initial_resources = kwargs.get(
            "initial_resources", self.max_resources.item()
        )

        self.selfishness = torch.nn.Parameter(
            torch.tensor(
                [kwargs.get("selfishness", 0.0)],
                device=device,
                dtype=torch.float,
            )
        )  # [0,1] 1 is selfish

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
                u_range=self.max_resources.item() / (self.n_agents * 2),
            )
            world.add_agent(agent)
            agent.eating_reward = torch.zeros(world.batch_dim, device=world.device)
            agent.consume_action = torch.zeros(world.batch_dim, device=world.device)

        self.t = 0
        self.consumed_resources = torch.zeros(world.batch_dim, device=world.device)
        self.eating_reward = torch.zeros(world.batch_dim, device=world.device)

        return world

    def parameters(self) -> List:
        return [self.selfishness]

    def to_log(self) -> Dict:
        return {}

    def reset_world_at(self, env_index: int = None):
        if env_index is None:
            self.current_resources = torch.full(
                (self.world.batch_dim,),
                self.initial_resources,
                device=self.world.device,
                dtype=torch.float,
            )

        else:
            self.current_resources = TorchUtils.where_from_index(
                env_index, self.initial_resources, self.current_resources
            )

    def process_action(self, agent: Agent):
        agent.consume_action = agent.action.u.squeeze(-1) + agent.action.u_range

        assert (agent.consume_action >= 0).all() and (
            agent.consume_action <= self.max_resources / self.n_agents
        ).all()

        agent.action.u = torch.zeros(
            (self.world.batch_dim, agent.dynamics.needed_action_size),
            device=self.world.device,
            dtype=torch.float,
        )

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]
        # is_last = agent == self.world.agents[-1]

        if is_first:
            if self.world.batch_dim == 1:
                import time

                self.t += 1
                print("Time", self.t)
                time.sleep(0.1)

            self.consumed_resources = torch.stack(
                [a.consume_action for a in self.world.agents], dim=-1
            ).sum(-1)
            for a in self.world.agents:
                a.eating_reward = a.consume_action - a.consume_action * (
                    self.consumed_resources / self.max_resources
                )
            self.eating_reward = torch.stack(
                [a.eating_reward for a in self.world.agents], dim=-1
            ).mean(-1)

        # if is_last:
        #     self.evolve_state()

        return (
            self.eating_reward * (1 - self.selfishness)
            + self.selfishness * agent.eating_reward
        )

    def evolve_state(self):
        # Remove consumed resources
        self.current_resources = (
            self.current_resources - self.consumed_resources
        ).clamp(min=1e-3)

        # Resources population growth --> Logistic growth model
        self.current_resources = self.current_resources + (
            self.resource_growth_rate
            * self.current_resources
            * (1 - self.current_resources / self.max_resources)
        )

    def observation(self, agent: Agent):
        return torch.cat(
            [
                self.current_resources.unsqueeze(-1),
            ],
            dim=-1,
        )

    def info(self, agent: Agent) -> AGENT_INFO_TYPE:
        return {
            "current_resources": self.current_resources,
            "total_eating_reward": self.eating_reward,
            "total_consumed_resources": self.consumed_resources,
            "agent_consumed_resources": agent.consume_action,
            "agent_eating_reward": agent.eating_reward,
        }


if __name__ == "__main__":
    render_interactively(__file__, control_two_agents=True)
