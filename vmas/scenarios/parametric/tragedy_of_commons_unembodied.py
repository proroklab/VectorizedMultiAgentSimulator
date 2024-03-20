#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
from typing import Dict, List

import torch
from genagg import GenAgg

from vmas import render_interactively
from vmas.simulator.core import Agent, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import AGENT_INFO_TYPE, TorchUtils


def agg_max(x, dim):
    return x.max(dim=dim, keepdim=True)[0]


def agg_mean(x, dim):
    return x.mean(dim=dim, keepdim=True)


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
                [kwargs.get("max_resources", 20.0)],
                device=device,
                dtype=torch.float,
            )
        )
        self.gen_agg_type = kwargs.get("gen_agg_type", None)
        if self.gen_agg_type is None:
            self.gen_agg = GenAgg().to(device)
        elif self.gen_agg_type == "max":
            self.gen_agg = agg_max
        elif self.gen_agg_type == "mean":
            self.gen_agg = agg_mean

        self.initial_resources_range = kwargs.get("initial_resources_range", 10)

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
        return self.gen_agg.parameters()

    def to_log(self) -> Dict:
        return {}

    def reset_world_at(self, env_index: int = None):
        if env_index is None:
            self.initial_resources = torch.empty(
                (self.world.batch_dim,),
                device=self.world.device,
                dtype=torch.float32,
            ).uniform_(1, self.initial_resources_range)
            self.current_resources = self.initial_resources.clone()
        else:
            self.initial_resources = TorchUtils.where_from_index(
                env_index,
                torch.empty(
                    (1,),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(1, self.initial_resources_range),
                self.initial_resources,
            )
            self.current_resources = TorchUtils.where_from_index(
                env_index, self.initial_resources, self.current_resources
            )

    def process_action(self, agent: Agent):
        # Actions are integers
        agent.consume_action = (
            (agent.action.u + agent.action.u_range).round().squeeze(-1)
        )
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
                # You cannot take more than available and you only take integers
                a.consumed_resources = torch.minimum(
                    a.consume_action, self.current_resources
                ).floor()
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
                self.current_resources.unsqueeze(-1),
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
