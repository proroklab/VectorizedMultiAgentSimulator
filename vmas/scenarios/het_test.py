#  Copyright (c) 2022.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
from typing import Dict

import torch
from torch import Tensor
from vmas import render_interactively
from vmas.simulator.core import Agent, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, X, Y


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.green_mass = kwargs.get("green_mass", 5)

        # Make world
        world = World(batch_dim, device)
        # Add agents
        agent = Agent(
            name=f"agent 0",
            collide=False,
            color=Color.GREEN,
            render_action=True,
            mass=self.green_mass,
        )
        world.add_agent(agent)
        agent = Agent(name=f"agent 1", collide=False, mass=1, render_action=True)
        world.add_agent(agent)

        return world

    def reset_world_at(self, env_index: int = None):
        for agent in self.world.agents:
            agent.set_pos(
                torch.zeros(
                    (1, self.world.dim_p)
                    if env_index is not None
                    else (self.world.batch_dim, self.world.dim_p),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(-1, 1),
                batch_index=env_index,
            )

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            self.energy_rew_1 = (self.world.agents[0].state.vel[:, X] - 0).abs()
            self.energy_rew_1 += (self.world.agents[0].state.vel[:, Y] - 0).abs()

            self.energy_rew_2 = (self.world.agents[1].state.vel[:, X] - 0).abs()
            self.energy_rew_2 += (self.world.agents[1].state.vel[:, Y] - 0).abs()

        return -self.energy_rew_1 + self.energy_rew_2

    def observation(self, agent: Agent):
        return torch.cat(
            [agent.state.pos, agent.state.vel],
            dim=-1,
        )

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        return {
            "energy_rew_1": self.energy_rew_1,
            "energy_rew_2": self.energy_rew_2,
        }


if __name__ == "__main__":
    render_interactively("het_test", control_two_agents=True)
