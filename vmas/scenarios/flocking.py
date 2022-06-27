#  Copyright (c) 2022. Jan Blumenkamp
#  All rights reserved.
import math
from typing import Dict

import torch
from torch import Tensor

from vmas import render_interactively
from vmas.simulator.core import Agent, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color

DEFAULT_ENERGY_COEFF = 0.02


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        n_agents = kwargs.get("n_agents", 4)
        self.energy_coeff = kwargs.get(
            "energy_coeff", DEFAULT_ENERGY_COEFF
        )  # Weight of team energy penalty

        # Make world
        world = World(batch_dim, device)
        # Add agents
        for i in range(n_agents):
            # Constraint: all agents have same action range and multiplier
            agent = Agent(name=f"agent {i}", collide=True)
            world.add_agent(agent)

        return world

    def reset_world_at(self, env_index: int = None):
        for agent in self.world.agents:
            # Random pos between -1 and 1
            agent.set_pos(
                torch.zeros(
                    (1, self.world.dim_p)
                    if env_index is not None
                    else (self.world.batch_dim, self.world.dim_p),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -1.0,
                    1.0,
                ),
                batch_index=env_index,
            )

    def reward(self, agent: Agent):
        self.pos_rew = torch.zeros(self.world.batch_dim, device=self.world.device)
        rew = self.pos_rew

        return rew

    def observation(self, agent: Agent):
        return torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
            ],
            dim=-1,
        )

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        try:
            info = {"pos_rew": self.pos_rew}
        # When reset is called before reward()
        except AttributeError:
            info = {}
        return info

    def done(self):
        return torch.zeros(self.world.batch_dim, device=self.world.device)


if __name__ == "__main__":
    render_interactively("flocking", n_agents=10, energy_coeff=DEFAULT_ENERGY_COEFF)
