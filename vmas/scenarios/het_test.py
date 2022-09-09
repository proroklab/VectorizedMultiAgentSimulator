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
from vmas.simulator.velocity_controller import VelocityController


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.green_mass = kwargs.get("green_mass", 5)
        self.plot_grid = True

        controller_params = [0.5, 1, 0.003]
        # Make world
        world = World(batch_dim, device, drag=0)
        # Add agents
        agent = Agent(
            name=f"agent 0",
            collide=False,
            color=Color.GREEN,
            render_action=True,
            mass=self.green_mass,
            f_range=2,
        )
        agent.controller = VelocityController(
            agent, world, controller_params, "standard"
        )
        world.add_agent(agent)
        agent = Agent(
            name=f"agent 1", collide=False, mass=1, render_action=True, f_range=2
        )
        agent.controller = VelocityController(
            agent, world, controller_params, "standard"
        )
        world.add_agent(agent)

        return world

    def reset_world_at(self, env_index: int = None):
        for agent in self.world.agents:
            agent.controller.reset(env_index)
            agent.set_pos(
                torch.zeros(
                    (1, self.world.dim_p)
                    if env_index is not None
                    else (self.world.batch_dim, self.world.dim_p),
                    device=self.world.device,
                    dtype=torch.float32,
                ),
                # ).uniform_(-1, 1),
                batch_index=env_index,
            )

    def process_action(self, agent: Agent):
        agent.action.u[:, Y] = 0
        agent.controller.process_force()

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            self.energy_rew_1 = (self.world.agents[0].action.u[:, X] - 0).abs()
            self.energy_rew_1 += (self.world.agents[0].action.u[:, Y] - 0).abs()

            self.energy_rew_2 = (self.world.agents[1].action.u[:, X] - 0).abs()
            self.energy_rew_2 += (self.world.agents[1].action.u[:, Y] - 0).abs()

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
