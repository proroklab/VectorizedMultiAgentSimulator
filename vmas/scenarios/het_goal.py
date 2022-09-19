#  Copyright (c) 2022.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import math
from typing import Dict

import torch
from torch import Tensor

from vmas import render_interactively
from vmas.simulator.core import Agent, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, Y
from vmas.simulator.velocity_controller import VelocityController


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.green_mass = kwargs.get("green_mass", 5)
        self.plot_grid = True

        self.agent_radius = 0.175

        controller_params = [4, 1.25, 0.001]

        # Make world
        world = World(batch_dim, device, linear_friction=0.1, drag=0)
        # Add agents
        agent = Agent(
            name=f"agent 0",
            collide=False,
            color=Color.GREEN,
            render_action=True,
            mass=self.green_mass,
            a_range=1,
            u_range=1,
        )
        agent.controller = VelocityController(
            agent, world, controller_params, "standard"
        )
        world.add_agent(agent)
        agent = Agent(
            name=f"agent 1", collide=False, render_action=True, a_range=1, u_range=1
        )
        agent.controller = VelocityController(
            agent, world, controller_params, "standard"
        )
        world.add_agent(agent)
        #
        # self.goal = Landmark(
        #     "goal",
        #     collide=True,
        #     shape=Sphere(radius=self.agent_radius),
        #     drag=0.25,
        #     movable=True,
        # )
        # world.add_landmark(self.goal)

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
                ).uniform_(0, 0),
                batch_index=env_index,
            )

    def process_action(self, agent: Agent):
        agent.action.u[:, Y] = 0
        agent.controller.process_force()

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:

            self.energy_expenditure = (
                -torch.stack(
                    [
                        torch.linalg.vector_norm(a.action.u, dim=-1)
                        / math.sqrt(self.world.dim_p * (a.a_range**2))
                        for a in self.world.agents
                    ],
                    dim=1,
                ).sum(-1)
                * 3
            )

        return self.energy_expenditure

    def observation(self, agent: Agent):
        return torch.cat(
            [agent.state.pos, agent.state.vel],
            dim=-1,
        )

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        return {
            "energy_expenditure": self.energy_expenditure,
        }


if __name__ == "__main__":
    render_interactively(__file__, control_two_agents=False)
