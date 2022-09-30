#  Copyright (c) 2022.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import math
from typing import Dict

import torch
from torch import Tensor

from vmas import render_interactively
from vmas.simulator.core import Agent, World, Landmark
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, X
from vmas.simulator.velocity_controller import VelocityController


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.green_mass = kwargs.get("green_mass", 1)
        self.plot_grid = True

        self.agent_radius = 0.16

        controller_params = [3, 2, 0.002]

        # Make world
        world = World(batch_dim, device, linear_friction=0.5, drag=0)
        # Add agents
        agent = Agent(
            name=f"agent 0",
            collide=True,
            color=Color.GREEN,
            render_action=True,
            mass=self.green_mass,
            f_range=1.5,
            u_range=1,
        )
        agent.controller = VelocityController(
            agent, world, controller_params, "standard"
        )
        world.add_agent(agent)
        agent = Agent(
            name=f"agent 1",
            collide=False,
            render_action=True,
            # f_range=30,
            u_range=1,
        )
        agent.controller = VelocityController(
            agent, world, controller_params, "standard"
        )
        world.add_agent(agent)
        agent = Agent(
            name=f"agent 1",
            collide=False,
            render_action=True,
            f_range=30,
            u_range=1,
        )
        agent.controller = VelocityController(
            agent, world, controller_params, "standard"
        )
        world.add_agent(agent)

        self.landmark = Landmark("landmark 0", collide=False, movable=True)
        world.add_landmark(self.landmark)

        return world

    def reset_world_at(self, env_index: int = None):
        for agent in self.world.agents:
            agent.controller.reset(env_index)
            agent.set_pos(
                torch.cat(
                    [
                        torch.zeros(
                            (1, 1)
                            if env_index is not None
                            else (self.world.batch_dim, 1),
                            device=self.world.device,
                            dtype=torch.float32,
                        ).uniform_(
                            -1,
                            -1,
                        ),
                        torch.zeros(
                            (1, 1)
                            if env_index is not None
                            else (self.world.batch_dim, 1),
                            device=self.world.device,
                            dtype=torch.float32,
                        ).uniform_(
                            0,
                            0,
                        ),
                    ],
                    dim=1,
                ),
                batch_index=env_index,
            )

    def process_action(self, agent: Agent):
        # agent.action.u[:, Y] = 0
        if agent == self.world.agents[1]:
            max_a = 1

            agent.vel_goal = agent.action.u[:, X]
            requested_a = (agent.vel_goal - agent.state.vel[:, X]) / self.world.dt
            achievable_a = torch.clamp(requested_a, -max_a, max_a)
            agent.action.u[:, X] = (achievable_a * self.world.dt) + agent.state.vel[
                :, X
            ]

        agent.controller.process_force()

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:

            self.energy_expenditure = (
                -torch.stack(
                    [
                        torch.linalg.vector_norm(a.action.u, dim=-1)
                        / math.sqrt(self.world.dim_p * 0)
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
