#  Copyright (c) 2022-2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import typing
from typing import List

import torch

from vmas import render_interactively
from vmas.simulator.core import Agent, World
from vmas.simulator.dynamics.diff_drive import DiffDriveDynamics
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.plot_grid = True

        # Make world
        world = World(batch_dim, device, substeps=10)

        agent = Agent(
            name="agent 0",
            collide=True,
            color=Color.GREEN,
            render_action=True,
            u_range=1,
            u_rot_range=1,
            u_rot_multiplier=0.001,
        )
        agent.dynamics = DiffDriveDynamics(agent, world, integration="rk4")

        world.add_agent(agent)

        return world

    def reset_world_at(self, env_index: int = None):
        pass

    def process_action(self, agent: Agent):
        for agent in self.world.agents:
            agent.dynamics.process_force()

    def reward(self, agent: Agent):
        return torch.zeros(self.world.batch_dim)

    def observation(self, agent: Agent):
        observations = [
            agent.state.pos,
            agent.state.vel,
        ]
        return torch.cat(
            observations,
            dim=-1,
        )

    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        from vmas.simulator import rendering

        geoms: List[Geom] = []

        # Agent rotation
        color = Color.BLACK.value
        line = rendering.Line(
            (0, 0),
            (0.1, 0),
            width=1,
        )
        xform = rendering.Transform()
        xform.set_rotation(self.world.agents[0].state.rot[env_index])
        xform.set_translation(*self.world.agents[0].state.pos[env_index])
        line.add_attr(xform)
        line.set_color(*color)
        geoms.append(line)

        return geoms


if __name__ == "__main__":
    render_interactively(__file__)
