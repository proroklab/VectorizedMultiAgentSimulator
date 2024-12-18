#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import typing
from typing import List

import torch

from vmas import render_interactively
from vmas.simulator.core import Agent, World
from vmas.simulator.dynamics.diff_drive import DiffDrive
from vmas.simulator.dynamics.holonomic_with_rot import HolonomicWithRotation
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        """
        Differential drive example scenario
        Run this file to try it out

        The first agent has differential drive dynamics.
        You can control its forward input with the LEFT and RIGHT arrows.
        You can control its rotation with UP and DOWN.

        The second agent has standard vmas holonomic dynamics.
        You can control it with WASD
        You can control its rotation with Q and E.

        """
        # T
        self.plot_grid = True
        self.n_agents = kwargs.pop("n_agents", 2)
        ScenarioUtils.check_kwargs_consumed(kwargs)

        # Make world
        world = World(batch_dim, device, substeps=10)

        for i in range(self.n_agents):
            if i == 0:
                agent = Agent(
                    name=f"diff_drive_{i}",
                    collide=True,
                    render_action=True,
                    u_range=[1, 1],
                    u_multiplier=[1, 1],
                    dynamics=DiffDrive(world, integration="rk4"),
                )
            else:
                agent = Agent(
                    name=f"holo_rot_{i}",
                    collide=True,
                    render_action=True,
                    u_range=[1, 1, 1],
                    u_multiplier=[1, 1, 0.001],
                    dynamics=HolonomicWithRotation(),
                )

            world.add_agent(agent)

        return world

    def reset_world_at(self, env_index: int = None):
        ScenarioUtils.spawn_entities_randomly(
            self.world.agents,
            self.world,
            env_index,
            min_dist_between_entities=0.1,
            x_bounds=(-1, 1),
            y_bounds=(-1, 1),
        )

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
        for agent in self.world.agents:
            color = Color.BLACK.value
            line = rendering.Line(
                (0, 0),
                (0.1, 0),
                width=1,
            )
            xform = rendering.Transform()
            xform.set_rotation(agent.state.rot[env_index])
            xform.set_translation(*agent.state.pos[env_index])
            line.add_attr(xform)
            line.set_color(*color)
            geoms.append(line)

        return geoms


if __name__ == "__main__":
    render_interactively(__file__, control_two_agents=True)
