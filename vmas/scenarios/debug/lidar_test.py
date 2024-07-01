#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import typing
from typing import List

import torch

from vmas import render_interactively
from vmas.simulator.core import Agent, Box, Landmark, Line, Sphere, World
from vmas.simulator.dynamics.diff_drive import DiffDrive
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
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

        agent = Agent(
            name="diff_drive",
            collide=True,
            render_action=True,
            u_range=[1, 1],
            u_multiplier=[1, 1],
            dynamics=DiffDrive(world, integration="rk4"),
            sensors=[
                Lidar(
                    world=world,
                    n_rays=100,
                    max_range=0.5,
                    render_color=Color.GRAY,
                    alpha=0.25,
                )
            ],
        )
        world.add_agent(agent)

        line = Line(0.5)
        sphere = Sphere(0.2)
        box = Box(0.2, 0.2)
        for i, shp in enumerate([line, sphere, box]):
            world.add_landmark(
                Landmark(
                    name=f"o_{i}",
                    shape=shp,
                    collision_filter=lambda e: e.movable,
                )
            )

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
        ScenarioUtils.spawn_entities_randomly(
            self.world.landmarks,
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
        agent.sensors[0].measure()
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
    render_interactively(__file__, control_two_agents=False)
