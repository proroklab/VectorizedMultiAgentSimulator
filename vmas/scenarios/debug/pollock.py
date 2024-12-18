#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import torch

from vmas import render_interactively
from vmas.simulator.core import Agent, Box, Landmark, Line, Sphere, World
from vmas.simulator.scenario import BaseScenario

from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import Color, ScenarioUtils


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.n_agents = kwargs.pop("n_agents", 15)
        self.n_lines = kwargs.pop("n_lines", 15)
        self.n_boxes = kwargs.pop("n_boxes", 15)
        self.lidar = kwargs.pop("lidar", False)
        self.vectorized_lidar = kwargs.pop("vectorized_lidar", True)
        ScenarioUtils.check_kwargs_consumed(kwargs)

        self.agent_radius = 0.05
        self.line_length = 0.3
        self.box_length = 0.2
        self.box_width = 0.1

        self.world_semidim = 1
        self.min_dist_between_entities = 0.1

        # Make world
        world = World(
            batch_dim,
            device,
            dt=0.1,
            drag=0.25,
            substeps=5,
            collision_force=500,
            x_semidim=self.world_semidim,
            y_semidim=self.world_semidim,
        )
        # Add agents
        for i in range(self.n_agents):
            agent = Agent(
                name=f"agent_{i}",
                shape=Sphere(radius=self.agent_radius),
                u_multiplier=0.7,
                rotatable=True,
                sensors=[Lidar(world, n_rays=16, max_range=0.5)] if self.lidar else [],
            )
            world.add_agent(agent)

        # Add lines
        for i in range(self.n_lines):
            landmark = Landmark(
                name=f"line {i}",
                collide=True,
                movable=True,
                rotatable=True,
                shape=Line(length=self.line_length),
                color=Color.BLACK,
            )
            world.add_landmark(landmark)
        for i in range(self.n_boxes):
            landmark = Landmark(
                name=f"box {i}",
                collide=True,
                movable=True,
                rotatable=True,
                shape=Box(length=self.box_length, width=self.box_width),
                color=Color.RED,
            )
            world.add_landmark(landmark)

        return world

    def reset_world_at(self, env_index: int = None):
        # Some things may be spawn on top of each other
        ScenarioUtils.spawn_entities_randomly(
            self.world.agents + self.world.landmarks,
            self.world,
            env_index,
            self.min_dist_between_entities,
            (-self.world_semidim, self.world_semidim),
            (-self.world_semidim, self.world_semidim),
        )

    def reward(self, agent: Agent):
        return torch.zeros(self.world.batch_dim, device=self.world.device)

    def observation(self, agent: Agent):
        return (
            torch.zeros(self.world.batch_dim, 1, device=self.world.device)
            if not self.lidar
            else agent.sensors[0].measure(vectorized=self.vectorized_lidar)
        )


if __name__ == "__main__":
    render_interactively(
        __file__,
        control_two_agents=True,
    )
