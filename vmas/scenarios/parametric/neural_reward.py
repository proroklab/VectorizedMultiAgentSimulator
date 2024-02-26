#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import typing
from typing import Dict, List

import torch
from torch.nn import Sequential, Tanh, Linear
from vmas import render_interactively
from vmas.simulator.core import Agent, World, Sphere, Landmark
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import ScenarioUtils

if typing.TYPE_CHECKING:
    pass


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.plot_grid = True
        self.n_agents = kwargs.get("n_agents", 2)
        self.collisions = kwargs.get("collisions", False)

        self.lidar_range = kwargs.get("lidar_range", 0.35)
        self.agent_radius = kwargs.get("agent_radius", 0.1)

        self.min_distance_between_entities = self.agent_radius * 2 + 0.05
        self.world_semidim = 1
        self.n_lidar_rays = 12

        self.reward_function = Sequential(
            Linear(6, 256),
            Tanh(),
            Linear(256, 256),
            Tanh(),
            Linear(256, 1),
        ).to(device)

        # Make world
        world = World(batch_dim, device, substeps=2)

        known_colors = [
            (0.22, 0.49, 0.72),
            (1.00, 0.50, 0),
            (0.30, 0.69, 0.29),
            (0.97, 0.51, 0.75),
            (0.60, 0.31, 0.64),
            (0.89, 0.10, 0.11),
            (0.87, 0.87, 0),
        ]
        colors = torch.randn(
            (max(self.n_agents - len(known_colors), 0), 3), device=device
        )

        # Add agents
        for i in range(self.n_agents):
            color = (
                known_colors[i]
                if i < len(known_colors)
                else colors[i - len(known_colors)]
            )

            # Constraint: all agents have same action range and multiplier
            agent = Agent(
                name=f"agent_{i}",
                collide=self.collisions,
                color=color,
                shape=Sphere(radius=self.agent_radius),
                render_action=True,
                sensors=[
                    Lidar(
                        world,
                        n_rays=self.n_lidar_rays,
                        max_range=self.lidar_range,
                    ),
                ]
                if self.collisions
                else None,
            )
            world.add_agent(agent)

            # Add goals
            goal = Landmark(
                name=f"goal_{i}",
                collide=False,
            )
            world.add_landmark(goal)

        return world

    def parameters(self) -> List:
        return self.reward_function.parameters()

    def to_log(self) -> Dict:
        return {}

    def reset_world_at(self, env_index: int = None):
        ScenarioUtils.spawn_entities_randomly(
            self.world.agents + self.world.landmarks,
            self.world,
            env_index,
            self.min_distance_between_entities,
            (-self.world_semidim, self.world_semidim),
            (-self.world_semidim, self.world_semidim),
        )

    def reward(self, agent: Agent):
        return self.reward_function(self.observation(agent))

    def observation(self, agent: Agent):
        obs = self.obs_from_pos(agent.state.pos)

        return obs

    def obs_from_pos(self, pos, env_index=None):
        rel_pos_to_landmarks = []
        for l in self.world.landmarks:
            rel_pos_to_landmarks.append(pos - l.state.pos)

        return torch.cat(
            [pos] + rel_pos_to_landmarks,
            dim=-1,
        )


if __name__ == "__main__":
    render_interactively(
        __file__,
        control_two_agents=True,
    )
