#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import torch

from vmas import render_interactively

from vmas.simulator.core import Agent, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):

        self.n_agents = 4
        self.agent_radius = 0.16

        # Make world
        world = World(batch_dim, device, x_semidim=0, y_semidim=0)

        self.colors = [Color.GREEN, Color.BLUE, Color.RED, Color.GRAY]

        # Add agents
        for i in range(self.n_agents):
            agent = Agent(
                name=f"agent_{i}",
                rotatable=False,
                shape=Sphere(radius=self.agent_radius),
                render_action=True,
                color=self.colors[i],
                collide=False,
            )
            world.add_agent(agent)

        return world

    def reset_world_at(self, env_index: int = None):

        ScenarioUtils.spawn_entities_randomly(
            self.world.agents,
            self.world,
            env_index,
            min_dist_between_entities=0,
            x_bounds=(
                0,
                0,
            ),
            y_bounds=(
                0,
                0,
            ),
        )

    def reward(self, agent: Agent):
        return torch.zeros(
            self.world.batch_dim, device=self.world.device, dtype=torch.float32
        )

    def observation(self, agent: Agent):
        return torch.zeros(
            self.world.batch_dim, 1, device=self.world.device, dtype=torch.float32
        )


if __name__ == "__main__":
    render_interactively(__file__, control_two_agents=True)
