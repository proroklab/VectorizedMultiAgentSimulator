#  Copyright (c) 2022. Matteo Bettini
#  All rights reserved.

import torch

from maps import render_interactively
from maps.simulator.core import Agent, World, Landmark, Sphere, Box, Line, Lidar
from maps.simulator.scenario import BaseScenario
from maps.simulator.utils import Color


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        n_agents = 1

        # Make world
        world = World(batch_dim, device, dt=0.1, damping=0.25)
        # Add agents
        for i in range(n_agents):
            agent = Agent(
                name=f"agent {i}", shape=Sphere(radius=0.04), u_multiplier=0.8, sensors=[Lidar(world)]
            )
            world.add_agent(agent)
        # Add landmarks
        for i in range(1):
            landmark = Landmark(
                name=f"landmark {i}",
                collide=True,
                movable=True,
                rotatable=False,
                shape=Box(length=0.3, width=0.1),
                color=Color.RED,
            )
            world.add_landmark(landmark)

        landmark_circ = Landmark(
            name=f"landmark {i}",
            collide=True,
            movable=True,
            rotatable=False,
            shape=Sphere(radius=0.2),
            color=Color.RED,
        )
        world.add_landmark(landmark_circ)


        return world

    def reset_world_at(self, env_index: int = None):
        for i, agent in enumerate(self.world.agents):
            agent.set_pos(
                torch.tensor(
                    [-0.2 + 0.1 * i, 1.0],
                    dtype=torch.float32,
                    device=self.world.device,
                ),
                batch_index=env_index,
            )
        self.world.landmarks[0].set_pos(
            torch.tensor(
                [0.0, 0.5],
                dtype=torch.float32,
                device=self.world.device,
            ),
            batch_index=env_index,
        )
        self.world.landmarks[0].set_rot(
            torch.tensor(
                [1.0],
                dtype=torch.float32,
                device=self.world.device,
            ),
            batch_index=env_index,
        )

    def reward(self, agent: Agent):
        dist2 = torch.linalg.vector_norm(
            agent.state.pos, dim=1
        )
        return -dist2

    def observation(self, agent: Agent):
        # get positions of all entities in this agent's reference frame
        return agent.state.pos


if __name__ == "__main__":
    render_interactively("test", n_agents=1)
