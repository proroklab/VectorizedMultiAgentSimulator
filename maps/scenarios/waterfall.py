#  Copyright (c) 2022. Matteo Bettini
#  All rights reserved.

import torch

from maps.simulator.core import Agent, World, Landmark, Sphere, Box, Line
from maps.simulator.scenario import BaseScenario
from maps.simulator.utils import Color


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        n_agents = kwargs.get("n_agents", 5)

        # Make world
        world = World(batch_dim, device, dt=0.1, damping=0.25)
        # Add agents
        for i in range(n_agents):
            agent = Agent(name=f"agent {i}", shape=Sphere(radius=0.04))
            world.add_agent(agent)
        # Add landmarks
        for i in range(5):
            landmark = Landmark(
                name=f"landmark {i}",
                collide=True,
                movable=True,
                shape=Box(length=0.3, width=0.1),
                color=Color.RED,
            )
            world.add_landmark(landmark)
        floor = Landmark(
            name="floor",
            collide=True,
            movable=False,
            shape=Line(length=2),
            color=Color.BLACK,
        )
        world.add_landmark(floor)

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
        for i, landmark in enumerate(self.world.landmarks[:-1]):
            landmark.set_pos(
                torch.tensor(
                    [0.2 if i % 2 else -0.2, 0.6 - 0.3 * i],
                    dtype=torch.float32,
                    device=self.world.device,
                ),
                batch_index=env_index,
            )
            landmark.set_rot(
                torch.tensor(
                    [torch.pi / 4 if i % 2 else -torch.pi / 4],
                    dtype=torch.float32,
                    device=self.world.device,
                ),
                batch_index=env_index,
            )
        floor = self.world.landmarks[-1]
        floor.set_pos(
            torch.tensor(
                [0, -1],
                dtype=torch.float32,
                device=self.world.device,
            ),
            batch_index=env_index,
        )

    def reward(self, agent: Agent):
        dist2 = torch.linalg.vector_norm(
            agent.state.pos - self.world.landmarks[-1].state.pos, dim=1
        )
        return -dist2

    def observation(self, agent: Agent):
        # get positions of all entities in this agent's reference frame
        return torch.cat(
            [agent.state.vel]
            + [
                landmark.state.pos - agent.state.pos
                for landmark in self.world.landmarks
            ],
            dim=-1,
        )
