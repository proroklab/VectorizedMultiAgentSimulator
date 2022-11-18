#  Copyright (c) 2022.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import torch
from vmas import render_interactively

from vmas.simulator.core import Agent, World, Landmark
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):

        # Make world
        world = World(batch_dim, device)
        # Add agents
        for i in range(1):
            agent = Agent(name=f"agent {i}", collide=False, color=Color.GRAY)
            world.add_agent(agent)
        # Add landmarks
        for i in range(1):
            landmark = Landmark(
                name=f"landmark {i}",
                collide=False,
                color=Color.RED,
            )
            world.add_landmark(landmark)

        return world

    def reset_world_at(self, env_index: int = None):
        for agent in self.world.agents:
            agent.set_pos(
                2
                * torch.rand(
                    self.world.dim_p, device=self.world.device, dtype=torch.float32
                )
                - 1,
                batch_index=env_index,
            )
        for landmark in self.world.landmarks:
            landmark.set_pos(
                2
                * torch.rand(
                    self.world.dim_p, device=self.world.device, dtype=torch.float32
                )
                - 1,
                batch_index=env_index,
            )

    def reward(self, agent: Agent):
        dist2 = torch.sum(
            torch.square(agent.state.pos - self.world.landmarks[0].state.pos), dim=-1
        )
        return -dist2

    def observation(self, agent: Agent):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in self.world.landmarks:
            entity_pos.append(entity.state.pos - agent.state.pos)
        return torch.cat([agent.state.vel, *entity_pos], dim=-1)


if __name__ == "__main__":
    render_interactively(__file__)
