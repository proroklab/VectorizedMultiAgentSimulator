#  Copyright (c) 2022.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import torch
from vmas import render_interactively

from vmas.simulator.core import Agent, Landmark, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        num_agents = kwargs.get("n_agents", 3)
        obs_agents = kwargs.get("obs_agents", True)

        self.obs_agents = obs_agents

        world = World(batch_dim=batch_dim, device=device)
        # set any world properties first
        num_landmarks = num_agents
        # Add agents
        for i in range(num_agents):
            agent = Agent(
                name=f"agent {i}",
                collide=True,
                shape=Sphere(radius=0.15),
                color=Color.BLUE,
            )
            world.add_agent(agent)
        # Add landmarks
        for i in range(num_landmarks):
            landmark = Landmark(
                name=f"landmark {i}",
                collide=False,
                color=Color.BLACK,
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
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = torch.zeros(
            self.world.batch_dim, device=self.world.device, dtype=torch.float32
        )

        for landmark in self.world.landmarks:
            closest = torch.min(
                torch.stack(
                    [
                        torch.linalg.vector_norm(
                            a.state.pos - landmark.state.pos, dim=1
                        )
                        for a in self.world.agents
                    ],
                    dim=1,
                ),
                dim=-1,
            )[0]
            rew -= closest

        if agent.collide:
            for a in self.world.agents:
                if a != agent:
                    rew[self.world.is_overlapping(a, agent)] -= 1

        return rew

    def observation(self, agent: Agent):
        # get positions of all landmarks in this agent's reference frame
        landmark_pos = []
        for landmark in self.world.landmarks:  # world.entities:
            landmark_pos.append(landmark.state.pos - agent.state.pos)
        # distance to all other agents
        other_pos = []
        for other in self.world.agents:
            if other != agent:
                other_pos.append(other.state.pos - agent.state.pos)
        return torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
                *landmark_pos,
                *(other_pos if self.obs_agents else []),
            ],
            dim=-1,
        )


if __name__ == "__main__":
    render_interactively(__file__, control_two_agents=True)
