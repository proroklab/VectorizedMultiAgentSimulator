#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import torch

from vmas import render_interactively
from vmas.simulator.core import Agent, Landmark, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        num_agents = kwargs.pop("n_agents", 3)
        obs_agents = kwargs.pop("obs_agents", True)
        ScenarioUtils.check_kwargs_consumed(kwargs)

        self.obs_agents = obs_agents

        world = World(batch_dim=batch_dim, device=device)
        # set any world properties first
        num_landmarks = num_agents
        # Add agents
        for i in range(num_agents):
            agent = Agent(
                name=f"agent_{i}",
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
                torch.zeros(
                    (
                        (1, self.world.dim_p)
                        if env_index is not None
                        else (self.world.batch_dim, self.world.dim_p)
                    ),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -1.0,
                    1.0,
                ),
                batch_index=env_index,
            )

        for landmark in self.world.landmarks:
            landmark.set_pos(
                torch.zeros(
                    (
                        (1, self.world.dim_p)
                        if env_index is not None
                        else (self.world.batch_dim, self.world.dim_p)
                    ),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -1.0,
                    1.0,
                ),
                batch_index=env_index,
            )

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]
        if is_first:
            # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
            self.rew = torch.zeros(
                self.world.batch_dim,
                device=self.world.device,
                dtype=torch.float32,
            )
            for single_agent in self.world.agents:
                for landmark in self.world.landmarks:
                    closest = torch.min(
                        torch.stack(
                            [
                                torch.linalg.vector_norm(
                                    a.state.pos - landmark.state.pos, dim=1
                                )
                                for a in self.world.agents
                            ],
                            dim=-1,
                        ),
                        dim=-1,
                    )[0]
                    self.rew -= closest

                if single_agent.collide:
                    for a in self.world.agents:
                        if a != single_agent:
                            self.rew[self.world.is_overlapping(a, single_agent)] -= 1

        return self.rew

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
