#  Copyright (c) 2022.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import torch

from vmas.simulator.core import World, Agent, Landmark, Sphere
from vmas.simulator.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):

        world = World(batch_dim=batch_dim, device=device, dim_c=3)
        # set any world properties first
        num_agents = 2
        num_landmarks = 3

        # Add agents
        for i in range(num_agents):
            speaker = True if i == 0 else False
            agent = Agent(
                name=f"agent {i}",
                collide=False,
                movable=False if speaker else True,
                silent=False if speaker else True,
                shape=Sphere(radius=0.075),
            )
            world.add_agent(agent)
        # Add landmarks
        for i in range(num_landmarks):
            landmark = Landmark(
                name=f"landmark {i}", collide=False, shape=Sphere(radius=0.04)
            )
            world.add_landmark(landmark)

        return world

    def reset_world_at(self, env_index: int = None):
        if env_index is None:
            # assign goals to agents
            for agent in self.world.agents:
                agent.goal_a = None
                agent.goal_b = None
            # want listener to go to the goal landmark
            self.world.agents[0].goal_a = self.world.agents[1]
            self.world.agents[0].goal_b = self.world.landmarks[
                torch.randint(0, len(self.world.landmarks), (1,)).item()
            ]
            # random properties for agents
            for i, agent in enumerate(self.world.agents):
                agent.color = torch.tensor(
                    [0.25, 0.25, 0.25], device=self.world.device, dtype=torch.float32
                )
            # random properties for landmarks
            self.world.landmarks[0].color = torch.tensor(
                [0.65, 0.15, 0.15], device=self.world.device, dtype=torch.float32
            )
            self.world.landmarks[1].color = torch.tensor(
                [0.15, 0.65, 0.15], device=self.world.device, dtype=torch.float32
            )
            self.world.landmarks[2].color = torch.tensor(
                [0.15, 0.15, 0.65], device=self.world.device, dtype=torch.float32
            )
            # special colors for goals
            self.world.agents[0].goal_a.color = self.world.agents[
                0
            ].goal_b.color + torch.tensor(
                [0.45, 0.45, 0.45], device=self.world.device, dtype=torch.float32
            )

        # set random initial states
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
        # squared distance from listener to landmark
        a = self.world.agents[0]
        dist2 = torch.sqrt(
            torch.sum(torch.square(a.goal_a.state.pos - a.goal_b.state.pos), dim=-1)
        )
        return -dist2

    def observation(self, agent):
        # goal color
        goal_color = torch.zeros(3, device=self.world.device, dtype=torch.float32)
        if agent.goal_b is not None:
            goal_color = agent.goal_b.color

        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in self.world.landmarks:
            entity_pos.append(entity.state.pos - agent.state.pos)

        # communication of all other agents
        comm = []
        for other in self.world.agents:
            if other is agent or (other.state.c is None):
                continue
            comm.append(other.state.c)

        # speaker
        if not agent.movable:
            return goal_color.repeat(self.world.batch_dim, 1)
        # listener
        if agent.silent:
            return torch.cat([agent.state.vel, *entity_pos, *comm], dim=-1)
