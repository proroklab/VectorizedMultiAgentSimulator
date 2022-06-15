#  Copyright (c) 2022. Matteo Bettini
#  All rights reserved.

import torch

from maps.simulator.core import Agent, Landmark, World, Line, Sphere
from maps.simulator.scenario import BaseScenario
from maps.simulator.utils import Color


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        n_agents = kwargs.get("n_agents", 4)
        line_length = kwargs.get("line_length", 2)
        line_mass = kwargs.get("line_mass", 30)

        # Make world
        world = World(batch_dim, device)
        # Add agents
        for i in range(n_agents):
            # Constraint: all agents have same action range and multiplier
            agent = Agent(name=f"agent {i}", u_multiplier=0.9)
            world.add_agent(agent)
        # Add landmarks
        line = Landmark(
            name="line",
            collide=True,
            rotatable=True,
            shape=Line(length=line_length),
            mass=line_mass,
            color=Color.BLACK,
        )
        world.add_landmark(line)
        center = Landmark(
            name="center",
            shape=Sphere(radius=0.02),
            collide=False,
            color=Color.BLACK,
        )
        world.add_landmark(center)

        return world

    def reset_world_at(self, env_index: int = None):
        for agent in self.world.agents:
            # Random pos between -1 and 1
            agent.set_pos(
                2
                * torch.rand(
                    self.world.dim_p
                    if env_index is not None
                    else (self.world.batch_dim, self.world.dim_p),
                    device=self.world.device,
                    dtype=torch.float32,
                )
                - 1,
                batch_index=env_index,
            )

        self.world.landmarks[0].set_rot(
            -torch.pi
            * torch.rand(
                1 if env_index is not None else (self.world.batch_dim, 1),
                device=self.world.device,
                dtype=torch.float32,
            )
            + torch.pi / 2,
            batch_index=env_index,
        )

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            self.rew = self.world.landmarks[0].state.ang_vel.abs()

        return self.rew

    def observation(self, agent: Agent):
        return torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
                self.world.landmarks[0].state.ang_vel,
            ],
            dim=-1,
        )
