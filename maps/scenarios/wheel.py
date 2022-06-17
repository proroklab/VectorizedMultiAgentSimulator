#  Copyright (c) 2022. Matteo Bettini
#  All rights reserved.

import torch

from maps import render_interactively
from maps.simulator.core import Agent, Landmark, World, Line, Sphere
from maps.simulator.scenario import BaseScenario
from maps.simulator.utils import Color


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        n_agents = kwargs.get("n_agents", 4)
        line_length = kwargs.get("line_length", 2)
        line_mass = kwargs.get("line_mass", 30)
        self.desired_velocity = kwargs.get("desired_velocity", 0.1)

        # Make world
        world = World(batch_dim, device)
        # Add agents
        for i in range(n_agents):
            # Constraint: all agents have same action range and multiplier
            agent = Agent(name=f"agent {i}", u_multiplier=0.7, shape=Sphere(0.03))
            world.add_agent(agent)
        # Add landmarks
        self.line = Landmark(
            name="line",
            collide=True,
            rotatable=True,
            shape=Line(length=line_length),
            mass=line_mass,
            color=Color.BLACK,
        )
        world.add_landmark(self.line)
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
                torch.zeros(
                    (1, self.world.dim_p)
                    if env_index is not None
                    else (self.world.batch_dim, self.world.dim_p),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -1.0,
                    1.0,
                ),
                batch_index=env_index,
            )

        self.line.set_rot(
            torch.zeros(
                (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                device=self.world.device,
                dtype=torch.float32,
            ).uniform_(
                -torch.pi / 2,
                torch.pi / 2,
            ),
            batch_index=env_index,
        )

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            self.rew = (self.line.state.ang_vel.abs() - self.desired_velocity).abs()

        return -self.rew

    def observation(self, agent: Agent):
        return torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
                (self.desired_velocity - self.line.state.ang_vel.abs()).abs(),
            ],
            dim=-1,
        )


if __name__ == "__main__":
    render_interactively(
        "wheel", desired_velocity=0.1, n_agents=4, line_length=2, line_mass=30
    )
