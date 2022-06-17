#  Copyright (c) 2022. Matteo Bettini
#  All rights reserved.

import torch

from maps import render_interactively
from maps.simulator.core import Agent, Landmark, Sphere, World, Line, Box
from maps.simulator.scenario import BaseScenario
from maps.simulator.utils import Color, LINE_MIN_DIST, Y


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.n_agents = kwargs.get("n_agents", 3)
        self.package_mass = kwargs.get("package_mass", 10)

        self.line_length = 0.8
        self.agent_radius = 0.03

        # Make world
        world = World(batch_dim, device, gravity=(0.0, -0.05), y_semidim=1)
        # Add agents
        for i in range(self.n_agents):
            agent = Agent(
                name=f"agent {i}", shape=Sphere(self.agent_radius), u_multiplier=0.7
            )
            world.add_agent(agent)

        goal = Landmark(
            name="goal",
            collide=False,
            shape=Sphere(),
            color=Color.LIGHT_GREEN,
        )
        world.add_landmark(goal)
        self.package = Landmark(
            name="package",
            collide=True,
            movable=True,
            shape=Sphere(),
            mass=self.package_mass,
            color=Color.RED,
        )
        self.package.goal = goal
        world.add_landmark(self.package)
        # Add landmarks

        self.line = Landmark(
            name="line",
            shape=Line(length=self.line_length),
            collide=True,
            movable=True,
            rotatable=True,
            mass=5,
            color=Color.BLACK,
        )
        world.add_landmark(self.line)

        floor = Landmark(
            name="floor",
            collide=False,
            shape=Box(length=10, width=1),
            color=Color.WHITE,
        )
        world.add_landmark(floor)

        return world

    def reset_world_at(self, env_index: int = None):
        goal_pos = torch.cat(
            [
                torch.zeros(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -1.0,
                    1.0,
                ),
                torch.zeros(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    0.0,
                    self.world.y_semidim,
                ),
            ],
            dim=1,
        )
        line_pos = torch.cat(
            [
                torch.zeros(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -1.0 + self.line_length / 2,
                    1.0 - self.line_length / 2,
                ),
                torch.zeros(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -0.95,
                    -0.8,
                ),
            ],
            dim=1,
        )
        package_rel_pos = torch.cat(
            [
                torch.zeros(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -self.line_length / 2 + self.package.shape.radius,
                    self.line_length / 2 - self.package.shape.radius,
                ),
                torch.full(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    self.package.shape.radius + LINE_MIN_DIST,
                    device=self.world.device,
                    dtype=torch.float32,
                ),
            ],
            dim=1,
        )

        for i, agent in enumerate(self.world.agents):
            agent.set_pos(
                line_pos
                + torch.tensor(
                    [
                        -(self.line_length - agent.shape.radius) / 2
                        + i
                        * (self.line_length - agent.shape.radius)
                        / (self.n_agents - 1),
                        -0.1,
                    ],
                    device=self.world.device,
                    dtype=torch.float32,
                ),
                batch_index=env_index,
            )

        self.line.set_pos(
            line_pos,
            batch_index=env_index,
        )
        self.package.goal.set_pos(
            goal_pos,
            batch_index=env_index,
        )
        self.line.set_rot(
            torch.zeros(1, device=self.world.device, dtype=torch.float32),
            batch_index=env_index,
        )
        self.package.set_pos(
            line_pos + package_rel_pos,
            batch_index=env_index,
        )
        floor = self.world.landmarks[3]
        floor.set_pos(
            torch.tensor(
                [0, -self.world.y_semidim - floor.shape.width / 2 - self.agent_radius],
                device=self.world.device,
            ),
            batch_index=env_index,
        )

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        rew = torch.zeros(
            self.world.batch_dim, device=self.world.device, dtype=torch.float32
        )

        if is_first:
            self.ball_on_the_ground = (
                self.package.state.pos[:, Y] <= -self.world.y_semidim
            )
            self.package_dist = torch.linalg.vector_norm(
                self.package.state.pos - self.package.goal.state.pos, dim=1
            )

        rew[self.ball_on_the_ground] -= 100
        rew -= self.package_dist

        return rew

    def observation(self, agent: Agent):
        # get positions of all entities in this agent's reference frame
        return torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
                agent.state.pos - self.package.state.pos,
                self.package.state.pos - self.package.goal.state.pos,
                self.world.is_overlapping(agent, self.line).unsqueeze(-1),
                self.world.is_overlapping(self.package, self.line).unsqueeze(-1),
            ],
            dim=-1,
        )

    def done(self):
        return self.ball_on_the_ground + self.world.is_overlapping(
            self.package, self.package.goal
        )


if __name__ == "__main__":
    render_interactively("balance", n_agents=4)
