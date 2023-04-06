#  Copyright (c) 2022-2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.


import torch

from vmas import render_interactively
from vmas.simulator.core import Agent, Box, Landmark, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        n_agents = kwargs.get("n_agents", 4)
        self.package_width = kwargs.get("package_width", 0.6)
        self.package_length = kwargs.get("package_length", 0.6)
        self.package_mass = kwargs.get("package_mass", 50)

        self.shaping_factor = 100

        # Make world
        world = World(batch_dim, device, contact_margin=6e-3)
        # Add agents
        for i in range(n_agents):
            agent = Agent(name=f"agent {i}", shape=Sphere(0.03), u_multiplier=0.5)
            world.add_agent(agent)
        # Add landmarks
        goal = Landmark(
            name="goal",
            collide=False,
            shape=Sphere(radius=0.09),
            color=Color.LIGHT_GREEN,
        )
        world.add_landmark(goal)

        self.package = Landmark(
            name=f"package {i}",
            collide=True,
            movable=True,
            mass=50,
            shape=Box(
                length=self.package_length, width=self.package_width, hollow=True
            ),
            color=Color.RED,
        )
        self.package.goal = goal
        world.add_landmark(self.package)

        return world

    def reset_world_at(self, env_index: int = None):
        package_pos = torch.zeros(
            (1, self.world.dim_p)
            if env_index is not None
            else (self.world.batch_dim, self.world.dim_p),
            device=self.world.device,
            dtype=torch.float32,
        ).uniform_(
            -1.0,
            1.0,
        )

        self.package.set_pos(
            package_pos,
            batch_index=env_index,
        )
        for i, agent in enumerate(self.world.agents):
            agent.set_pos(
                torch.cat(
                    [
                        torch.zeros(
                            (1, 1)
                            if env_index is not None
                            else (self.world.batch_dim, 1),
                            device=self.world.device,
                            dtype=torch.float32,
                        ).uniform_(
                            -self.package_length / 2 + agent.shape.radius,
                            self.package_length / 2 - agent.shape.radius,
                        ),
                        torch.zeros(
                            (1, 1)
                            if env_index is not None
                            else (self.world.batch_dim, 1),
                            device=self.world.device,
                            dtype=torch.float32,
                        ).uniform_(
                            -self.package_width / 2 + agent.shape.radius,
                            self.package_width / 2 - agent.shape.radius,
                        ),
                    ],
                    dim=1,
                )
                + package_pos,
                batch_index=env_index,
            )

        self.package.goal.set_pos(
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

        if env_index is None:
            self.package.global_shaping = (
                torch.linalg.vector_norm(
                    self.package.state.pos - self.package.goal.state.pos, dim=1
                )
                * self.shaping_factor
            )
            self.package.on_goal = torch.zeros(
                self.world.batch_dim, dtype=torch.bool, device=self.world.device
            )
        else:
            self.package.global_shaping[env_index] = (
                torch.linalg.vector_norm(
                    self.package.state.pos[env_index]
                    - self.package.goal.state.pos[env_index]
                )
                * self.shaping_factor
            )
            self.package.on_goal[env_index] = False

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            self.rew = torch.zeros(
                self.world.batch_dim, device=self.world.device, dtype=torch.float32
            )

            self.package.dist_to_goal = torch.linalg.vector_norm(
                self.package.state.pos - self.package.goal.state.pos, dim=1
            )
            self.package.on_goal = self.world.is_overlapping(
                self.package, self.package.goal
            )
            self.package.color = torch.tensor(
                Color.RED.value, device=self.world.device, dtype=torch.float32
            ).repeat(self.world.batch_dim, 1)
            self.package.color[self.package.on_goal] = torch.tensor(
                Color.GREEN.value, device=self.world.device, dtype=torch.float32
            )

            package_shaping = self.package.dist_to_goal * self.shaping_factor
            self.rew[~self.package.on_goal] += (
                self.package.global_shaping[~self.package.on_goal]
                - package_shaping[~self.package.on_goal]
            )
            self.package.global_shaping = package_shaping

            self.rew[~self.package.on_goal] += (
                self.package.global_shaping[~self.package.on_goal]
                - package_shaping[~self.package.on_goal]
            )
            self.package.global_shaping = package_shaping

        return self.rew

    def observation(self, agent: Agent):
        return torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
                self.package.state.vel,
                self.package.state.pos - agent.state.pos,
                self.package.state.pos - self.package.goal.state.pos,
            ],
            dim=-1,
        )

    def done(self):
        return self.package.on_goal


if __name__ == "__main__":
    render_interactively(
        __file__,
        control_two_agents=True,
        n_agents=4,
        package_width=0.6,
        package_length=0.6,
        package_mass=50,
    )
