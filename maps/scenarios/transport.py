#  Copyright (c) 2022. Matteo Bettini
#  All rights reserved.

import torch

from maps.simulator.core import Agent, Box, Landmark, Sphere, World
from maps.simulator.scenario import BaseScenario
from maps.simulator.utils import Color


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        n_agents = kwargs.get("n_agents", 4)
        self.n_packages = kwargs.get("n_packages", 1)
        self.package_width = kwargs.get("package_width", 0.15)
        self.package_length = kwargs.get("package_length", 0.15)
        self.package_mass = kwargs.get("package_mass", 50)

        # Make world
        world = World(batch_dim, device)
        # Add agents
        for i in range(n_agents):
            agent = Agent(name=f"agent {i}", shape=Sphere(0.03), u_multiplier=0.9)
            world.add_agent(agent)
        # Add landmarks
        goal = Landmark(
            name="goal",
            collide=False,
            shape=Sphere(radius=0.15),
            color=Color.LIGHT_GREEN,
        )
        world.add_landmark(goal)
        for i in range(self.n_packages):
            package = Landmark(
                name=f"package {i}",
                collide=True,
                movable=True,
                mass=50,
                shape=Box(length=self.package_length, width=self.package_width),
                color=Color.RED,
            )
            package.goal = goal
            world.add_landmark(package)

        return world

    def reset_world_at(self, env_index: int = None):
        for i, agent in enumerate(self.world.agents):
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
        for i, package in enumerate(self.world.landmarks[1:]):
            package.set_pos(
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
        goal = self.world.landmarks[0]
        goal.set_pos(
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

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        rew = torch.zeros(
            self.world.batch_dim, device=self.world.device, dtype=torch.float32
        )

        for i, package in enumerate(self.world.landmarks[1:]):
            if is_first:
                package.dist_to_goal = torch.linalg.vector_norm(
                    package.state.pos - package.goal.state.pos, dim=1
                )
                package.on_goal = self.world.is_overlapping(package, package.goal)
                package.color = torch.tensor(
                    Color.RED.value, device=self.world.device, dtype=torch.float32
                ).repeat(self.world.batch_dim, 1)
                package.color[package.on_goal] = torch.tensor(
                    Color.GREEN.value, device=self.world.device, dtype=torch.float32
                )

            rew[~package.on_goal] += package.dist_to_goal[~package.on_goal]

        return -rew

    def observation(self, agent: Agent):
        # get positions of all entities in this agent's reference frame
        package_pos = []
        for package in self.world.landmarks[1:]:
            package_pos.append(package.state.pos - package.goal.state.pos)
            package_pos.append(package.state.pos - agent.state.pos)
        return torch.cat([agent.state.pos, agent.state.vel, *package_pos], dim=-1)

    def done(self):
        return torch.all(
            torch.stack(
                [package.on_goal for package in self.world.landmarks[1:]],
                dim=1,
            ),
            dim=-1,
        )
