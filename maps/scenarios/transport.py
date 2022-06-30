#  Copyright (c) 2022. Matteo Bettini
#  All rights reserved.

import torch

from maps import render_interactively
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

        self.shaping_factor = 100

        # Make world
        world = World(batch_dim, device)
        # Add agents
        for i in range(n_agents):
            agent = Agent(name=f"agent {i}", shape=Sphere(0.03), u_multiplier=0.6)
            world.add_agent(agent)
        # Add landmarks
        goal = Landmark(
            name="goal",
            collide=False,
            shape=Sphere(radius=0.15),
            color=Color.LIGHT_GREEN,
        )
        world.add_landmark(goal)
        self.packages = []
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
            self.packages.append(package)
            world.add_landmark(package)

        return world

    def reset_world_at(self, env_index: int = None):
        goal = self.world.landmarks[0]
        goal.set_pos(
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
        for i, package in enumerate(self.packages):
            package.set_pos(
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
            package.on_goal = self.world.is_overlapping(package, package.goal)
            if env_index is None:
                package.global_shaping = (
                    torch.linalg.vector_norm(
                        package.state.pos - package.goal.state.pos, dim=1
                    )
                    * self.shaping_factor
                )
            else:
                package.global_shaping[env_index] = (
                    torch.linalg.vector_norm(
                        package.state.pos[env_index] - package.goal.state.pos[env_index]
                    )
                    * self.shaping_factor
                )
        for i, agent in enumerate(self.world.agents):
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
            for package in self.packages:
                while self.world.is_overlapping(
                    agent, package, env_index=env_index
                ).any():
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

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            self.rew = torch.zeros(
                self.world.batch_dim, device=self.world.device, dtype=torch.float32
            )

            for i, package in enumerate(self.packages):
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

                package_shaping = package.dist_to_goal * self.shaping_factor
                self.rew[~package.on_goal] += (
                    package.global_shaping[~package.on_goal]
                    - package_shaping[~package.on_goal]
                )
                package.global_shaping = package_shaping

        return self.rew

    def observation(self, agent: Agent):
        # get positions of all entities in this agent's reference frame
        package_obs = []
        for package in self.packages:
            package_obs.append(package.state.pos - package.goal.state.pos)
            package_obs.append(package.state.pos - agent.state.pos)
            package_obs.append(package.state.vel)
            package_obs.append(package.on_goal.unsqueeze(-1))

        return torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
                *package_obs,
            ],
            dim=-1,
        )

    def done(self):
        return torch.all(
            torch.stack(
                [package.on_goal for package in self.packages],
                dim=1,
            ),
            dim=-1,
        )


if __name__ == "__main__":
    render_interactively(
        "transport",
        n_agents=4,
        n_packages=1,
        package_width=0.15,
        package_length=0.15,
        package_mass=50,
    )
