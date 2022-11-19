#  Copyright (c) 2022.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import torch

from vmas import render_interactively
from vmas.simulator.core import Agent, Landmark, Sphere, World, Line, Box
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, Y


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.n_agents = kwargs.get("n_agents", 3)
        self.package_mass = kwargs.get("package_mass", 5)
        self.random_package_pos_on_line = kwargs.get("random_package_pos_on_line", True)

        assert self.n_agents > 1

        self.line_length = 0.8
        self.agent_radius = 0.03

        self.shaping_factor = 100
        self.fall_reward = -10

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
            collide=True,
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
                torch.full(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    -self.world.y_semidim + self.agent_radius * 2,
                    device=self.world.device,
                    dtype=torch.float32,
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
                    -self.line_length / 2 + self.package.shape.radius
                    if self.random_package_pos_on_line
                    else 0.0,
                    self.line_length / 2 - self.package.shape.radius
                    if self.random_package_pos_on_line
                    else 0.0,
                ),
                torch.full(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    self.package.shape.radius,
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
                        -self.agent_radius * 2,
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
        if env_index is None:
            self.global_shaping = (
                torch.linalg.vector_norm(
                    self.package.state.pos - self.package.goal.state.pos, dim=1
                )
                * self.shaping_factor
            )
        else:
            self.global_shaping[env_index] = (
                torch.linalg.vector_norm(
                    self.package.state.pos[env_index]
                    - self.package.goal.state.pos[env_index]
                )
                * self.shaping_factor
            )

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            self.pos_rew = torch.zeros(
                self.world.batch_dim, device=self.world.device, dtype=torch.float32
            )
            self.ground_rew = torch.zeros(
                self.world.batch_dim, device=self.world.device, dtype=torch.float32
            )

            self.on_the_ground = (
                self.package.state.pos[:, Y] <= -self.world.y_semidim
            ) + (self.line.state.pos[:, Y] <= -self.world.y_semidim)
            self.package_dist = torch.linalg.vector_norm(
                self.package.state.pos - self.package.goal.state.pos, dim=1
            )

            self.ground_rew[self.on_the_ground] = self.fall_reward

            global_shaping = self.package_dist * self.shaping_factor
            self.pos_rew = self.global_shaping - global_shaping
            self.global_shaping = global_shaping

        return self.ground_rew + self.pos_rew

    def observation(self, agent: Agent):
        # get positions of all entities in this agent's reference frame
        return torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
                agent.state.pos - self.package.state.pos,
                agent.state.pos - self.line.state.pos,
                self.package.state.pos - self.package.goal.state.pos,
                self.package.state.vel,
                self.line.state.vel,
                self.line.state.ang_vel,
                self.line.state.rot % torch.pi,
            ],
            dim=-1,
        )

    def done(self):
        return self.on_the_ground + self.world.is_overlapping(
            self.package, self.package.goal
        )

    def info(self, agent: Agent):
        info = {"pos_rew": self.pos_rew, "ground_rew": self.ground_rew}
        # When reset is called before reward()
        return info


class HeuristicPolicy(BaseHeuristicPolicy):
    def compute_action(self, observation: torch.Tensor, u_range: float) -> torch.Tensor:
        batch_dim = observation.shape[0]

        index_package_goal_pos = 8
        dist_package_goal = observation[
            :, index_package_goal_pos : index_package_goal_pos + 2
        ]
        y_distance_ge_0 = dist_package_goal[:, Y] >= 0

        if self.continuous_actions:
            action_agent = torch.clamp(
                torch.stack(
                    [
                        torch.zeros(batch_dim, device=observation.device),
                        -dist_package_goal[:, Y],
                    ],
                    dim=1,
                ),
                min=-u_range,
                max=u_range,
            )
            action_agent[:, Y][y_distance_ge_0] = 0
        else:
            action_agent = torch.full((batch_dim,), 4, device=observation.device)
            action_agent[y_distance_ge_0] = 0
        return action_agent


if __name__ == "__main__":
    render_interactively(
        __file__,
        n_agents=3,
        package_mass=5,
        random_package_pos_on_line=True,
        control_two_agents=True,
    )
