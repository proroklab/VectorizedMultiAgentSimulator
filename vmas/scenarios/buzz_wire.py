#  Copyright (c) 2022-2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
from typing import Dict

import torch
from torch import Tensor
from vmas import render_interactively
from vmas.simulator.core import Agent, Landmark, Sphere, World, Line
from vmas.simulator.joints import Joint
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.random_start_angle = kwargs.get("random_start_angle", True)
        self.pos_shaping_factor = kwargs.get("pos_shaping_factor", 1)
        self.collision_reward = kwargs.get("collision_reward", -10)
        self.max_speed_1 = kwargs.get("max_speed_1", None)  # 0.05

        self.pos_shaping_factor = 1

        self.n_agents = 2

        self.wall_length = 2
        self.agent_spacing = 0.5
        self.agent_radius = 0.03
        self.ball_radius = self.agent_radius

        # Make world
        world = World(
            batch_dim,
            device,
            substeps=15,
            joint_force=900,
            collision_force=1500,
        )
        # Add agents
        agent = Agent(
            name="agent 0",
            shape=Sphere(self.agent_radius),
            u_multiplier=1,
            mass=1,
        )
        world.add_agent(agent)
        agent = Agent(
            name="agent 1",
            shape=Sphere(self.agent_radius),
            u_multiplier=1,
            mass=1,
            max_speed=self.max_speed_1,
        )
        world.add_agent(agent)

        self.goal = Landmark(
            name="goal",
            shape=Sphere(radius=self.ball_radius),
            collide=False,
            color=Color.GREEN,
        )
        world.add_landmark(self.goal)

        self.ball = Landmark(
            name="ball",
            shape=Sphere(radius=self.ball_radius),
            collide=True,
            movable=True,
        )
        world.add_landmark(self.ball)

        self.joints = []
        for i in range(2):
            self.joints.append(
                Joint(
                    world.agents[i],
                    self.ball,
                    anchor_a=(0, 0),
                    anchor_b=(0, 0),
                    dist=self.agent_spacing / 2,
                    rotate_a=True,
                    rotate_b=True,
                    collidable=False,
                    width=0,
                    mass=1,
                )
            )
            world.add_joint(self.joints[i])

        self.build_path_line(world)

        self.pos_rew = torch.zeros(batch_dim, device=device, dtype=torch.float32)
        self.collision_rew = self.pos_rew.clone()
        self.collided = torch.full((world.batch_dim,), False, device=device)

        return world

    def reset_world_at(self, env_index: int = None):
        start_angle = torch.zeros(
            (1, 1) if env_index is not None else (self.world.batch_dim, 1),
            device=self.world.device,
            dtype=torch.float32,
        ).uniform_(
            -torch.pi / 2 + torch.pi / 3 if self.random_start_angle else 0,
            torch.pi / 2 - torch.pi / 3 if self.random_start_angle else 0,
        )

        start_delta_x = (self.agent_spacing / 2) * torch.cos(start_angle)
        min_x_start = -self.agent_radius
        max_x_start = self.agent_radius
        start_delta_y = (self.agent_spacing / 2) * torch.sin(start_angle)
        min_y_start = -self.wall_length / 2 + 2 * self.agent_radius
        max_y_start = -self.agent_radius

        min_x_goal = min_x_start
        max_x_goal = max_x_start
        min_y_goal = -min_y_start
        max_y_goal = -max_x_start

        ball_position = torch.cat(
            [
                (min_x_start - max_x_start)
                * torch.rand(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                )
                + max_x_start,
                (min_y_start - max_y_start)
                * torch.rand(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                )
                + max_y_start,
            ],
            dim=1,
        )
        goal_pos = torch.cat(
            [
                (min_x_goal - max_x_goal)
                * torch.rand(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                )
                + max_x_goal,
                (min_y_goal - max_y_goal)
                * torch.rand(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                )
                + max_y_goal,
            ],
            dim=1,
        )

        self.goal.set_pos(
            goal_pos,
            batch_index=env_index,
        )
        self.ball.set_pos(ball_position, batch_index=env_index)

        for i, agent in enumerate(self.world.agents):
            agent.set_pos(
                ball_position
                + torch.cat([start_delta_x, start_delta_y], dim=1)
                * (-1 if i == 0 else 1),
                batch_index=env_index,
            )

        for i, joint in enumerate(self.joints):
            joint.landmark.set_pos(
                ball_position
                + (
                    (torch.cat([start_delta_x, start_delta_y], dim=1) / 2)
                    * (-1 if i == 0 else 1)
                ),
                batch_index=env_index,
            )
            joint.landmark.set_rot(
                start_angle + (torch.pi if i == 1 else 0), batch_index=env_index
            )

        self.spawn_path_line(env_index)
        if env_index is None:
            self.pos_shaping = (
                torch.linalg.vector_norm(
                    self.ball.state.pos - self.goal.state.pos, dim=1
                )
                * self.pos_shaping_factor
            )
            self.collided[:] = False
        else:
            self.pos_shaping[env_index] = (
                torch.linalg.vector_norm(
                    self.ball.state.pos[env_index] - self.goal.state.pos[env_index],
                )
                * self.pos_shaping_factor
            )
            self.collided[env_index] = False

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            self.rew = torch.zeros(
                self.world.batch_dim, device=self.world.device, dtype=torch.float32
            )
            self.pos_rew[:] = 0
            self.collision_rew[:] = 0
            self.collided[:] = False

            dist_to_goal = torch.linalg.vector_norm(
                self.ball.state.pos - self.goal.state.pos,
                dim=1,
            )
            pos_shaping = dist_to_goal * self.pos_shaping_factor
            self.pos_rew += self.pos_shaping - pos_shaping
            self.pos_shaping = pos_shaping

            # Agent collisions
            for collidable in self.world.agents + [self.ball]:
                for entity in self.walls + self.floors:
                    is_overlap = self.world.is_overlapping(collidable, entity)
                    self.collision_rew[is_overlap] += self.collision_reward
                    self.collided += is_overlap

            self.rew = self.pos_rew + self.collision_rew

        return self.rew

    def observation(self, agent: Agent):
        return torch.cat(
            [agent.state.pos, agent.state.vel, agent.state.pos - self.goal.state.pos],
            dim=-1,
        )

    def done(self):
        return (
            torch.linalg.vector_norm(self.ball.state.pos - self.goal.state.pos, dim=1)
            <= 0.01
        ) + self.collided

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        return {
            "pos_rew": self.pos_rew,
            "collision_rew": self.collision_rew,
        }

    def build_path_line(self, world: World):
        self.walls = []
        for i in range(2):
            self.walls.append(
                Landmark(
                    name=f"wall {i}",
                    collide=True,
                    shape=Line(length=self.wall_length),
                    color=Color.BLACK,
                )
            )
            world.add_landmark(self.walls[i])
        self.floors = []
        for i in range(2):
            self.floors.append(
                Landmark(
                    name=f"floor {i}",
                    collide=True,
                    shape=Line(length=self.agent_spacing / 2),
                    color=Color.BLACK,
                )
            )
            world.add_landmark(self.floors[i])

    def spawn_path_line(self, env_index):
        for i, wall in enumerate(self.walls):
            wall.set_pos(
                torch.tensor(
                    [
                        (self.agent_spacing / 4) * (-1 if i == 0 else 1),
                        0.0,
                    ],
                    device=self.world.device,
                ),
                batch_index=env_index,
            )
            wall.set_rot(
                torch.tensor(torch.pi / 2, device=self.world.device),
                batch_index=env_index,
            )

        for i, floor in enumerate(self.floors):
            floor.set_pos(
                torch.tensor(
                    [
                        0,
                        (self.wall_length / 2) * (-1 if i == 0 else 1),
                    ],
                    device=self.world.device,
                ),
                batch_index=env_index,
            )


if __name__ == "__main__":
    render_interactively(__file__, control_two_agents=True)
