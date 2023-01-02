#  Copyright (c) 2022-2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

from typing import Dict

import torch
from torch import Tensor

from vmas import render_interactively
from vmas.simulator.core import Agent, Box, Landmark, Sphere, World, Line
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, Y, X


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.n_passages = kwargs.get("n_passages", 1)
        self.fixed_passage = kwargs.get("fixed_passage", False)
        self.random_start_angle = kwargs.get("random_start_angle", True)

        assert 1 <= self.n_passages <= 20

        self.pos_shaping_factor = 1
        self.collision_reward = -0.06

        self.n_agents = 2

        self.agent_spacing = 0.5
        self.agent_radius = 0.03333
        self.ball_radius = self.agent_radius
        self.passage_width = 0.2
        self.passage_length = 0.103

        # Make world
        world = World(
            batch_dim,
            device,
            x_semidim=1,
            y_semidim=1,
            drag=0,
            linear_friction=0.0,
        )
        # Add agents
        for i in range(2):
            agent = Agent(
                name=f"agent {i}",
                shape=Sphere(self.agent_radius),
                u_multiplier=0.7,
                mass=2,
                drag=0.25,
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
            mass=1,
            color=Color.BLACK,
            linear_friction=0.02,
        )
        world.add_landmark(self.ball)

        self.create_passage_map(world)

        self.pos_rew = torch.zeros(batch_dim, device=device, dtype=torch.float32)
        self.collision_rew = self.pos_rew.clone()

        return world

    def reset_world_at(self, env_index: int = None):

        start_angle = torch.zeros(
            (1, 1) if env_index is not None else (self.world.batch_dim, 1),
            device=self.world.device,
            dtype=torch.float32,
        ).uniform_(
            -torch.pi / 2 if self.random_start_angle else -torch.pi / 2,
            torch.pi / 2 if self.random_start_angle else -torch.pi / 2,
        )

        start_delta_x = (self.agent_spacing / 2) * torch.cos(start_angle)
        start_delta_x_abs = start_delta_x.abs()
        min_x_start = -self.world.x_semidim + (self.agent_radius + start_delta_x_abs)
        max_x_start = self.world.x_semidim - (self.agent_radius + start_delta_x_abs)
        start_delta_y = (self.agent_spacing / 2) * torch.sin(start_angle)
        start_delta_y_abs = start_delta_y.abs()
        min_y_start = -self.world.y_semidim + (self.agent_radius + start_delta_y_abs)
        max_y_start = (
            -2 * self.agent_radius - self.passage_width / 2 - start_delta_y_abs
        )

        min_x_goal = -self.world.x_semidim + self.agent_radius
        max_x_goal = self.world.x_semidim - self.agent_radius
        min_y_goal = 2 * self.agent_radius + self.passage_width / 2
        max_y_goal = self.world.y_semidim - self.agent_radius

        ball_pos = torch.cat(
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

        self.ball.set_pos(
            ball_pos,
            batch_index=env_index,
        )

        for i, agent in enumerate(self.world.agents):
            if i == 0:
                agent.set_pos(
                    ball_pos - torch.cat([start_delta_x, start_delta_y], dim=1),
                    batch_index=env_index,
                )
            else:
                agent.set_pos(
                    ball_pos + torch.cat([start_delta_x, start_delta_y], dim=1),
                    batch_index=env_index,
                )

        self.goal.set_pos(
            torch.cat(
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
            ),
            batch_index=env_index,
        )

        self.spawn_passage_map(env_index)

        if env_index is None:
            self.ball.pos_shaping_pre = (
                torch.stack(
                    [
                        torch.linalg.vector_norm(
                            self.ball.state.pos - p.state.pos, dim=1
                        )
                        for p in self.passages
                        if not p.collide
                    ],
                    dim=1,
                ).min(dim=1)[0]
                * self.pos_shaping_factor
            )
            self.ball.pos_shaping_post = (
                torch.linalg.vector_norm(
                    self.ball.state.pos - self.goal.state.pos, dim=1
                )
                * self.pos_shaping_factor
            )

        else:
            self.ball.pos_shaping_pre[env_index] = (
                torch.stack(
                    [
                        torch.linalg.vector_norm(
                            self.ball.state.pos[env_index] - p.state.pos[env_index],
                        ).unsqueeze(-1)
                        for p in self.passages
                        if not p.collide
                    ],
                    dim=1,
                ).min(dim=1)[0]
                * self.pos_shaping_factor
            )
            self.ball.pos_shaping_post[env_index] = (
                torch.linalg.vector_norm(
                    self.ball.state.pos[env_index] - self.goal.state.pos[env_index],
                )
                * self.pos_shaping_factor
            )

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            self.rew = torch.zeros(
                self.world.batch_dim, device=self.world.device, dtype=torch.float32
            )
            self.pos_rew[:] = 0
            self.collision_rew[:] = 0

            ball_passed = self.ball.state.pos[:, Y] > 0

            # Pos shaping
            ball_dist_to_closest_pass = torch.stack(
                [
                    torch.linalg.vector_norm(self.ball.state.pos - p.state.pos, dim=1)
                    for p in self.passages
                    if not p.collide
                ],
                dim=1,
            ).min(dim=1)[0]
            ball_shaping = ball_dist_to_closest_pass * self.pos_shaping_factor
            self.pos_rew[~ball_passed] += (self.ball.pos_shaping_pre - ball_shaping)[
                ~ball_passed
            ]
            self.ball.pos_shaping_pre = ball_shaping

            ball_dist_to_goal = torch.linalg.vector_norm(
                self.ball.state.pos - self.goal.state.pos,
                dim=1,
            )
            ball_shaping = ball_dist_to_goal * self.pos_shaping_factor
            self.pos_rew[ball_passed] += (self.ball.pos_shaping_post - ball_shaping)[
                ball_passed
            ]
            self.ball.pos_shaping_post = ball_shaping

            # Agent collisions
            for a in self.world.agents:
                for passage in self.passages:
                    if passage.collide:
                        self.collision_rew[
                            self.world.is_overlapping(a, passage)
                        ] += self.collision_reward

            # Ball collisions
            for i, p in enumerate(self.passages):
                if p.collide:
                    self.collision_rew[
                        self.world.is_overlapping(p, self.ball)
                    ] += self.collision_reward

            self.rew = self.pos_rew + self.collision_rew

        return self.rew

    def observation(self, agent: Agent):
        # get positions of all entities in this agent's reference frame
        passage_obs = []
        for passage in self.passages:
            if not passage.collide:
                passage_obs.append(agent.state.pos - passage.state.pos)
        return torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
                agent.state.pos - self.goal.state.pos,
                agent.state.pos - self.ball.state.pos,
                *passage_obs,
            ],
            dim=-1,
        )

    def done(self):
        return (
            (
                torch.linalg.vector_norm(
                    self.ball.state.pos - self.goal.state.pos, dim=1
                )
                <= 0.01
            )
            + (-self.world.x_semidim + self.ball_radius >= self.ball.state.pos[:, X])
            + (self.ball.state.pos[:, X] >= self.world.x_semidim - self.ball_radius)
            + (-self.world.y_semidim + self.ball_radius >= self.ball.state.pos[:, Y])
            + (self.ball.state.pos[:, Y] >= self.world.y_semidim - self.ball_radius)
        )

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        return {
            "pos_rew": self.pos_rew,
            "collision_rew": self.collision_rew,
        }

    def create_passage_map(self, world: World):
        # Add landmarks
        self.passages = []
        n_boxes = int(
            (2 * world.x_semidim + 2 * self.agent_radius) // self.passage_length
        )

        def removed(i):
            return (
                (n_boxes // 2) - self.n_passages / 2
                <= i
                < (n_boxes // 2) + self.n_passages / 2
            )

        for i in range(n_boxes):
            passage = Landmark(
                name=f"passage {i}",
                collide=not removed(i),
                movable=False,
                shape=Box(length=self.passage_length, width=self.passage_width),
                color=Color.RED,
                collision_filter=lambda e: not isinstance(e.shape, Box),
            )
            self.passages.append(passage)
            world.add_landmark(passage)

    def spawn_passage_map(self, env_index):
        if not self.fixed_passage:
            order = torch.randperm(len(self.passages)).tolist()
            self.passages_to_place = [self.passages[i] for i in order]
        else:
            self.passages_to_place = self.passages
        for i, passage in enumerate(self.passages_to_place):
            if not passage.collide:
                passage.is_rendering[:] = False
            passage.neighbour = False
            try:
                passage.neighbour += not self.passages_to_place[i - 1].collide
            except IndexError:
                pass
            try:
                passage.neighbour += not self.passages_to_place[i + 1].collide
            except IndexError:
                pass
            pos = torch.tensor(
                [
                    -1
                    - self.agent_radius
                    + self.passage_length / 2
                    + self.passage_length * i,
                    0.0,
                ],
                dtype=torch.float32,
                device=self.world.device,
            )
            passage.neighbour *= passage.collide
            passage.set_pos(
                pos,
                batch_index=env_index,
            )

    def extra_render(self, env_index: int = 0):
        from vmas.simulator import rendering

        geoms = []
        # Perimeter
        for i in range(4):
            geom = Line(length=2 + self.agent_radius * 2).get_geometry()
            xform = rendering.Transform()
            geom.add_attr(xform)

            xform.set_translation(
                0.0
                if i % 2
                else (
                    self.world.x_semidim + self.agent_radius
                    if i == 0
                    else -self.world.x_semidim - self.agent_radius
                ),
                0.0
                if not i % 2
                else (
                    self.world.x_semidim + self.agent_radius
                    if i == 1
                    else -self.world.x_semidim - self.agent_radius
                ),
            )
            xform.set_rotation(torch.pi / 2 if not i % 2 else 0.0)
            color = Color.BLACK.value
            if isinstance(color, torch.Tensor) and len(color.shape) > 1:
                color = color[env_index]
            geom.set_color(*color)
            geoms.append(geom)

        return geoms


if __name__ == "__main__":
    render_interactively(__file__, control_two_agents=True)
