#  Copyright (c) 2022.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import math
from typing import Dict

import torch
from torch import Tensor
from vmas import render_interactively
from vmas.simulator.core import Agent, Box, Landmark, Sphere, World, Line
from vmas.simulator.joints import Joint
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, Y, X


def get_line_angle_0_90(rot: Tensor):
    angle = torch.abs(rot) % torch.pi
    other_angle = torch.pi - angle
    return torch.minimum(angle, other_angle)


def get_line_angle_0_180(rot):
    angle = rot % torch.pi
    return angle


def get_line_angle_dist_0_180(angle, goal):
    angle = get_line_angle_0_180(angle)
    goal = get_line_angle_0_180(goal)
    return torch.minimum(
        (angle - goal).abs(),
        torch.minimum(
            (angle - (goal - torch.pi)).abs(), ((angle - torch.pi) - goal).abs()
        ),
    ).squeeze(-1)


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.n_passages = kwargs.get("n_passages", 1)
        self.fixed_passage = kwargs.get("fixed_passage", False)
        self.joint_length = kwargs.get("joint_length", 0.5)
        self.random_start_angle = kwargs.get("random_start_angle", True)
        self.random_goal_angle = kwargs.get("random_goal_angle", True)
        self.observe_joint_angle = kwargs.get("observe_joint_angle", True)
        self.joint_angle_obs_noise = kwargs.get("joint_angle_obs_noise", 0.0)
        self.asym_package = kwargs.get("asym_package", False)
        self.mass_ratio = kwargs.get("mass_ratio", 1)
        self.max_speed_1 = kwargs.get("max_speed_1", None)  # 0.1
        self.pos_shaping_factor = kwargs.get("pos_shaping_factor", 1)
        self.rot_shaping_factor = kwargs.get("rot_shaping_factor", 1)
        self.collision_reward = kwargs.get("collision_reward", -0.06)
        self.all_passed_rot = kwargs.get("all_passed_rot", False)

        assert 1 <= self.n_passages <= 20
        if not self.observe_joint_angle:
            assert self.joint_angle_obs_noise == 0

        self.middle_angle = torch.pi / 2

        self.n_agents = 2

        self.agent_radius = 0.03333
        self.passage_width = 0.2
        self.passage_length = 0.103

        # Make world
        world = World(
            batch_dim,
            device,
            x_semidim=1,
            y_semidim=1,
            substeps=7,
            joint_force=400,
            collision_force=1500,
        )
        # Add agents
        agent = Agent(
            name=f"agent 0", shape=Sphere(self.agent_radius), u_multiplier=0.7
        )
        world.add_agent(agent)
        agent = Agent(
            name=f"agent 1",
            shape=Sphere(self.agent_radius),
            u_multiplier=0.7,
            mass=1 if self.asym_package else self.mass_ratio,
            max_speed=self.max_speed_1,
        )
        world.add_agent(agent)

        self.joint = Joint(
            world.agents[0],
            world.agents[1],
            anchor_a=(0, 0),
            anchor_b=(0, 0),
            dist=self.joint_length,
            rotate_a=True,
            rotate_b=True,
            collidable=True,
            width=0,
            mass=1,
        )

        if self.asym_package:
            self.mass = Landmark(
                name="mass",
                shape=Sphere(radius=self.agent_radius),
                collide=True,
                movable=True,
                color=Color.BLACK,
                mass=self.mass_ratio,
            )
            world.add_landmark(self.mass)

            joint = Joint(
                self.mass,
                self.joint.landmark,
                anchor_a=(0, 0),
                anchor_b=(0.5, 0),
                dist=0,
                rotate_a=False,
                rotate_b=False,
            )
            world.add_joint(joint)

        self.goal = Landmark(
            name="joint_goal",
            shape=Line(length=self.joint_length),
            collide=False,
            color=Color.GREEN,
        )
        world.add_landmark(self.goal)

        def joint_collision_filter(e):
            try:
                return e.neighbour
            except AttributeError:
                return False

        self.joint.landmark.collision_filter = joint_collision_filter
        world.add_joint(self.joint)

        self.create_passage_map(world)

        return world

    def reset_world_at(self, env_index: int = None):
        start_angle = torch.zeros(
            (1, 1) if env_index is not None else (self.world.batch_dim, 1),
            device=self.world.device,
            dtype=torch.float32,
        ).uniform_(
            -torch.pi / 2 if self.random_start_angle else 0,
            torch.pi / 2 if self.random_start_angle else 0,
        )

        goal_angle = torch.zeros(
            (1, 1) if env_index is not None else (self.world.batch_dim, 1),
            device=self.world.device,
            dtype=torch.float32,
        ).uniform_(
            -torch.pi / 2 if self.random_goal_angle else 0,
            torch.pi / 2 if self.random_goal_angle else 0,
        )

        start_delta_x = (self.joint_length / 2) * torch.cos(start_angle)
        start_delta_x_abs = start_delta_x.abs()
        min_x_start = -self.world.x_semidim + (self.agent_radius + start_delta_x_abs)
        max_x_start = self.world.x_semidim - (self.agent_radius + start_delta_x_abs)
        start_delta_y = (self.joint_length / 2) * torch.sin(start_angle)
        start_delta_y_abs = start_delta_y.abs()
        min_y_start = -self.world.y_semidim + (self.agent_radius + start_delta_y_abs)
        max_y_start = (
            -2 * self.agent_radius - self.passage_width / 2 - start_delta_y_abs
        )

        goal_delta_x = (self.joint_length / 2) * torch.cos(goal_angle)
        goal_delta_x_abs = goal_delta_x.abs()
        min_x_goal = -self.world.x_semidim + (self.agent_radius + goal_delta_x_abs)
        max_x_goal = self.world.x_semidim - (self.agent_radius + goal_delta_x_abs)
        goal_delta_y = (self.joint_length / 2) * torch.sin(goal_angle)
        goal_delta_y_abs = goal_delta_y.abs()
        min_y_goal = 2 * self.agent_radius + self.passage_width / 2 + goal_delta_y_abs
        max_y_goal = self.world.y_semidim - (self.agent_radius + goal_delta_y_abs)

        joint_pos = torch.cat(
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
        self.goal.set_rot(goal_angle, batch_index=env_index)

        for i, agent in enumerate(self.world.agents):
            if i == 0:
                agent.set_pos(
                    joint_pos - torch.cat([start_delta_x, start_delta_y], dim=1),
                    batch_index=env_index,
                )
            else:
                agent.set_pos(
                    joint_pos + torch.cat([start_delta_x, start_delta_y], dim=1),
                    batch_index=env_index,
                )
        self.joint.landmark.set_pos(joint_pos, batch_index=env_index)
        self.joint.landmark.set_rot(start_angle, batch_index=env_index)

        if self.asym_package:
            self.mass.set_pos(
                joint_pos + 0.5 * torch.cat([start_delta_x, start_delta_y], dim=1),
                batch_index=env_index,
            )

        self.spawn_passage_map(env_index)

        if env_index is None:
            self.joint.pos_shaping_pre = (
                torch.stack(
                    [
                        torch.linalg.vector_norm(
                            self.joint.landmark.state.pos - p.state.pos, dim=1
                        )
                        for p in self.passages
                        if not p.collide
                    ],
                    dim=1,
                ).min(dim=1)[0]
                * self.pos_shaping_factor
            )
            self.joint.pos_shaping_post = (
                torch.linalg.vector_norm(
                    self.joint.landmark.state.pos - self.goal.state.pos, dim=1
                )
                * self.pos_shaping_factor
            )

            self.joint.rot_shaping_pre = (
                get_line_angle_dist_0_180(
                    self.joint.landmark.state.rot, self.middle_angle
                )
                * self.rot_shaping_factor
            )
            self.joint.rot_shaping_post = (
                get_line_angle_dist_0_180(
                    self.joint.landmark.state.rot, self.goal.state.rot
                )
                * self.rot_shaping_factor
            )

        else:
            self.joint.pos_shaping_pre[env_index] = (
                torch.stack(
                    [
                        torch.linalg.vector_norm(
                            self.joint.landmark.state.pos[env_index]
                            - p.state.pos[env_index],
                        ).unsqueeze(-1)
                        for p in self.passages
                        if not p.collide
                    ],
                    dim=1,
                ).min(dim=1)[0]
                * self.pos_shaping_factor
            )
            self.joint.pos_shaping_post[env_index] = (
                torch.linalg.vector_norm(
                    self.joint.landmark.state.pos[env_index]
                    - self.goal.state.pos[env_index],
                )
                * self.pos_shaping_factor
            )
            self.joint.rot_shaping_pre[env_index] = (
                get_line_angle_dist_0_180(
                    self.joint.landmark.state.rot[env_index], self.middle_angle
                )
                * self.rot_shaping_factor
            )
            self.joint.rot_shaping_post[env_index] = (
                get_line_angle_dist_0_180(
                    self.joint.landmark.state.rot[env_index],
                    self.goal.state.rot[env_index],
                )
                * self.rot_shaping_factor
            )

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            self.rew = torch.zeros(
                self.world.batch_dim, device=self.world.device, dtype=torch.float32
            )
            self.pos_rew = self.rew.clone()
            self.rot_rew = self.rew.clone()
            self.collision_rew = self.rew.clone()

            joint_passed = self.joint.landmark.state.pos[:, Y] > 0
            all_passed = (
                torch.stack([a.state.pos[:, Y] for a in self.world.agents], dim=1)
                > self.passage_width / 2
            ).all(dim=1)

            # Pos shaping
            joint_dist_to_closest_pass = torch.stack(
                [
                    torch.linalg.vector_norm(
                        self.joint.landmark.state.pos - p.state.pos, dim=1
                    )
                    for p in self.passages
                    if not p.collide
                ],
                dim=1,
            ).min(dim=1)[0]
            joint_shaping = joint_dist_to_closest_pass * self.pos_shaping_factor
            self.pos_rew[~joint_passed] += (self.joint.pos_shaping_pre - joint_shaping)[
                ~joint_passed
            ]
            self.joint.pos_shaping_pre = joint_shaping

            joint_dist_to_goal = torch.linalg.vector_norm(
                self.joint.landmark.state.pos - self.goal.state.pos,
                dim=1,
            )
            joint_shaping = joint_dist_to_goal * self.pos_shaping_factor
            self.pos_rew[joint_passed] += (self.joint.pos_shaping_post - joint_shaping)[
                joint_passed
            ]
            self.joint.pos_shaping_post = joint_shaping

            # Rot shaping
            rot_passed = all_passed if self.all_passed_rot else joint_passed
            joint_dist_to_90_rot = get_line_angle_dist_0_180(
                self.joint.landmark.state.rot, self.middle_angle
            )
            joint_shaping = joint_dist_to_90_rot * self.rot_shaping_factor
            self.rot_rew[~rot_passed] += (self.joint.rot_shaping_pre - joint_shaping)[
                ~rot_passed
            ]
            self.joint.rot_shaping_pre = joint_shaping

            joint_dist_to_goal_rot = get_line_angle_dist_0_180(
                self.joint.landmark.state.rot, self.goal.state.rot
            )
            joint_shaping = joint_dist_to_goal_rot * self.rot_shaping_factor
            self.rot_rew[rot_passed] += (self.joint.rot_shaping_post - joint_shaping)[
                rot_passed
            ]
            self.joint.rot_shaping_post = joint_shaping

            # Agent collisions
            for a in self.world.agents:
                for passage in self.passages:
                    if passage.collide:
                        self.collision_rew[
                            self.world.is_overlapping(a, passage)
                        ] += self.collision_reward

            # Joint collisions
            for i, p in enumerate(self.passages):
                if p.collide and p.neighbour:
                    self.collision_rew[
                        self.world.is_overlapping(p, self.joint.landmark)
                    ] += self.collision_reward

            self.rew = self.pos_rew + self.rot_rew + self.collision_rew

        return self.rew

    def observation(self, agent: Agent):
        if self.observe_joint_angle:
            joint_angle = get_line_angle_0_180(self.joint.landmark.state.rot)
            angle_noise = (
                torch.randn(
                    *joint_angle.shape, device=self.world.device, dtype=torch.float32
                )
                * self.joint_angle_obs_noise
                if self.joint_angle_obs_noise
                else 0.0
            )
            joint_angle += angle_noise

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
                *passage_obs,
                get_line_angle_0_180(self.goal.state.rot),
            ]
            + ([joint_angle] if self.observe_joint_angle else []),
            dim=-1,
        )

    def done(self):
        return torch.all(
            (
                torch.linalg.vector_norm(
                    self.joint.landmark.state.pos - self.goal.state.pos, dim=1
                )
                <= 0.01
            )
            * (
                get_line_angle_dist_0_180(
                    self.joint.landmark.state.rot, self.goal.state.rot
                ).unsqueeze(-1)
                <= 0.01
            ),
            dim=1,
        )

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        return {
            "pos_rew": self.pos_rew,
            "rot_rew": self.rot_rew,
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
            self.passages = [self.passages[i] for i in order]
        for i, passage in enumerate(self.passages):
            if not passage.collide:
                passage.is_rendering[:] = False
            passage.neighbour = False
            try:
                passage.neighbour += not self.passages[i - 1].collide
            except IndexError:
                pass
            try:
                passage.neighbour += not self.passages[i + 1].collide
            except IndexError:
                pass
            passage.neighbour *= passage.collide
            passage.set_pos(
                torch.tensor(
                    [
                        -1
                        - self.agent_radius
                        + self.passage_length / 2
                        + self.passage_length * i,
                        0.0,
                    ],
                    dtype=torch.float32,
                    device=self.world.device,
                ),
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

        # Agent goal circles
        color = self.goal.color
        if isinstance(color, torch.Tensor) and len(color.shape) > 1:
            color = color[env_index]
        goal_agent_1 = rendering.make_circle(self.agent_radius)
        xform = rendering.Transform()
        goal_agent_1.add_attr(xform)
        xform.set_translation(
            self.goal.state.pos[env_index][X]
            - self.joint_length / 2 * math.cos(self.goal.state.rot[env_index]),
            self.goal.state.pos[env_index][Y]
            - self.joint_length / 2 * math.sin(self.goal.state.rot[env_index]),
        )

        goal_agent_1.set_color(*color)
        geoms.append(goal_agent_1)
        goal_agent_2 = rendering.make_circle(self.agent_radius)
        xform = rendering.Transform()
        goal_agent_2.add_attr(xform)
        xform.set_translation(
            self.goal.state.pos[env_index][X]
            + self.joint_length / 2 * math.cos(self.goal.state.rot[env_index]),
            self.goal.state.pos[env_index][Y]
            + self.joint_length / 2 * math.sin(self.goal.state.rot[env_index]),
        )
        goal_agent_2.set_color(*color)
        geoms.append(goal_agent_2)

        return geoms


if __name__ == "__main__":
    render_interactively("joint", control_two_agents=True)
