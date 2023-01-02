#  Copyright (c) 2022-2023.
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
from vmas.simulator.velocity_controller import VelocityController


def angle_to_vector(angle):
    return torch.cat([torch.cos(angle), torch.sin(angle)], dim=1)


def get_line_angle_0_90(rot: Tensor):
    angle = torch.abs(rot) % torch.pi
    other_angle = torch.pi - angle
    return torch.minimum(angle, other_angle)


def get_line_angle_0_180(rot):
    angle = rot % torch.pi
    return angle


def get_line_angle_dist_0_360(angle, goal):
    angle = angle_to_vector(angle)
    goal = angle_to_vector(goal)
    return -torch.einsum("bs,bs->b", angle, goal)


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
        self.fixed_passage = kwargs.get("fixed_passage", False)
        self.joint_length = kwargs.get("joint_length", 0.52)
        self.random_start_angle = kwargs.get("random_start_angle", False)
        self.random_goal_angle = kwargs.get("random_goal_angle", False)
        self.observe_joint_angle = kwargs.get("observe_joint_angle", False)
        self.joint_angle_obs_noise = kwargs.get("joint_angle_obs_noise", 0.0)
        self.asym_package = kwargs.get("asym_package", False)
        self.mass_ratio = kwargs.get("mass_ratio", 1)
        self.mass_position = kwargs.get("mass_position", 0.75)
        self.max_speed_1 = kwargs.get("max_speed_1", None)  # 0.1
        self.pos_shaping_factor = kwargs.get("pos_shaping_factor", 1)
        self.rot_shaping_factor = kwargs.get("rot_shaping_factor", 1)
        self.collision_reward = kwargs.get("collision_reward", 0)
        self.energy_reward_coeff = kwargs.get("energy_reward_coeff", 0)
        self.obs_noise = kwargs.get("obs_noise", 0.0)

        self.plot_grid = False

        # Make world
        world = World(
            batch_dim,
            device,
            x_semidim=1,
            y_semidim=1,
            substeps=5 if not self.asym_package else 10,
            joint_force=700 if self.asym_package else 400,
            collision_force=2500 if self.asym_package else 1500,
            drag=0.25 if not self.asym_package else 0.15,
        )

        if not self.observe_joint_angle:
            assert self.joint_angle_obs_noise == 0

        self.n_agents = 2

        self.middle_angle = torch.zeros((world.batch_dim, 1), device=world.device)

        self.agent_radius = 0.03333
        self.agent_radius_2 = 3 * self.agent_radius
        self.mass_radius = self.agent_radius * (2 / 3)
        self.passage_width = 0.2
        self.passage_length = 0.1476
        self.scenario_length = 2 + 2 * self.agent_radius
        self.n_boxes = int(self.scenario_length // self.passage_length)
        self.min_collision_distance = 0.005

        cotnroller_params = [2.0, 10, 0.00001]

        # Add agents
        agent = Agent(
            name="agent 0",
            shape=Sphere(self.agent_radius),
            u_range=0.5,
            obs_noise=self.obs_noise,
            render_action=True,
            f_range=10,
        )
        agent.controller = VelocityController(
            agent, world, cotnroller_params, "standard"
        )
        world.add_agent(agent)
        agent = Agent(
            name="agent 1",
            shape=Sphere(self.agent_radius_2),
            u_range=0.5,
            mass=1 if self.asym_package else self.mass_ratio,
            max_speed=self.max_speed_1,
            obs_noise=self.obs_noise,
            render_action=True,
            f_range=10,
        )
        agent.controller = VelocityController(
            agent, world, cotnroller_params, "standard"
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
            collidable=False,
            width=0,
            mass=1,
        )
        world.add_joint(self.joint)

        if self.asym_package:

            def mass_collision_filter(e):
                return not isinstance(e.shape, Sphere)

            self.mass = Landmark(
                name="mass",
                shape=Sphere(radius=self.mass_radius),
                collide=True,
                movable=True,
                color=Color.BLACK,
                mass=self.mass_ratio,
                collision_filter=mass_collision_filter,
            )

            world.add_landmark(self.mass)

            joint = Joint(
                self.mass,
                self.joint.landmark,
                anchor_a=(0, 0),
                anchor_b=(self.mass_position, 0),
                dist=0,
                rotate_a=True,
                rotate_b=True,
            )
            world.add_joint(joint)

        self.goal = Landmark(
            name="joint_goal",
            shape=Line(length=self.joint_length),
            collide=False,
            color=Color.GREEN,
        )
        world.add_landmark(self.goal)

        self.walls = []
        for i in range(4):
            wall = Landmark(
                name=f"wall {i}",
                collide=True,
                shape=Line(length=2 + self.agent_radius * 2),
                color=Color.BLACK,
            )
            world.add_landmark(wall)
            self.walls.append(wall)

        self.create_passage_map(world)

        self.pos_rew = torch.zeros(batch_dim, device=device)
        self.rot_rew = self.pos_rew.clone()
        self.collision_rew = self.pos_rew.clone()
        self.energy_rew = self.pos_rew.clone()
        self.all_passed = torch.full((batch_dim,), False, device=device)

        return world

    def reset_world_at(self, env_index: int = None):
        start_angle = torch.rand(
            (1, 1) if env_index is not None else (self.world.batch_dim, 1),
            device=self.world.device,
        )
        start_angle[start_angle >= 0.5] = torch.pi / 2
        start_angle[start_angle < 0.5] = -torch.pi / 2

        goal_angle = torch.zeros(
            (1, 1) if env_index is not None else (self.world.batch_dim, 1),
            device=self.world.device,
            dtype=torch.float32,
        ).uniform_(
            -torch.pi / 2 if self.random_goal_angle else torch.pi,
            torch.pi / 2 if self.random_goal_angle else torch.pi,
        )

        bigger_radius = max(self.agent_radius, self.agent_radius_2)

        start_delta_x = (self.joint_length / 2) * torch.cos(start_angle)
        start_delta_x_abs = start_delta_x.abs()
        min_x_start = -self.world.x_semidim + (bigger_radius + start_delta_x_abs)
        max_x_start = self.world.x_semidim - (bigger_radius + start_delta_x_abs)
        start_delta_y = (self.joint_length / 2) * torch.sin(start_angle)
        start_delta_y_abs = start_delta_y.abs()
        min_y_start = -self.world.y_semidim + (bigger_radius + start_delta_y_abs)
        max_y_start = -2 * bigger_radius - self.passage_width / 2 - start_delta_y_abs

        goal_delta_x = (self.joint_length / 2) * torch.cos(goal_angle)
        goal_delta_x_abs = goal_delta_x.abs()
        min_x_goal = -self.world.x_semidim + (bigger_radius + goal_delta_x_abs)
        max_x_goal = self.world.x_semidim - (bigger_radius + goal_delta_x_abs)
        goal_delta_y = (self.joint_length / 2) * torch.sin(goal_angle)
        goal_delta_y_abs = goal_delta_y.abs()
        min_y_goal = 2 * bigger_radius + self.passage_width / 2 + goal_delta_y_abs
        max_y_goal = self.world.y_semidim - (bigger_radius + goal_delta_y_abs)

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

        agents = self.world.agents
        for i, agent in enumerate(agents):
            agent.controller.reset(env_index)
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

        if self.asym_package:
            self.mass.set_pos(
                joint_pos
                + self.mass_position
                * torch.cat([start_delta_x, start_delta_y], dim=1)
                * (1 if agents[0] == self.world.agents[0] else -1),
                batch_index=env_index,
            )

        self.spawn_passage_map(env_index)

        self.spawn_walls(env_index)

        if env_index is None:
            self.passed = torch.zeros((self.world.batch_dim,), device=self.world.device)

            self.joint.pos_shaping_pre = (
                torch.linalg.vector_norm(
                    self.joint.landmark.state.pos - self.pass_center, dim=1
                )
                * self.pos_shaping_factor
            )
            self.joint.pos_shaping_post = (
                torch.linalg.vector_norm(
                    self.joint.landmark.state.pos - self.goal.state.pos, dim=1
                )
                * self.pos_shaping_factor
            )

            self.joint.rot_shaping_pre = (
                get_line_angle_dist_0_360(
                    self.joint.landmark.state.rot, self.middle_angle
                )
                * self.rot_shaping_factor
            )
            # self.joint.rot_shaping_post = (
            #     get_line_angle_dist_0_180(
            #         self.joint.landmark.state.rot, self.goal.state.rot
            #     )
            #     * self.rot_shaping_factor
            # )

        else:
            self.passed[env_index] = 0

            self.joint.pos_shaping_pre[env_index] = (
                torch.linalg.vector_norm(
                    self.joint.landmark.state.pos[env_index]
                    - self.pass_center[env_index]
                )
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
                get_line_angle_dist_0_360(
                    self.joint.landmark.state.rot[env_index].unsqueeze(-1),
                    self.middle_angle[env_index].unsqueeze(-1),
                )
                * self.rot_shaping_factor
            )
            # self.joint.rot_shaping_post[env_index] = (
            #     get_line_angle_dist_0_180(
            #         self.joint.landmark.state.rot[env_index],
            #         self.goal.state.rot[env_index],
            #     )
            #     * self.rot_shaping_factor
            # )

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            self.rew = torch.zeros(
                self.world.batch_dim, device=self.world.device, dtype=torch.float32
            )
            self.pos_rew[:] = 0
            self.rot_rew[:] = 0
            self.collision_rew[:] = 0
            self.energy_rew[:] = 0

            joint_passed = self.joint.landmark.state.pos[:, Y] > 0
            self.all_passed = (
                torch.stack([a.state.pos[:, Y] for a in self.world.agents], dim=1)
                > self.passage_width / 2
            ).all(dim=1)

            # Pos shaping
            joint_dist_to_closest_pass = (
                torch.linalg.vector_norm(
                    self.joint.landmark.state.pos - self.pass_center, dim=1
                )
                * self.pos_shaping_factor
            )
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
            joint_dist_to_90_rot = get_line_angle_dist_0_360(
                self.joint.landmark.state.rot, self.middle_angle
            )
            joint_shaping = joint_dist_to_90_rot * self.rot_shaping_factor
            self.rot_rew += self.joint.rot_shaping_pre - joint_shaping
            self.joint.rot_shaping_pre = joint_shaping

            # joint_dist_to_goal_rot = get_line_angle_dist_0_180(
            #     self.joint.landmark.state.rot, self.goal.state.rot
            # )
            # joint_shaping = joint_dist_to_goal_rot * self.rot_shaping_factor
            # self.rot_rew[rot_passed] += (self.joint.rot_shaping_post - joint_shaping)[
            #     rot_passed
            # ]
            # self.joint.rot_shaping_post = joint_shaping

            # Agent collisions
            if self.collision_reward != 0:
                for a in self.world.agents + ([self.mass] if self.asym_package else []):
                    for passage in self.passages:
                        if passage.collide:
                            self.collision_rew[
                                self.world.get_distance(a, passage)
                                <= self.min_collision_distance
                            ] += self.collision_reward
                    for wall in self.walls:
                        self.collision_rew[
                            self.world.get_distance(a, wall)
                            <= self.min_collision_distance
                        ] += self.collision_reward

            # Energy reward
            if self.energy_reward_coeff != 0:
                self.energy_expenditure = torch.stack(
                    [
                        torch.linalg.vector_norm(a.action.u, dim=-1)
                        / math.sqrt(
                            self.world.dim_p * ((a.u_range * a.u_multiplier) ** 2)
                        )
                        for a in self.world.agents
                    ],
                    dim=1,
                ).sum(-1)
                self.energy_rew = -self.energy_expenditure * self.energy_reward_coeff

            self.rew = (
                self.pos_rew + self.rot_rew + self.collision_rew + self.energy_rew
            )

        return self.rew

    def process_action(self, agent: Agent):
        vel_is_zero = torch.linalg.vector_norm(agent.action.u, dim=1) < 1e-3
        agent.controller.reset(vel_is_zero)
        agent.controller.process_force()

    def is_out_or_touching_perimeter(self, agent: Agent):
        is_out_or_touching_perimeter = torch.full(
            (self.world.batch_dim,), False, device=self.world.device
        )
        is_out_or_touching_perimeter += agent.state.pos[:, X] >= self.world.x_semidim
        is_out_or_touching_perimeter += agent.state.pos[:, X] <= -self.world.x_semidim
        is_out_or_touching_perimeter += agent.state.pos[:, Y] >= self.world.y_semidim
        is_out_or_touching_perimeter += agent.state.pos[:, Y] <= -self.world.y_semidim
        return is_out_or_touching_perimeter

    def observation(self, agent: Agent):
        if self.observe_joint_angle:
            joint_angle = self.joint.landmark.state.rot
            angle_noise = (
                torch.randn(
                    *joint_angle.shape, device=self.world.device, dtype=torch.float32
                )
                * self.joint_angle_obs_noise
                if self.joint_angle_obs_noise
                else 0.0
            )
            joint_angle += angle_noise

        observations = [
            agent.state.pos,
            agent.state.vel,
            agent.state.pos - self.goal.state.pos,
            agent.state.pos - self.big_passage_pos,
            agent.state.pos - self.small_passage_pos,
            angle_to_vector(self.goal.state.rot),
        ] + ([angle_to_vector(joint_angle)] if self.observe_joint_angle else [])

        if self.obs_noise > 0:
            for i, obs in enumerate(observations):
                noise = torch.zeros(*obs.shape, device=self.world.device,).uniform_(
                    -self.obs_noise,
                    self.obs_noise,
                )
                observations[i] = obs + noise
        return torch.cat(
            observations,
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
        is_first = self.world.agents[0] == agent
        if is_first:
            just_passed = self.all_passed * (self.passed == 0)
            self.passed[just_passed] = 100
            self.info_stored = {
                "pos_rew": self.pos_rew,
                "rot_rew": self.rot_rew,
                "collision_rew": self.collision_rew,
                "energy_rew": self.energy_rew,
                "passed": just_passed.to(torch.int),
            }
        return self.info_stored

    def create_passage_map(self, world: World):
        # Add landmarks
        self.passages = []
        self.collide_passages = []
        self.non_collide_passages = []
        self.n_passages = 3

        def is_passage(i):
            return i < self.n_passages

        for i in range(self.n_boxes):
            passage = Landmark(
                name=f"passage {i}",
                collide=not is_passage(i),
                movable=False,
                shape=Box(length=self.passage_length, width=self.passage_width),
                color=Color.RED,
                collision_filter=lambda e: not isinstance(e.shape, Box),
            )
            if not passage.collide:
                self.non_collide_passages.append(passage)
            else:
                self.collide_passages.append(passage)
            self.passages.append(passage)
            world.add_landmark(passage)

    def spawn_passage_map(self, env_index):
        if self.fixed_passage:
            big_passage_start_index = torch.full(
                (self.world.batch_dim,) if env_index is None else (1,),
                5,
                device=self.world.device,
            )
            small_left_or_right = torch.full(
                (self.world.batch_dim,) if env_index is None else (1,),
                1,
                device=self.world.device,
            )
        else:
            big_passage_start_index = torch.randint(
                0,
                self.n_boxes - 1,
                (self.world.batch_dim,) if env_index is None else (1,),
                device=self.world.device,
            )
            small_left_or_right = torch.randint(
                0,
                2,
                (self.world.batch_dim,) if env_index is None else (1,),
                device=self.world.device,
            )

        small_left_or_right[big_passage_start_index > self.n_boxes - 1 - 4] = 0
        small_left_or_right[big_passage_start_index < 3] = 1
        small_left_or_right[small_left_or_right == 0] -= 3
        small_left_or_right[small_left_or_right == 1] += 3

        def is_passage(i):
            is_pass = big_passage_start_index == i
            is_pass += big_passage_start_index == i - 1
            is_pass += big_passage_start_index + small_left_or_right == i
            return is_pass

        def get_pos(i):
            pos = torch.tensor(
                [
                    -1 - self.agent_radius + self.passage_length / 2,
                    0.0,
                ],
                dtype=torch.float32,
                device=self.world.device,
            ).repeat(i.shape[0], 1)
            pos[:, X] += self.passage_length * i
            return pos

        for index, i in enumerate(
            [
                big_passage_start_index,
                big_passage_start_index + 1,
                big_passage_start_index + small_left_or_right,
            ]
        ):
            self.non_collide_passages[index].is_rendering[:] = False
            self.non_collide_passages[index].set_pos(get_pos(i), batch_index=env_index)

        big_passage_pos = (
            get_pos(big_passage_start_index) + get_pos(big_passage_start_index + 1)
        ) / 2
        small_passage_pos = get_pos(big_passage_start_index + small_left_or_right)
        pass_center = (big_passage_pos + small_passage_pos) / 2

        if env_index is None:
            self.small_left_or_right = small_left_or_right
            self.pass_center = pass_center
            self.big_passage_pos = big_passage_pos
            self.small_passage_pos = small_passage_pos
            self.middle_angle[small_left_or_right > 0] = torch.pi
            self.middle_angle[small_left_or_right < 0] = 0
        else:
            self.pass_center[env_index] = pass_center
            self.small_left_or_right[env_index] = small_left_or_right
            self.big_passage_pos[env_index] = big_passage_pos
            self.small_passage_pos[env_index] = small_passage_pos
            self.middle_angle[env_index] = (
                0 if small_left_or_right.item() < 0 else torch.pi
            )

        i = torch.zeros(
            (self.world.batch_dim,) if env_index is None else (1,),
            dtype=torch.int,
            device=self.world.device,
        )
        for passage in self.collide_passages:
            is_pass = is_passage(i)
            while is_pass.any():
                i[is_pass] += 1
                is_pass = is_passage(i)
            passage.set_pos(get_pos(i), batch_index=env_index)
            i += 1

    def spawn_walls(self, env_index):
        for i, wall in enumerate(self.walls):
            wall.set_pos(
                torch.tensor(
                    [
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
                            self.world.y_semidim + self.agent_radius
                            if i == 1
                            else -self.world.y_semidim - self.agent_radius
                        ),
                    ],
                    device=self.world.device,
                ),
                batch_index=env_index,
            )
            wall.set_rot(
                torch.tensor(
                    [torch.pi / 2 if not i % 2 else 0.0], device=self.world.device
                ),
                batch_index=env_index,
            )

    def extra_render(self, env_index: int = 0):
        from vmas.simulator import rendering

        geoms = []

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
    render_interactively(__file__, control_two_agents=True)
