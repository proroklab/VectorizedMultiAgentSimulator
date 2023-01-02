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


def angle_to_vector(angle):
    return torch.cat([torch.cos(angle), torch.sin(angle)], dim=1)


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.n_passages = kwargs.get("n_passages", 1)
        self.fixed_passage = kwargs.get("fixed_passage", True)
        self.joint_length = kwargs.get("joint_length", 0.5)
        self.random_start_angle = kwargs.get("random_start_angle", True)
        self.random_goal_angle = kwargs.get("random_goal_angle", True)
        self.observe_joint_angle = kwargs.get("observe_joint_angle", False)
        self.joint_angle_obs_noise = kwargs.get("joint_angle_obs_noise", 0.0)
        self.asym_package = kwargs.get("asym_package", True)
        self.mass_ratio = kwargs.get("mass_ratio", 5)
        self.mass_position = kwargs.get("mass_position", 0.75)
        self.max_speed_1 = kwargs.get("max_speed_1", None)  # 0.1
        self.pos_shaping_factor = kwargs.get("pos_shaping_factor", 1)
        self.rot_shaping_factor = kwargs.get("rot_shaping_factor", 1)
        self.collision_reward = kwargs.get("collision_reward", 0)
        self.energy_reward_coeff = kwargs.get("energy_reward_coeff", 0)
        self.all_passed_rot = kwargs.get("all_passed_rot", True)
        self.obs_noise = kwargs.get("obs_noise", 0.0)
        self.use_controller = kwargs.get("use_controller", False)

        self.plot_grid = True
        # Make world
        world = World(
            batch_dim,
            device,
            x_semidim=1,
            y_semidim=1,
            substeps=7 if not self.asym_package else 10,
            joint_force=900 if self.asym_package else 400,
            collision_force=2500 if self.asym_package else 1500,
            drag=0.25 if not self.asym_package else 0.15,
        )

        if not self.observe_joint_angle:
            assert self.joint_angle_obs_noise == 0

        self.middle_angle = torch.pi / 2

        self.n_agents = 2

        self.agent_radius = 0.03333
        self.mass_radius = self.agent_radius * (2 / 3)
        self.passage_width = 0.2
        self.passage_length = 0.1476
        self.scenario_length = 2 * world.x_semidim + 2 * self.agent_radius
        self.n_boxes = int(self.scenario_length // self.passage_length)
        self.min_collision_distance = 0.005

        assert 1 <= self.n_passages <= int(self.scenario_length // self.passage_length)

        cotnroller_params = [2.0, 10, 0.00001]

        # Add agents
        agent = Agent(
            name="agent 0",
            shape=Sphere(self.agent_radius),
            obs_noise=self.obs_noise,
            render_action=True,
            u_multiplier=0.8,
            f_range=0.8,
        )
        agent.controller = VelocityController(
            agent, world, cotnroller_params, "standard"
        )
        world.add_agent(agent)

        agent = Agent(
            name="agent 1",
            shape=Sphere(self.agent_radius),
            mass=1 if self.asym_package else self.mass_ratio,
            color=Color.BLUE,
            max_speed=self.max_speed_1,
            obs_noise=self.obs_noise,
            render_action=True,
            u_multiplier=0.8,
            f_range=0.8,
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
            collidable=True,
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

        order = torch.randperm(self.n_agents).tolist()
        agents = [self.world.agents[i] for i in order]
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
            self.passed[env_index] = 0

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
            self.pos_rew[:] = 0
            self.rot_rew[:] = 0
            self.collision_rew[:] = 0

            joint_passed = self.joint.landmark.state.pos[:, Y] > 0
            self.all_passed = (
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
            rot_passed = self.all_passed if self.all_passed_rot else joint_passed
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

            # Joint collisions
            for i, p in enumerate(self.passages):
                if p.collide:
                    self.collision_rew[
                        self.world.get_distance(p, self.joint.landmark)
                        <= self.min_collision_distance
                    ] += self.collision_reward

            # Energy reward
            self.energy_expenditure = torch.stack(
                [
                    torch.linalg.vector_norm(a.action.u, dim=-1)
                    / math.sqrt(self.world.dim_p * (a.f_range**2))
                    for a in self.world.agents
                ],
                dim=1,
            ).sum(-1)
            self.energy_rew = -self.energy_expenditure * self.energy_reward_coeff

            self.rew = (
                self.pos_rew + self.rot_rew + self.collision_rew + self.energy_rew
            )

        return self.rew

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

        # get positions of all entities in this agent's reference frame
        passage_obs = []
        for passage in self.passages:
            if not passage.collide:
                passage_obs.append(agent.state.pos - passage.state.pos)

        observations = [
            agent.state.pos,
            agent.state.vel,
            agent.state.pos - self.goal.state.pos,
            *passage_obs,
            angle_to_vector(self.goal.state.rot),
        ] + ([angle_to_vector(joint_angle)] if self.observe_joint_angle else [])

        for i, obs in enumerate(observations):
            noise = torch.zeros(*obs.shape, device=self.world.device,).uniform_(
                -self.obs_noise,
                self.obs_noise,
            )
            # noise = (
            #     torch.randn(*obs.shape, device=self.world.device, dtype=torch.float32)
            #     * agent.obs_noise
            #     if agent.obs_noise
            #     else 0.0
            # )
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

    def process_action(self, agent: Agent):
        if self.use_controller:
            vel_is_zero = torch.linalg.vector_norm(agent.action.u, dim=1) < 1e-3
            agent.controller.reset(vel_is_zero)
            agent.controller.process_force()

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

        def removed(i):
            return (
                (self.n_boxes // 2) - self.n_passages / 2
                <= i
                < (self.n_boxes // 2) + self.n_passages / 2
            )

        for i in range(self.n_boxes):
            passage = Landmark(
                name=f"passage {i}",
                collide=not removed(i),
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

        def joint_collides(e):
            if e in self.collide_passages and self.fixed_passage:
                return e.neighbour
            elif e in self.collide_passages:
                return True
            return False

        self.joint.landmark.collision_filter = joint_collides

    def spawn_passage_map(self, env_index):

        passage_indexes = []
        j = self.n_boxes // 2
        for i in range(self.n_passages):
            if self.fixed_passage:
                j += i * (-1 if i % 2 == 0 else 1)
                passage_index = torch.full(
                    (self.world.batch_dim,) if env_index is None else (1,),
                    j,
                    device=self.world.device,
                )
            else:
                passage_index = torch.randint(
                    0,
                    self.n_boxes - 1,
                    (self.world.batch_dim,) if env_index is None else (1,),
                    device=self.world.device,
                )
            passage_indexes.append(passage_index)

        def is_passage(i):
            is_pass = torch.full(i.shape, False, device=self.world.device)
            for index in passage_indexes:
                is_pass += i == index
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

        for index, i in enumerate(passage_indexes):
            self.non_collide_passages[index].is_rendering[:] = False
            self.non_collide_passages[index].set_pos(get_pos(i), batch_index=env_index)

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
            if self.fixed_passage:
                passage.neighbour = (is_passage(i - 1) + is_passage(i + 1)).all()
            elif env_index is None:
                passage.neighbour = is_passage(i - 1) + is_passage(i + 1)
            else:
                passage.neighbour[env_index] = is_passage(i - 1) + is_passage(i + 1)
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

        # for passage in self.passages:
        #     color = Color.BLACK.value
        #     try:
        #         if passage.neighbour[env_index]:
        #             color = Color.BLUE.value
        #     except AttributeError:
        #         pass
        #
        #     p = rendering.make_circle(self.agent_radius)
        #     xform = rendering.Transform()
        #     p.add_attr(xform)
        #     xform.set_translation(*passage.state.pos[env_index])
        #     p.set_color(*color)
        #     geoms.append(p)

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
