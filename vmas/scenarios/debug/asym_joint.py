#  Copyright (c) 2022-2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import math
import typing
from typing import Dict, List

import torch
from torch import Tensor

from vmas import render_interactively
from vmas.simulator.core import Agent, Landmark, Sphere, World
from vmas.simulator.joints import Joint
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom


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
        self.joint_length = kwargs.get("joint_length", 0.5)
        self.random_start_angle = kwargs.get("random_start_angle", False)
        self.observe_joint_angle = kwargs.get("observe_joint_angle", False)
        self.joint_angle_obs_noise = kwargs.get("joint_angle_obs_noise", 0.0)
        self.asym_package = kwargs.get("asym_package", True)
        self.mass_ratio = kwargs.get("mass_ratio", 5)
        self.mass_position = kwargs.get("mass_position", 0.75)
        self.max_speed_1 = kwargs.get("max_speed_1", None)  # 0.1
        self.obs_noise = kwargs.get("obs_noise", 0.2)

        # Reward
        self.rot_shaping_factor = kwargs.get("rot_shaping_factor", 1)
        self.energy_reward_coeff = kwargs.get("energy_reward_coeff", 0.08)

        # Make world
        world = World(
            batch_dim,
            device,
            substeps=7 if not self.asym_package else 10,
            joint_force=900 if self.asym_package else 400,
            drag=0.25 if not self.asym_package else 0.15,
        )

        if not self.observe_joint_angle:
            assert self.joint_angle_obs_noise == 0

        self.goal_angle = torch.pi / 2

        self.n_agents = 2

        self.agent_radius = 0.03333
        self.mass_radius = self.agent_radius * (2 / 3)

        # Add agents
        agent = Agent(
            name=f"agent 0",
            shape=Sphere(self.agent_radius),
            u_multiplier=0.8,
            obs_noise=self.obs_noise,
            render_action=True,
        )
        world.add_agent(agent)
        agent = Agent(
            name=f"agent 1",
            shape=Sphere(self.agent_radius),
            u_multiplier=0.8,
            mass=1 if self.asym_package else self.mass_ratio,
            max_speed=self.max_speed_1,
            obs_noise=self.obs_noise,
            render_action=True,
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
                collide=False,
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

        self.rot_rew = torch.zeros(batch_dim, device=device)
        self.energy_rew = self.rot_rew.clone()

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

        start_delta_x = (self.joint_length / 2) * torch.cos(start_angle)
        start_delta_x_abs = start_delta_x.abs()
        min_x_start = 0
        max_x_start = 0
        start_delta_y = (self.joint_length / 2) * torch.sin(start_angle)
        start_delta_y_abs = start_delta_y.abs()
        min_y_start = 0
        max_y_start = 0

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

        order = torch.randperm(self.n_agents).tolist()
        agents = [self.world.agents[i] for i in order]
        for i, agent in enumerate(agents):
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

        if env_index is None:

            self.joint.rot_shaping_pre = (
                get_line_angle_dist_0_180(
                    self.joint.landmark.state.rot, self.goal_angle
                )
                * self.rot_shaping_factor
            )

        else:

            self.joint.rot_shaping_pre[env_index] = (
                get_line_angle_dist_0_180(
                    self.joint.landmark.state.rot[env_index], self.goal_angle
                )
                * self.rot_shaping_factor
            )

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            self.rot_rew[:] = 0

            # Rot shaping
            joint_dist_to_90_rot = get_line_angle_dist_0_180(
                self.joint.landmark.state.rot, self.goal_angle
            )
            joint_shaping = joint_dist_to_90_rot * self.rot_shaping_factor
            self.rot_rew += self.joint.rot_shaping_pre - joint_shaping
            self.joint.rot_shaping_pre = joint_shaping

            # Energy reward
            self.energy_expenditure = torch.stack(
                [
                    torch.linalg.vector_norm(a.action.u, dim=-1)
                    / math.sqrt(self.world.dim_p * ((a.u_range * a.u_multiplier) ** 2))
                    for a in self.world.agents
                ],
                dim=1,
            ).sum(-1)
            self.energy_rew = -self.energy_expenditure * self.energy_reward_coeff

            self.rew = self.rot_rew + self.energy_rew

        return self.rew

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
        ] + ([angle_to_vector(joint_angle)] if self.observe_joint_angle else [])

        for i, obs in enumerate(observations):
            noise = torch.zeros(*obs.shape, device=self.world.device,).uniform_(
                -self.obs_noise,
                self.obs_noise,
            )
            observations[i] = obs.clone() + noise
        return torch.cat(
            observations,
            dim=-1,
        )

    def done(self):
        return torch.all(
            (
                get_line_angle_dist_0_180(
                    self.joint.landmark.state.rot, self.goal_angle
                ).unsqueeze(-1)
                <= 0.01
            ),
            dim=1,
        )

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        return {
            "rot_rew": self.rot_rew,
            "energy_rew": self.energy_rew,
        }

    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        from vmas.simulator import rendering

        geoms = []

        color = Color.GREEN.value
        origin = rendering.make_circle(0.01)
        xform = rendering.Transform()
        origin.add_attr(xform)
        xform.set_translation(0, 0)
        origin.set_color(*color)
        geoms.append(origin)

        return geoms


if __name__ == "__main__":
    render_interactively(__file__, control_two_agents=True)
