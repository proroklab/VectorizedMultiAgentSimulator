#  Copyright (c) 2022-2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
from typing import Dict

import torch
from torch import Tensor

from vmas import render_interactively
from vmas.simulator.core import Agent, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, X, Y, TorchUtils
from vmas.simulator.velocity_controller import VelocityController


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.u_range = kwargs.get("u_range", 1)
        self.a_range = kwargs.get("a_range", 1)
        self.obs_noise = kwargs.get("obs_noise", 0.0)
        self.dt_delay = kwargs.get("dt_delay", 0)
        self.min_input_norm = kwargs.get("min_input_norm", 0.08)
        self.linear_friction = kwargs.get("linear_friction", 0.1)

        self.agent_radius = 0.16
        self.desired_radius = 1.5

        self.viewer_zoom = 2

        # Make world
        world = World(
            batch_dim, device, linear_friction=self.linear_friction, dt=0.05, drag=0
        )

        controller_params = [2, 6, 0.002]

        self.f_range = self.a_range + self.linear_friction

        null_action = torch.zeros(world.batch_dim, world.dim_p, device=world.device)
        self.input_queue = [null_action.clone() for _ in range(self.dt_delay)]
        # control delayed by n dts

        # Add agents
        self.agent = Agent(
            name=f"agent",
            shape=Sphere(self.agent_radius),
            f_range=self.f_range,
            u_range=self.u_range,
            render_action=True,
        )
        self.agent.controller = VelocityController(
            self.agent, world, controller_params, "standard"
        )
        world.add_agent(self.agent)

        self.pos_rew = torch.zeros(batch_dim, device=device)
        self.dot_product = self.pos_rew.clone()

        return world

    def process_action(self, agent: Agent):
        # Use queue for delay
        self.input_queue.append(agent.action.u.clone())
        agent.action.u = self.input_queue.pop(0)

        # Clamp square to circle
        agent.action.u = TorchUtils.clamp_with_norm(agent.action.u, self.u_range)

        # Zero small input
        action_norm = torch.linalg.vector_norm(agent.action.u, dim=1)
        agent.action.u[action_norm < self.min_input_norm] = 0

        agent.vel_action = agent.action.u.clone()
        agent.controller.process_force()

    def reset_world_at(self, env_index: int = None):
        self.agent.controller.reset(env_index)
        self.agent.set_pos(
            torch.zeros(
                (1, self.world.dim_p)
                if env_index is not None
                else (self.world.batch_dim, self.world.dim_p),
                device=self.world.device,
                dtype=torch.float32,
            ).uniform_(
                -self.desired_radius,
                self.desired_radius,
            ),
            batch_index=env_index,
        )

    def reward(self, agent: Agent):
        closest_point = self.get_closest_point_circle(agent)
        self.pos_rew = (
            -(torch.linalg.vector_norm(agent.state.pos - closest_point, dim=1) ** 0.5)
            * 1
        )

        tangent = self.get_tangent_to_circle(agent, closest_point)
        self.dot_product = torch.einsum("bs,bs->b", tangent, agent.state.vel) * 0.5

        return self.pos_rew + self.dot_product

    def get_closest_point_circle(self, agent: Agent):
        pos_norm = torch.linalg.vector_norm(agent.state.pos, dim=1)
        agent_pos_normalized = agent.state.pos / pos_norm.unsqueeze(-1)

        agent_pos_normalized *= self.desired_radius

        return torch.nan_to_num(agent_pos_normalized)

    def get_next_closest_point_circle(self, agent: Agent):
        closest_point = self.get_closest_point_circle(agent)
        angle = torch.atan2(closest_point[:, Y], closest_point[:, X])
        angle += torch.pi / 24
        new_point = (
            torch.stack([torch.cos(angle), torch.sin(angle)], dim=1)
            * self.desired_radius
        )
        return new_point

    def get_tangent_to_circle(self, agent: Agent, closest_point=None):
        if closest_point is None:
            closest_point = self.get_closest_point_circle(agent)
        distance_to_circle = agent.state.pos - closest_point
        inside_circle = (
            torch.linalg.vector_norm(agent.state.pos, dim=1) < self.desired_radius,
        )

        rotated_vector = TorchUtils.rotate_vector(
            distance_to_circle, torch.tensor(torch.pi / 2, device=self.world.device)
        )
        rotated_vector[inside_circle] = TorchUtils.rotate_vector(
            distance_to_circle[inside_circle],
            torch.tensor(-torch.pi / 2, device=self.world.device),
        )
        angle = rotated_vector / torch.linalg.vector_norm(
            rotated_vector, dim=1
        ).unsqueeze(-1)
        angle = torch.nan_to_num(angle)
        return angle

    def observation(self, agent: Agent):
        observations = [agent.state.pos, agent.state.vel, agent.state.pos]
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

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        return {
            "pos_rew": self.pos_rew,
            "dot_product": self.dot_product,
        }

    def extra_render(self, env_index: int = 0):
        from vmas.simulator import rendering

        geoms = []

        # agent = self.world.agents[0]
        # color = Color.BLACK.value
        # circle = rendering.make_circle(self.agent_radius, filled=True)
        # xform = rendering.Transform()
        # circle.add_attr(xform)
        # point = self.get_next_closest_point_circle(agent)
        # xform.set_translation(*point[env_index])
        # circle.set_color(*color)
        # geoms.append(circle)

        # Trajectory goal circle
        color = Color.BLACK.value
        circle = rendering.make_circle(self.desired_radius, filled=False)
        xform = rendering.Transform()
        circle.add_attr(xform)
        xform.set_translation(0, 0)
        circle.set_color(*color)
        geoms.append(circle)

        # Trajectory vel
        tangent = self.get_tangent_to_circle(self.agent)
        color = Color.BLACK.value
        circle = rendering.Line(
            (0, 0),
            (tangent[env_index]),
            width=1,
        )
        xform = rendering.Transform()
        circle.add_attr(xform)
        circle.set_color(*color)
        geoms.append(circle)

        return geoms


if __name__ == "__main__":
    render_interactively(__file__)
