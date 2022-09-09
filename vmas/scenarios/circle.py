#  Copyright (c) 2022.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
from typing import Dict

import torch
from torch import Tensor

from vmas import render_interactively
from vmas.simulator.core import Agent, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, X, Y
from vmas.simulator.velocity_controller import VelocityController


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.pos_shaping_factor = kwargs.get("pos_shaping_factor", 1)
        self.pos_shaping_factor2 = kwargs.get("pos_shaping_factor2", 0)
        self.obs_noise = kwargs.get("obs_noise", 0)

        self.agent_radius = 0.03
        self.desired_speed = 1
        self.desired_radius = 0.5

        # Make world
        world = World(batch_dim, device, drag=0.25, linear_friction=0.1)
        # Add agents
        self.agent = Agent(
            name=f"agent",
            shape=Sphere(self.agent_radius),
            mass=2,
            f_range=30,
            u_range=0.5,
        )
        self.agent.controller = VelocityController(
            self.agent, world, [0.5, 0.5, 0.01], "standard"
        )
        world.add_agent(self.agent)

        return world

    def process_action(self, agent: Agent):
        # print( agent.state.vel );
        agent.controller.process_force()

    def reset_world_at(self, env_index: int = None):
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

        self.next_point = self.get_next_closest_point_circle(self.world.agents[0])

        if env_index is None:

            self.pos_shaping = (
                torch.linalg.vector_norm(
                    self.agent.state.pos - self.next_point,
                    dim=1,
                )
                ** 0.5
                * self.pos_shaping_factor
            )

            self.pos_shaping_2 = (
                torch.linalg.vector_norm(
                    self.agent.state.pos
                    - self.get_closest_point_circle(self.world.agents[0]),
                    dim=1,
                )
                ** 0.5
                * self.pos_shaping_factor2
            )

        else:

            self.pos_shaping[env_index] = (
                torch.linalg.vector_norm(
                    self.agent.state.pos[env_index] - self.next_point[env_index]
                )
                ** 0.5
                * self.pos_shaping_factor
            )

            self.pos_shaping_2[env_index] = (
                torch.linalg.vector_norm(
                    self.agent.state.pos[env_index]
                    - self.get_closest_point_circle(self.world.agents[0])[env_index],
                )
                ** 0.5
                * self.pos_shaping_factor2
            )

    def reward(self, agent: Agent):

        pos_shaping = (
            torch.linalg.vector_norm(self.agent.state.pos - self.next_point, dim=1)
        ) ** 0.5 * self.pos_shaping_factor
        self.pos_rew = self.pos_shaping - pos_shaping
        self.pos_shaping = pos_shaping
        self.next_point = self.get_next_closest_point_circle(agent)

        pos_shaping2 = (
            torch.linalg.vector_norm(
                self.agent.state.pos - self.get_closest_point_circle(agent), dim=1
            )
        ) ** 0.5 * self.pos_shaping_factor2
        self.pos_rew2 = self.pos_shaping_2 - pos_shaping2
        self.pos_shaping_2 = pos_shaping2

        # speed = torch.linalg.vector_norm(self.agent.state.vel, dim=1)
        # speed_shaping = (
        #     self.desired_speed - speed
        # ).abs() * self.speed_shaping_factor
        # self.speed_rew += self.speed_shaping - speed_shaping
        # self.speed_shaping = speed_shaping

        return self.pos_rew + self.pos_rew2

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

    def observation(self, agent: Agent):
        distance_to_circle = agent.state.pos - self.get_closest_point_circle(agent)
        rotated_vector = World._rotate_vector(
            distance_to_circle, torch.tensor(torch.pi, device=self.world.device)
        )
        angle = rotated_vector / torch.linalg.vector_norm(
            rotated_vector, dim=1
        ).unsqueeze(-1)
        angle = torch.nan_to_num(angle)
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
            "pos_rew2": self.pos_rew2,
        }

    def extra_render(self, env_index: int = 0):
        from vmas.simulator import rendering

        geoms = []

        agent = self.world.agents[0]
        color = Color.BLACK.value
        circle = rendering.make_circle(self.agent_radius, filled=True)
        xform = rendering.Transform()
        circle.add_attr(xform)
        point = self.get_next_closest_point_circle(agent)
        xform.set_translation(*point[env_index])
        circle.set_color(*color)
        geoms.append(circle)

        # Trajectory goal circle
        color = Color.BLACK.value
        circle = rendering.make_circle(self.desired_radius, filled=False)
        xform = rendering.Transform()
        circle.add_attr(xform)
        xform.set_translation(0, 0)
        circle.set_color(*color)
        geoms.append(circle)

        # Trajectory vel
        # if self.use_velocity_traj:
        #     color = Color.BLACK.value
        #     circle = rendering.Line(
        #         (0, 0),
        #         (
        #             self.get_velocity_trajectory()[env_index][X],
        #             self.get_velocity_trajectory()[env_index][Y],
        #         ),
        #         width=1,
        #     )
        #     xform = rendering.Transform()
        #     circle.add_attr(xform)
        #     circle.set_color(*color)
        #     geoms.append(circle)

        return geoms


if __name__ == "__main__":
    render_interactively("circle")
