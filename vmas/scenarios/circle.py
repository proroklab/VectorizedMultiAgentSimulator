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
        self.speed_shaping_factor = kwargs.get("speed_shaping_factor", 1)
        self.use_velocity_traj = kwargs.get("use_velocity_traj", False)

        self.dt = 0.05

        self.agent_radius = 0.03
        self.desired_speed = 1
        self.desired_radius = 0.5

        # Make world
        world = World(batch_dim, device, drag=0.25, linear_friction=0.1)
        # Add agents
        self.agent = Agent(
            name=f"agent", shape=Sphere(self.agent_radius), mass=2, f_range=15
        )
        self.agent.controller = VelocityController(
            self.agent, world.dt, [1, 0.1, 0.01], "standard"
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

        if env_index is None:
            self.t = torch.zeros(
                (self.world.batch_dim,),
                device=self.world.device,
                dtype=torch.float32,
            )

            self.pos_shaping = (
                torch.linalg.vector_norm(
                    self.agent.state.pos
                    - self.get_closest_point_circle(self.world.agents[0]),
                    dim=1,
                )
                ** 0.5
                * self.pos_shaping_factor
            )

            self.speed_shaping = (
                self.desired_speed
                - torch.linalg.vector_norm(self.agent.state.vel, dim=1)
            ).abs() * self.speed_shaping_factor
        else:
            self.t[env_index] = 0

            self.pos_shaping[env_index] = (
                torch.linalg.vector_norm(
                    self.agent.state.pos[env_index]
                    - self.get_closest_point_circle(self.world.agents[0])[env_index]
                )
                ** 0.5
                * self.pos_shaping_factor
            )

            self.speed_shaping[env_index] = (
                self.desired_speed
                - torch.linalg.vector_norm(self.agent.state.vel[env_index])
            ).abs() * self.speed_shaping_factor

    def reward(self, agent: Agent):

        self.t += self.dt

        if self.use_velocity_traj:
            reward = (
                agent.state.vel[:, X] - self.get_velocity_trajectory()[:, X]
            ).abs()
            reward += (
                agent.state.vel[:, Y] - self.get_velocity_trajectory()[:, Y]
            ).abs()
            return -reward
        else:
            self.pos_rew = torch.zeros(self.world.batch_dim, device=self.world.device)
            self.speed_rew = torch.zeros(self.world.batch_dim, device=self.world.device)

            pos_shaping = (
                torch.linalg.vector_norm(
                    self.agent.state.pos - self.get_closest_point_circle(agent), dim=1
                )
                ** 0.5
            ) * self.pos_shaping_factor
            self.pos_rew += self.pos_shaping - pos_shaping
            self.pos_shaping = pos_shaping

            speed = torch.linalg.vector_norm(self.agent.state.vel, dim=1)
            speed_shaping = (
                self.desired_speed - speed
            ).abs() * self.speed_shaping_factor
            self.speed_rew += self.speed_shaping - speed_shaping
            self.speed_shaping = speed_shaping

            return self.pos_rew + self.speed_rew

    def get_closest_point_circle(self, agent: Agent):
        pos_norm = torch.linalg.vector_norm(agent.state.pos, dim=1)
        agent_pos_normalized = agent.state.pos / pos_norm.unsqueeze(-1)

        agent_pos_normalized *= self.desired_radius

        return torch.nan_to_num(agent_pos_normalized)

    def get_velocity_trajectory(self):
        return 0.1 * torch.stack([torch.cos(self.t), torch.sin(self.t)], dim=1)

    def observation(self, agent: Agent):
        distance_to_circle = agent.state.pos - self.get_closest_point_circle(agent)
        rotated_vector = World._rotate_vector(
            distance_to_circle, torch.tensor(torch.pi, device=self.world.device)
        )
        angle = rotated_vector / torch.linalg.vector_norm(
            rotated_vector, dim=1
        ).unsqueeze(-1)
        angle = torch.nan_to_num(angle)
        return torch.cat(
            [agent.state.pos, agent.state.vel, agent.state.pos]
            + ([self.get_velocity_trajectory()] if self.use_velocity_traj else []),
            dim=-1,
        )

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        return (
            {
                "pos_rew": self.pos_rew,
                "speed_rew": self.speed_rew,
            }
            if not self.use_velocity_traj
            else {}
        )

    def extra_render(self, env_index: int = 0):
        from vmas.simulator import rendering

        geoms = []
        # Trajectory goal circle
        color = Color.BLACK.value
        circle = rendering.make_circle(self.desired_radius, filled=False)
        xform = rendering.Transform()
        circle.add_attr(xform)
        xform.set_translation(0, 0)
        circle.set_color(*color)
        geoms.append(circle)

        # Trajectory vel
        if self.use_velocity_traj:
            color = Color.BLACK.value
            circle = rendering.Line(
                (0, 0),
                (
                    self.get_velocity_trajectory()[env_index][X],
                    self.get_velocity_trajectory()[env_index][Y],
                ),
                width=1,
            )
            xform = rendering.Transform()
            circle.add_attr(xform)
            circle.set_color(*color)
            geoms.append(circle)

        return geoms


if __name__ == "__main__":
    render_interactively("circle")
