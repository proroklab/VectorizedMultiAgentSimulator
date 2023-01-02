#  Copyright (c) 2022-2023.
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
        self.obs_noise = kwargs.get("obs_noise", 0)

        self.agent_radius = 0.03
        self.line_length = 3

        # Make world
        world = World(batch_dim, device, drag=0.1)
        # Add agents
        self.agent = Agent(
            name=f"agent",
            shape=Sphere(self.agent_radius),
            mass=2,
            f_range=0.5,
            u_range=1,
            render_action=True,
        )
        self.agent.controller = VelocityController(
            self.agent, world, [4, 1.25, 0.001], "standard"
        )
        world.add_agent(self.agent)

        self.tangent = torch.zeros((world.batch_dim, world.dim_p), device=world.device)
        self.tangent[:, Y] = 1

        self.pos_rew = torch.zeros(batch_dim, device=device)
        self.dot_product = self.pos_rew.clone()
        self.steady_rew = self.pos_rew.clone()

        return world

    def process_action(self, agent: Agent):
        self.vel_action = agent.action.u.clone()
        agent.controller.process_force()

    def reset_world_at(self, env_index: int = None):
        self.agent.controller.reset(env_index)
        self.agent.set_pos(
            torch.cat(
                [
                    torch.zeros(
                        (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                        device=self.world.device,
                        dtype=torch.float32,
                    ).uniform_(
                        -1,
                        1,
                    ),
                    torch.zeros(
                        (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                        device=self.world.device,
                        dtype=torch.float32,
                    ).uniform_(
                        -1,
                        0,
                    ),
                ],
                dim=1,
            ),
            batch_index=env_index,
        )

    def reward(self, agent: Agent):
        closest_point = agent.state.pos.clone()
        closest_point[:, X] = 0
        self.pos_rew = (
            -(torch.linalg.vector_norm(agent.state.pos - closest_point, dim=1) ** 0.5)
            * 1
        )
        self.dot_product = torch.einsum("bs,bs->b", self.tangent, agent.state.vel) * 0.5

        normalized_vel = agent.state.vel / torch.linalg.vector_norm(
            agent.state.vel, dim=1
        ).unsqueeze(-1)
        normalized_vel = torch.nan_to_num(normalized_vel)

        normalized_vel_action = self.vel_action / torch.linalg.vector_norm(
            self.vel_action, dim=1
        ).unsqueeze(-1)
        normalized_vel_action = torch.nan_to_num(normalized_vel_action)

        self.steady_rew = (
            torch.einsum("bs,bs->b", normalized_vel, normalized_vel_action) * 0.2
        )

        return self.pos_rew + self.dot_product + self.steady_rew

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

    def done(self):
        return self.world.agents[0].state.pos[:, Y] > self.line_length - 1

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        return {
            "pos_rew": self.pos_rew,
            "dot_product": self.dot_product,
            "steady_rew": self.steady_rew,
        }

    def extra_render(self, env_index: int = 0):
        from vmas.simulator import rendering

        geoms = []

        # Trajectory goal circle
        color = Color.BLACK.value
        line = rendering.Line(
            (0, -1),
            (0, -1 + self.line_length),
            width=1,
        )
        line.set_color(*color)
        geoms.append(line)

        return geoms


if __name__ == "__main__":
    render_interactively(__file__)
