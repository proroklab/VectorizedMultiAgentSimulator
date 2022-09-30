#  Copyright (c) 2022.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import math
from typing import Dict

import torch
from torch import Tensor

from vmas import render_interactively
from vmas.simulator.core import Agent, World, Landmark, Sphere
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, clamp_with_norm
from vmas.simulator.velocity_controller import VelocityController


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.u_range = kwargs.get("u_range", 1)
        self.f_range = kwargs.get("f_range", 1.5)
        self.obs_noise = kwargs.get("obs_noise", 0.0)
        self.dt_delay = kwargs.get("dt_delay", 0)
        self.min_input_norm = kwargs.get("min_input_norm", 0.08)

        self.pos_shaping_factor = kwargs.get("pos_shaping_factor", 1.0)
        self.act_shaping_factor = kwargs.get("act_shaping_factor", 0.0)
        self.energy_reward_coeff = kwargs.get("energy_rew_coeff", 0.0)

        self.viewer_size = (1600, 700)
        self.viewer_zoom = 2

        self.plot_grid = True
        self.agent_radius = 0.16

        self.lab_length = 6
        self.lab_width = 3

        controller_params = [3, 2, 0.002]

        # Make world
        world = World(batch_dim, device, linear_friction=0.5, drag=0)

        null_action = torch.zeros(world.batch_dim, world.dim_p, device=world.device)
        self.input_queue = [null_action.clone() for _ in range(self.dt_delay)]
        # control delayed by n dts

        # Goal
        self.goal = Landmark(
            "goal",
            collide=False,
            movable=False,
            shape=Sphere(radius=self.agent_radius / 2),
        )
        world.add_landmark(self.goal)
        # Add agents
        agent = Agent(
            name=f"agent 0",
            collide=True,
            color=Color.GREEN,
            render_action=True,
            shape=Sphere(radius=self.agent_radius),
            f_range=self.f_range,
            u_range=self.u_range,
        )
        agent.controller = VelocityController(
            agent, world, controller_params, "standard"
        )
        agent.goal = self.goal
        world.add_agent(agent)
        # agent = Agent(
        #     name=f"agent 1",
        #     collide=False,
        #     render_action=True,
        #     # f_range=30,
        #     u_range=1,
        # )
        # agent.controller = VelocityController(
        #     agent, world, controller_params, "standard"
        # )
        # world.add_agent(agent)
        # agent = Agent(
        #     name=f"agent 1",
        #     collide=False,
        #     render_action=True,
        #     f_range=30,
        #     u_range=1,
        # )
        # agent.controller = VelocityController(
        #     agent, world, controller_params, "standard"
        # )
        # world.add_agent(agent)

        return world

    def reset_world_at(self, env_index: int = None):
        for agent in self.world.agents:
            agent.controller.reset(env_index)
            agent.set_pos(
                torch.cat(
                    [
                        torch.zeros(
                            (1, 1)
                            if env_index is not None
                            else (self.world.batch_dim, 1),
                            device=self.world.device,
                            dtype=torch.float32,
                        ).uniform_(
                            -self.lab_length / 2,
                            self.lab_length / 2,
                        ),
                        torch.zeros(
                            (1, 1)
                            if env_index is not None
                            else (self.world.batch_dim, 1),
                            device=self.world.device,
                            dtype=torch.float32,
                        ).uniform_(
                            -self.lab_width / 2,
                            self.lab_width / 2,
                        ),
                    ],
                    dim=1,
                ),
                batch_index=env_index,
            )

        for landmark in self.world.landmarks:
            # Random pos between -1 and 1
            landmark.set_pos(
                torch.cat(
                    [
                        torch.zeros(
                            (1, 1)
                            if env_index is not None
                            else (self.world.batch_dim, 1),
                            device=self.world.device,
                            dtype=torch.float32,
                        ).uniform_(
                            -self.lab_length / 2,
                            self.lab_length / 2,
                        ),
                        torch.zeros(
                            (1, 1)
                            if env_index is not None
                            else (self.world.batch_dim, 1),
                            device=self.world.device,
                            dtype=torch.float32,
                        ).uniform_(
                            -self.lab_width / 2,
                            self.lab_width / 2,
                        ),
                    ],
                    dim=1,
                ),
                batch_index=env_index,
            )
            if env_index is None:
                landmark.eaten = torch.full(
                    (self.world.batch_dim,), False, device=self.world.device
                )
                landmark.reset_render()
                landmark.pos_shaping = (
                    torch.stack(
                        [
                            torch.linalg.vector_norm(
                                landmark.state.pos - a.state.pos, dim=1
                            )
                            for a in self.world.agents
                        ],
                        dim=1,
                    ).min(dim=1)[0]
                    * self.pos_shaping_factor
                )

            else:
                landmark.eaten[env_index] = False
                landmark.is_rendering[env_index] = True

                landmark.pos_shaping[env_index] = (
                    torch.stack(
                        [
                            torch.linalg.vector_norm(
                                landmark.state.pos[env_index] - a.state.pos[env_index],
                            ).unsqueeze(-1)
                            for a in self.world.agents
                        ],
                        dim=1,
                    ).min(dim=1)[0]
                    * self.pos_shaping_factor
                )

    def process_action(self, agent: Agent):
        # Use queue for delay
        self.input_queue.append(agent.action.u.clone())
        agent.action.u = self.input_queue.pop(0)

        # Clamp square to circle
        agent.action.u = clamp_with_norm(agent.action.u, self.u_range)

        # Zero small input
        action_norm = torch.linalg.vector_norm(agent.action.u, dim=1)
        agent.action.u[action_norm < self.min_input_norm] = 0

        agent.vel_action = agent.action.u.clone()
        agent.controller.process_force()

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            self.pos_rew = torch.zeros(self.world.batch_dim, device=self.world.device)

            # Pos shaping
            goal_dist = torch.stack(
                [
                    torch.linalg.vector_norm(self.goal.state.pos - a.state.pos, dim=1)
                    for a in self.world.agents
                ],
                dim=1,
            ).min(dim=1)[0]
            pos_shaping = goal_dist * self.pos_shaping_factor
            self.pos_rew = self.goal.pos_shaping - pos_shaping
            self.goal.pos_shaping = pos_shaping

        # Steady action
        normalized_vel_action = agent.vel_action / torch.linalg.vector_norm(
            agent.vel_action, dim=1
        ).unsqueeze(-1)
        normalized_vel_action = torch.nan_to_num(normalized_vel_action)

        normalized_dir = self.goal.state.pos - agent.state.pos
        normalized_dir /= torch.linalg.vector_norm(normalized_dir, dim=1).unsqueeze(-1)

        dot_product = torch.einsum("bs,bs->b", normalized_dir, normalized_vel_action)

        # angle = torch.arccos(dot_product.clamp(-1, 1))
        # angle[
        #     (normalized_vel == 0).all(dim=1) * (normalized_vel_action == 0).all(dim=1)
        # ] = 0
        # agent.action_rew = -(angle**2) * self.act_shaping_factor
        # Energy reward

        agent.energy_expenditure = torch.stack(
            [
                torch.linalg.vector_norm(a.action.u, dim=-1)
                / math.sqrt(self.world.dim_p * (a.f_range**2))
                for a in self.world.agents
            ],
            dim=1,
        ).sum(-1)
        agent.energy_rew = -agent.energy_expenditure * self.energy_reward_coeff

        agent.action_rew = (dot_product - 1) * self.act_shaping_factor

        return self.pos_rew + agent.action_rew + agent.energy_rew

    def observation(self, agent: Agent):
        observations = [
            agent.state.pos,
            agent.state.vel,
            agent.state.pos - self.goal.state.pos,
        ]

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

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        return {
            "pos_rew": self.pos_rew,
            "energy_rew": agent.energy_rew,
        }

    # def done(self):
    #     return torch.any(
    #         torch.stack(
    #             [
    #                 torch.linalg.vector_norm(a.state.pos - self.goal.state.pos, dim=1)
    #                 < a.shape.radius + self.goal.shape.radius
    #                 for a in self.world.agents
    #             ],
    #             dim=1,
    #         ),
    #         dim=-1,
    #     )


if __name__ == "__main__":
    render_interactively(__file__, control_two_agents=False)
