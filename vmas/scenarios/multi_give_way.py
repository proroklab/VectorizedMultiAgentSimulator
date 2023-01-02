#  Copyright (c) 2022-2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import typing
from typing import List

import torch

from vmas import render_interactively
from vmas.simulator.core import Agent, World, Landmark, Sphere, Line, Box
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, TorchUtils
from vmas.simulator.velocity_controller import VelocityController

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.u_range = kwargs.get("u_range", 0.5)
        self.a_range = kwargs.get("a_range", 1)
        self.obs_noise = kwargs.get("obs_noise", 0)
        self.box_agents = kwargs.get("box_agents", False)
        self.linear_friction = kwargs.get("linear_friction", 0.1)
        self.min_input_norm = kwargs.get("min_input_norm", 0.08)
        self.comms_range = kwargs.get("comms_range", 5)
        self.shared_rew = kwargs.get("shared_rew", True)

        self.pos_shaping_factor = kwargs.get("pos_shaping_factor", 1)  # max is 8
        self.final_reward = kwargs.get("final_reward", 0.01)
        # self.energy_reward_coeff = kwargs.get("energy_rew_coeff", 0)

        self.agent_collision_penalty = kwargs.get("agent_collision_penalty", -0.1)
        # self.passage_collision_penalty = kwargs.get("passage_collision_penalty", 0)
        # self.obstacle_collision_penalty = kwargs.get("obstacle_collision_penalty", 0)

        self.viewer_zoom = 1.7

        controller_params = [2, 6, 0.002]

        self.n_agents = 4
        self.f_range = self.a_range + self.linear_friction

        # Make world#
        world = World(
            batch_dim,
            device,
            drag=0,
            dt=0.1,
            linear_friction=self.linear_friction,
            substeps=16 if self.box_agents else 5,
            collision_force=10000 if self.box_agents else 500,
        )

        self.agent_radius = 0.16
        self.agent_box_length = 0.32
        self.agent_box_width = 0.24

        self.min_collision_distance = 0.005

        self.colors = [Color.GREEN, Color.BLUE, Color.RED, Color.GRAY]

        # Add agents
        for i in range(self.n_agents):
            agent = Agent(
                name=f"agent {i}",
                rotatable=False,
                linear_friction=self.linear_friction,
                shape=Sphere(radius=self.agent_radius)
                if not self.box_agents
                else Box(length=self.agent_box_length, width=self.agent_box_width),
                u_range=self.u_range,
                f_range=self.f_range,
                render_action=True,
                color=self.colors[i],
            )
            agent.controller = VelocityController(
                agent, world, controller_params, "standard"
            )
            goal = Landmark(
                name=f"goal {i}",
                collide=False,
                shape=Sphere(radius=self.agent_radius / 2),
                color=self.colors[i],
            )
            agent.goal = goal
            agent.pos_rew = torch.zeros(batch_dim, device=device)
            agent.agent_collision_rew = agent.pos_rew.clone()
            world.add_agent(agent)
            world.add_landmark(goal)

        self.spawn_map(world)

        self.pos_rew = torch.zeros(batch_dim, device=device)
        self.final_rew = self.pos_rew.clone()

        return world

    def reset_world_at(self, env_index: int = None):
        for i, agent in enumerate(self.world.agents):
            agent.controller.reset(env_index)
            next_i = (i + 1) % self.n_agents
            if i in [0, 2]:
                agent.set_pos(
                    torch.tensor(
                        [
                            (self.scenario_length / 2 - self.agent_dist_from_wall)
                            * (-1 if i == 0 else 1),
                            0.0,
                        ],
                        dtype=torch.float32,
                        device=self.world.device,
                    ),
                    batch_index=env_index,
                )
                self.world.agents[next_i].goal.set_pos(
                    torch.tensor(
                        [
                            (self.scenario_length / 2 - self.goal_dist_from_wall)
                            * (-1 if i == 0 else 1),
                            0.0,
                        ],
                        dtype=torch.float32,
                        device=self.world.device,
                    ),
                    batch_index=env_index,
                )
            else:
                agent.set_pos(
                    torch.tensor(
                        [
                            0.0,
                            (self.scenario_length / 2 - self.agent_dist_from_wall)
                            * (1 if i == 1 else -1),
                        ],
                        dtype=torch.float32,
                        device=self.world.device,
                    ),
                    batch_index=env_index,
                )
                self.world.agents[next_i].goal.set_pos(
                    torch.tensor(
                        [
                            0.0,
                            (self.scenario_length / 2 - self.goal_dist_from_wall)
                            * (1 if i == 1 else -1),
                        ],
                        dtype=torch.float32,
                        device=self.world.device,
                    ),
                    batch_index=env_index,
                )

        for i, agent in enumerate(self.world.agents):
            if env_index is None:
                agent.shaping = (
                    torch.linalg.vector_norm(
                        agent.state.pos - agent.goal.state.pos, dim=1
                    )
                    * self.pos_shaping_factor
                )
            else:
                agent.shaping[env_index] = (
                    torch.linalg.vector_norm(
                        agent.state.pos[env_index] - agent.goal.state.pos[env_index]
                    )
                    * self.pos_shaping_factor
                )

        self.reset_map(env_index)

        if env_index is None:
            self.reached_goal = torch.full(
                (self.world.batch_dim,), False, device=self.world.device
            )
        else:
            self.reached_goal[env_index] = False

    def process_action(self, agent: Agent):
        # Clamp square to circle
        agent.action.u = TorchUtils.clamp_with_norm(agent.action.u, self.u_range)

        # Zero small input
        action_norm = torch.linalg.vector_norm(agent.action.u, dim=1)
        agent.action.u[action_norm < self.min_input_norm] = 0

        # Copy action
        agent.vel_action = agent.action.u.clone()

        # Reset controller
        vel_is_zero = torch.linalg.vector_norm(agent.action.u, dim=1) < 1e-3
        agent.controller.reset(vel_is_zero)

        agent.controller.process_force()

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            self.pos_rew[:] = 0
            self.final_rew[:] = 0

            for a in self.world.agents:
                a.distance_to_goal = torch.linalg.vector_norm(
                    a.state.pos - a.goal.state.pos,
                    dim=-1,
                )
                a.on_goal = a.distance_to_goal < a.goal.shape.radius

                pos_shaping = a.distance_to_goal * self.pos_shaping_factor
                a.pos_rew = (
                    (a.shaping - pos_shaping)
                    if self.pos_shaping_factor != 0
                    else -a.distance_to_goal * 0.0001
                )
                a.shaping = pos_shaping

                self.pos_rew += a.pos_rew

            self.all_goal_reached = torch.all(
                torch.stack([a.on_goal for a in self.world.agents], dim=-1), dim=-1
            )

            self.final_rew[self.all_goal_reached] = self.final_reward
            self.reached_goal += self.all_goal_reached

        agent.agent_collision_rew[:] = 0
        # agent.obstacle_collision_rew = torch.zeros(
        #     (self.world.batch_dim,), device=self.world.device
        # )
        for a in self.world.agents:
            if a != agent:
                agent.agent_collision_rew[
                    self.world.get_distance(agent, a) <= self.min_collision_distance
                ] += self.agent_collision_penalty
        # for l in self.world.landmarks:
        #     if self.world._collides(agent, l):
        #         if l in [*self.passage_1, *self.passage_2]:
        #             penalty = self.passage_collision_penalty
        #         else:
        #             penalty = self.obstacle_collision_penalty
        #         agent.obstacle_collision_rew[
        #             self.world.get_distance(agent, l) <= self.min_collision_distance
        #         ] += penalty

        # Energy reward
        # agent.energy_expenditure = torch.linalg.vector_norm(
        #     agent.action.u, dim=-1
        # ) / math.sqrt(self.world.dim_p * (agent.f_range**2))
        #
        # agent.energy_rew = -agent.energy_expenditure * self.energy_reward_coeff

        return (
            (self.pos_rew if self.shared_rew else agent.pos_rew)
            # + agent.obstacle_collision_rew
            + agent.agent_collision_rew
            # + agent.energy_rew
            + self.final_rew
        )

    def observation(self, agent: Agent):
        observations = [
            agent.state.pos,
            agent.state.vel,
            agent.state.pos - agent.goal.state.pos,
            torch.linalg.vector_norm(
                agent.state.pos - agent.goal.state.pos,
                dim=-1,
            ).unsqueeze(-1),
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

    def info(self, agent: Agent):
        return {
            "pos_rew": self.pos_rew if self.shared_rew else agent.pos_rew,
            "final_rew": self.final_rew,
            # "energy_rew": agent.energy_rew,
            "agent_collision_rew": agent.agent_collision_rew,
            # "obstacle_collision_rew": agent.obstacle_collision_rew,
        }

    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        from vmas.simulator import rendering

        geoms: List[Geom] = []

        # Communication lines
        for i, agent1 in enumerate(self.world.agents):
            for j, agent2 in enumerate(self.world.agents):
                if j <= i:
                    continue
                agent_dist = torch.linalg.vector_norm(
                    agent1.state.pos - agent2.state.pos, dim=-1
                )
                if agent_dist[env_index] <= self.comms_range:
                    color = Color.BLACK.value
                    line = rendering.Line(
                        (agent1.state.pos[env_index]),
                        (agent2.state.pos[env_index]),
                        width=1,
                    )
                    xform = rendering.Transform()
                    line.add_attr(xform)
                    line.set_color(*color)
                    geoms.append(line)

        return geoms

    def spawn_map(self, world: World):

        self.scenario_length = 5
        self.scenario_width = 0.4

        self.long_wall_length = (self.scenario_length / 2) - (self.scenario_width / 2)
        self.short_wall_length = self.scenario_width
        self.goal_dist_from_wall = self.agent_radius + 0.05
        self.agent_dist_from_wall = 0.5

        self.long_walls = []
        for i in range(8):
            landmark = Landmark(
                name=f"wall {i}",
                collide=True,
                shape=Line(length=self.long_wall_length),
                color=Color.BLACK,
            )
            self.long_walls.append(landmark)
            world.add_landmark(landmark)
        self.short_walls = []
        for i in range(4):
            landmark = Landmark(
                name=f"short wall {i}",
                collide=True,
                shape=Line(length=self.short_wall_length),
                color=Color.BLACK,
            )
            self.short_walls.append(landmark)
            world.add_landmark(landmark)

    def reset_map(self, env_index):
        for i, landmark in enumerate(self.short_walls):
            if i < 2:
                landmark.set_pos(
                    torch.tensor(
                        [
                            -self.scenario_length / 2
                            if i % 2 == 0
                            else self.scenario_length / 2,
                            0.0,
                        ],
                        dtype=torch.float32,
                        device=self.world.device,
                    ),
                    batch_index=env_index,
                )
                landmark.set_rot(
                    torch.tensor(
                        [torch.pi / 2],
                        dtype=torch.float32,
                        device=self.world.device,
                    ),
                    batch_index=env_index,
                )
            else:
                landmark.set_pos(
                    torch.tensor(
                        [
                            0.0,
                            -self.scenario_length / 2
                            if i % 2 == 0
                            else self.scenario_length / 2,
                        ],
                        dtype=torch.float32,
                        device=self.world.device,
                    ),
                    batch_index=env_index,
                )

        long_wall_pos = self.long_wall_length / 2 - self.scenario_length / 2
        for i, landmark in enumerate(self.long_walls):
            if i < 4:
                landmark.set_pos(
                    torch.tensor(
                        [
                            long_wall_pos * (1 if i < 2 else -1),
                            self.scenario_width / 2 * (-1 if i % 2 == 0 else 1),
                        ],
                        dtype=torch.float32,
                        device=self.world.device,
                    ),
                    batch_index=env_index,
                )
            else:
                landmark.set_pos(
                    torch.tensor(
                        [
                            self.scenario_width / 2 * (-1 if i % 2 == 0 else 1),
                            long_wall_pos * (1 if i < 6 else -1),
                        ],
                        dtype=torch.float32,
                        device=self.world.device,
                    ),
                    batch_index=env_index,
                )
                landmark.set_rot(
                    torch.tensor(
                        [torch.pi / 2],
                        dtype=torch.float32,
                        device=self.world.device,
                    ),
                    batch_index=env_index,
                )


if __name__ == "__main__":
    render_interactively(__file__, control_two_agents=True)
