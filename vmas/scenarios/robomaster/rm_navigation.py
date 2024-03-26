#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
from typing import Dict

import torch
from torch import Tensor

from vmas import render_interactively
from vmas.simulator.controllers.velocity_controller import VelocityController
from vmas.simulator.core import Agent, Box, Landmark, Sphere, World
from vmas.simulator.dynamics.holonomic import Holonomic
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import ScenarioUtils, TorchUtils


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        # Robomaster
        self.v_range = kwargs.get("v_range", 1)
        self.a_range = kwargs.get("a_range", 1)
        self.box_agents = kwargs.get("box_agents", False)
        self.linear_friction = kwargs.get("linear_friction", 0.1)
        self.min_input_norm = kwargs.get("min_input_norm", 0.08)
        self.agent_radius = 0.16
        self.agent_box_length = 0.32
        self.agent_box_width = 0.24

        # Controller
        self.use_velocity_controller = kwargs.get("use_velocity_controller", True)
        controller_params = [2, 6, 0.002]
        self.f_range = self.a_range + self.linear_friction
        self.u_range = self.v_range if self.use_velocity_controller else self.f_range

        self.plot_grid = True
        self.viewer_zoom = 2
        self.n_agents = kwargs.get("n_agents", 2)

        self.pos_shaping_factor = kwargs.get("pos_shaping_factor", 1)
        self.final_reward = kwargs.get("final_reward", 0.01)
        self.agent_collision_penalty = kwargs.get("agent_collision_penalty", -5)

        self.min_distance_between_entities = self.agent_radius * 2 + 0.05
        self.world_semidim = 3
        self.min_collision_distance = 0.05

        # Make world
        world = World(
            batch_dim,
            device,
            drag=0,
            dt=0.05,
            linear_friction=self.linear_friction,
            substeps=16 if self.box_agents else 5,
            collision_force=100000 if self.box_agents else 500,
        )

        known_colors = [
            (0.22, 0.49, 0.72),
            (1.00, 0.50, 0),
            (0.30, 0.69, 0.29),
            (0.97, 0.51, 0.75),
            (0.60, 0.31, 0.64),
            (0.89, 0.10, 0.11),
            (0.87, 0.87, 0),
        ]
        colors = torch.randn(
            (max(self.n_agents - len(known_colors), 0), 3), device=device
        )

        # Add agents
        for i in range(self.n_agents):
            color = (
                known_colors[i]
                if i < len(known_colors)
                else colors[i - len(known_colors)]
            )

            # Constraint: all agents have same action range and multiplier
            agent = Agent(
                name=f"agent_{i}",
                collide=True,
                color=color,
                rotatable=False,
                shape=(
                    Sphere(radius=self.agent_radius)
                    if not self.box_agents
                    else Box(length=self.agent_box_length, width=self.agent_box_width)
                ),
                u_range=self.u_range,
                f_range=self.f_range,
                v_range=self.v_range,
                render_action=True,
                dynamics=Holonomic(),
            )
            if self.use_velocity_controller:
                agent.controller = VelocityController(
                    agent, world, controller_params, "standard"
                )

            agent.pos_rew = torch.zeros(batch_dim, device=device)
            agent.agent_collision_rew = agent.pos_rew.clone()
            agent.final_rew = agent.pos_rew.clone()

            world.add_agent(agent)

            # Add goals
            goal = Landmark(
                name=f"goal {i}", collide=False, color=color, shape=Sphere(radius=0.1)
            )
            world.add_landmark(goal)
            agent.goal = goal

        return world

    def reset_world_at(self, env_index: int = None):
        ScenarioUtils.spawn_entities_randomly(
            self.world.agents,
            self.world,
            env_index,
            self.min_distance_between_entities,
            (-self.world_semidim, self.world_semidim),
            (-self.world_semidim, self.world_semidim),
        )

        occupied_positions = torch.stack(
            [agent.state.pos for agent in self.world.agents], dim=1
        )
        if env_index is not None:
            occupied_positions = occupied_positions[env_index].unsqueeze(0)

        goal_poses = []
        for _ in self.world.agents:
            position = ScenarioUtils.find_random_pos_for_entity(
                occupied_positions=occupied_positions,
                env_index=env_index,
                world=self.world,
                min_dist_between_entities=self.min_distance_between_entities,
                x_bounds=(-self.world_semidim, self.world_semidim),
                y_bounds=(-self.world_semidim, self.world_semidim),
            )
            goal_poses.append(position.squeeze(1))
            occupied_positions = torch.cat([occupied_positions, position], dim=1)

        for i, agent in enumerate(self.world.agents):
            if self.use_velocity_controller:
                agent.controller.reset(env_index)

            agent.goal.set_pos(goal_poses[i], batch_index=env_index)

            if env_index is None:
                agent.pos_shaping = (
                    torch.linalg.vector_norm(
                        agent.state.pos - agent.goal.state.pos,
                        dim=1,
                    )
                    * self.pos_shaping_factor
                )
            else:
                agent.pos_shaping[env_index] = (
                    torch.linalg.vector_norm(
                        agent.state.pos[env_index] - agent.goal.state.pos[env_index]
                    )
                    * self.pos_shaping_factor
                )

    def process_action(self, agent: Agent):
        if self.use_velocity_controller:
            # Clamp square to circle
            agent.action.u = TorchUtils.clamp_with_norm(
                agent.action.u, agent.action.u_range
            )

            # Zero small input
            action_norm = torch.linalg.vector_norm(agent.action.u, dim=1)
            agent.action.u[action_norm < self.min_input_norm] = 0

            # Reset controller
            vel_is_zero = torch.linalg.vector_norm(agent.action.u, dim=1) < 1e-3
            agent.controller.reset(vel_is_zero)

            agent.controller.process_force()

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            for a in self.world.agents:
                self.agent_reward(a)
                a.agent_collision_rew[:] = 0

            for i, a in enumerate(self.world.agents):
                for j, b in enumerate(self.world.agents):
                    if i <= j:
                        continue
                    distance = self.world.get_distance(a, b)
                    a.agent_collision_rew[
                        distance <= self.min_collision_distance
                    ] += self.agent_collision_penalty
                    b.agent_collision_rew[
                        distance <= self.min_collision_distance
                    ] += self.agent_collision_penalty

        return agent.pos_rew + agent.final_rew + agent.agent_collision_rew

    def agent_reward(self, agent: Agent):
        agent.distance_to_goal = torch.linalg.vector_norm(
            agent.state.pos - agent.goal.state.pos,
            dim=-1,
        )
        agent.on_goal = agent.distance_to_goal < agent.goal.shape.radius

        agent.final_rew = torch.where(
            agent.on_goal, self.final_reward, torch.zeros_like(agent.final_rew)
        )

        pos_shaping = agent.distance_to_goal * self.pos_shaping_factor
        agent.pos_rew = agent.pos_shaping - pos_shaping
        agent.pos_shaping = pos_shaping

    def observation(self, agent: Agent):
        return torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
                agent.state.pos - agent.goal.state.pos,
            ],
            dim=-1,
        )

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        return {
            "pos_rew": agent.pos_rew,
            "final_rew": agent.final_rew,
            "collision_rew": agent.agent_collision_rew,
        }

    # def extra_render(self, env_index: int = 0):
    #     from vmas.simulator import rendering
    #
    #     geoms = []
    #     for i, entity_a in enumerate(self.world.agents):
    #         for j, entity_b in enumerate(self.world.agents):
    #             if i <= j:
    #                 continue
    #             if not self.box_agents:
    #                 point_a = entity_a.state.pos
    #                 point_b = entity_b.state.pos
    #                 dist = self.world.get_distance_from_point(
    #                     entity_a, entity_b.state.pos, env_index
    #                 )
    #                 dist = dist - entity_b.shape.radius
    #             else:
    #                 point_a, point_b = _get_closest_box_box(
    #                     entity_a.state.pos,
    #                     entity_a.state.rot,
    #                     entity_a.shape.width,
    #                     entity_a.shape.length,
    #                     entity_b.state.pos,
    #                     entity_b.state.rot,
    #                     entity_b.shape.width,
    #                     entity_b.shape.length,
    #                 )
    #                 dist = torch.linalg.vector_norm(point_a - point_b, dim=-1)
    #
    #             for point in [point_a, point_b]:
    #                 color = (
    #                     Color.BLACK.value
    #                     if dist > self.min_collision_distance
    #                     else Color.RED.value
    #                 )
    #                 line = rendering.make_circle(0.03)
    #                 xform = rendering.Transform()
    #                 xform.set_translation(point[env_index][0], point[env_index][1])
    #                 line.add_attr(xform)
    #                 line.set_color(*color)
    #                 geoms.append(line)
    #     return geoms
    #


if __name__ == "__main__":
    render_interactively(
        __file__,
        control_two_agents=True,
    )
