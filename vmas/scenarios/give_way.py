#  Copyright (c) 2022-2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import math

import torch

from vmas import render_interactively
from vmas.simulator.core import Agent, World, Landmark, Sphere, Line, Box
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, TorchUtils
from vmas.simulator.velocity_controller import VelocityController


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.v_range = kwargs.get("v_range", 0.5)
        self.a_range = kwargs.get("a_range", 1)
        self.obs_noise = kwargs.get("obs_noise", 0)
        self.box_agents = kwargs.get("box_agents", False)
        self.linear_friction = kwargs.get("linear_friction", 0.1)
        self.mirror_passage = kwargs.get("mirror_passage", False)

        # Reward params
        self.pos_shaping_factor = kwargs.get("pos_shaping_factor", 1.0)
        self.final_reward = kwargs.get("final_reward", 0.01)
        self.energy_reward_coeff = kwargs.get("energy_rew_coeff", 0)

        self.agent_collision_penalty = kwargs.get("agent_collision_penalty", 0)
        self.passage_collision_penalty = kwargs.get("passage_collision_penalty", 0)
        self.obstacle_collision_penalty = kwargs.get("obstacle_collision_penalty", 0)

        # Params for real world deployment on robomasters
        self.use_velocity_controller = kwargs.get("use_velocity_controller", True)
        self.min_input_norm = kwargs.get("min_input_norm", 0.08)
        self.dt_delay = kwargs.get("dt_delay", 0)

        self.viewer_size = (1600, 700)

        controller_params = [2, 6, 0.002]

        self.f_range = self.a_range + self.linear_friction
        self.u_range = self.v_range if self.use_velocity_controller else self.f_range

        # Make world
        world = World(
            batch_dim,
            device,
            drag=0,
            dt=0.05,
            linear_friction=self.linear_friction,
            substeps=16 if self.box_agents else 5,
            collision_force=10000 if self.box_agents else 500,
        )

        self.agent_radius = 0.16
        self.agent_box_length = 0.32
        self.agent_box_width = 0.24

        self.spawn_pos_noise = 0.02
        self.min_collision_distance = 0.005

        # Add agents
        blue_agent = Agent(
            name="blue agent",
            rotatable=False,
            linear_friction=self.linear_friction,
            shape=Sphere(radius=self.agent_radius)
            if not self.box_agents
            else Box(length=self.agent_box_length, width=self.agent_box_width),
            u_range=self.u_range,
            f_range=self.f_range,
            v_range=self.v_range,
            render_action=True,
        )
        if self.use_velocity_controller:
            blue_agent.controller = VelocityController(
                blue_agent, world, controller_params, "standard"
            )
        blue_goal = Landmark(
            name="blue goal",
            collide=False,
            shape=Sphere(radius=self.agent_radius / 2),
            color=Color.BLUE,
        )
        blue_agent.goal = blue_goal
        world.add_agent(blue_agent)
        world.add_landmark(blue_goal)

        green_agent = Agent(
            name="green agent",
            color=Color.GREEN,
            linear_friction=self.linear_friction,
            shape=Sphere(radius=self.agent_radius)
            if not self.box_agents
            else Box(length=self.agent_box_length, width=self.agent_box_width),
            rotatable=False,
            u_range=self.u_range,
            f_range=self.f_range,
            v_range=self.v_range,
            render_action=True,
        )
        if self.use_velocity_controller:
            green_agent.controller = VelocityController(
                green_agent, world, controller_params, "standard"
            )
        green_goal = Landmark(
            name="green goal",
            collide=False,
            shape=Sphere(radius=self.agent_radius / 2),
            color=Color.GREEN,
        )
        green_agent.goal = green_goal
        world.add_agent(green_agent)
        world.add_landmark(green_goal)

        null_action = torch.zeros(world.batch_dim, world.dim_p, device=world.device)
        blue_agent.input_queue = [null_action.clone() for _ in range(self.dt_delay)]
        green_agent.input_queue = [null_action.clone() for _ in range(self.dt_delay)]
        # control delayed by n dts

        self.spawn_map(world)

        for agent in world.agents:
            agent.energy_rew = torch.zeros(batch_dim, device=device)
            agent.agent_collision_rew = agent.energy_rew.clone()
            agent.obstacle_collision_rew = agent.agent_collision_rew.clone()

        self.pos_rew = torch.zeros(batch_dim, device=device)
        self.final_rew = self.pos_rew.clone()

        return world

    def reset_world_at(self, env_index: int = None):
        self.world.agents[0].set_pos(
            torch.tensor(
                [-(self.scenario_length / 2 - self.agent_dist_from_wall), 0.0],
                dtype=torch.float32,
                device=self.world.device,
            )
            + torch.zeros(self.world.dim_p, device=self.world.device,).uniform_(
                -self.spawn_pos_noise,
                self.spawn_pos_noise,
            ),
            batch_index=env_index,
        )
        if self.use_velocity_controller:
            self.world.agents[0].controller.reset(env_index)
        self.world.landmarks[0].set_pos(
            torch.tensor(
                [(self.scenario_length / 2 - self.goal_dist_from_wall), 0.0],
                dtype=torch.float32,
                device=self.world.device,
            ),
            batch_index=env_index,
        )
        self.world.agents[1].set_pos(
            torch.tensor(
                [self.scenario_length / 2 - self.agent_dist_from_wall, 0.0],
                dtype=torch.float32,
                device=self.world.device,
            )
            + torch.zeros(self.world.dim_p, device=self.world.device,).uniform_(
                -self.spawn_pos_noise,
                self.spawn_pos_noise,
            ),
            batch_index=env_index,
        )
        if self.use_velocity_controller:
            self.world.agents[1].controller.reset(env_index)
        self.world.landmarks[1].set_pos(
            torch.tensor(
                [-(self.scenario_length / 2 - self.goal_dist_from_wall), 0.0],
                dtype=torch.float32,
                device=self.world.device,
            ),
            batch_index=env_index,
        )

        self.reset_map(env_index)
        for agent in self.world.agents:
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

        if env_index is None:
            self.reached_goal = torch.full(
                (self.world.batch_dim,), False, device=self.world.device
            )
        else:
            self.reached_goal[env_index] = False

    def process_action(self, agent: Agent):
        if self.use_velocity_controller:
            # Use queue for delay
            agent.input_queue.append(agent.action.u.clone())
            agent.action.u = agent.input_queue.pop(0)

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

        blue_agent = self.world.agents[0]
        green_agent = self.world.agents[-1]

        if is_first:
            self.pos_rew[:] = 0
            self.final_rew[:] = 0

            self.blue_distance = torch.linalg.vector_norm(
                blue_agent.state.pos - blue_agent.goal.state.pos,
                dim=1,
            )
            self.green_distance = torch.linalg.vector_norm(
                green_agent.state.pos - green_agent.goal.state.pos,
                dim=1,
            )
            self.blue_on_goal = self.blue_distance < blue_agent.goal.shape.radius
            self.green_on_goal = self.green_distance < green_agent.goal.shape.radius
            self.goal_reached = self.green_on_goal * self.blue_on_goal

            green_shaping = self.green_distance * self.pos_shaping_factor
            self.green_rew = green_agent.shaping - green_shaping
            green_agent.shaping = green_shaping

            blue_shaping = self.blue_distance * self.pos_shaping_factor
            self.blue_rew = blue_agent.shaping - blue_shaping
            blue_agent.shaping = blue_shaping

            self.pos_rew += self.blue_rew
            self.pos_rew += self.green_rew

            self.final_rew[self.goal_reached] = self.final_reward
            self.reached_goal += self.goal_reached

        agent.agent_collision_rew[:] = 0
        agent.obstacle_collision_rew[:] = 0
        for a in self.world.agents:
            if a != agent:
                agent.agent_collision_rew[
                    self.world.get_distance(agent, a) <= self.min_collision_distance
                ] += self.agent_collision_penalty
        for l in self.world.landmarks:
            if self.world.collides(agent, l):
                if l in (
                    [*self.passage_1, *self.passage_2]
                    if self.mirror_passage is True
                    else [*self.passage_1]
                ):
                    penalty = self.passage_collision_penalty
                else:
                    penalty = self.obstacle_collision_penalty
                agent.obstacle_collision_rew[
                    self.world.get_distance(agent, l) <= self.min_collision_distance
                ] += penalty

        # Energy reward
        agent.energy_expenditure = torch.linalg.vector_norm(
            agent.action.u, dim=-1
        ) / math.sqrt(self.world.dim_p * (agent.f_range**2))

        agent.energy_rew = -agent.energy_expenditure * self.energy_reward_coeff

        return (
            self.pos_rew
            + agent.obstacle_collision_rew
            + agent.agent_collision_rew
            + agent.energy_rew
            + self.final_rew
        )

    def observation(self, agent: Agent):
        observations = [
            agent.state.pos,
            agent.state.vel,
            agent.state.pos,
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
            "pos_rew": self.pos_rew,
            "final_rew": self.final_rew,
            "energy_rew": agent.energy_rew,
            "agent_collision_rew": agent.agent_collision_rew,
            "obstacle_collision_rew": agent.obstacle_collision_rew,
        }

    def spawn_map(self, world: World):

        self.scenario_length = 5
        self.passage_length = 0.4
        self.passage_width = 0.48  # box obstacle length
        self.corridor_width = self.passage_length
        self.small_ceiling_length = (self.scenario_length / 2) - (
            self.passage_length / 2
        )
        self.goal_dist_from_wall = self.agent_radius + 0.05
        self.agent_dist_from_wall = 0.5

        self.walls = []
        for i in range(2):
            landmark = Landmark(
                name=f"wall {i}",
                collide=True,
                shape=Line(length=self.corridor_width),
                color=Color.BLACK,
            )
            self.walls.append(landmark)
            world.add_landmark(landmark)
        self.small_ceilings_1 = []
        for i in range(2):
            landmark = Landmark(
                name=f"ceil 1 {i}",
                collide=True,
                shape=Line(length=self.small_ceiling_length),
                color=Color.BLACK,
            )
            self.small_ceilings_1.append(landmark)
            world.add_landmark(landmark)
        self.passage_1 = []
        for i in range(3):
            landmark = Landmark(
                name=f"ceil 2 {i}",
                collide=True,
                shape=Line(
                    length=self.passage_length if i == 2 else self.passage_width
                ),
                color=Color.BLACK,
            )
            self.passage_1.append(landmark)
            world.add_landmark(landmark)

        if self.mirror_passage:
            self.small_ceilings_2 = []
            for i in range(2):
                landmark = Landmark(
                    name=f"ceil 12 {i}",
                    collide=True,
                    shape=Line(length=self.small_ceiling_length),
                    color=Color.BLACK,
                )
                self.small_ceilings_2.append(landmark)
                world.add_landmark(landmark)
            self.passage_2 = []
            for i in range(3):
                landmark = Landmark(
                    name=f"ceil 22 {i}",
                    collide=True,
                    shape=Line(
                        length=self.passage_length if i == 2 else self.passage_width
                    ),
                    color=Color.BLACK,
                )
                self.passage_2.append(landmark)
                world.add_landmark(landmark)
        else:
            # Add landmarks
            landmark = Landmark(
                name="floor",
                collide=True,
                shape=Line(length=self.scenario_length),
                color=Color.BLACK,
            )
            self.floor = landmark
            world.add_landmark(landmark)

    def reset_map(self, env_index):
        # Walls
        for i, landmark in enumerate(self.walls):
            landmark.set_pos(
                torch.tensor(
                    [
                        -self.scenario_length / 2
                        if i == 0
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

        # Ceiling
        small_ceiling_pos = self.small_ceiling_length / 2 - self.scenario_length / 2
        for i, landmark in enumerate(self.small_ceilings_1):
            landmark.set_pos(
                torch.tensor(
                    [
                        -small_ceiling_pos if i == 0 else small_ceiling_pos,
                        self.passage_length / 2,
                    ],
                    dtype=torch.float32,
                    device=self.world.device,
                ),
                batch_index=env_index,
            )

        # Asymmetric hole
        for i, landmark in enumerate(self.passage_1[:-1]):
            landmark.set_pos(
                torch.tensor(
                    [
                        -self.passage_length / 2 if i == 0 else self.passage_length / 2,
                        self.passage_length / 2 + self.passage_width / 2,
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
        self.passage_1[-1].set_pos(
            torch.tensor(
                [0, self.passage_length / 2 + self.passage_width],
                dtype=torch.float32,
                device=self.world.device,
            ),
            batch_index=env_index,
        )

        if self.mirror_passage:
            # Ceiling
            for i, landmark in enumerate(self.small_ceilings_2):
                landmark.set_pos(
                    torch.tensor(
                        [
                            -small_ceiling_pos if i == 0 else small_ceiling_pos,
                            -self.passage_length / 2,
                        ],
                        dtype=torch.float32,
                        device=self.world.device,
                    ),
                    batch_index=env_index,
                )

            # Asymmetric hole
            for i, landmark in enumerate(self.passage_2[:-1]):
                landmark.set_pos(
                    torch.tensor(
                        [
                            -self.passage_length / 2
                            if i == 0
                            else self.passage_length / 2,
                            -self.passage_length / 2 - self.passage_width / 2,
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
            self.passage_2[-1].set_pos(
                torch.tensor(
                    [0, -self.passage_length / 2 - self.passage_width],
                    dtype=torch.float32,
                    device=self.world.device,
                ),
                batch_index=env_index,
            )
        else:
            # Floor
            self.floor.set_pos(
                torch.tensor(
                    [0, -self.passage_length / 2],
                    dtype=torch.float32,
                    device=self.world.device,
                ),
                batch_index=env_index,
            )


if __name__ == "__main__":
    render_interactively(__file__, control_two_agents=True)
