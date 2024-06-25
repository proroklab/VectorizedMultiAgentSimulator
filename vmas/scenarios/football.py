#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import math
import typing
from typing import List

import torch
from torch import Tensor

from vmas import render_interactively
from vmas.simulator.core import Agent, Box, Landmark, Line, Sphere, World
from vmas.simulator.dynamics.holonomic import Holonomic
from vmas.simulator.dynamics.holonomic_with_rot import HolonomicWithRotation
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, TorchUtils, X, Y

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.init_params(**kwargs)
        world = self.init_world(batch_dim, device)
        self.init_agents(world)
        self.init_ball(world)
        self.init_background(world)
        self.init_walls(world)
        self.init_goals(world)
        self.init_traj_pts(world)
        self.left_goal_pos = torch.tensor(
            [-self.pitch_length / 2 - self.ball_size / 2, 0],
            device=device,
            dtype=torch.float,
        )
        self.right_goal_pos = -self.left_goal_pos
        self._done = torch.zeros(batch_dim, device=device, dtype=torch.bool)
        self._sparse_reward = torch.zeros(batch_dim, device=device, dtype=torch.float32)
        self._render_field = True

        self._reset_agent_range = torch.tensor(
            [self.pitch_length / 2, self.pitch_width],
            device=device,
        )
        self._reset_agent_offset_blue = torch.tensor(
            [-self.pitch_length / 2 + self.agent_size, -self.pitch_width / 2],
            device=device,
        )
        self._reset_agent_offset_red = torch.tensor(
            [-self.agent_size, -self.pitch_width / 2], device=device
        )
        return world

    def reset_world_at(self, env_index: int = None):
        self.reset_agents(env_index)
        self.reset_ball(env_index)
        self.reset_walls(env_index)
        self.reset_goals(env_index)
        self.reset_controllers(env_index)
        if env_index is None:
            self._done[:] = False
        else:
            self._done[env_index] = False

    def init_params(self, **kwargs):
        # Scenario config
        self.viewer_size = kwargs.get("viewer_size", (1200, 800))

        # Agents config
        self.n_blue_agents = kwargs.get("n_blue_agents", 3)
        self.n_red_agents = kwargs.get("n_red_agents", 3)
        self.ai_red_agents = kwargs.get("ai_red_agents", True)
        self.ai_blue_agents = kwargs.get("ai_blue_agents", False)

        # Ai config
        self.n_traj_points = kwargs.get("n_traj_points", 0)
        self.ai_strength = kwargs.get("ai_strength", 1)
        self.disable_ai_red = kwargs.get("disable_ai_red", False)

        # Task sizes
        self.agent_size = kwargs.get("agent_size", 0.025)
        self.goal_size = kwargs.get("goal_size", 0.35)
        self.goal_depth = kwargs.get("goal_depth", 0.1)
        self.pitch_length = kwargs.get("pitch_length", 3.0)
        self.pitch_width = kwargs.get("pitch_width", 1.5)
        self.ball_mass = kwargs.get("ball_mass", 0.25)
        self.ball_size = kwargs.get("ball_size", 0.02)

        # Actions
        self.u_multiplier = kwargs.get("u_multiplier", 0.1)

        # Actions shooting
        self.enable_shooting = kwargs.get("enable_shooting", False)
        self.u_rot_multiplier = kwargs.get("u_rot_multiplier", 0.5)
        self.u_shoot_multiplier = kwargs.get("u_shoot_multiplier", 0.25)
        self.shooting_radius = kwargs.get("shooting_radius", 0.1)
        self.shooting_angle = kwargs.get("shooting_angle", torch.pi / 2)

        # Speeds
        self.max_speed = kwargs.get("max_speed", 0.15)
        self.ball_max_speed = kwargs.get("ball_max_speed", 0.3)

        # Rewards
        self.dense_reward = kwargs.get("dense_reward", True)
        self.pos_shaping_factor_ball_goal = kwargs.get(
            "pos_shaping_factor_ball_goal", 10.0
        )
        self.pos_shaping_factor_agent_ball = kwargs.get(
            "pos_shaping_factor_agent_ball", 0.1
        )
        self.only_closest_agent_ball_reward = kwargs.get(
            "only_closest_agent_ball_reward", True
        )
        self.distance_to_ball_trigger = kwargs.get("distance_to_ball_trigger", 0.4)
        self.scoring_reward = kwargs.get("scoring_reward", 100.0)

        # Observations
        self.observe_teammates = kwargs.get("observe_teammates", True)
        self.observe_adversaries = kwargs.get("observe_adversaries", True)
        self.dict_obs = kwargs.get("dict_obs", False)

        if kwargs.get("dense_reward_ratio", None) is not None:
            raise ValueError(
                "dense_reward_ratio in football is deprecated, please use `dense_reward` "
                "which is a bool that turns on/off the dense reward"
            )

    def init_world(self, batch_dim: int, device: torch.device):
        # Make world
        world = World(
            batch_dim,
            device,
            dt=0.1,
            drag=0.05,
            x_semidim=self.pitch_length / 2 + self.goal_depth - self.agent_size,
            y_semidim=self.pitch_width / 2 - self.agent_size,
        )
        world.agent_size = self.agent_size
        world.pitch_width = self.pitch_width
        world.pitch_length = self.pitch_length
        world.goal_size = self.goal_size
        world.goal_depth = self.goal_depth
        return world

    def init_agents(self, world):
        # Add agents
        self.red_controller = (
            AgentPolicy(
                team="Red",
                disabled=self.disable_ai_red,
                strength=self.ai_strength,
            )
            if self.ai_red_agents
            else None
        )
        self.blue_controller = (
            AgentPolicy(team="Blue", strength=self.ai_strength)
            if self.ai_blue_agents
            else None
        )

        blue_agents = []
        for i in range(self.n_blue_agents):
            agent = Agent(
                name=f"agent_blue_{i}",
                shape=Sphere(radius=self.agent_size),
                action_script=self.blue_controller.run if self.ai_blue_agents else None,
                u_multiplier=[self.u_multiplier, self.u_multiplier]
                if not self.enable_shooting
                else [
                    self.u_multiplier,
                    self.u_multiplier,
                    self.u_rot_multiplier,
                    self.u_shoot_multiplier,
                ],
                max_speed=self.max_speed,
                dynamics=Holonomic()
                if not self.enable_shooting
                else HolonomicWithRotation(),
                action_size=2 if not self.enable_shooting else 4,
                color=(0.22, 0.49, 0.72),
                alpha=1,
            )
            world.add_agent(agent)
            blue_agents.append(agent)
        self.blue_agents = blue_agents
        world.blue_agents = blue_agents

        red_agents = []
        for i in range(self.n_red_agents):
            agent = Agent(
                name=f"agent_red_{i}",
                shape=Sphere(radius=self.agent_size),
                action_script=self.red_controller.run if self.ai_red_agents else None,
                u_multiplier=self.u_multiplier,
                max_speed=self.max_speed,
                color=(0.89, 0.10, 0.11),
                alpha=1,
            )
            world.add_agent(agent)
            red_agents.append(agent)
        self.red_agents = red_agents
        world.red_agents = red_agents

        for agent in self.blue_agents:
            agent.pos_rew = torch.zeros(
                world.batch_dim, device=agent.device, dtype=torch.float32
            )
            agent.ball_within_angle = torch.zeros(
                world.batch_dim, device=agent.device, dtype=torch.bool
            )
            agent.ball_within_range = torch.zeros(
                world.batch_dim, device=agent.device, dtype=torch.bool
            )

    def reset_agents(self, env_index: int = None):
        for agent in self.blue_agents:
            agent.set_pos(
                torch.rand(
                    (
                        (1, self.world.dim_p)
                        if env_index is not None
                        else (self.world.batch_dim, self.world.dim_p)
                    ),
                    device=self.world.device,
                )
                * self._reset_agent_range
                + self._reset_agent_offset_blue,
                batch_index=env_index,
            )
        for agent in self.red_agents:
            agent.set_pos(
                torch.rand(
                    (
                        (1, self.world.dim_p)
                        if env_index is not None
                        else (self.world.batch_dim, self.world.dim_p)
                    ),
                    device=self.world.device,
                )
                * self._reset_agent_range
                + self._reset_agent_offset_red,
                batch_index=env_index,
            )
        for agent in self.red_agents + self.blue_agents:
            if env_index is None:
                agent.pos_shaping = (
                    torch.linalg.vector_norm(
                        agent.state.pos - self.ball.state.pos,
                        dim=-1,
                    )
                    * self.pos_shaping_factor_agent_ball
                )
            else:
                agent.pos_shaping[env_index] = (
                    torch.linalg.vector_norm(
                        agent.state.pos[env_index] - self.ball.state.pos[env_index]
                    )
                    * self.pos_shaping_factor_agent_ball
                )

    def reset_controllers(self, env_index: int = None):
        if self.red_controller is not None:
            if not self.red_controller.initialised:
                self.red_controller.init(self.world)
            self.red_controller.reset(env_index)
        if self.blue_controller is not None:
            if not self.blue_controller.initialised:
                self.blue_controller.init(self.world)
            self.blue_controller.reset(env_index)

    def init_ball(self, world):
        # Add Ball
        ball = Agent(
            name="Ball",
            shape=Sphere(radius=self.ball_size),
            action_script=ball_action_script,
            max_speed=self.ball_max_speed,
            mass=self.ball_mass,
            alpha=1,
            color=Color.BLACK,
        )
        ball.pos_rew = torch.zeros(
            world.batch_dim, device=world.device, dtype=torch.float32
        )
        ball.pos_rew_agent = ball.pos_rew.clone()
        ball.kicking_action = ball.pos_rew = torch.zeros(
            world.batch_dim, world.dim_p, device=world.device, dtype=torch.float32
        )
        world.add_agent(ball)
        world.ball = ball
        self.ball = ball

    def reset_ball(self, env_index: int = None):
        min_agent_dist_to_ball = self.get_closest_agent_to_ball(env_index)
        if env_index is None:
            self.ball.pos_shaping = (
                torch.linalg.vector_norm(
                    self.ball.state.pos - self.right_goal_pos,
                    dim=-1,
                )
                * self.pos_shaping_factor_ball_goal
            )
            self.ball.pos_shaping_agent = (
                min_agent_dist_to_ball * self.pos_shaping_factor_agent_ball
            )
            if self.enable_shooting:
                self.ball.kicking_action[:] = 0.0
        else:
            self.ball.pos_shaping[env_index] = (
                torch.linalg.vector_norm(
                    self.ball.state.pos[env_index] - self.right_goal_pos
                )
                * self.pos_shaping_factor_ball_goal
            )
            self.ball.pos_shaping_agent[env_index] = (
                min_agent_dist_to_ball * self.pos_shaping_factor_agent_ball
            )
            if self.enable_shooting:
                self.ball.kicking_action[env_index] = 0.0

    def get_closest_agent_to_ball(self, env_index):
        if self.only_closest_agent_ball_reward:
            pos = torch.stack(
                [a.state.pos for a in self.world.policy_agents], dim=-2
            )  # shape == (batch_dim, n_agents, 2)
            ball_pos = self.ball.state.pos.unsqueeze(-2)
            if isinstance(env_index, int):
                pos = pos[env_index].unsqueeze(0)
                ball_pos = ball_pos[env_index].unsqueeze(0)
            dist = torch.cdist(pos, ball_pos)
            dist = dist.squeeze(-1)
            min_dist = dist.min(dim=-1)[0]
            if isinstance(env_index, int):
                min_dist = min_dist.squeeze(0)
        else:
            min_dist = torch.zeros(
                self.world.batch_dim, device=self.world.device, dtype=torch.float32
            )
            if isinstance(env_index, int):
                min_dist = min_dist[env_index]
        return min_dist

    def init_background(self, world):
        # Add landmarks
        self.background = Landmark(
            name="Background",
            collide=False,
            movable=False,
            shape=Box(length=self.pitch_length, width=self.pitch_width),
            color=Color.GREEN,
        )

        self.centre_circle_outer = Landmark(
            name="Centre Circle Outer",
            collide=False,
            movable=False,
            shape=Sphere(radius=self.goal_size / 2),
            color=Color.WHITE,
        )

        self.centre_circle_inner = Landmark(
            name="Centre Circle Inner",
            collide=False,
            movable=False,
            shape=Sphere(self.goal_size / 2 - 0.02),
            color=Color.GREEN,
        )

        centre_line = Landmark(
            name="Centre Line",
            collide=False,
            movable=False,
            shape=Line(length=self.pitch_width - 2 * self.agent_size),
            color=Color.WHITE,
        )

        right_line = Landmark(
            name="Right Line",
            collide=False,
            movable=False,
            shape=Line(length=self.pitch_width - 2 * self.agent_size),
            color=Color.WHITE,
        )

        left_line = Landmark(
            name="Left Line",
            collide=False,
            movable=False,
            shape=Line(length=self.pitch_width - 2 * self.agent_size),
            color=Color.WHITE,
        )

        top_line = Landmark(
            name="Top Line",
            collide=False,
            movable=False,
            shape=Line(length=self.pitch_length - 2 * self.agent_size),
            color=Color.WHITE,
        )

        bottom_line = Landmark(
            name="Bottom Line",
            collide=False,
            movable=False,
            shape=Line(length=self.pitch_length - 2 * self.agent_size),
            color=Color.WHITE,
        )

        self.background_entities = [
            self.background,
            self.centre_circle_outer,
            self.centre_circle_inner,
            centre_line,
            right_line,
            left_line,
            top_line,
            bottom_line,
        ]

    def render_field(self, render: bool):
        self._render_field = render
        self.left_top_wall.is_rendering[:] = render
        self.left_bottom_wall.is_rendering[:] = render
        self.right_top_wall.is_rendering[:] = render
        self.right_bottom_wall.is_rendering[:] = render

    def init_walls(self, world):
        self.right_top_wall = Landmark(
            name="Right Top Wall",
            collide=True,
            movable=False,
            shape=Line(
                length=self.pitch_width / 2 - self.agent_size - self.goal_size / 2,
            ),
            color=Color.WHITE,
        )
        world.add_landmark(self.right_top_wall)

        self.left_top_wall = Landmark(
            name="Left Top Wall",
            collide=True,
            movable=False,
            shape=Line(
                length=self.pitch_width / 2 - self.agent_size - self.goal_size / 2,
            ),
            color=Color.WHITE,
        )
        world.add_landmark(self.left_top_wall)

        self.right_bottom_wall = Landmark(
            name="Right Bottom Wall",
            collide=True,
            movable=False,
            shape=Line(
                length=self.pitch_width / 2 - self.agent_size - self.goal_size / 2,
            ),
            color=Color.WHITE,
        )
        world.add_landmark(self.right_bottom_wall)

        self.left_bottom_wall = Landmark(
            name="Left Bottom Wall",
            collide=True,
            movable=False,
            shape=Line(
                length=self.pitch_width / 2 - self.agent_size - self.goal_size / 2,
            ),
            color=Color.WHITE,
        )
        world.add_landmark(self.left_bottom_wall)

    def reset_walls(self, env_index: int = None):
        for landmark in self.world.landmarks:
            if landmark.name == "Left Top Wall":
                landmark.set_pos(
                    torch.tensor(
                        [
                            -self.pitch_length / 2,
                            self.pitch_width / 4 + self.goal_size / 4,
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

            elif landmark.name == "Left Bottom Wall":
                landmark.set_pos(
                    torch.tensor(
                        [
                            -self.pitch_length / 2,
                            -self.pitch_width / 4 - self.goal_size / 4,
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

            elif landmark.name == "Right Top Wall":
                landmark.set_pos(
                    torch.tensor(
                        [
                            self.pitch_length / 2,
                            self.pitch_width / 4 + self.goal_size / 4,
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
            elif landmark.name == "Right Bottom Wall":
                landmark.set_pos(
                    torch.tensor(
                        [
                            self.pitch_length / 2,
                            -self.pitch_width / 4 - self.goal_size / 4,
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

    def init_goals(self, world):
        right_goal_back = Landmark(
            name="Right Goal Back",
            collide=True,
            movable=False,
            shape=Line(length=self.goal_size),
            color=Color.WHITE,
        )
        world.add_landmark(right_goal_back)

        left_goal_back = Landmark(
            name="Left Goal Back",
            collide=True,
            movable=False,
            shape=Line(length=self.goal_size),
            color=Color.WHITE,
        )
        world.add_landmark(left_goal_back)

        right_goal_top = Landmark(
            name="Right Goal Top",
            collide=True,
            movable=False,
            shape=Line(length=self.goal_depth),
            color=Color.WHITE,
        )
        world.add_landmark(right_goal_top)

        left_goal_top = Landmark(
            name="Left Goal Top",
            collide=True,
            movable=False,
            shape=Line(length=self.goal_depth),
            color=Color.WHITE,
        )
        world.add_landmark(left_goal_top)

        right_goal_bottom = Landmark(
            name="Right Goal Bottom",
            collide=True,
            movable=False,
            shape=Line(length=self.goal_depth),
            color=Color.WHITE,
        )
        world.add_landmark(right_goal_bottom)

        left_goal_bottom = Landmark(
            name="Left Goal Bottom",
            collide=True,
            movable=False,
            shape=Line(length=self.goal_depth),
            color=Color.WHITE,
        )
        world.add_landmark(left_goal_bottom)

        blue_net = Landmark(
            name="Blue Net",
            collide=False,
            movable=False,
            shape=Box(length=self.goal_depth, width=self.goal_size),
            color=(0.5, 0.5, 0.5, 0.5),
        )
        world.add_landmark(blue_net)

        red_net = Landmark(
            name="Red Net",
            collide=False,
            movable=False,
            shape=Box(length=self.goal_depth, width=self.goal_size),
            color=(0.5, 0.5, 0.5, 0.5),
        )
        world.add_landmark(red_net)

        self.blue_net = blue_net
        self.red_net = red_net
        world.blue_net = blue_net
        world.red_net = red_net

    def reset_goals(self, env_index: int = None):
        for landmark in self.world.landmarks:
            if landmark.name == "Left Goal Back":
                landmark.set_pos(
                    torch.tensor(
                        [
                            -self.pitch_length / 2 - self.goal_depth + self.agent_size,
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
            elif landmark.name == "Right Goal Back":
                landmark.set_pos(
                    torch.tensor(
                        [
                            self.pitch_length / 2 + self.goal_depth - self.agent_size,
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
            elif landmark.name == "Left Goal Top":
                landmark.set_pos(
                    torch.tensor(
                        [
                            -self.pitch_length / 2
                            - self.goal_depth / 2
                            + self.agent_size,
                            self.goal_size / 2,
                        ],
                        dtype=torch.float32,
                        device=self.world.device,
                    ),
                    batch_index=env_index,
                )
            elif landmark.name == "Left Goal Bottom":
                landmark.set_pos(
                    torch.tensor(
                        [
                            -self.pitch_length / 2
                            - self.goal_depth / 2
                            + self.agent_size,
                            -self.goal_size / 2,
                        ],
                        dtype=torch.float32,
                        device=self.world.device,
                    ),
                    batch_index=env_index,
                )
            elif landmark.name == "Right Goal Top":
                landmark.set_pos(
                    torch.tensor(
                        [
                            self.pitch_length / 2
                            + self.goal_depth / 2
                            - self.agent_size,
                            self.goal_size / 2,
                        ],
                        dtype=torch.float32,
                        device=self.world.device,
                    ),
                    batch_index=env_index,
                )
            elif landmark.name == "Right Goal Bottom":
                landmark.set_pos(
                    torch.tensor(
                        [
                            self.pitch_length / 2
                            + self.goal_depth / 2
                            - self.agent_size,
                            -self.goal_size / 2,
                        ],
                        dtype=torch.float32,
                        device=self.world.device,
                    ),
                    batch_index=env_index,
                )
            elif landmark.name == "Red Net":
                landmark.set_pos(
                    torch.tensor(
                        [
                            self.pitch_length / 2
                            + self.goal_depth / 2
                            - self.agent_size / 2,
                            0.0,
                        ],
                        dtype=torch.float32,
                        device=self.world.device,
                    ),
                    batch_index=env_index,
                )
            elif landmark.name == "Blue Net":
                landmark.set_pos(
                    torch.tensor(
                        [
                            -self.pitch_length / 2
                            - self.goal_depth / 2
                            + self.agent_size / 2,
                            0.0,
                        ],
                        dtype=torch.float32,
                        device=self.world.device,
                    ),
                    batch_index=env_index,
                )

    def init_traj_pts(self, world):
        world.traj_points = {"Red": {}, "Blue": {}}
        if self.ai_red_agents:
            for i, agent in enumerate(world.red_agents):
                world.traj_points["Red"][agent] = []
                for j in range(self.n_traj_points):
                    pointj = Landmark(
                        name="Red {agent} Trajectory {pt}".format(agent=i, pt=j),
                        collide=False,
                        movable=False,
                        shape=Sphere(radius=0.01),
                        color=Color.GRAY,
                    )
                    world.add_landmark(pointj)
                    world.traj_points["Red"][agent].append(pointj)
        if self.ai_blue_agents:
            for i, agent in enumerate(world.blue_agents):
                world.traj_points["Blue"][agent] = []
                for j in range(self.n_traj_points):
                    pointj = Landmark(
                        name="Blue {agent} Trajectory {pt}".format(agent=i, pt=j),
                        collide=False,
                        movable=False,
                        shape=Sphere(radius=0.01),
                        color=Color.GRAY,
                    )
                    world.add_landmark(pointj)
                    world.traj_points["Blue"][agent].append(pointj)

    def process_action(self, agent: Agent):
        if self.enable_shooting and agent.action_script is None:
            rel_pos = self.ball.state.pos - agent.state.pos
            agent.ball_within_range = (
                torch.linalg.vector_norm(rel_pos, dim=-1) <= self.shooting_radius
            )

            rel_pos_angle = torch.atan2(rel_pos[:, Y], rel_pos[:, X])
            a = (agent.state.rot.squeeze(-1) - rel_pos_angle + torch.pi) % (
                2 * torch.pi
            ) - torch.pi
            agent.ball_within_angle = (
                -self.shooting_angle / 2 <= a <= self.shooting_angle / 2
            )

            shoot_force = torch.zeros(
                self.world.batch_dim, 2, device=self.world.device, dtype=torch.float32
            )
            shoot_force[..., X] = agent.action.u[..., -1] + self.u_shoot_multiplier / 2
            shoot_force = TorchUtils.rotate_vector(shoot_force, agent.state.rot)
            shoot_force = torch.where(
                (agent.ball_within_angle * agent.ball_within_range).unsqueeze(-1),
                shoot_force,
                0.0,
            )

            self.ball.kicking_action += shoot_force
            agent.action.u = agent.action.u[:, :-1]

    def pre_step(self):
        if self.enable_shooting:
            self.ball.action.u += self.ball.kicking_action
            self.ball.kicking_action[:] = 0

    def reward(self, agent: Agent):
        # Called with agent=None when only AIs are playing to compute the _done
        if agent is None or agent == self.world.agents[0]:
            # Sparse Reward
            over_right_line = (
                self.ball.state.pos[:, 0] > self.pitch_length / 2 + self.ball_size / 2
            )
            over_left_line = (
                self.ball.state.pos[:, 0] < -self.pitch_length / 2 - self.ball_size / 2
            )
            blue_score = over_right_line
            red_score = over_left_line
            self._sparse_reward = (
                self.scoring_reward * blue_score - self.scoring_reward * red_score
            )
            self._reward = self._sparse_reward
            self._done = blue_score | red_score
            # Dense Reward
            if self.dense_reward and agent is not None:
                self._reward = self._reward + self.reward_ball_to_goal()
                if self.only_closest_agent_ball_reward:
                    self._reward = self._reward + self.reward_all_agent_to_ball()
        if self.dense_reward and agent is not None:
            reward = self._reward
            if not self.only_closest_agent_ball_reward:
                reward = self._reward + self.reward_agent_to_ball(agent)
        else:
            reward = self._reward
        return reward

    def reward_ball_to_goal(self):
        self.ball.distance_to_goal = torch.linalg.vector_norm(
            self.ball.state.pos - self.right_goal_pos,
            dim=-1,
        )

        pos_shaping = self.ball.distance_to_goal * self.pos_shaping_factor_ball_goal
        self.ball.pos_rew = self.ball.pos_shaping - pos_shaping
        self.ball.pos_shaping = pos_shaping
        return self.ball.pos_rew

    def reward_agent_to_ball(self, agent: Agent):
        agent.distance_to_goal = torch.linalg.vector_norm(
            agent.state.pos - self.ball.state.pos,
            dim=-1,
        )
        pos_shaping = agent.distance_to_goal * self.pos_shaping_factor_agent_ball
        ball_moving = torch.linalg.vector_norm(self.ball.state.vel, dim=-1) > 1e-6
        agent_close_to_goal = agent.distance_to_goal < self.distance_to_ball_trigger
        agent.pos_rew = torch.where(
            ball_moving + agent_close_to_goal,
            0.0,
            agent.pos_shaping - pos_shaping,
        )
        agent.pos_shaping = pos_shaping
        return agent.pos_rew

    def reward_all_agent_to_ball(self):
        min_dist_to_ball = self.get_closest_agent_to_ball(env_index=None)
        pos_shaping = min_dist_to_ball * self.pos_shaping_factor_agent_ball

        ball_moving = torch.linalg.vector_norm(self.ball.state.vel, dim=-1) > 1e-6
        agent_close_to_goal = min_dist_to_ball < self.distance_to_ball_trigger
        self.ball.pos_rew_agent = torch.where(
            agent_close_to_goal + ball_moving,
            0.0,
            self.ball.pos_shaping_agent - pos_shaping,
        )
        self.ball.pos_shaping_agent = pos_shaping
        return self.ball.pos_rew_agent

    def observation(
        self,
        agent: Agent,
        agent_pos=None,
        agent_vel=None,
        agent_force=None,
        teammate_poses=None,
        teammate_forces=None,
        teammate_vels=None,
        adversary_poses=None,
        adversary_forces=None,
        adversary_vels=None,
        ball_pos=None,
        ball_vel=None,
        ball_force=None,
        env_index=Ellipsis,
    ):

        actual_adversary_poses = []
        actual_adversary_forces = []
        actual_adversary_vels = []
        if self.observe_adversaries:
            for a in self.red_agents:
                actual_adversary_poses.append(a.state.pos[env_index])
                actual_adversary_vels.append(a.state.vel[env_index])
                actual_adversary_forces.append(a.state.force[env_index])

        actual_teammate_poses = []
        actual_teammate_forces = []
        actual_teammate_vels = []
        if self.observe_teammates:
            for a in self.blue_agents:
                if a != agent:
                    actual_teammate_poses.append(a.state.pos[env_index])
                    actual_teammate_vels.append(a.state.vel[env_index])
                    actual_teammate_forces.append(a.state.force[env_index])

        obs = self.observation_base(
            agent.state.pos[env_index] if agent_pos is None else agent_pos,
            agent.state.vel[env_index] if agent_vel is None else agent_vel,
            agent.state.force[env_index] if agent_force is None else agent_force,
            ball_pos=self.ball.state.pos[env_index] if ball_pos is None else ball_pos,
            ball_vel=self.ball.state.vel[env_index] if ball_vel is None else ball_vel,
            ball_force=self.ball.state.force[env_index]
            if ball_force is None
            else ball_force,
            adversary_poses=actual_adversary_poses
            if adversary_poses is None
            else adversary_poses,
            adversary_forces=actual_adversary_forces
            if adversary_forces is None
            else adversary_forces,
            adversary_vels=actual_adversary_vels
            if adversary_vels is None
            else adversary_vels,
            teammate_poses=actual_teammate_poses
            if teammate_poses is None
            else teammate_poses,
            teammate_forces=actual_teammate_forces
            if teammate_forces is None
            else teammate_forces,
            teammate_vels=actual_teammate_vels
            if teammate_vels is None
            else teammate_vels,
        )
        return obs

    def observation_base(
        self,
        agent_pos,
        agent_vel,
        agent_force,
        teammate_poses,
        teammate_forces,
        teammate_vels,
        adversary_poses,
        adversary_forces,
        adversary_vels,
        ball_pos,
        ball_vel,
        ball_force,
    ):
        # Make all inputs same batch size (this is needed when this function is called for rendering
        input = [
            agent_pos,
            agent_vel,
            agent_force,
            ball_pos,
            ball_vel,
            ball_force,
            teammate_poses,
            teammate_forces,
            teammate_vels,
            adversary_poses,
            adversary_forces,
            adversary_vels,
        ]
        for o in input:
            if isinstance(o, Tensor) and len(o.shape) > 1:
                batch_dim = o.shape[0]
                break
        for j in range(len(input)):
            if isinstance(input[j], Tensor):
                if len(input[j].shape) == 1:
                    input[j] = input[j].unsqueeze(0).expand(batch_dim, *input[j].shape)
            else:
                o = input[j]
                for i in range(len(o)):
                    if len(o[i].shape) == 1:
                        o[i] = o[i].unsqueeze(0).expand(batch_dim, *o[i].shape)

        (
            agent_pos,
            agent_vel,
            agent_force,
            ball_pos,
            ball_vel,
            ball_force,
            teammate_poses,
            teammate_forces,
            teammate_vels,
            adversary_poses,
            adversary_forces,
            adversary_vels,
        ) = input
        #  End rendering code

        obs = {
            "obs": [
                agent_force,
                agent_pos - ball_pos,
                agent_vel - ball_vel,
                ball_pos - self.right_goal_pos,
                ball_vel,
                ball_force,
            ],
            "pos": [agent_pos - self.right_goal_pos],
            "vel": [agent_vel],
        }

        if self.observe_adversaries and len(adversary_poses):
            obs["adversaries"] = []
            for adversary_pos, adversary_force, adversary_vel in zip(
                adversary_poses, adversary_forces, adversary_vels
            ):
                obs["adversaries"].append(
                    torch.cat(
                        [
                            agent_pos - adversary_pos,
                            agent_vel - adversary_vel,
                            adversary_vel,
                            adversary_force,
                        ],
                        dim=-1,
                    )
                )
            obs["adversaries"] = [
                torch.stack(obs["adversaries"], dim=-2)
                if self.dict_obs
                else torch.cat(obs["adversaries"], dim=-1)
            ]

        if self.observe_teammates:
            obs["teammates"] = []
            for teammate_pos, teammate_force, teammate_vel in zip(
                teammate_poses, teammate_forces, teammate_vels
            ):
                obs["teammates"].append(
                    torch.cat(
                        [
                            agent_pos - teammate_pos,
                            agent_vel - teammate_vel,
                            teammate_vel,
                            teammate_force,
                        ],
                        dim=-1,
                    )
                )
            obs["teammates"] = [
                torch.stack(obs["teammates"], dim=-2)
                if self.dict_obs
                else torch.cat(obs["teammates"], dim=-1)
            ]

        for key, value in obs.items():
            obs[key] = torch.cat(value, dim=-1)
        if self.dict_obs:
            return obs
        else:
            return torch.cat(list(obs.values()), dim=-1)

    def done(self):
        if self.ai_blue_agents and self.ai_red_agents:
            self.reward(None)
        return self._done

    def info(self, agent: Agent):
        return {
            "sparse_reward": self._sparse_reward,
            "agent_ball_pos_rew": agent.pos_rew,
            "ball_goal_pos_rew": self.ball.pos_rew,
            "all_agent_ball_pos_rew": self.ball.pos_rew_agent,
        }

    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        from vmas.simulator import rendering
        from vmas.simulator.rendering import Geom

        # Background
        geoms: List[Geom] = (
            self._get_background_geoms(self.background_entities)
            if self._render_field
            else self._get_background_geoms(self.background_entities[3:])
        )

        # Agent rotation
        if self.enable_shooting:
            for agent in self.world.policy_agents:
                color = agent.color
                if (
                    agent.ball_within_angle[env_index]
                    and agent.ball_within_range[env_index]
                ):
                    color = Color.PINK.value
                sector = rendering.make_circle(
                    radius=self.shooting_radius, angle=self.shooting_angle, filled=True
                )
                xform = rendering.Transform()
                xform.set_rotation(agent.state.rot[env_index])
                xform.set_translation(*agent.state.pos[env_index])
                sector.add_attr(xform)
                sector.set_color(*color, alpha=agent._alpha / 2)
                geoms.append(sector)

        return geoms

    def _get_background_geoms(self, objects):
        def _get_geom(entity, pos, rot=0.0):
            from vmas.simulator import rendering

            geom = entity.shape.get_geometry()
            xform = rendering.Transform()
            geom.add_attr(xform)
            xform.set_translation(*pos)
            xform.set_rotation(rot)
            color = entity.color
            geom.set_color(*color)
            return geom

        geoms = []
        for landmark in objects:
            if landmark.name == "Centre Line":
                geoms.append(_get_geom(landmark, [0.0, 0.0], torch.pi / 2))
            elif landmark.name == "Right Line":
                geoms.append(
                    _get_geom(
                        landmark,
                        [self.pitch_length / 2 - self.agent_size, 0.0],
                        torch.pi / 2,
                    )
                )
            elif landmark.name == "Left Line":
                geoms.append(
                    _get_geom(
                        landmark,
                        [-self.pitch_length / 2 + self.agent_size, 0.0],
                        torch.pi / 2,
                    )
                )
            elif landmark.name == "Top Line":
                geoms.append(
                    _get_geom(landmark, [0.0, self.pitch_width / 2 - self.agent_size])
                )
            elif landmark.name == "Bottom Line":
                geoms.append(
                    _get_geom(landmark, [0.0, -self.pitch_width / 2 + self.agent_size])
                )
            else:
                geoms.append(_get_geom(landmark, [0, 0]))
        return geoms


# Ball Physics


def ball_action_script(ball, world):
    # Avoid getting stuck against the wall
    dist_thres = world.agent_size * 2
    vel_thres = 0.3
    impulse = 0.05
    upper = (
        1
        - torch.minimum(
            world.pitch_width / 2 - ball.state.pos[:, 1],
            torch.tensor(dist_thres, device=world.device),
        )
        / dist_thres
    )
    lower = (
        1
        - torch.minimum(
            world.pitch_width / 2 + ball.state.pos[:, 1],
            torch.tensor(dist_thres, device=world.device),
        )
        / dist_thres
    )
    right = (
        1
        - torch.minimum(
            world.pitch_length / 2 - ball.state.pos[:, 0],
            torch.tensor(dist_thres, device=world.device),
        )
        / dist_thres
    )
    left = (
        1
        - torch.minimum(
            world.pitch_length / 2 + ball.state.pos[:, 0],
            torch.tensor(dist_thres, device=world.device),
        )
        / dist_thres
    )
    vertical_vel = (
        1
        - torch.minimum(
            torch.abs(ball.state.vel[:, 1]),
            torch.tensor(vel_thres, device=world.device),
        )
        / vel_thres
    )
    horizontal_vel = (
        1
        - torch.minimum(
            torch.abs(ball.state.vel[:, 1]),
            torch.tensor(vel_thres, device=world.device),
        )
        / vel_thres
    )
    dist_action = torch.stack([left - right, lower - upper], dim=1)
    vel_action = torch.stack([horizontal_vel, vertical_vel], dim=1)
    actions = dist_action * vel_action * impulse
    goal_mask = (ball.state.pos[:, 1] < world.goal_size / 2) * (
        ball.state.pos[:, 1] > -world.goal_size / 2
    )
    actions[goal_mask, 0] = 0
    ball.action.u = actions


# Agent Policy


class AgentPolicy:
    def __init__(
        self,
        team: str,
        strength=1.0,
        disabled: bool = False,
    ):
        self.team_name = team
        self.otherteam_name = "Blue" if (self.team_name == "Red") else "Red"

        self.pos_lookahead = 0.01
        self.vel_lookahead = 0.01
        self.strength = strength
        self.strength_multiplier = 25.0

        self.dribble_speed = 0.16
        self.dribble_slowdown_dist = 0.3
        self.initial_vel_dist_behind_target_frac = 0.6
        self.ball_pos_eps = 0.08

        self.max_shoot_dist = 0.6
        self.valid_start_pos_angle = math.cos(torch.pi / 4)
        self.valid_start_vel_angle = math.cos(torch.pi / 4)
        self.valid_start_dist = 0.12
        self.dist_to_hit_speed = 1.7
        self.start_vel_mag_shoot = 1.0
        self.touch_eps = 0.01
        self.shoot_on_goal_dist = 0.4

        self.possession_lookahead = 0.5

        self.passing_angle = (2 * torch.pi / 128) * 1
        self.shooting_angle = (2 * torch.pi / 128) * 3
        self.shooting_dist = self.max_shoot_dist
        self.passing_dist = self.max_shoot_dist

        self.nsamples = 1
        self.sigma = 0.5
        self.replan_margin = 0.0

        self.initialised = False
        self.disabled = disabled

    def init(self, world):
        self.initialised = True
        self.world = world

        self.ball = self.world.ball
        if self.team_name == "Red":
            self.teammates = self.world.red_agents
            self.opposition = self.world.blue_agents
            self.own_net = self.world.red_net
            self.target_net = self.world.blue_net
        elif self.team_name == "Blue":
            self.teammates = self.world.blue_agents
            self.opposition = self.world.red_agents
            self.own_net = self.world.blue_net
            self.target_net = self.world.red_net

        self.actions = {
            agent: {
                "dribbling": torch.zeros(
                    self.world.batch_dim, device=world.device, dtype=torch.bool
                ),
                "shooting": torch.zeros(
                    self.world.batch_dim, device=world.device, dtype=torch.bool
                ),
                "pre-shooting": torch.zeros(
                    self.world.batch_dim, device=world.device, dtype=torch.bool
                ),
            }
            for agent in self.teammates
        }

        self.objectives = {
            agent: {
                "target_pos": torch.zeros(
                    self.world.batch_dim, self.world.dim_p, device=world.device
                ),
                "target_vel": torch.zeros(
                    self.world.batch_dim, self.world.dim_p, device=world.device
                ),
                "start_pos": torch.zeros(
                    self.world.batch_dim, self.world.dim_p, device=world.device
                ),
                "start_vel": torch.zeros(
                    self.world.batch_dim, self.world.dim_p, device=world.device
                ),
            }
            for agent in self.teammates
        }

        self.agent_possession = {
            agent: torch.zeros(
                self.world.batch_dim, device=world.device, dtype=torch.bool
            )
            for agent in self.teammates
        }

        self.team_possession = torch.zeros(
            self.world.batch_dim, device=world.device, dtype=torch.bool
        )

        self.team_disps = {agent: None for agent in self.teammates}

    def reset(self, env_index=Ellipsis):
        for agent in self.teammates:
            self.actions[agent]["dribbling"][env_index] = False
            self.actions[agent]["shooting"][env_index] = False
            self.actions[agent]["pre-shooting"][env_index] = False
            self.objectives[agent]["target_pos"][env_index] = torch.zeros(
                self.world.dim_p, device=self.world.device
            )
            self.objectives[agent]["target_vel"][env_index] = torch.zeros(
                self.world.dim_p, device=self.world.device
            )
            self.objectives[agent]["start_pos"][env_index] = torch.zeros(
                self.world.dim_p, device=self.world.device
            )
            self.objectives[agent]["start_vel"][env_index] = torch.zeros(
                self.world.dim_p, device=self.world.device
            )

    def dribble_policy(self, agent):
        possession_mask = self.agent_possession[agent]
        self.dribble_to_goal(agent, env_index=possession_mask)
        move_mask = ~possession_mask
        best_pos = self.check_better_positions(agent)
        self.go_to(
            agent,
            pos=best_pos[move_mask],
            vel=torch.zeros(
                move_mask.sum(), self.world.dim_p, device=self.world.device
            ),
            env_index=move_mask,
        )

    def disable(self):
        self.disabled = True

    def enable(self):
        self.disabled = False

    def run(self, agent, world):
        if not self.disabled:
            if "0" in agent.name:
                self.check_possession()
            self.dribble_policy(agent)
            control = self.get_action(agent)
            control = torch.clamp(control, min=-agent.u_range, max=agent.u_range)
            agent.action.u = control * agent.u_multiplier
        else:
            agent.action.u = torch.zeros(
                self.world.batch_dim,
                self.world.dim_p,
                device=self.world.device,
                dtype=torch.float,
            )

    def dribble_to_goal(self, agent, env_index=Ellipsis):
        self.dribble(agent, self.target_net.state.pos, env_index=env_index)

    def dribble(self, agent, pos, env_index=Ellipsis):
        if isinstance(env_index, int):
            env_index = [env_index]
        self.actions[agent]["dribbling"][env_index] = True
        self.update_dribble(
            agent,
            pos=pos[env_index],
            env_index=env_index,
        )

    def update_dribble(self, agent, pos, env_index=Ellipsis):
        agent_pos = agent.state.pos[env_index]
        ball_pos = self.ball.state.pos[env_index]
        ball_disp = pos - ball_pos
        ball_dist = ball_disp.norm(dim=-1)
        direction = ball_disp / ball_dist[:, None]
        hit_vel = direction * self.dribble_speed
        start_vel = self.get_start_vel(ball_pos, hit_vel, agent_pos)
        start_vel_mag = start_vel.norm(dim=-1)
        offset = start_vel.clone()
        start_vel_mag_mask = start_vel_mag > 0
        offset[start_vel_mag_mask] /= start_vel_mag.unsqueeze(-1)[start_vel_mag_mask]
        new_direction = direction + 0.5 * offset
        new_direction /= new_direction.norm(dim=-1)[:, None]
        hit_pos = (
            ball_pos
            - new_direction * (self.ball.shape.radius + agent.shape.radius) * 0.7
        )
        self.go_to(agent, hit_pos, hit_vel, start_vel=start_vel, env_index=env_index)

    def go_to(self, agent, pos, vel, start_vel=None, env_index=Ellipsis):
        start_pos = agent.state.pos[env_index]
        if start_vel is None:
            start_vel = self.get_start_vel(pos, vel, start_pos)
        self.objectives[agent]["target_pos"][env_index] = pos
        self.objectives[agent]["target_vel"][env_index] = vel
        self.objectives[agent]["start_pos"][env_index] = start_pos
        self.objectives[agent]["start_vel"][env_index] = start_vel
        self.plot_traj(agent, env_index=env_index)

    def get_start_vel(self, pos, vel, start_pos):
        goal_disp = pos - start_pos
        goal_dist = goal_disp.norm(dim=-1)
        vel_mag = vel.norm(dim=-1)
        vel_dir = vel.clone()
        vel_dir[vel_mag > 0] /= vel_mag[vel_mag > 0, None]
        dist_behind_target = self.initial_vel_dist_behind_target_frac * goal_dist
        target_pos = pos - vel_dir * dist_behind_target[:, None]
        target_disp = target_pos - start_pos
        target_dist = target_disp.norm(dim=1)
        start_vel_aug_dir = target_disp
        start_vel_aug_dir[target_dist > 0] /= target_dist[target_dist > 0, None]
        start_vel = start_vel_aug_dir * vel_mag[:, None]
        return start_vel

    def get_action(self, agent, env_index=Ellipsis):
        curr_pos = agent.state.pos[env_index, :]
        curr_vel = agent.state.vel[env_index, :]
        u_start = torch.zeros(curr_pos.shape[0], device=self.world.device)
        des_curr_pos = self.hermite(
            self.objectives[agent]["start_pos"][env_index, :],
            self.objectives[agent]["target_pos"][env_index, :],
            self.objectives[agent]["start_vel"][env_index, :],
            self.objectives[agent]["target_vel"][env_index, :],
            u=torch.minimum(
                u_start + self.pos_lookahead,
                torch.tensor(1.0, device=self.world.device),
            ),
            deriv=0,
        )
        des_curr_vel = self.hermite(
            self.objectives[agent]["start_pos"][env_index, :],
            self.objectives[agent]["target_pos"][env_index, :],
            self.objectives[agent]["start_vel"][env_index, :],
            self.objectives[agent]["target_vel"][env_index, :],
            u=torch.minimum(
                u_start + self.vel_lookahead,
                torch.tensor(1.0, device=self.world.device),
            ),
            deriv=1,
        )

        des_curr_pos = torch.as_tensor(des_curr_pos, device=self.world.device)
        des_curr_vel = torch.as_tensor(des_curr_vel, device=self.world.device)
        control = 0.5 * (des_curr_pos - curr_pos) + 0.5 * (des_curr_vel - curr_vel)
        return control * self.strength * self.strength_multiplier

    def hermite(self, p0, p1, p0dot, p1dot, u=0.1, deriv=0):
        # Formatting
        u = u.reshape((-1,))

        # Calculation
        U = torch.stack(
            [
                self.nPr(3, deriv) * (u ** max(0, 3 - deriv)),
                self.nPr(2, deriv) * (u ** max(0, 2 - deriv)),
                self.nPr(1, deriv) * (u ** max(0, 1 - deriv)),
                self.nPr(0, deriv) * (u**0),
            ],
            dim=1,
        ).float()
        A = torch.tensor(
            [
                [2.0, -2.0, 1.0, 1.0],
                [-3.0, 3.0, -2.0, -1.0],
                [0.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
            device=U.device,
        )
        P = torch.stack([p0, p1, p0dot, p1dot], dim=1)
        ans = U[:, None, :] @ A[None, :, :] @ P
        ans = ans.squeeze(1)
        return ans

    def plot_traj(self, agent, env_index=0):
        for i, u in enumerate(
            torch.linspace(0, 1, len(self.world.traj_points[self.team_name][agent]))
        ):
            pointi = self.world.traj_points[self.team_name][agent][i]
            num_envs = self.objectives[agent]["start_pos"][env_index, :].shape[0]
            posi = self.hermite(
                self.objectives[agent]["start_pos"][env_index, :],
                self.objectives[agent]["target_pos"][env_index, :],
                self.objectives[agent]["start_vel"][env_index, :],
                self.objectives[agent]["target_vel"][env_index, :],
                u=torch.tensor([u] * num_envs, device=self.world.device),
                deriv=0,
            )
            if env_index == Ellipsis or (
                isinstance(env_index, torch.Tensor)
                and env_index.dtype == torch.bool
                and torch.all(env_index)
            ):
                pointi.set_pos(
                    torch.as_tensor(posi, device=self.world.device),
                    batch_index=None,
                )
            elif isinstance(env_index, int):
                pointi.set_pos(
                    torch.as_tensor(posi, device=self.world.device),
                    batch_index=env_index,
                )
            elif isinstance(env_index, list):
                for envi in env_index:
                    pointi.set_pos(
                        torch.as_tensor(posi, device=self.world.device)[envi, :],
                        batch_index=env_index[envi],
                    )
            elif (
                isinstance(env_index, torch.Tensor)
                and env_index.dtype == torch.bool
                and torch.any(env_index)
            ):
                envs = torch.where(env_index)
                for i, envi in enumerate(envs):
                    pointi.set_pos(
                        torch.as_tensor(posi, device=self.world.device)[i, :],
                        batch_index=envi[0],
                    )

    def clamp_pos(self, pos, return_bool=False):
        orig_pos = pos.clone()
        agent_size = self.world.agent_size
        pitch_y = self.world.pitch_width / 2 - agent_size
        pitch_x = self.world.pitch_length / 2 - agent_size
        goal_y = self.world.goal_size / 2 - agent_size
        goal_x = self.world.goal_depth
        pos[:, Y] = torch.clamp(pos[:, Y], -pitch_y, pitch_y)
        inside_goal_y_mask = torch.abs(pos[:, Y]) < goal_y
        pos[~inside_goal_y_mask, X] = torch.clamp(
            pos[~inside_goal_y_mask, X], -pitch_x, pitch_x
        )
        pos[inside_goal_y_mask, X] = torch.clamp(
            pos[inside_goal_y_mask, X], -pitch_x - goal_x, pitch_x + goal_x
        )
        if return_bool:
            return torch.any(pos != orig_pos, dim=-1)
        else:
            return pos

    def nPr(self, n, r):
        if r > n:
            return 0
        ans = 1
        for k in range(n, max(1, n - r), -1):
            ans = ans * k
        return ans

    def check_possession(self, env_index=Ellipsis):
        agents_pos = torch.stack(
            [agent.state.pos[env_index] for agent in self.teammates + self.opposition],
            dim=1,
        )
        agents_vel = torch.stack(
            [agent.state.vel[env_index] for agent in self.teammates + self.opposition],
            dim=1,
        )
        ball_pos = self.ball.state.pos[env_index]
        ball_vel = self.ball.state.vel[env_index]
        disps = ball_pos[:, None, :] - agents_pos
        relvels = ball_vel[:, None, :] - agents_vel
        dists = (disps + relvels * self.possession_lookahead).norm(dim=-1)
        mindist_agent = torch.argmin(dists[:, : len(self.teammates)], dim=-1)
        mindist_team = torch.argmin(dists, dim=-1) < len(self.teammates)
        for i, agent in enumerate(self.teammates):
            self.agent_possession[agent][env_index] = mindist_agent == i
        self.team_possession[env_index] = mindist_team

    def check_better_positions(self, agent, env_index=Ellipsis):
        self.team_disps[agent] = None
        ball_pos = self.ball.state.pos[env_index]
        curr_target = self.objectives[agent]["target_pos"]
        samples = (
            torch.randn(
                self.nsamples,
                ball_pos.shape[0],
                self.world.dim_p,
                device=self.world.device,
            )
            * self.sigma
            + ball_pos[None, :, :]
        )
        test_pos = torch.cat(
            [curr_target[None, :, :], samples], dim=0
        )  # curr_pos[None,:,:],
        test_pos_shape = test_pos.shape
        test_pos = self.clamp_pos(
            test_pos.view(test_pos_shape[0] * test_pos_shape[1], test_pos_shape[2])
        ).view(*test_pos_shape)
        values = torch.stack(
            [
                self.get_pos_value(test_pos[i], agent=agent, env_index=env_index)
                for i in range(test_pos.shape[0])
            ],
            dim=0,
        )
        values[0, :] += self.replan_margin
        highest_value = values.argmax(dim=0)
        best_pos = torch.gather(
            test_pos,
            dim=0,
            index=highest_value.unsqueeze(0)
            .unsqueeze(-1)
            .expand(-1, -1, self.world.dim_p),
        )
        return best_pos[0, :, :]

    def get_pos_value(self, pos, agent, env_index=Ellipsis):

        ball_dist = (pos - self.ball.state.pos[env_index]).norm(dim=-1)
        ball_dist_value = 2.0 * -((ball_dist - self.max_shoot_dist) ** 2)

        side_dot_prod = (
            (self.ball.state.pos - pos) * (self.target_net.state.pos - pos)
        ).sum(dim=-1)
        side_value = 0.5 * side_dot_prod

        if len(self.teammates) > 1:
            if self.team_disps[agent] is not None:
                team_disps = self.team_disps[agent]
            else:
                team_disps = self.get_separations(
                    agent=agent,
                    teammate=True,
                    target=True,
                    wall=False,
                    opposition=False,
                    env_index=env_index,
                )
                team_disps = torch.stack(team_disps, dim=1)
                self.team_disps[agent] = team_disps

            team_dists = (team_disps - pos[:, None, :]).norm(dim=-1)
            other_agent_value = -0.2 * (team_dists**-2).mean(dim=-1)
        else:
            other_agent_value = 0

        wall_disps = self.get_separations(
            pos,
            agent,
            teammate=False,
            wall=True,
            opposition=False,
            env_index=env_index,
        )
        wall_disps = torch.stack(wall_disps, dim=1)

        wall_dists = wall_disps.norm(dim=-1)
        wall_value = -0.01 * (wall_dists**-2).mean(dim=-1)

        return wall_value + other_agent_value + ball_dist_value + side_value

    def get_separations(
        self,
        pos=None,
        agent=None,
        teammate=True,
        wall=True,
        opposition=False,
        env_index=Ellipsis,
        target=False,
    ):
        disps = []
        if wall:
            top_wall_dist = -pos[:, Y] + self.world.pitch_width / 2
            bottom_wall_dist = pos[:, Y] + self.world.pitch_width / 2
            left_wall_dist = pos[:, X] + self.world.pitch_length / 2
            right_wall_dist = -pos[:, X] + self.world.pitch_length / 2
            vertical_wall_disp = torch.zeros(pos.shape, device=self.world.device)
            vertical_wall_disp[:, Y] = torch.minimum(top_wall_dist, bottom_wall_dist)
            vertical_wall_disp[bottom_wall_dist < top_wall_dist, Y] *= -1
            horizontal_wall_disp = torch.zeros(pos.shape, device=self.world.device)
            horizontal_wall_disp[:, X] = torch.minimum(left_wall_dist, right_wall_dist)
            horizontal_wall_disp[left_wall_dist < right_wall_dist, X] *= -1
            disps.append(vertical_wall_disp)
            disps.append(horizontal_wall_disp)
        if teammate:
            for otheragent in self.teammates:
                if otheragent != agent:
                    if target:
                        agent_disp = self.objectives[otheragent]["target_pos"][
                            env_index
                        ]
                    else:
                        agent_disp = otheragent.state.pos[env_index]
                    if pos is not None:
                        agent_disp -= pos
                    disps.append(agent_disp)
        if opposition:
            for otheragent in self.opposition:
                if otheragent != agent:
                    agent_disp = otheragent.state.pos[env_index]
                    if pos is not None:
                        agent_disp -= pos
                    disps.append(agent_disp)
        return disps


# Run
if __name__ == "__main__":
    render_interactively(
        __file__,
        control_two_agents=False,
        n_blue_agents=2,
        n_red_agents=0,
        ai_blue_agents=False,
        dense_reward=True,
        ai_strength=1,
        n_traj_points=8,
    )
