#  Copyright (c) 2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import math
import typing

import torch
from vmas import render_interactively
from vmas.simulator.core import Agent, Landmark, World, Sphere, Line
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import Color, Y, X

if typing.TYPE_CHECKING:
    pass


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.plot_grid = False
        self.viewer_zoom = 1.7
        self.agents_start_y = 2
        self.min_collision_distance = 0.005

        self.n_agents = kwargs.get("n_agents", 2)
        self.u_range = kwargs.get("u_range", 1)
        self.n_obstacle_rows = kwargs.get("n_obstacle_rows", 5)
        self.n_obstacle_columns = kwargs.get("n_obstacle_columns", 7)
        self.start_with_sidegaps = kwargs.get("start_with_sidegaps", False)
        self.passages_wiggle_room_radius = kwargs.get("passages_wiggle_room_radius", 0)
        self.agents_distance_from_obstacle_layer = kwargs.get(
            "agents_distance_from_obstacle_layer", 0.7
        )
        self.agent_radius = kwargs.get("agent_radius", 0.1)
        self.obstacle_radius = kwargs.get("obstacle_radius", 0.1)
        self.agents_x_spawning_distance = kwargs.get("agents_x_spawning_distance", 0.1)
        self.rand_agent_order = kwargs.get("rand_agent_order", True)
        self.lidar_range = kwargs.get("lidar_range", 0.3)
        self.lidar = kwargs.get("lidar", False)
        self.boundaries = kwargs.get("boundaries", False)
        self.spawn_vertically = kwargs.get("spawn_vertically", True)
        self.random_x_start = kwargs.get("random_x_start", False)
        self.goal_2d = kwargs.get("goal_2d", False)
        self.side_space = kwargs.get("side_space", 4 * self.agent_radius)
        self.goal_distance_from_obstacles = kwargs.get(
            "goal_distance_from_obstacles",
            self.agents_distance_from_obstacle_layer + 0.5,
        )

        self.shared_rew = kwargs.get("shared_rew", True)
        self.pos_shaping_factor = kwargs.get("pos_shaping_factor", 1)
        self.final_reward = kwargs.get("final_reward", 0)
        self.agent_collision_penalty = kwargs.get("agent_collision_penalty", -1)

        self._compute_startup_positions_and_sizes(device)

        # Make world
        world = World(batch_dim, device, substeps=3, x_semidim=self.world_x_semidim)

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
        entity_filter_agents = lambda e: isinstance(e, Landmark)

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
                shape=Sphere(radius=self.agent_radius),
                render_action=True,
                u_range=self.u_range,
                sensors=[
                    Lidar(
                        world,
                        n_rays=8,
                        max_range=self.lidar_range,
                        entity_filter=entity_filter_agents,
                    ),
                ],
            )
            agent.pos_rew = torch.zeros(batch_dim, device=device)
            agent.agent_collision_rew = agent.pos_rew.clone()
            world.add_agent(agent)

        # Add obstacles
        for i in range(self.n_obstacles):
            obstacle = Landmark(
                name=f"obstacle_{i}",
                collide=True,
                movable=False,
                rotatable=False,
                shape=Sphere(radius=self.obstacle_radius),
            )
            world.add_landmark(obstacle)

        # Lines
        if self.boundaries:
            self.lines = []
            for i in range(2):
                line = Landmark(
                    name=f"line_{i}",
                    collide=True,
                    movable=False,
                    rotatable=False,
                    shape=Line(length=2 * self.flock_semidist + 2 * self.agent_radius),
                )
                self.lines.append(line)
                world.add_landmark(line)
            line = Landmark(
                name=f"line_{2}",
                collide=True,
                movable=False,
                rotatable=False,
                shape=Line(length=2 * self.agent_radius),
            )
            self.lines.append(line)
            world.add_landmark(line)

        self.pos_rew = torch.zeros(batch_dim, device=device)
        self.final_rew = self.pos_rew.clone()

        return world

    def _reset_lines(self, env_index):
        for i, line in enumerate(self.lines[:-1]):
            line.set_pos(
                torch.tensor(
                    [
                        -self.agent_radius if i == 0 else self.agent_radius,
                        self.agents_start_y + self.flock_semidist,
                    ],
                    device=self.world.device,
                    dtype=torch.float,
                ),
                batch_index=env_index,
            )
            line.set_rot(
                torch.tensor(
                    [torch.pi / 2],
                    device=self.world.device,
                    dtype=torch.float,
                ),
                batch_index=env_index,
            )
        self.lines[-1].set_pos(
            torch.tensor(
                [
                    0.0,
                    self.agents_start_y + 2 * self.flock_semidist + self.agent_radius,
                ],
                device=self.world.device,
                dtype=torch.float,
            ),
            batch_index=env_index,
        )

    def _compute_startup_positions_and_sizes(self, device):
        # How many obstacles
        self.n_obstacles = (self.n_obstacle_rows // 2) * (
            self.n_obstacle_columns + 1
        ) + (self.n_obstacle_rows // 2) * self.n_obstacle_columns
        if self.n_obstacle_rows % 2:
            self.n_obstacles += (
                self.n_obstacle_columns
                if self.start_with_sidegaps
                else self.n_obstacle_columns + 1
            )
        # Agents spacing
        self.agents_x_spawning_distance += self.agent_radius * 2
        # Gap radius
        self.gap_radius = self.passages_wiggle_room_radius + self.agent_radius
        # Side barriers radius
        self.world_x_semidim = (
            self.obstacle_radius + self.gap_radius
        ) * self.n_obstacle_columns + self.side_space
        # X spacing of obstacles
        self.obstacle_x_spacing = self.obstacle_radius * 2 + self.gap_radius * 2
        # Y spacing of obstacles
        self.y_distance_rows = math.sqrt(
            (2 * self.obstacle_radius + 2 * self.gap_radius) ** 2
            - (self.gap_radius + self.obstacle_radius) ** 2
        )
        self.y_obstacles_finish = (
            self.agents_start_y
            - self.agents_distance_from_obstacle_layer
            - (self.n_obstacle_rows - 1) * self.y_distance_rows
        )

        # Goal line
        self.goal = torch.tensor(
            [
                0,
                self.y_obstacles_finish - self.goal_distance_from_obstacles - 0.4,
            ],
            device=device,
            dtype=torch.float,
        )
        # Flock semidist
        self.flock_semidist = (
            (
                self.agents_x_spawning_distance / 2
                + self.agents_x_spawning_distance * ((self.n_agents // 2) - 1)
            )
            if not self.n_agents % 2
            else self.agents_x_spawning_distance * (self.n_agents // 2)
        )

    def reset_world_at(self, env_index: int = None):
        if self.boundaries:
            self._reset_lines(env_index)
        start_pos = torch.cat(
            [
                torch.zeros(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    (
                        -self.world_x_semidim / 2
                        + (self.flock_semidist if not self.spawn_vertically else 0)
                        if self.random_x_start
                        else 0
                    ),
                    (
                        (
                            self.world_x_semidim / 2 - self.flock_semidist
                            if not self.spawn_vertically
                            else 0
                        )
                        if self.random_x_start
                        else 0
                    ),
                ),
                torch.zeros(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    self.agents_start_y,
                    self.agents_start_y,
                ),
            ],
            dim=-1,
        )

        order = torch.randperm(self.n_agents).tolist()
        agents = [self.world.agents[i] for i in order]
        for i, agent in enumerate(
            agents if self.rand_agent_order else self.world.agents
        ):
            agent_pos = start_pos.clone()
            if self.spawn_vertically:
                agent_pos[..., Y] += self.agents_x_spawning_distance * i
            else:
                agent_pos[..., X] += (
                    -self.flock_semidist + self.agents_x_spawning_distance * i
                )

            agent.set_pos(
                agent_pos,
                batch_index=env_index,
            )
            if env_index is None:
                agent.pos_shaping = (
                    torch.abs(agent.state.pos[..., Y] - self.goal[Y])
                    if not self.goal_2d
                    else torch.linalg.vector_norm(agent.state.pos - self.goal, dim=-1)
                    * self.pos_shaping_factor
                )

            else:
                agent.pos_shaping[env_index] = (
                    torch.abs(agent.state.pos[env_index, Y] - self.goal[Y])
                    if not self.goal_2d
                    else torch.linalg.vector_norm(
                        agent.state.pos[env_index] - self.goal, dim=-1
                    )
                    * self.pos_shaping_factor
                )

        obstacle_total_index = 0
        for row in range(self.n_obstacle_rows):
            if not (row % 2):
                n_obstacles_this_row = (
                    self.n_obstacle_columns
                    if self.start_with_sidegaps
                    else self.n_obstacle_columns + 1
                )
            else:
                n_obstacles_this_row = (
                    self.n_obstacle_columns + 1
                    if self.start_with_sidegaps
                    else self.n_obstacle_columns
                )
            for obstacle_index in range(n_obstacles_this_row):
                if not n_obstacles_this_row % 2:
                    pos = [
                        -self.obstacle_x_spacing / 2
                        - self.obstacle_x_spacing * ((n_obstacles_this_row // 2) - 1)
                        + self.obstacle_x_spacing * obstacle_index,
                        self.agents_start_y
                        - self.agents_distance_from_obstacle_layer
                        - self.y_distance_rows * row,
                    ]
                else:
                    pos = [
                        -self.obstacle_x_spacing * (n_obstacles_this_row // 2)
                        + self.obstacle_x_spacing * obstacle_index,
                        self.agents_start_y
                        - self.agents_distance_from_obstacle_layer
                        - self.y_distance_rows * row,
                    ]
                obstacle = self.world.landmarks[obstacle_total_index]
                obstacle.set_pos(
                    torch.tensor(
                        pos,
                        device=self.world.device,
                    ),
                    batch_index=env_index,
                )
                obstacle_total_index += 1
        assert obstacle_total_index == self.n_obstacles

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            self.pos_rew[:] = 0
            self.final_rew[:] = 0

            for a in self.world.agents:
                self.pos_rew += self.agent_reward(a)
                a.agent_collision_rew[:] = 0

            self.all_goal_reached = torch.all(
                torch.stack([a.on_goal for a in self.world.agents], dim=-1), dim=-1
            )

            self.final_rew[self.all_goal_reached] = self.final_reward

            for i, a in enumerate(self.world.agents):
                for j, b in enumerate(self.world.agents):
                    if i <= j:
                        continue
                    if self.world.collides(a, b) and self.agent_collision_penalty < 0:
                        safe_pos = a.state.pos[..., Y] < self.y_obstacles_finish

                        distance = self.world.get_distance(a, b)
                        close = distance <= self.min_collision_distance
                        mask = close * ~safe_pos

                        a.agent_collision_rew[mask] += self.agent_collision_penalty
                        b.agent_collision_rew[mask] += self.agent_collision_penalty

        pos_reward = self.pos_rew if self.shared_rew else agent.pos_rew
        return pos_reward + self.final_rew + agent.agent_collision_rew

    def agent_reward(self, agent: Agent):
        agent.distance_to_goal = (
            torch.abs(agent.state.pos[..., Y] - self.goal[Y])
            if not self.goal_2d
            else torch.linalg.vector_norm(agent.state.pos - self.goal, dim=-1)
        )

        agent.on_goal = agent.distance_to_goal < agent.shape.radius

        pos_shaping = agent.distance_to_goal * self.pos_shaping_factor
        agent.pos_rew = agent.pos_shaping - pos_shaping
        agent.pos_shaping = pos_shaping
        return agent.pos_rew

    def observation(self, agent: Agent):
        return torch.cat(
            [
                agent.state.pos[..., Y].unsqueeze(-1),
            ]
            + ([self.lidar_range - agent.sensors[0].measure()] if self.lidar else []),
            dim=-1,
        )

    def extra_render(self, env_index: int = 0):
        from vmas.simulator import rendering

        geoms = []

        # Perimeter
        for i in range(2):
            geom = Line(length=50).get_geometry()
            xform = rendering.Transform()
            geom.add_attr(xform)

            dist = (
                self.world_x_semidim
                + self.agent_radius
                + self.passages_wiggle_room_radius
            )

            xform.set_translation(
                -dist if i == 0 else dist,
                0.0,
            )
            xform.set_rotation(torch.pi / 2)
            color = Color.BLACK.value
            if isinstance(color, torch.Tensor) and len(color.shape) > 1:
                color = color[env_index]
            geom.set_color(*color)
            geoms.append(geom)

        # Y goal

        geom = Line(length=dist * 2).get_geometry()
        xform = rendering.Transform()
        geom.add_attr(xform)

        xform.set_translation(*self.goal.tolist())
        color = Color.RED.value
        if isinstance(color, torch.Tensor) and len(color.shape) > 1:
            color = color[env_index]
        geom.set_color(*color)
        geoms.append(geom)

        return geoms


if __name__ == "__main__":
    render_interactively(
        __file__,
        control_two_agents=True,
    )
