#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.


import torch
from torch.distributions import MultivariateNormal

from vmas import render_interactively

from vmas.simulator.core import Agent, Landmark, Line, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils, TorchUtils, X, Y


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):

        self.left_corridor_multplier = 2
        self.right_corridor_multplier = 2
        self.up_corridor_multplier = 3
        self.down_corridor_multplier = 1

        self.shared_rew = kwargs.get("shared_rew", True)

        self.viewer_zoom = 3.1

        self.n_agents = 4
        self.agent_radius = 0.16

        self.xdim = kwargs.get("xdim", 5)
        self.ydim = kwargs.get("ydim", 5)
        self.grid_spacing = kwargs.get("grid_spacing", self.agent_radius * 2)

        self.plot_grid = True
        self.n_x_cells = int((2 * self.xdim) / self.grid_spacing)
        self.n_y_cells = int((2 * self.ydim) / self.grid_spacing)
        self.max_pdf = torch.zeros((batch_dim,), device=device, dtype=torch.float32)
        self.alpha_plot: float = 0.5

        self.cov = kwargs.get("cov", 0.05)
        self.n_gaussians = 4
        self.covs = (
            [self.cov] * self.n_gaussians if isinstance(self.cov, float) else self.cov
        )

        self.x_semidim = self.xdim - self.agent_radius
        self.y_semidim = self.ydim - self.agent_radius

        self.sampled = torch.zeros(
            (batch_dim, self.n_x_cells, self.n_y_cells),
            device=device,
            dtype=torch.bool,
        )

        self.locs = [
            torch.zeros((batch_dim, 2), device=device, dtype=torch.float32)
            for _ in range(self.n_gaussians)
        ]
        self.cov_matrices = [
            torch.tensor([[cov, 0], [0, cov]], dtype=torch.float32, device=device)
            .expand(batch_dim, 2, 2)
            .clone()
            for cov in self.covs
        ]

        # Make world#
        world = World(
            batch_dim,
            device,
            x_semidim=self.x_semidim,
            y_semidim=self.y_semidim,
        )

        self.colors = [Color.GREEN, Color.BLUE, Color.RED, Color.GRAY]

        # Add agents
        for i in range(self.n_agents):
            agent = Agent(
                name=f"agent_{i}",
                rotatable=False,
                shape=Sphere(radius=self.agent_radius),
                render_action=True,
                color=self.colors[i],
                collide=True,
                collision_filter=lambda e: isinstance(e.shape, Line),
            )

            world.add_agent(agent)

        self.spawn_map(world)

        return world

    def reset_world_at(self, env_index: int = None):
        ScenarioUtils.spawn_entities_randomly(
            self.world.agents,
            self.world,
            env_index,
            min_dist_between_entities=0,
            x_bounds=(
                -0,
                0,
            ),
            y_bounds=(
                -0,
                0,
            ),
        )

        self.reset_map(env_index)
        self.reset_distribution(env_index)

    def reset_distribution(self, env_index):
        for i in self.locs:
            x = torch.zeros(
                (1,) if env_index is not None else (self.world.batch_dim, 1),
                device=self.world.device,
                dtype=torch.float32,
            ).uniform_(-self.xdim, self.xdim)
            y = torch.zeros(
                (1,) if env_index is not None else (self.world.batch_dim, 1),
                device=self.world.device,
                dtype=torch.float32,
            ).uniform_(-self.ydim, self.ydim)
            new_loc = torch.cat([x, y], dim=-1)
            if env_index is None:
                self.locs[i] = new_loc
            else:
                self.locs[i][env_index] = new_loc

        self.gaussians = [
            MultivariateNormal(
                loc=loc,
                covariance_matrix=cov_matrix,
            )
            for loc, cov_matrix in zip(self.locs, self.cov_matrices)
        ]

        if env_index is None:
            self.max_pdf = torch.zeros_like(self.max_pdf)
            self.sampled = torch.zeros_like(self.sampled)
        else:
            self.max_pdf = TorchUtils.where_from_index(env_index, 0, self.max_pdf)
            self.sampled = TorchUtils.where_from_index(env_index, False, self.sampled)

        self.normalize_pdf(env_index=env_index)

    def normalize_pdf(self, env_index):
        xpoints = torch.arange(
            -self.xdim, self.xdim, self.grid_spacing, device=self.world.device
        )
        ypoints = torch.arange(
            -self.ydim, self.ydim, self.grid_spacing, device=self.world.device
        )
        if env_index is not None:
            ygrid, xgrid = torch.meshgrid(ypoints, xpoints, indexing="ij")
            pos = torch.stack((xgrid, ygrid), dim=-1).reshape(-1, 2)
            sample = self.sample_single_env(pos, env_index, norm=False)
            self.max_pdf = TorchUtils.where_from_index(
                env_index, sample.max(), self.max_pdf
            )
        else:
            for x in xpoints:
                for y in ypoints:
                    pos = torch.tensor(
                        [x, y], device=self.world.device, dtype=torch.float32
                    ).repeat(self.world.batch_dim, 1)
                    sample = self.sample(pos, norm=False)
                    self.max_pdf = torch.maximum(self.max_pdf, sample)

    def sample(
        self,
        pos,
        update_sampled_flag: bool = False,
        norm: bool = True,
    ):
        out_of_bounds = (
            (pos[:, X] < -self.xdim)
            + (pos[:, X] > self.xdim)
            + (pos[:, Y] < -self.ydim)
            + (pos[:, Y] > self.ydim)
        )
        pos[:, X].clamp_(-self.world.x_semidim, self.world.x_semidim)
        pos[:, Y].clamp_(-self.world.y_semidim, self.world.y_semidim)

        index = pos / self.grid_spacing
        index[:, X] += self.n_x_cells / 2
        index[:, Y] += self.n_y_cells / 2
        index = index.to(torch.long)
        v = torch.stack(
            [gaussian.log_prob(pos).exp() for gaussian in self.gaussians], dim=-1
        ).sum(-1)
        if norm:
            v = v / self.max_pdf

        sampled = self.sampled[
            torch.arange(self.world.batch_dim), index[:, 0], index[:, 1]
        ]

        v[sampled + out_of_bounds] = 0
        if update_sampled_flag:
            self.sampled[
                torch.arange(self.world.batch_dim), index[:, 0], index[:, 1]
            ] = True

        return v

    def sample_single_env(
        self,
        pos,
        env_index,
        norm: bool = True,
    ):
        pos = pos.view(-1, self.world.dim_p)

        out_of_bounds = (
            (pos[:, X] < -self.xdim)
            + (pos[:, X] > self.xdim)
            + (pos[:, Y] < -self.ydim)
            + (pos[:, Y] > self.ydim)
        )
        pos[:, X].clamp_(-self.x_semidim, self.x_semidim)
        pos[:, Y].clamp_(-self.y_semidim, self.y_semidim)

        index = pos / self.grid_spacing
        index[:, X] += self.n_x_cells / 2
        index[:, Y] += self.n_y_cells / 2
        index = index.to(torch.long)

        pos = pos.unsqueeze(1).expand(pos.shape[0], self.world.batch_dim, 2)

        v = torch.stack(
            [gaussian.log_prob(pos).exp() for gaussian in self.gaussians], dim=-1
        ).sum(-1)[:, env_index]
        if norm:
            v = v / self.max_pdf[env_index]

        sampled = self.sampled[env_index, index[:, 0], index[:, 1]]

        v[sampled + out_of_bounds] = 0

        return v

    def reward(self, agent: Agent):
        # _is_first = agent == self.world.agents[0]

        return torch.zeros(
            self.world.batch_dim, device=self.world.device, dtype=torch.float32
        )

    def observation(self, agent: Agent):

        return agent.state.pos

    def spawn_map(self, world: World):
        self.inner_box_semidim = 1
        self.corridors_width = self.agent_radius * 4
        self.corridors_length = 2
        self.room_semidim = 1.5

        self.left_corridor_length = self.corridors_length * self.left_corridor_multplier
        self.right_corridor_length = (
            self.corridors_length * self.right_corridor_multplier
        )
        self.up_corridor_length = self.corridors_length * self.up_corridor_multplier
        self.down_corridor_length = self.corridors_length * self.down_corridor_multplier

        self.inner_box_lines_length = self.inner_box_semidim - self.corridors_width / 2

        self.inner_box_lines = []
        for i in range(8):
            landmark = Landmark(
                name=f"inner_wall_{i}",
                collide=True,
                shape=Line(length=self.inner_box_lines_length),
                color=Color.BLACK,
            )
            self.inner_box_lines.append(landmark)
            world.add_landmark(landmark)
        self.corridor_lines = []
        self.corridors_lengths = [
            self.left_corridor_length,
            self.right_corridor_length,
            self.up_corridor_length,
            self.down_corridor_length,
        ]
        for i in range(8):
            landmark = Landmark(
                name=f"corridor_{i}",
                collide=True,
                shape=Line(length=self.corridors_lengths[i % 4]),
                color=Color.BLACK,
            )
            self.corridor_lines.append(landmark)
            world.add_landmark(landmark)
        self.room_lines = []
        for i in range(12):
            landmark = Landmark(
                name=f"room_line_{i}",
                collide=True,
                shape=Line(length=self.room_semidim * 2),
                color=Color.BLACK,
            )
            self.room_lines.append(landmark)
            world.add_landmark(landmark)
        self.closing_lines = []
        for i in range(8):
            landmark = Landmark(
                name=f"closing_line_{i}",
                collide=True,
                shape=Line(length=self.room_semidim - self.corridors_width / 2),
                color=Color.BLACK,
            )
            self.closing_lines.append(landmark)
            world.add_landmark(landmark)

    def reset_map(self, env_index):
        self.reset_inner_walls(env_index)
        self.reset_corridors(env_index)
        self.reset_room_lines(env_index)
        self.reset_closing_lines(env_index)

    def reset_closing_lines(self, env_index):
        for i in [2, 3, 6, 7]:
            self.closing_lines[i].set_pos(
                torch.tensor(
                    [
                        -(self.room_semidim / 2 + self.corridors_width / 4)
                        if i in [2, 3]
                        else (self.room_semidim / 2 + self.corridors_width / 4),
                        (self.inner_box_semidim + self.corridors_lengths[i % 4])
                        * (-1 if i % 2 else 1),
                    ],
                    dtype=torch.float32,
                    device=self.world.device,
                ),
                batch_index=env_index,
            )
        for i in [0, 1, 4, 5]:
            self.closing_lines[i].set_pos(
                torch.tensor(
                    [
                        (self.inner_box_semidim + self.corridors_lengths[i % 4])
                        * (-1 if i % 2 else 1),
                        -(self.room_semidim / 2 + self.corridors_width / 4)
                        if i in [0, 1]
                        else (self.room_semidim / 2 + self.corridors_width / 4),
                    ],
                    dtype=torch.float32,
                    device=self.world.device,
                ),
                batch_index=env_index,
            )
            self.closing_lines[i].set_rot(
                torch.tensor(
                    [torch.pi / 2],
                    dtype=torch.float32,
                    device=self.world.device,
                ),
                batch_index=env_index,
            )

    def reset_room_lines(self, env_index):
        for i in [2, 3]:
            self.room_lines[i].set_pos(
                torch.tensor(
                    [
                        0,
                        (
                            self.inner_box_semidim
                            + self.corridors_lengths[i % 4]
                            + self.room_semidim * 2
                        )
                        * (-1 if i % 2 else 1),
                    ],
                    dtype=torch.float32,
                    device=self.world.device,
                ),
                batch_index=env_index,
            )

        for i in [0, 1]:
            self.room_lines[i].set_pos(
                torch.tensor(
                    [
                        (
                            self.inner_box_semidim
                            + self.corridors_lengths[i % 4]
                            + self.room_semidim * 2
                        )
                        * (-1 if i % 2 else 1),
                        0,
                    ],
                    dtype=torch.float32,
                    device=self.world.device,
                ),
                batch_index=env_index,
            )
            self.room_lines[i].set_rot(
                torch.tensor(
                    [torch.pi / 2],
                    dtype=torch.float32,
                    device=self.world.device,
                ),
                batch_index=env_index,
            )
        for i in [6, 7, 10, 11]:
            self.room_lines[i].set_pos(
                torch.tensor(
                    [
                        -self.room_semidim if i in [6, 7] else self.room_semidim,
                        (
                            self.inner_box_semidim
                            + self.corridors_lengths[i % 4]
                            + self.room_semidim
                        )
                        * (-1 if i % 2 else 1),
                    ],
                    dtype=torch.float32,
                    device=self.world.device,
                ),
                batch_index=env_index,
            )
            self.room_lines[i].set_rot(
                torch.tensor(
                    [torch.pi / 2],
                    dtype=torch.float32,
                    device=self.world.device,
                ),
                batch_index=env_index,
            )
        for i in [4, 5, 8, 9]:
            self.room_lines[i].set_pos(
                torch.tensor(
                    [
                        (
                            self.inner_box_semidim
                            + self.corridors_lengths[i % 4]
                            + self.room_semidim
                        )
                        * (-1 if i % 2 else 1),
                        -self.room_semidim if i in [4, 5] else self.room_semidim,
                    ],
                    dtype=torch.float32,
                    device=self.world.device,
                ),
                batch_index=env_index,
            )

    def reset_inner_walls(self, env_index):
        for i, landmark in enumerate(self.inner_box_lines):
            if i < 4:
                landmark.set_pos(
                    torch.tensor(
                        [
                            -self.inner_box_semidim
                            if i in [0, 2]
                            else self.inner_box_semidim,
                            self.inner_box_lines_length / 2 + self.corridors_width / 2
                            if i in [0, 1]
                            else -(
                                self.inner_box_lines_length / 2
                                + self.corridors_width / 2
                            ),
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
                            self.inner_box_lines_length / 2 + self.corridors_width / 2
                            if i in [4, 6]
                            else -(
                                self.inner_box_lines_length / 2
                                + self.corridors_width / 2
                            ),
                            -self.inner_box_semidim
                            if i in [4, 5]
                            else self.inner_box_semidim,
                        ],
                        dtype=torch.float32,
                        device=self.world.device,
                    ),
                    batch_index=env_index,
                )

    def reset_corridors(self, env_index):
        for i in [0, 4]:
            self.corridor_lines[i].set_pos(
                torch.tensor(
                    [
                        -(self.inner_box_semidim + self.corridors_lengths[i % 4] / 2),
                        -self.corridors_width / 2
                        if i == 0
                        else self.corridors_width / 2,
                    ],
                    dtype=torch.float32,
                    device=self.world.device,
                ),
                batch_index=env_index,
            )
        for i in [1, 5]:
            self.corridor_lines[i].set_pos(
                torch.tensor(
                    [
                        (self.inner_box_semidim + self.corridors_lengths[i % 4] / 2),
                        -self.corridors_width / 2
                        if i == 1
                        else self.corridors_width / 2,
                    ],
                    dtype=torch.float32,
                    device=self.world.device,
                ),
                batch_index=env_index,
            )
        for i in [2, 6]:
            self.corridor_lines[i].set_pos(
                torch.tensor(
                    [
                        -self.corridors_width / 2
                        if i == 2
                        else self.corridors_width / 2,
                        (self.inner_box_semidim + self.corridors_lengths[i % 4] / 2),
                    ],
                    dtype=torch.float32,
                    device=self.world.device,
                ),
                batch_index=env_index,
            )
            self.corridor_lines[i].set_rot(
                torch.tensor(
                    [torch.pi / 2],
                    dtype=torch.float32,
                    device=self.world.device,
                ),
                batch_index=env_index,
            )
        for i in [3, 7]:
            self.corridor_lines[i].set_pos(
                torch.tensor(
                    [
                        -self.corridors_width / 2
                        if i == 3
                        else self.corridors_width / 2,
                        -(self.inner_box_semidim + self.corridors_lengths[i % 4] / 2),
                    ],
                    dtype=torch.float32,
                    device=self.world.device,
                ),
                batch_index=env_index,
            )
            self.corridor_lines[i].set_rot(
                torch.tensor(
                    [torch.pi / 2],
                    dtype=torch.float32,
                    device=self.world.device,
                ),
                batch_index=env_index,
            )


if __name__ == "__main__":
    render_interactively(__file__, control_two_agents=True)
