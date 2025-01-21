#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.


import torch

from vmas import render_interactively

from vmas.simulator.core import Agent, Landmark, Line, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils, TorchUtils, X, Y


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):

        self.corridors_length = kwargs.get("corridors_length", 1)
        self.lrud_ratio = kwargs.get("lrud_ratio", (2, 2, 3, 1))
        self.shared_rew = kwargs.get("shared_rew", True)
        self.observe_position = kwargs.get("observe_position", False)
        self.spawn_same_pos = kwargs.get("spawn_same_pos", True)

        self.left_corridor_multplier = self.lrud_ratio[0]
        self.right_corridor_multplier = self.lrud_ratio[1]
        self.up_corridor_multplier = self.lrud_ratio[2]
        self.down_corridor_multplier = self.lrud_ratio[3]

        self.viewer_zoom = 3.1

        self.n_agents = 4
        self.agent_radius = 0.16

        # Make world
        world = World(
            batch_dim,
            device,
        )
        self.spawn_map(world)

        self.xdim = kwargs.get("xdim", 5)
        self.ydim = kwargs.get("ydim", 5)
        self.grid_spacing = kwargs.get("grid_spacing", self.agent_radius * 2)

        self.plot_grid = True
        self.alpha_plot: float = 0.5

        self.x_semidim = self.room_semidim - self.agent_radius
        self.y_semidim = self.room_semidim - self.agent_radius
        self.n_x_cells = int((2 * self.room_semidim) / self.grid_spacing)
        self.n_y_cells = int((2 * self.room_semidim) / self.grid_spacing)

        self.distributions = []
        for _ in range(4):
            dist = torch.distributions.uniform.Uniform(
                low=torch.tensor([-self.room_semidim], device=device, dtype=torch.float)
                .unsqueeze(0)
                .expand(batch_dim, 2)
                .clone(),
                high=torch.tensor([self.room_semidim], device=device, dtype=torch.float)
                .unsqueeze(0)
                .expand(batch_dim, 2)
                .clone(),
            )
            self.distributions.append(dist)

            dist.sampled = torch.zeros(
                (batch_dim, self.n_x_cells, self.n_y_cells),
                device=device,
                dtype=torch.bool,
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
            agent.sample = torch.zeros(batch_dim, device=device, dtype=torch.float)
            world.add_agent(agent)

        return world

    def reset_world_at(self, env_index: int = None):

        ScenarioUtils.spawn_entities_randomly(
            self.world.agents,
            self.world,
            env_index,
            min_dist_between_entities=0,
            x_bounds=(
                -(self.inner_box_semidim - self.agent_radius)
                if not self.spawn_same_pos
                else 0,
                (self.inner_box_semidim - self.agent_radius)
                if not self.spawn_same_pos
                else 0,
            ),
            y_bounds=(
                -(self.inner_box_semidim - self.agent_radius)
                if not self.spawn_same_pos
                else 0,
                (self.inner_box_semidim - self.agent_radius)
                if not self.spawn_same_pos
                else 0,
            ),
        )

        self.reset_map(env_index)
        self.reset_distribution(env_index)

    def reset_distribution(self, env_index):
        for dist in self.distributions:
            if env_index is None:
                dist.sampled = torch.zeros_like(dist.sampled)
            else:
                dist.sampled = TorchUtils.where_from_index(
                    env_index, False, dist.sampled
                )

    def translate_pos(self, pos, room_index, inverse=False):
        if room_index == 0:
            shift = (
                self.left_corridor_length + self.room_semidim + self.inner_box_semidim
            )
            pos[..., X] = pos[..., X] - (shift if not inverse else -shift)
        if room_index == 1:
            shift = (
                self.right_corridor_length + self.room_semidim + self.inner_box_semidim
            )
            pos[..., X] = pos[..., X] + (shift if not inverse else -shift)
        if room_index == 2:
            shift = self.up_corridor_length + self.room_semidim + self.inner_box_semidim
            pos[..., Y] = pos[..., Y] + (shift if not inverse else -shift)
        if room_index == 3:
            shift = (
                self.down_corridor_length + self.room_semidim + self.inner_box_semidim
            )
            pos[..., Y] = pos[..., Y] - (shift if not inverse else -shift)
        return pos

    def sample(
        self,
        pos,
        update_sampled_flag: bool = False,
    ):
        values = []
        for i, dist in enumerate(self.distributions):
            trans_pos = self.translate_pos(pos.clone(), i, inverse=True)
            out_of_bounds = (
                (trans_pos < -self.room_semidim) + (trans_pos > self.room_semidim)
            ).any(-1)
            if out_of_bounds.all():
                continue
            trans_pos[:, X].clamp_(-self.x_semidim, self.x_semidim)
            trans_pos[:, Y].clamp_(-self.y_semidim, self.y_semidim)

            index = trans_pos / self.grid_spacing
            index[:, X] += self.n_x_cells / 2
            index[:, Y] += self.n_y_cells / 2
            index = index.to(torch.long)
            v = dist.log_prob(trans_pos).exp().prod(-1)
            v = v * (2 * self.room_semidim) ** 2  # Make it 1

            sampled = dist.sampled[
                torch.arange(self.world.batch_dim), index[:, X], index[:, Y]
            ]
            invalid_sample = sampled + out_of_bounds
            v[invalid_sample] = 0
            if update_sampled_flag:
                dist.sampled[
                    torch.arange(self.world.batch_dim), index[:, X], index[:, Y]
                ] += ~invalid_sample
            values.append(v)
        if len(values) == 0:
            return torch.zeros_like(pos).sum(dim=-1)
        return torch.stack(values, dim=-1).sum(-1)

    def sample_single_env(
        self,
        pos,
        env_index,
    ):
        pos = pos.view(-1, self.world.dim_p)
        values = []
        for i, dist in enumerate(self.distributions):
            trans_pos = self.translate_pos(pos.clone(), i, inverse=True)
            out_of_bounds = (
                (trans_pos < -self.room_semidim) + (trans_pos > self.room_semidim)
            ).any(-1)

            if out_of_bounds.all():
                continue
            trans_pos[:, X].clamp_(-self.x_semidim, self.x_semidim)
            trans_pos[:, Y].clamp_(-self.y_semidim, self.y_semidim)

            index = trans_pos / self.grid_spacing
            index[:, X] += self.n_x_cells / 2
            index[:, Y] += self.n_y_cells / 2
            index = index.to(torch.long)

            trans_pos = trans_pos.unsqueeze(1).expand(
                trans_pos.shape[0], self.world.batch_dim, 2
            )

            v = dist.log_prob(trans_pos).exp().prod(-1)[:, env_index]
            v = v * (2 * self.room_semidim) ** 2  # Make it 1

            sampled = dist.sampled[env_index, index[:, 0], index[:, 1]]

            v[sampled + out_of_bounds] = 0

            values.append(v)

        if len(values) == 0:
            return torch.zeros_like(pos).sum(dim=-1)
        return torch.stack(values, dim=-1).sum(-1)

    def reward(self, agent: Agent):
        is_first = self.world.agents.index(agent) == 0
        if is_first:
            for a in self.world.agents:
                a.sample = self.sample(a.state.pos, update_sampled_flag=True)
            self.sampling_rew = torch.stack(
                [a.sample for a in self.world.agents], dim=-1
            ).sum(-1)

        return self.sampling_rew if self.shared_rew else agent.sample

    def observation(self, agent: Agent):
        observations = self.observation_from_pos(agent.state.pos)
        return observations

    def observation_from_pos(self, pos, env_index=None):
        samples = []
        in_pos = pos
        for delta in [
            [self.grid_spacing, 0],
            [-self.grid_spacing, 0],
            [0, self.grid_spacing],
            [0, -self.grid_spacing],
            [-self.grid_spacing, -self.grid_spacing],
            [self.grid_spacing, -self.grid_spacing],
            [-self.grid_spacing, self.grid_spacing],
            [self.grid_spacing, self.grid_spacing],
        ]:
            pos = in_pos + torch.tensor(
                delta,
                device=self.world.device,
                dtype=torch.float32,
            )
            if env_index is not None:
                samples.append(
                    self.sample_single_env(pos, env_index=env_index).unsqueeze(-1)
                )
            else:
                samples.append(
                    self.sample(pos, update_sampled_flag=False).unsqueeze(-1)
                )
        if self.observe_position:
            samples.append(pos)

        return torch.cat(
            samples,
            dim=-1,
        )

    def info(self, agent: Agent):
        return {"agent_sample": agent.sample}

    def density_for_plot(self, env_index):
        def f(x):
            sample = self.sample_single_env(
                torch.tensor(x, dtype=torch.float32, device=self.world.device),
                env_index=env_index,
            )
            return sample

        return f

    def extra_render(self, env_index: int = 0):

        from vmas.simulator.rendering import render_function_util

        geoms = []

        # Function
        for x_bounds, y_bounds in zip(self.x_bounds, self.y_bounds):
            res = render_function_util(
                f=self.density_for_plot(env_index=env_index),
                plot_range=(x_bounds, y_bounds),
                cmap_alpha=self.alpha_plot,
                precision=self.agent_radius / 2,
                cmap_range=(0, 1),
            )
            geoms.append(res)

        return geoms

    def spawn_map(self, world: World):
        self.corridors_width = self.agent_radius * 4
        self.inner_box_semidim = self.corridors_width / 2 + 1e-5

        self.room_semidim = self.corridors_width / 2 + 1e-6

        self.left_corridor_length = self.corridors_length * self.left_corridor_multplier
        self.right_corridor_length = (
            self.corridors_length * self.right_corridor_multplier
        )
        self.up_corridor_length = self.corridors_length * self.up_corridor_multplier
        self.down_corridor_length = self.corridors_length * self.down_corridor_multplier

        self.inner_box_lines_length = self.inner_box_semidim - self.corridors_width / 2

        self.x_bounds = [
            (
                -(
                    self.left_corridor_length
                    + self.inner_box_semidim
                    + self.room_semidim * 2
                ),
                -(self.left_corridor_length + self.inner_box_semidim),
            ),
            (
                +(self.right_corridor_length + self.inner_box_semidim),
                +(
                    self.right_corridor_length
                    + self.inner_box_semidim
                    + self.room_semidim * 2
                ),
            ),
            (
                -self.room_semidim,
                self.room_semidim,
            ),
            (
                -self.room_semidim,
                self.room_semidim,
            ),
        ]
        self.y_bounds = [
            (
                -self.room_semidim,
                self.room_semidim,
            ),
            (
                -self.room_semidim,
                self.room_semidim,
            ),
            (
                (self.up_corridor_length + self.inner_box_semidim),
                self.up_corridor_length
                + self.inner_box_semidim
                + self.room_semidim * 2,
            ),
            (
                -(
                    self.down_corridor_length
                    + self.inner_box_semidim
                    + self.room_semidim * 2
                ),
                -(self.down_corridor_length + self.inner_box_semidim),
            ),
        ]

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
