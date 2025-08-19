#  Copyright (c) 2024-2025.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.


import typing

import torch
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn import SoftmaxAggregation

from vmas import render_interactively
from vmas.simulator.core import Agent, Box, Landmark, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import AGENT_INFO_TYPE, Color, ScenarioUtils, X

if typing.TYPE_CHECKING:
    pass


def agg_max(x, dim):
    return x.max(dim=dim, keepdim=True)[0]


def agg_min(x, dim):
    return x.min(dim=dim, keepdim=True)[0]


def agg_mean(x, dim):
    return x.mean(dim=dim, keepdim=True)


def agg_sum(x, dim):
    return x.sum(dim=dim, keepdim=True)


def agg_logsumexp(x, dim):
    return torch.logsumexp(x, dim=dim, keepdim=True)


class Square:
    def forward(self, x):
        return (x + 1e-7) ** 2

    def inverse(self, x):
        return (x.abs() + 1e-7).sqrt()


class PowerMeanAggregation(torch.nn.Module):
    def __init__(self, p: float = 1.0, learn: bool = False):
        super().__init__()

        self._init_p = p
        self.learn = learn

        self.p = Parameter(torch.empty(1)) if learn else p
        self.reset_parameters()

    def reset_parameters(self):
        if isinstance(self.p, Tensor):
            self.p.data.fill_(self._init_p)

    def forward(self, x, dim: int = -2) -> Tensor:

        p = self.p + 1e-2
        x = x.pow(p)
        x = x.mean(dim=dim, keepdim=True)
        x = x.pow(1.0 / p)

        return x


def tanh_squash(x, low, high):
    tanh_x = torch.tanh(x)
    scale = (high - low) / 2
    add = (high + low) / 2
    return tanh_x * scale + add


def tanh_unsquash(x, low, high):
    scale = (high - low) / 2
    add = (high + low) / 2
    return torch.atanh((x - add) / scale)


class PowerSumAggregation(torch.nn.Module):
    def __init__(self, t: float, low, high, device, learn: bool = True):
        super().__init__()

        self.low = torch.tensor(low, device=device)
        self.high = torch.tensor(high, device=device)
        self._init_inner_t = tanh_unsquash(t, self.low, self.high)
        if not learn:
            self._init_inner_t.requires_grad_(False)

        self.learn = learn
        self.dist = torch.distributions.Normal(loc=0, scale=1)

        self._inner_t = (
            Parameter(torch.empty(1, device=device)) if learn else self._init_inner_t
        )
        self.reset_parameters()

    def reset_parameters(self):
        if isinstance(self._inner_t, Tensor):
            self._inner_t.data.fill_(self._init_inner_t)

    @property
    def t(self):
        return tanh_squash(self._inner_t, self.low, self.high).clamp(
            min=self.low + 1e-1, max=self.high - 1e-1
        )

    def forward(self, x, dim: int = -2) -> Tensor:
        x = x.pow(self.t)
        x = x.sum(dim=dim, keepdim=True)

        return x


def get_aggregation_function(name, device):
    if name == "softmax":
        return SoftmaxAggregation(t=0, learn=True).to(device)
    elif name == "softmax_5":
        return SoftmaxAggregation(t=5, learn=True).to(device)
    elif name == "softmax_-5":
        return SoftmaxAggregation(t=-5, learn=True).to(device)
    elif name == "max":
        return agg_max
    elif name == "mean":
        return agg_mean
    elif name == "min":
        return agg_min
    elif name == "sum":
        return agg_sum
    elif name == "powersum":
        return PowerSumAggregation(t=1, learn=True, low=0.3, high=6, device=device)
    elif name == "powersum_04":
        return PowerSumAggregation(t=0.4, learn=False, low=0.3, high=6, device=device)
    elif name == "powersum_5":
        return PowerSumAggregation(t=5, learn=False, low=0.3, high=6, device=device)
    else:
        raise AssertionError


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.plot_grid = False
        self.n_agents = kwargs.pop("n_agents", 2)
        self.n_adversaries = kwargs.pop("n_adversaries", 0)
        self.spawn_agents_in_same_pos = kwargs.pop("spawn_agents_in_same_pos", False)

        self.n_flags = kwargs.pop("n_flags", 2)

        self.world_spawning_x = kwargs.pop("world_spawning_x", 1)
        self.world_spawning_y = kwargs.pop("world_spawning_y", 1)
        self.base_width = kwargs.pop("base_width", 0.2)

        self.agent_radius = kwargs.pop("agent_radius", 0.05)

        self.use_lidar = kwargs.pop("use_lidar", False)
        self.lidar_range = kwargs.pop("lidar_range", 0.35)

        # One of: "distance", "deltas", "percentage"
        self.reward_type = kwargs.pop("reward_type", "percentage")

        self.pos_shaping_factor = kwargs.pop("pos_shaping_factor", 1)
        self.flag_capture_reward = kwargs.pop("flag_capture_reward", 0.01)
        self.reach_flag_line_rew = kwargs.pop("reach_flag_line_rew", False)

        self.gen_agg_type_task = kwargs.pop("gen_agg_type_task", "mean")
        self.gen_agg_type_agent = kwargs.pop("gen_agg_type_agent", "mean")

        self.task_agg = get_aggregation_function(self.gen_agg_type_task, device)
        self.agent_agg = get_aggregation_function(self.gen_agg_type_agent, device)

        ScenarioUtils.check_kwargs_consumed(kwargs)

        assert not (self.spawn_agents_in_same_pos and self.use_lidar)

        self.min_distance_between_agents = self.agent_radius * 2 + 0.05
        self.min_distance_between_flags = 0.8
        self.map_inflate_radius = 0.2

        self.min_collision_distance = 0.005

        self.x_map_bound = (
            self.world_spawning_x + self.base_width + self.map_inflate_radius
        )
        self.y_map_bound = self.world_spawning_y + self.map_inflate_radius

        # Make world
        world = World(
            batch_dim,
            device,
            substeps=2,
            x_semidim=self.x_map_bound,
            y_semidim=self.y_map_bound,
        )

        self.flag_distances = None
        self.final_rew = torch.zeros(batch_dim, device=device)

        self.blue_agents = []
        # Add agents
        for i in range(self.n_agents):
            # Constraint: all agents have same action range and multiplier
            agent = Agent(
                name=f"agent_{i}",
                collide=self.use_lidar,
                color=Color.BLUE,
                shape=Sphere(radius=self.agent_radius),
                render_action=True,
                sensors=(
                    [
                        Lidar(
                            world,
                            n_rays=12,
                            max_range=self.lidar_range,
                            entity_filter=lambda e: isinstance(e, Agent),
                        ),
                    ]
                    if self.use_lidar
                    else None
                ),
            )
            self.blue_agents.append(agent)
            world.add_agent(agent)
        self.blue_flags = []
        for i in range(self.n_flags):
            # Add goals
            flag = Landmark(
                name=f"goal {i}",
                collide=False,
                color=Color.GREEN,
                shape=Box(0.07, 0.04),
            )
            self.blue_flags.append(flag)
            world.add_landmark(flag)

        return world

    def parameters(self) -> typing.List:
        params = []
        if hasattr(self.task_agg, "parameters"):
            params += self.task_agg.parameters()
        if hasattr(self.agent_agg, "parameters"):
            params += self.agent_agg.parameters()
        return params

    def to_log(self) -> typing.Dict:
        result = {}
        if hasattr(self.task_agg, "t"):
            result["task_agg_t"] = self.task_agg.t.item()
        if hasattr(self.agent_agg, "t"):
            result["agent_agg_t"] = self.agent_agg.t.item()

        return result

    def reset_world_at(self, env_index: int = None):
        if self.spawn_agents_in_same_pos:
            x = torch.zeros(
                (1,) if env_index is not None else (self.world.batch_dim, 1),
                device=self.world.device,
                dtype=torch.float32,
            ).uniform_(-self.world_spawning_x - self.base_width, -self.world_spawning_x)
            y = torch.zeros(
                (1,) if env_index is not None else (self.world.batch_dim, 1),
                device=self.world.device,
                dtype=torch.float32,
            ).uniform_(-self.world_spawning_y, self.world_spawning_y)
            pos = torch.cat([x, y], dim=-1)
            for agent in self.blue_agents:
                agent.set_pos(pos, batch_index=env_index)
        else:
            ScenarioUtils.spawn_entities_randomly(
                self.blue_agents,
                self.world,
                env_index,
                self.min_distance_between_agents,
                (-self.world_spawning_x - self.base_width, -self.world_spawning_x),
                (-self.world_spawning_y, self.world_spawning_y),
            )

        ScenarioUtils.spawn_entities_randomly(
            self.blue_flags,
            self.world,
            env_index,
            self.min_distance_between_flags,
            (
                self.world_spawning_x + self.base_width / 2,
                self.world_spawning_x + self.base_width / 2,
            ),
            (-self.world_spawning_y, self.world_spawning_y),
        )
        if env_index is None:
            self.flag_distances = self._get_distance_to_flags(env_index)
            self.initial_flag_distances = self.flag_distances.clone()
        else:
            self.flag_distances[env_index] = self._get_distance_to_flags(env_index)
            self.initial_flag_distances[env_index] = self.flag_distances[env_index]

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            flag_distances = self._get_distance_to_flags()  # batch, n_agents, n_flags
            self.flag_shaping = (
                self.flag_distances - flag_distances
            ) * self.pos_shaping_factor
            sum_distances = flag_distances.sum(dim=-1, keepdim=True)
            self.flag_percentage = -flag_distances / sum_distances
            self.flag_distances = flag_distances.detach()

            if self.reward_type == "distance":
                self.matrix = -self.flag_distances / 10
            elif self.reward_type == "deltas":
                self.matrix = self.flag_shaping
            elif self.reward_type == "percentage":
                self.matrix = self.flag_percentage
            else:
                raise AssertionError

            self.on_goals = (
                -torch.max(-flag_distances, dim=-2)[0].min(dim=-1)[0]
                < self.agent_radius
            )
            self.final_rew = self.on_goals * self.flag_capture_reward

        if self.gen_agg_type_agent.startswith("powersum"):
            task_matrix = (
                self.agent_agg(1 + self.matrix, dim=-2).squeeze(-2) / 2
            ) - 1  # Output between -1 and 0
        else:
            task_matrix = self.agent_agg(self.matrix, dim=-2).squeeze(-2)

        if self.gen_agg_type_task.startswith("powersum"):
            self.rew = (
                self.task_agg(1 + task_matrix, dim=-1).squeeze(-1) / 2
            ) - 1  # Output between -1 and 0
        else:
            self.rew = self.task_agg(task_matrix, dim=-1).squeeze(-1)

        if self.reach_flag_line_rew:
            flag_line_rew = (
                -(
                    (
                        agent.state.pos[:, X]
                        - (self.world_spawning_x + self.base_width / 2)
                    )
                    ** 2
                )
                / 50
            )
        else:
            flag_line_rew = 0

        return self.rew / 10 + self.final_rew + flag_line_rew

    def _get_distance_to_flags(self, env_index: typing.Optional[int] = None):
        if env_index is None:
            distances = torch.zeros(
                self.world.batch_dim,
                self.n_agents,
                self.n_flags,
                device=self.world.device,
            )
            for i, agent in enumerate(self.world.agents):
                for j, flag in enumerate(self.blue_flags):
                    distances[:, i, j] = torch.linalg.vector_norm(
                        agent.state.pos - flag.state.pos,
                        dim=-1,
                    )
        else:
            distances = torch.zeros(
                self.n_agents, self.n_flags, device=self.world.device
            )
            for i, agent in enumerate(self.world.agents):
                for j, flag in enumerate(self.blue_flags):
                    distances[i, j] = torch.linalg.vector_norm(
                        agent.state.pos[env_index] - flag.state.pos[env_index],
                        dim=-1,
                    )
        return distances

    def observation(self, agent: Agent):
        flag_poses = []

        for flag in self.blue_flags:
            flag_poses.append(agent.state.pos - flag.state.pos)

        return torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
            ]
            + flag_poses
            + (
                [agent.sensors[0]._max_range - agent.sensors[0].measure()]
                if self.use_lidar
                else []
            ),  # maybe use rel pos directly
            dim=-1,
        )

    def info(self, agent: Agent) -> AGENT_INFO_TYPE:
        return {
            "final_rew": self.final_rew,
        }


if __name__ == "__main__":
    render_interactively(
        __file__,
        control_two_agents=True,
    )
