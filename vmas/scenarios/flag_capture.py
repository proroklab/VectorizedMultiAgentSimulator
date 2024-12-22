#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import typing

import torch
from torch_geometric.nn import SoftmaxAggregation

from vmas import render_interactively
from vmas.simulator.core import Agent, Box, Landmark, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils

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


class Square:
    def forward(self, x):
        return (x + 1e-7) ** 2

    def inverse(self, x):
        return (x.abs() + 1e-7).sqrt()


def get_aggregation_function(name, device):
    if name == "softmax":
        return SoftmaxAggregation(t=1, learn=True).to(device)
    elif name == "max":
        return agg_max
    elif name == "mean":
        return agg_mean
    elif name == "min":
        return agg_min
    elif name == "sum":
        return agg_sum
    else:
        raise AssertionError


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.plot_grid = False
        self.n_agents = kwargs.pop("n_agents", 2)
        self.n_adversaries = kwargs.pop("n_adversaries", 0)
        self.spawn_agents_in_same_pos = kwargs.pop("spawn_agents_in_same_pos", True)

        self.n_flags = kwargs.pop("n_flags", 2)
        self.n_adversary_flags = kwargs.pop("n_adversary_flags", 0)

        self.world_spawning_x = kwargs.pop("world_spawning_x", 1)
        self.world_spawning_y = kwargs.pop("world_spawning_y", 1)
        self.base_width = kwargs.pop("base_width", 0.2)

        self.agent_radius = kwargs.pop("agent_radius", 0.05)

        self.pos_shaping_factor = kwargs.pop("pos_shaping_factor", 1)
        self.flag_capture_reward = kwargs.pop("flag_capture_reward", 0)
        self.flag_drop_reward = kwargs.pop("flag_drop_reward", 0)
        self.flag_return_reward = kwargs.pop("flag_return_reward", 0)

        self.gen_agg_type_task = kwargs.pop("gen_agg_type_task", "max")
        self.gen_agg_type_agent = kwargs.pop("gen_agg_type_agent", "max")

        self.task_agg = get_aggregation_function(self.gen_agg_type_task, device)
        self.agent_agg = get_aggregation_function(self.gen_agg_type_agent, device)

        ScenarioUtils.check_kwargs_consumed(kwargs)

        self.min_distance_between_entities = self.agent_radius * 2 + 0.05
        self.min_collision_distance = 0.005

        # Make world
        world = World(
            batch_dim,
            device,
            substeps=2,
        )

        self.flag_distances = None

        self.blue_agents = []
        # Add agents
        for i in range(self.n_agents):
            # Constraint: all agents have same action range and multiplier
            agent = Agent(
                name=f"agent_{i}",
                collide=False,
                color=Color.BLUE,
                shape=Sphere(radius=self.agent_radius),
                render_action=True,
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
                self.min_distance_between_entities,
                (-self.world_spawning_x - self.base_width, -self.world_spawning_x),
                (-self.world_spawning_y, self.world_spawning_y),
            )

        ScenarioUtils.spawn_entities_randomly(
            self.blue_flags,
            self.world,
            env_index,
            self.min_distance_between_entities,
            (self.world_spawning_x, self.world_spawning_x + self.base_width),
            (-self.world_spawning_y, self.world_spawning_y),
        )
        if env_index is None:
            self.flag_distances = self._get_distance_to_flags(env_index)
        else:
            self.flag_distances[env_index] = self._get_distance_to_flags(env_index)

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            flag_distances = self._get_distance_to_flags()
            self.flag_rews = (
                self.flag_distances - flag_distances
            ) * self.pos_shaping_factor
            self.flag_distances = flag_distances

        task_matrix = self.agent_agg(-self.flag_distances, dim=-2).squeeze(-2)
        self.rew = self.task_agg(task_matrix, dim=-1).squeeze(-1)
        return self.rew

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
            + flag_poses,
            dim=-1,
        )


if __name__ == "__main__":
    render_interactively(
        __file__,
        control_two_agents=True,
    )
