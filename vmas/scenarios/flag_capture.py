#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import typing

import torch

from vmas import render_interactively
from vmas.simulator.core import Agent, Box, Landmark, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils

if typing.TYPE_CHECKING:
    pass


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.plot_grid = False
        self.n_agents = kwargs.pop("n_agents", 2)
        self.n_adversaries = kwargs.pop("n_adversaries", 0)

        self.n_flags = kwargs.pop("n_flags", 2)
        self.n_adversary_flags = kwargs.pop("n_adversary_flags", 0)

        self.world_spawning_x = kwargs.pop("world_spawning_x", 1)
        self.world_spawning_y = kwargs.pop("world_spawning_y", 1)
        self.base_width = kwargs.pop("base_width", 0.2)

        self.agent_radius = kwargs.pop("agent_radius", 0.05)

        self.pos_shaping_factor = kwargs.pop("pos_shaping_factor", 1)
        self.flag_capture_reward = kwargs.pop("flag_capture_reward", 1)
        self.flag_drop_reward = kwargs.pop("flag_drop_reward", 1)
        self.flag_return_reward = kwargs.pop("flag_return_reward", 1)

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

    def reset_world_at(self, env_index: int = None):
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

        self.flag_distances = self._get_distance_to_flags(env_index)

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            flag_distances = self._get_distance_to_flags()
            self.flag_rews = (
                self.flag_distances - flag_distances
            ) * self.pos_shaping_factor
            self.flag_distances = flag_distances

        rew = self.flag_rews.mean(-1).max(-1)[0]
        return rew

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
