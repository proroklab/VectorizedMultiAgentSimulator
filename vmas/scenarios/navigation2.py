#  Copyright (c) 2022-2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

'''
Same as navigation.py except modified so it will work with the expert
'''

import typing
from typing import Dict, Callable, List

import random

import torch
from torch import Tensor
from vmas import render_interactively
from vmas.simulator.core import Agent, Box, Landmark, World, Sphere, Entity
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import Color, ScenarioUtils
from vmas.simulator.controllers.velocity_controller import VelocityController


if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.plot_grid = False
        self.n_agents = kwargs.get("n_agents", 4)
        self.collisions = kwargs.get("collisions", True)

        self.observe_all_goals = kwargs.get("observe_all_goals", False)

        self.agent_radius = kwargs.get("agent_radius", 0.1)
        self.lidar_range = kwargs.get("lidar_range", self.agent_radius*2)
        self.comms_range = kwargs.get("comms_range", 0)

        self.shared_rew = kwargs.get("shared_rew", True)
        self.pos_shaping_factor = kwargs.get("pos_shaping_factor", 1)
        self.final_reward = kwargs.get("final_reward", 0.01)

        self.agent_collision_penalty = kwargs.get("agent_collision_penalty", -1)

        self.episodes = kwargs.get('episodes')
        self.map = kwargs.get('map')
        self.first = True
        
        self.min_collision_distance = 0.005

        controller_params = [2, 6, 0.002]

         # Make world
        world = World(batch_dim, device, substeps=2)

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
        entity_filter_agents: Callable[[Entity], bool] = lambda e: isinstance(e, Agent)

        # Add agents
        for i in range(self.n_agents):
            color = (
                known_colors[i]
                if i < len(known_colors)
                else colors[i - len(known_colors)]
            )

            # Constraint: all agents have same action range and multiplier
            agent = Agent(
                name=f"agent {i}",
                collide=self.collisions,
                color=color,
                shape=Sphere(radius=self.agent_radius),
                render_action=True,
                sensors=[
                    Lidar(
                        world,
                        n_rays=12,
                        max_range=self.lidar_range,
                        entity_filter=entity_filter_agents,
                    ),
                ]
                if self.collisions
                else None,
            )
            agent.controller = VelocityController(
                agent, world, controller_params, "standard"
            )
            agent.pos_rew = torch.zeros(batch_dim, device=device)
            agent.agent_collision_rew = agent.pos_rew.clone()
            world.add_agent(agent)

            # Add goals
            goal = Landmark(
                name=f"goal {i}",
                collide=False,
                color=color,
            )
            world.add_landmark(goal)
            agent.goal = goal

        #Add in the map obstacles
        indices = torch.nonzero(self.map == 1)
        for coord in indices:
            obstacle = Landmark(
                name = f"Obstacle at {coord}",
                collide = True,
                shape = Box(1,1)
            )
            world.add_landmark(obstacle)            

        self.pos_rew = torch.zeros(batch_dim, device=device)
        self.final_rew = self.pos_rew.clone()

        return world

    def reset_world_at(self, env_index: int = None):
        episode_name, episode_agents = random.choice(list(self.episodes.items()))

        if self.first:
            #TODO: is there a way to do this in the constructor since it only needs to happen once?
            counter = 0
            indices = torch.nonzero(self.map == 1)
            for i,coord in enumerate(indices):
                self.world.landmarks[i+self.n_agents].set_pos(
                    torch.tensor(
                        [coord[1], coord[0]],
                        dtype=torch.float32,
                        device=self.world.device
                    ),
                    batch_index = None
                )
            self.first=False

        for i, agent in enumerate(self.world.agents):
            agent.set_pos(
                torch.tensor(
                    [
                        episode_agents[i][0][1], episode_agents[i][0][2]
                    ],
                    dtype = torch.float32,
                    device = self.world.device
                ),
                batch_index=env_index
            )

            pos = torch.tensor(
                [episode_agents[i][-1][1], episode_agents[i][-1][2]]
            )

            agent.goal.set_pos(pos, batch_index=env_index)
            self.world.landmarks[i].set_pos(pos, batch_index=env_index)

            if env_index is None:
                agent.episode_name = torch.tensor([[episode_name]]*self.world.batch_dim, device = self.world.device)
                agent.pos_shaping = (
                    torch.linalg.vector_norm(
                        agent.state.pos - agent.goal.state.pos,
                        dim=1,
                    )
                    * self.pos_shaping_factor
                )
            else:
                agent.episode_name[env_index] = [episode_name]
                agent.pos_shaping[env_index] = (
                    torch.linalg.vector_norm(
                        agent.state.pos[env_index] - agent.goal.state.pos[env_index]
                    )
                    * self.pos_shaping_factor
                )

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
                    if self.world.collides(a, b):
                        distance = self.world.get_distance(a, b)
                        a.agent_collision_rew[
                            distance <= self.min_collision_distance
                        ] += self.agent_collision_penalty
                        b.agent_collision_rew[
                            distance <= self.min_collision_distance
                        ] += self.agent_collision_penalty

        pos_reward = self.pos_rew if self.shared_rew else agent.pos_rew
        return pos_reward + self.final_rew + agent.agent_collision_rew

    def agent_reward(self, agent: Agent):
        agent.distance_to_goal = torch.linalg.vector_norm(
            agent.state.pos - agent.goal.state.pos,
            dim=-1,
        )
        agent.on_goal = agent.distance_to_goal < agent.goal.shape.radius

        pos_shaping = agent.distance_to_goal * self.pos_shaping_factor
        agent.pos_rew = agent.pos_shaping - pos_shaping
        agent.pos_shaping = pos_shaping
        return agent.pos_rew

    def observation(self, agent: Agent):
        goal_poses = []
        if self.observe_all_goals:
            for a in self.world.agents:
                goal_poses.append(agent.state.pos - a.goal.state.pos)
        else:
            goal_poses.append(agent.state.pos - agent.goal.state.pos)
        return torch.cat(
            [
                agent.episode_name,
                agent.state.pos,
                agent.state.vel,
            ]
            + goal_poses
            + (
                [agent.sensors[0]._max_range - agent.sensors[0].measure()]
                if self.collisions
                else []
            ),
            dim=-1,
        )

    def done(self):
        return torch.stack(
            [
                torch.linalg.vector_norm(
                    agent.state.pos - agent.goal.state.pos,
                    dim=-1,
                )
                < agent.shape.radius
                for agent in self.world.agents
            ],
            dim=-1,
        ).all(-1)

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        return {
            "pos_rew": self.pos_rew if self.shared_rew else agent.pos_rew,
            "final_rew": self.final_rew,
            "agent_collisions": agent.agent_collision_rew,
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


if __name__ == "__main__":
    render_interactively(
        __file__,
        control_two_agents=True,
    )
