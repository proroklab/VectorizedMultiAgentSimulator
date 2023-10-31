#  Copyright (c) 2022-2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import typing
from typing import Dict, Callable, List

import torch
from torch import Tensor
from vmas import render_interactively
from vmas.simulator.core import Agent, Landmark, World, Sphere, Entity
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import Color, ScenarioUtils

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.n_agents = kwargs.get("n_agents", 2)
        self.n_goals = kwargs.get("n_goals", 2)
        self.collision_penalty = kwargs.get("collision_penalty", -0.25)
        self.region_reward = kwargs.get("region_reward", 0.1)
        self.agent_radius = kwargs.get("agent_radius", 0.1)
        self.goal_radius = kwargs.get("goal_radius", 0.3)
        self.reward_observation = kwargs.get("reward_observation", False)
        self.collision_objective = kwargs.get("collision_objective", False)
        self.scalarisation_min = kwargs.get("scalarisation_min", True)
        self.scalarisation_weights = kwargs.get("scalarisation_weights", [1.] * (self.n_goals + self.collision_objective))
        self.scalarisation_over_episode = kwargs.get("scalarisation_over_episode", True)

        self.min_distance_between_entities = self.agent_radius * 2 + 0.05
        self.min_collision_distance = 0.005
        self.world_semidim = 1

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

        # Add agents
        for i in range(self.n_agents):
            # Constraint: all agents have same action range and multiplier
            agent = Agent(
                name=f"agent {i}",
                collide=True,
                shape=Sphere(radius=self.agent_radius),
                render_action=True,
            )
            agent.goal_rew = torch.zeros(batch_dim, self.n_goals, device=device)
            agent.collision_rew = torch.zeros(batch_dim, device=device)
            world.add_agent(agent)

        world.goals = []
        for i in range(self.n_goals):
            color = known_colors[i] if i < len(known_colors) else colors[i - len(known_colors)]
            # Add goals
            goal = Landmark(
                name=f"goal {i}",
                collide=False,
                shape=Sphere(radius=self.goal_radius),
                color=color,
            )
            world.add_landmark(goal)
            world.goals.append(goal)

        self.agent_rewards = {}
        self.agent_accrued_rewards = {}

        return world

    def reset_world_at(self, env_index: int = None):
        ScenarioUtils.spawn_entities_randomly(
            entities=self.world.agents,
            world=self.world,
            env_index=env_index,
            min_dist_between_entities=self.min_distance_between_entities,
            x_bounds=(-self.world_semidim, self.world_semidim),
            y_bounds=(-self.world_semidim, self.world_semidim),
        )

        occupied_positions = torch.stack([agent.state.pos for agent in self.world.agents], dim=1)
        if env_index is not None:
            occupied_positions = occupied_positions[env_index].unsqueeze(0)

        ScenarioUtils.spawn_entities_randomly(
            entities=self.world.goals,
            world=self.world,
            env_index=env_index,
            min_dist_between_entities=self.min_distance_between_entities,
            x_bounds=(-self.world_semidim, self.world_semidim),
            y_bounds=(-self.world_semidim, self.world_semidim),
            occupied_positions=occupied_positions,
        )

        for agent in self.world.agents:
            if agent in self.agent_accrued_rewards:
                self.agent_accrued_rewards[agent][:] = 0
            else:
                self.agent_accrued_rewards[agent] = \
                    torch.zeros(self.world.batch_dim, self.n_goals+int(self.collision_objective),
                                device=self.world.device)


    def reward(self, agent: Agent):

        agent.goal_rew[:] = 0
        agent.collision_rew[:] = 0

        for i, goal in enumerate(self.world.goals):
            agent.distance_to_goal = torch.norm(agent.state.pos - goal.state.pos, dim=-1)
            mask = agent.distance_to_goal < goal.shape.radius
            agent.goal_rew[mask,i] += self.region_reward

        if self.collision_penalty != 0:
            for other in self.world.agents:
                if (agent is not other) and self.world.collides(agent, other):
                    distance = self.world.get_distance(agent, other)
                    mask = distance <= self.min_collision_distance
                    agent.collision_rew[mask] += self.collision_penalty

        if self.collision_objective:
            agent_rew = torch.cat([agent.goal_rew, agent.collision_rew.unsqueeze(-1)], dim=-1)
        else:
            agent_rew = agent.goal_rew.clone()

        if self.scalarisation_over_episode:
            utility_before = self.scalarisation(self.agent_accrued_rewards[agent])
            utility_after = self.scalarisation(agent_rew + self.agent_accrued_rewards[agent])
            utility = utility_after - utility_before
        else:
            utility = self.scalarisation(agent_rew)

        self.agent_rewards[agent] = agent_rew
        self.agent_accrued_rewards[agent] += agent_rew

        return utility

    def scalarisation(self, agent_rew):
        if not isinstance(self.scalarisation_weights, torch.Tensor):
            self.scalarisation_weights = torch.Tensor(self.scalarisation_weights)
        weighted_obj = agent_rew * self.scalarisation_weights[None,:] / torch.norm(self.scalarisation_weights)
        if self.scalarisation_min:
            return torch.min(weighted_obj, dim=-1).values
        else:
            return torch.sum(weighted_obj, dim=-1)

    def observation(self, agent: Agent):
        pos = agent.state.pos
        rel_goal_pos = torch.cat([
            pos - goal.state.pos
            for goal in self.world.goals], dim=-1)
        if self.reward_observation:
            accrued_reward = self.symlog(self.agent_accrued_rewards[agent])
            return torch.cat([rel_goal_pos, accrued_reward], dim=-1)
        else:
            return rel_goal_pos

    def done(self):
        return torch.zeros(self.world.batch_dim).bool()

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        if agent not in self.agent_rewards:
            self.reward(agent)
        return {
            "reward": self.agent_rewards[agent]
        }

    def extra_render(self, env_index: int = 0):
       return []

    def symlog(self, val):
        return torch.sign(val) * torch.log(torch.abs(val)+1)


if __name__ == "__main__":
    render_interactively(
        __file__,
        control_two_agents=True,
        scalarisation_weights = [1.,1.],
        scalarisation_over_episode=True,
        scalarisation_min=True,
        reward_observation=True,
    )
