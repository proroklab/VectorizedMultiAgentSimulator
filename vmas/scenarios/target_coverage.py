#  Copyright (c) 2022. Jan Blumenkamp
#  All rights reserved.
import math
from typing import Dict, Callable, Optional

import torch
from torch import Tensor

from vmas import render_interactively
from vmas.simulator.core import Agent, Landmark, Sphere, World, Entity, Line
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color
from vmas.simulator.sensors import Lidar


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        n_agents = kwargs.get("n_agents", 4)
        n_targets = kwargs.get("n_targets", 5)
        self._min_dist_between_entities = kwargs.get("min_dist_between_entities", 0.2)
        self._lidar_range = kwargs.get("lidar_range", 0.5)
        self._covering_range = kwargs.get("covering_range", 0.3)
        self._agents_per_target = kwargs.get("agents_per_target", 2)

        # Make world
        world = World(batch_dim, device, x_semidim=1, y_semidim=1)
        # Add agents
        entity_filter_agents: Callable[[Entity], bool] = lambda e: e.name.startswith(
            "agent"
        )
        entity_filter_targets: Callable[[Entity], bool] = lambda e: e.name.startswith(
            "target"
        )
        for i in range(n_agents):
            # Constraint: all agents have same action range and multiplier
            agent = Agent(
                name=f"agent_{i}",
                collide=True,
                sensors=[
                    Lidar(
                        world,
                        angle_start=0.05,
                        angle_end=2 * torch.pi + 0.05,
                        n_rays=12,
                        max_range=self._lidar_range,
                        entity_filter=entity_filter_agents,
                        render_color=Color.BLUE,
                    ),
                    Lidar(
                        world,
                        n_rays=12,
                        max_range=self._lidar_range,
                        entity_filter=entity_filter_targets,
                        render_color=Color.GREEN,
                    ),
                ],
            )
            world.add_agent(agent)

        self._targets = []
        for i in range(n_targets):
            target = Landmark(
                name=f"target_{i}",
                collide=True,
                movable=False,
                shape=Sphere(radius=0.05),
                color=Color.GREEN,
            )
            world.add_landmark(target)
            self._targets.append(target)

        return world

    def _find_random_pos_for_entity(
        self, occupied_positions: torch.Tensor, env_index: int = None
    ):
        batch_size = 1 if env_index is not None else self.world.batch_dim
        pos = None
        while True:
            proposed_pos = torch.empty(
                (batch_size, 1, self.world.dim_p),
                device=self.world.device,
                dtype=torch.float32,
            ).uniform_(-1.0, 1.0)
            if pos is None:
                pos = proposed_pos
            if occupied_positions.shape[1] == 0:
                break

            dist = torch.cdist(occupied_positions, pos)
            overlaps = torch.any(
                (dist < self._min_dist_between_entities).squeeze(2), dim=1
            )
            if torch.any(overlaps, dim=0):
                pos[overlaps] = proposed_pos[overlaps]
            else:
                break
        return pos

    def reset_world_at(self, env_index: int = None):
        batch_size = 1 if env_index is not None else self.world.batch_dim
        occupied_positions = torch.zeros(
            (batch_size, 0, self.world.dim_p), device=self.world.device
        )
        placable_entities = self._targets + self.world.agents
        for entity in placable_entities:
            pos = self._find_random_pos_for_entity(occupied_positions, env_index)
            occupied_positions = torch.cat([occupied_positions, pos], dim=1)
            entity.set_pos(pos.squeeze(1), batch_index=env_index)

    def reward(self, agent: Agent):
        # Avoid collisions with each other
        self.collision_rew = torch.zeros(self.world.batch_dim, device=self.world.device)
        for a in self.world.agents:
            if a != agent:
                self.collision_rew[self.world.is_overlapping(a, agent)] -= 1.0

        if agent == self.world.agents[0]:
            agent_pos = torch.stack([a.state.pos for a in self.world.agents], dim=1)
            target_pos = torch.stack([t.state.pos for t in self._targets], dim=1)
            agent_target_dists = torch.cdist(agent_pos, target_pos)
            agents_per_target = torch.sum(
                (agent_target_dists < self._covering_range).type(torch.int), dim=1
            )
            covered_targets = agents_per_target == self._agents_per_target

            for i, target in enumerate(self._targets):
                occupied_positions = [agent_pos] + [
                    o.state.pos.unsqueeze(1) for o in self._targets if o is not target
                ]
                occupied_positions = torch.cat(occupied_positions, dim=1)
                pos = self._find_random_pos_for_entity(occupied_positions)
                target.state.pos[covered_targets[:, i]] = pos[
                    covered_targets[:, i]
                ].squeeze(1)

        return self.collision_rew

    def observation(self, agent: Agent):
        return torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
                agent.sensors[0].measure(),
                agent.sensors[1].measure(),
            ],
            dim=-1,
        )

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        try:
            info = {
                "collision_rew": self.collision_rew,
            }
        # When reset is called before reward()
        except AttributeError:
            info = {}
        return info


if __name__ == "__main__":
    render_interactively("target_coverage", n_agents=4)
