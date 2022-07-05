#  Copyright (c) 2022. Jan Blumenkamp
#  All rights reserved.
import math
from typing import Dict, Callable

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
        self._min_dist_between_entities = kwargs.get("min_dist_between_entities", 0.25)

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
                        max_range=0.5,
                        entity_filter=entity_filter_agents,
                        render_color=Color.BLUE,
                    ),
                    Lidar(
                        world,
                        n_rays=12,
                        max_range=0.5,
                        entity_filter=entity_filter_targets,
                        render_color=Color.GREEN,
                    ),
                ],
            )
            world.add_agent(agent)

        self._border = []
        for i in range(4):
            border = Landmark(
                name=f"border {i}",
                shape=Line(length=2.0),
                collide=False,
                color=Color.BLACK,
            )
            self._border.append(border)
            world.add_landmark(border)

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
        pos = None
        while True:
            p_shape = (
                (1, 1, self.world.dim_p)
                if env_index is not None
                else (self.world.batch_dim, 1, self.world.dim_p)
            )
            proposed_pos = torch.empty(
                p_shape,
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
        p_shape = (
            (1, self.world.dim_p)
            if env_index is not None
            else (self.world.batch_dim, self.world.dim_p)
        )
        occupied_positions = torch.zeros(
            (p_shape[0], 0, p_shape[1]), device=self.world.device
        )
        placable_entities = self._targets + self.world.agents
        for entity in placable_entities:
            pos = self._find_random_pos_for_entity(occupied_positions, env_index)
            occupied_positions = torch.cat([occupied_positions, pos], dim=1)
            entity.set_pos(pos.squeeze(1), batch_index=env_index)

        for i, border in enumerate(self._border):
            border.set_pos(
                torch.tensor(
                    [
                        0.0
                        if i % 2
                        else (
                            self.world.x_semidim if i == 0 else -self.world.x_semidim
                        ),
                        0.0
                        if not i % 2
                        else (
                            self.world.x_semidim if i == 1 else -self.world.x_semidim
                        ),
                    ],
                    dtype=torch.float32,
                    device=self.world.device,
                ),
                batch_index=env_index,
            )
            border.set_rot(
                torch.tensor(
                    [
                        torch.pi / 2 if not i % 2 else 0.0,
                    ],
                    dtype=torch.float32,
                    device=self.world.device,
                ),
                batch_index=env_index,
            )

    def reward(self, agent: Agent):
        # Avoid collisions with each other
        self.collision_rew = torch.zeros(self.world.batch_dim, device=self.world.device)
        for a in self.world.agents:
            if a != agent:
                self.collision_rew[self.world.is_overlapping(a, agent)] -= 1.0

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
                "velocity_rew": self.velocity_rew,
                "separation_rew": self.separation_rew,
                "cohesion_rew": self.cohesion_rew,
            }
        # When reset is called before reward()
        except AttributeError:
            info = {}
        return info


if __name__ == "__main__":
    render_interactively("target_coverage", n_agents=5)
