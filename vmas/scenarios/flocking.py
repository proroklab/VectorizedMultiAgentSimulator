#  Copyright (c) 2022. Jan Blumenkamp
#  All rights reserved.
import math
from typing import Dict, Callable

import torch
from torch import Tensor

from vmas import render_interactively
from vmas.simulator.core import Agent, Landmark, Sphere, World, Entity, Box
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color
from vmas.simulator.sensors import Lidar

DEFAULT_ENERGY_COEFF = 0.02


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        n_agents = kwargs.get("n_agents", 4)
        self.energy_coeff = kwargs.get(
            "energy_coeff", DEFAULT_ENERGY_COEFF
        )  # Weight of team energy penalty

        # Make world
        world = World(batch_dim, device)
        # Add agents
        goal_entity_filter: Callable[[Entity], bool] = lambda e: e.name == "goal"
        for i in range(n_agents):
            # Constraint: all agents have same action range and multiplier
            agent = Agent(
                name=f"agent {i}",
                collide=True,
                sensors=[
                    Lidar(
                        world,
                        n_rays=12,
                        max_range=0.5,
                        entity_filter=goal_entity_filter,
                    )
                ],
            )
            world.add_agent(agent)

        # Add landmarks
        goal = Landmark(
            name="goal",
            collide=False,
            shape=Sphere(radius=0.03),
            color=Color.GREEN,
        )
        world.add_landmark(goal)

        self.obstacles = []
        for i in range(5):
            obstacle = Landmark(
                name=f"obstacle_{i}",
                collide=True,
                movable=False,
                shape=Box(width=0.1, length=0.1),
                color=Color.RED,
            )
            world.add_landmark(obstacle)
            self.obstacles.append(obstacle)

        return world

    def reset_world_at(self, env_index: int = None):
        for agent in self.world.agents:
            # Random pos between -1 and 1
            agent.set_pos(
                torch.zeros(
                    (1, self.world.dim_p)
                    if env_index is not None
                    else (self.world.batch_dim, self.world.dim_p),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -1.0,
                    1.0,
                ),
                batch_index=env_index,
            )
        for obstacle in self.obstacles:
            # Random pos between -1 and 1
            obstacle.set_pos(
                torch.zeros(
                    (1, self.world.dim_p)
                    if env_index is not None
                    else (self.world.batch_dim, self.world.dim_p),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -1.0,
                    1.0,
                ),
                batch_index=env_index,
            )
            obstacle.set_rot(
                torch.zeros(
                    (1,) if env_index is not None else (self.world.batch_dim,),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -torch.pi,
                    torch.pi,
                ),
                batch_index=env_index,
            )

    def reward(self, agent: Agent):
        # Avoid collisions with each other
        self.collision_rew = torch.zeros(self.world.batch_dim, device=self.world.device)
        for a in self.world.agents:
            if a != agent:
                self.collision_rew[self.world.is_overlapping(a, agent)] -= 1.0

        # stay close together (separation)
        agents_rel_pos = [agent.state.pos - a.state.pos for a in self.world.agents]
        agents_rel_dist = torch.linalg.norm(torch.stack(agents_rel_pos, dim=1), dim=2)
        agents_max_dist, _ = torch.max(agents_rel_dist, dim=1)
        self.separation_rew = -agents_max_dist

        # keep moving (reward velocity)
        self.velocity_rew = torch.linalg.norm(agent.state.vel, dim=1)

        # stay close to target (cohesion)
        dist_target = torch.linalg.norm(
            agent.state.pos - self.world.landmarks[0].state.pos, dim=1
        )
        self.cohesion_rew = -dist_target

        return (
            self.collision_rew
            + self.velocity_rew
            + self.separation_rew
            + self.cohesion_rew
        )

    def observation(self, agent: Agent):
        return torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
                self.world.landmarks[0].state.pos - agent.state.pos,
                agent.sensors[0].measure(),
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
    render_interactively("flocking", n_agents=10, energy_coeff=DEFAULT_ENERGY_COEFF)
