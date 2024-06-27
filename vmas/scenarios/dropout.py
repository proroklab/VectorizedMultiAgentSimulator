#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import math
from typing import Dict

import torch
from torch import Tensor

from vmas import render_interactively
from vmas.simulator.core import Agent, Landmark, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils

DEFAULT_ENERGY_COEFF = 0.02


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        n_agents = kwargs.pop("n_agents", 4)
        self.energy_coeff = kwargs.pop(
            "energy_coeff", DEFAULT_ENERGY_COEFF
        )  # Weight of team energy penalty
        self.start_same_point = kwargs.pop("start_same_point", False)
        ScenarioUtils.check_kwargs_consumed(kwargs)
        self.agent_radius = 0.05
        self.goal_radius = 0.03

        # Make world
        world = World(batch_dim, device)
        # Add agents
        for i in range(n_agents):
            # Constraint: all agents have same action range and multiplier
            agent = Agent(
                name=f"agent_{i}",
                collide=False,
                shape=Sphere(radius=self.agent_radius),
            )
            world.add_agent(agent)
        # Add landmarks
        goal = Landmark(
            name="goal",
            collide=False,
            shape=Sphere(radius=self.goal_radius),
            color=Color.GREEN,
        )
        world.add_landmark(goal)

        self.pos_rew = torch.zeros(batch_dim, device=device)
        self.energy_rew = self.pos_rew.clone()
        self._done = torch.zeros(batch_dim, device=device, dtype=torch.bool)

        return world

    def reset_world_at(self, env_index: int = None):
        if self.start_same_point:
            for agent in self.world.agents:
                agent.set_pos(
                    torch.zeros(
                        (1, 2) if env_index is not None else (self.world.batch_dim, 2),
                        device=self.world.device,
                        dtype=torch.float,
                    ),
                    batch_index=env_index,
                )
            ScenarioUtils.spawn_entities_randomly(
                self.world.landmarks,
                self.world,
                env_index,
                self.goal_radius + self.agent_radius + 0.01,
                x_bounds=(-1, 1),
                y_bounds=(-1, 1),
                occupied_positions=torch.zeros(
                    1 if env_index is not None else self.world.batch_dim,
                    1,
                    2,
                    device=self.world.device,
                    dtype=torch.float,
                ),
            )
        else:
            ScenarioUtils.spawn_entities_randomly(
                self.world.policy_agents + self.world.landmarks,
                self.world,
                env_index,
                self.goal_radius + self.agent_radius + 0.01,
                x_bounds=(-1, 1),
                y_bounds=(-1, 1),
            )

        for landmark in self.world.landmarks:
            if env_index is None:
                landmark.eaten = torch.full(
                    (self.world.batch_dim,), False, device=self.world.device
                )
                landmark.reset_render()
                self._done[:] = False
            else:
                landmark.eaten[env_index] = False
                landmark.is_rendering[env_index] = True
                self._done[env_index] = False

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]
        is_last = agent == self.world.agents[-1]

        if is_first:
            self.any_eaten = self._done = torch.any(
                torch.stack(
                    [
                        torch.linalg.vector_norm(
                            a.state.pos - self.world.landmarks[0].state.pos,
                            dim=1,
                        )
                        < a.shape.radius + self.world.landmarks[0].shape.radius
                        for a in self.world.agents
                    ],
                    dim=1,
                ),
                dim=-1,
            )

        self.pos_rew[:] = 0
        self.pos_rew[self.any_eaten * ~self.world.landmarks[0].eaten] = 1

        if is_last:
            self.world.landmarks[0].eaten[self.any_eaten] = True
            self.world.landmarks[0].is_rendering[self.any_eaten] = False

        # Assumption: all agents have same action range and multiplier
        if is_first:
            self.energy_rew = self.energy_coeff * -torch.stack(
                [
                    torch.linalg.vector_norm(a.action.u, dim=-1)
                    / math.sqrt(self.world.dim_p * ((a.u_range * a.u_multiplier) ** 2))
                    for a in self.world.agents
                ],
                dim=1,
            ).sum(-1)

        rew = self.pos_rew + self.energy_rew

        return rew

    def observation(self, agent: Agent):
        return torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
                self.world.landmarks[0].state.pos - agent.state.pos,
                self.world.landmarks[0].eaten.unsqueeze(-1),
            ],
            dim=-1,
        )

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        info = {"pos_rew": self.pos_rew, "energy_rew": self.energy_rew}
        return info

    def done(self):
        return self._done


if __name__ == "__main__":
    render_interactively(
        __file__,
        control_two_agents=True,
        n_agents=4,
        energy_coeff=DEFAULT_ENERGY_COEFF,
    )
