#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.


import torch

from vmas import render_interactively
from vmas.simulator.core import Agent, World
from vmas.simulator.dynamics.static import Static
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import ScenarioUtils


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.n_agents = kwargs.pop("n_agents", 1)
        self.k = kwargs.pop("k", 0)
        self.horizon = kwargs.pop("horizon", 100)
        self.u_range = kwargs.pop("u_range", 1)
        ScenarioUtils.check_kwargs_consumed(kwargs)

        # Make world
        world = World(
            batch_dim,
            device,
        )
        # Add agents
        for i in range(self.n_agents):
            agent = Agent(
                name=f"agent_{i}",
                u_range=self.u_range,
                render_action=True,
                dynamics=Static(),
                action_size=2,
            )
            world.add_agent(agent)

        self.actions = torch.zeros(
            batch_dim,
            self.n_agents,
            self.horizon + 1,
            2,
            device=device,
            dtype=torch.float,
        )
        self.t = torch.zeros(batch_dim, device=device, dtype=torch.int)
        return world

    def reset_world_at(self, env_index: int = None):
        if env_index is None:
            self.actions = torch.zeros(
                self.world.batch_dim,
                self.n_agents,
                self.horizon + 1,
                2,
                device=self.world.device,
                dtype=torch.float,
            ).uniform_(-self.u_range, self.u_range)
            self.t = torch.zeros(
                self.world.batch_dim, device=self.world.device, dtype=torch.int
            )
        else:
            self.actions[env_index] = torch.zeros(
                self.n_agents,
                self.horizon + 1,
                2,
                device=self.world.device,
                dtype=torch.float,
            ).uniform_(-self.u_range, self.u_range)
            self.t[env_index] = 0

    def reward(self, agent: Agent):
        rew = torch.where(
            self.t >= self.k,
            -torch.linalg.vector_norm(
                agent.action.u
                - self.actions[torch.arange(self.world.batch_dim), :, self.t - self.k][
                    :, self.world.agents.index(agent)
                ],
                dim=-1,
            ),
            0,
        )
        is_last = agent == self.world.agents[-1]
        if is_last:
            self.t += 1
        return rew

    # def extra_render(self, env_index: int = 0) -> "List[Geom]":
    #     from vmas.simulator import rendering
    #
    #     geoms = []
    #
    #     # Agent rotation
    #     for agent in self.world.agents:
    #         color = Color.BLACK.value
    #
    #         line = rendering.Line(
    #             (0, 0),
    #             (
    #                 self.actions[
    #                     env_index, self.world.agents.index(agent), self.t, X
    #                 ].item(),
    #                 self.actions[
    #                     env_index, self.world.agents.index(agent), self.t, Y
    #                 ].item(),
    #             ),
    #             width=1,
    #         )
    #         xform = rendering.Transform()
    #         xform.set_translation(*agent.state.pos[env_index])
    #         line.add_attr(xform)
    #         line.set_color(*color)
    #         geoms.append(line)
    #
    #         if agent.action.u is not None:
    #             line = rendering.Line(
    #                 (0, 0),
    #                 (agent.action.u[env_index, X], agent.action.u[env_index, Y]),
    #                 width=1,
    #             )
    #             xform = rendering.Transform()
    #             xform.set_translation(*agent.state.pos[env_index])
    #             line.add_attr(xform)
    #             line.set_color(*agent.color)
    #             geoms.append(line)
    #
    #     return geoms

    def observation(self, agent: Agent):
        # get positions of all entities in this agent's reference frame

        return self.actions[torch.arange(self.world.batch_dim), :, self.t][
            :, self.world.agents.index(agent)
        ]

    def done(self):
        rand = torch.rand(
            self.world.batch_dim, dtype=torch.float, device=self.world.device
        )
        rand_termination = rand < 0.05
        return (self.t >= self.horizon - 1) + rand_termination


if __name__ == "__main__":
    render_interactively(
        __file__,
    )
