#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import typing
from typing import List

import torch

from vmas import render_interactively
from vmas.simulator.core import Agent, World
from vmas.simulator.dynamics.drone import Drone
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        """
        Drone example scenario
        Run this file to try it out.

        You can control the three input torques using left/right arrows, up/down arrows, and m/n.
        """
        self.plot_grid = True
        self.n_agents = kwargs.pop("n_agents", 2)
        ScenarioUtils.check_kwargs_consumed(kwargs)

        # Make world
        world = World(batch_dim, device, substeps=10)

        for i in range(self.n_agents):
            agent = Agent(
                name=f"drone_{i}",
                collide=True,
                render_action=True,
                u_range=[
                    0.00001,
                    0.00001,
                    0.00001,
                ],  # torque_x, torque_y, torque_z
                u_multiplier=[1, 1, 1],
                action_size=3,  # We feed only the torque actions to interactively control the drone in the debug scenario
                # In non-debug cases, remove this line and the `process_action` function in this file
                dynamics=Drone(world, integration="rk4"),
            )
            world.add_agent(agent)

        return world

    def reset_world_at(self, env_index: int = None):
        ScenarioUtils.spawn_entities_randomly(
            self.world.agents,
            self.world,
            env_index,
            min_dist_between_entities=0.1,
            x_bounds=(-1, 1),
            y_bounds=(-1, 1),
        )

    def reward(self, agent: Agent):
        return torch.zeros(self.world.batch_dim, device=self.world.device)

    def process_action(self, agent: Agent):
        torque = agent.action.u
        thrust = torch.full(
            (self.world.batch_dim, 1),
            agent.mass * agent.dynamics.g,
            device=self.world.device,
        )  # Add a fixed thrust to make sure the agent is not falling
        agent.action.u = torch.cat([thrust, torque], dim=-1)

    def observation(self, agent: Agent):
        observations = [
            agent.state.pos,
            agent.state.vel,
        ]
        return torch.cat(
            observations,
            dim=-1,
        )

    def done(self):
        return torch.any(
            torch.stack(
                [agent.dynamics.needs_reset() for agent in self.world.agents],
                dim=-1,
            ),
            dim=-1,
        )

    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        from vmas.simulator import rendering

        geoms: List[Geom] = []

        # Agent rotation
        for agent in self.world.agents:
            color = Color.BLACK.value
            line = rendering.Line(
                (0, 0),
                (0.1, 0),
                width=1,
            )
            xform = rendering.Transform()
            xform.set_rotation(agent.state.rot[env_index])
            xform.set_translation(*agent.state.pos[env_index])
            line.add_attr(xform)
            line.set_color(*color)
            geoms.append(line)

        return geoms


if __name__ == "__main__":
    render_interactively(__file__, control_two_agents=True)
