#  Copyright (c) 2022.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import torch

from vmas import render_interactively
from vmas.simulator.core import Agent, World, Landmark, Sphere, Color
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.velocity_controller import VelocityController


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        n_agents = kwargs.get("n_agents", 1)

        # Make world
        world = World(batch_dim, device, drag=0.5)
        # Add agents
        for i in range(n_agents):
            # Constraint: all agents have same action range and multiplier
            agent = Agent(
                name=f"agent {i}", collide=True, u_range=1, u_multiplier=1, f_range=None
            )
            world.add_agent(agent)
            agent.controller = VelocityController(agent, world.dt, [1.5, 0.15, 0.01], "standard")
        goal = Landmark(
            name="goal",
            collide=True,
            movable=True,
            shape=Sphere(radius=0.3),
            color=Color.GREEN,
            mass=0.5,
            linear_friction = 0.01,
        )
        world.add_landmark(goal);
        return world

    def reset_world_at(self, env_index: int = None):
        for agent in self.world.agents:
            agent.set_pos(
                torch.zeros(
                    (1, self.world.dim_p)
                    if env_index is not None
                    else (self.world.batch_dim, self.world.dim_p),
                    device=self.world.device,
                    dtype=torch.float32,
                ),
                batch_index=env_index,
            )
        for landmark in self.world.landmarks:
            landmark.set_pos(
                torch.tensor([0,1], device=self.world.device),
                batch_index=env_index,
            )    

    def process_action(self, agent: Agent):
        # print( agent.state.vel );
        agent.controller.process_force()

    def reward(self, agent: Agent):
        return torch.zeros(self.world.batch_dim, device=self.world.device)

    def observation(self, agent: Agent):
        return torch.cat(
            [agent.state.pos, agent.state.vel],
            dim=-1,
        )


if __name__ == "__main__":
    render_interactively("velocity_control", control_two_agents=False, n_agents=1)
