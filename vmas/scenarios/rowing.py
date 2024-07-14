#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import torch

from vmas import render_interactively
from vmas.simulator.core import Agent, Box, World
from vmas.simulator.dynamics.forward import Forward
from vmas.simulator.joints import Joint
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import ScenarioUtils


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.n_agents = kwargs.pop("n_agents", 2)
        self.rotatable_modules = kwargs.pop("rotatable_modules", False)
        self.module_mass = kwargs.pop("module_mass", 10)

        ScenarioUtils.check_kwargs_consumed(kwargs)
        assert not self.n_agents % 2
        self.width = 0.2
        self.box_width = 0.05
        self.box_length = 0.15
        self.row_dist = 0.3

        # Make world
        world = World(
            batch_dim, device, substeps=5, torque_constraint_force=12, joint_force=300
        )
        # Add agents
        for i in range(self.n_agents):
            agent = Agent(
                name=f"agent_{i}",
                shape=Box(length=self.box_length, width=self.box_width),
                dynamics=Forward(),
                rotatable=True,
                collide=False,
            )
            world.add_agent(agent)

        # Add joints
        i = 0
        agent_joints = []
        while i < self.n_agents - 1:
            joint = Joint(
                world.agents[i],
                world.agents[i + 1],
                anchor_a=(0, 0),
                anchor_b=(0, 0),
                dist=self.width,
                rotate_a=False,
                rotate_b=False,
                fixed_rotation_a=-torch.pi / 2,
                fixed_rotation_b=-torch.pi / 2,
                collidable=False,
                width=0,
                mass=self.module_mass - 2,
            )
            agent_joints.append(joint)
            world.add_joint(joint)
            i += 2

        for i in range(len(agent_joints) - 1):
            joint = Joint(
                agent_joints[i].landmark,
                agent_joints[i + 1].landmark,
                anchor_a=(0, 0),
                anchor_b=(0, 0),
                dist=self.row_dist,
                rotate_a=self.rotatable_modules,
                rotate_b=self.rotatable_modules,
                fixed_rotation_a=torch.pi / 2 if not self.rotatable_modules else None,
                fixed_rotation_b=torch.pi / 2 if not self.rotatable_modules else None,
                collidable=False,
                width=0,
                mass=1,
            )

            world.add_joint(joint)

        return world

    def reset_world_at(self, env_index: int = None):
        i = 0
        row = 0
        while i < self.n_agents - 1:

            self.world.agents[i].set_pos(
                torch.tensor(
                    [
                        (-self.width / 2),
                        -row * self.row_dist,
                    ],
                    dtype=torch.float32,
                    device=self.world.device,
                ),
                batch_index=env_index,
            )
            self.world.agents[i + 1].set_pos(
                torch.tensor(
                    [
                        (self.width / 2),
                        -row * self.row_dist,
                    ],
                    dtype=torch.float32,
                    device=self.world.device,
                ),
                batch_index=env_index,
            )
            rotation = torch.tensor(
                [torch.pi / 2],
                dtype=torch.float32,
                device=self.world.device,
            )
            self.world.agents[i].set_rot(
                rotation,
                batch_index=env_index,
            )
            self.world.agents[i + 1].set_rot(
                rotation,
                batch_index=env_index,
            )
            i += 2
            row += 1

    def reward(self, agent: Agent):
        dist2 = torch.linalg.vector_norm(agent.state.pos, dim=1)
        return -dist2

    def observation(self, agent: Agent):
        # get positions of all entities in this agent's reference frame
        return torch.cat(
            [agent.state.pos, agent.state.vel],
            dim=-1,
        )


if __name__ == "__main__":
    render_interactively(
        __file__,
        control_two_agents=True,
        n_agents=6,
    )
