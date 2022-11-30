#  Copyright (c) 2022.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import torch

from vmas import render_interactively
from vmas.simulator.core import Agent, World, Sphere, Landmark, Box, Line
from vmas.simulator.joints import Joint
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.n_agents = kwargs.get("n_agents", 5)
        self.with_joints = kwargs.get("joints", True)

        self.agent_dist = 0.1
        self.agent_radius = 0.04

        # Make world
        world = World(
            batch_dim, device, dt=0.1, drag=0.25, substeps=5, collision_force=500
        )
        # Add agents
        for i in range(self.n_agents):
            agent = Agent(
                name=f"agent {i}",
                shape=Sphere(radius=self.agent_radius),
                u_multiplier=0.7,
                rotatable=True,
            )
            world.add_agent(agent)
        if self.with_joints:
            # Add joints
            for i in range(self.n_agents - 1):
                joint = Joint(
                    world.agents[i],
                    world.agents[i + 1],
                    anchor_a=(1, 0),
                    anchor_b=(-1, 0),
                    dist=self.agent_dist,
                    rotate_a=True,
                    rotate_b=True,
                    collidable=True,
                    width=0,
                    mass=1,
                )
                world.add_joint(joint)
            landmark = Landmark(
                name="joined landmark",
                collide=True,
                movable=True,
                rotatable=True,
                shape=Box(length=self.agent_radius * 2, width=0.3),
                color=Color.GREEN,
            )
            world.add_landmark(landmark)
            joint = Joint(
                world.agents[-1],
                landmark,
                anchor_a=(1, 0),
                anchor_b=(-1, 0),
                dist=self.agent_dist,
                rotate_a=True,
                rotate_b=True,
                collidable=True,
                width=0,
                mass=1,
            )
            world.add_joint(joint)

        # Add landmarks
        for i in range(5):
            landmark = Landmark(
                name=f"landmark {i}",
                collide=True,
                movable=True,
                rotatable=True,
                shape=Box(length=0.3, width=0.1),
                color=Color.RED,
                # collision_filter=lambda e: False
                # if isinstance(e.shape, Box) and e.name != "joined landmark"
                # else True,
            )
            world.add_landmark(landmark)
        floor = Landmark(
            name="floor",
            collide=True,
            movable=False,
            shape=Line(length=2),
            color=Color.BLACK,
        )
        world.add_landmark(floor)

        return world

    def reset_world_at(self, env_index: int = None):
        for i, agent in enumerate(
            self.world.agents + [self.world.landmarks[self.n_agents - 1]]
        ):
            agent.set_pos(
                torch.tensor(
                    [-0.2 + (self.agent_dist + 2 * self.agent_radius) * i, 1.0],
                    dtype=torch.float32,
                    device=self.world.device,
                ),
                batch_index=env_index,
            )
        for i, landmark in enumerate(
            self.world.landmarks[(self.n_agents + 1) if self.with_joints else 0 : -1]
        ):
            landmark.set_pos(
                torch.tensor(
                    [0.2 if i % 2 else -0.2, 0.6 - 0.3 * i],
                    dtype=torch.float32,
                    device=self.world.device,
                ),
                batch_index=env_index,
            )
            landmark.set_rot(
                torch.tensor(
                    [torch.pi / 4 if i % 2 else -torch.pi / 4],
                    dtype=torch.float32,
                    device=self.world.device,
                ),
                batch_index=env_index,
            )
        floor = self.world.landmarks[-1]
        floor.set_pos(
            torch.tensor(
                [0, -1],
                dtype=torch.float32,
                device=self.world.device,
            ),
            batch_index=env_index,
        )

    def reward(self, agent: Agent):
        dist2 = torch.linalg.vector_norm(
            agent.state.pos - self.world.landmarks[-1].state.pos, dim=1
        )
        return -dist2

    def observation(self, agent: Agent):
        # get positions of all entities in this agent's reference frame
        return torch.cat(
            [agent.state.pos, agent.state.vel]
            + [
                landmark.state.pos - agent.state.pos
                for landmark in self.world.landmarks
            ],
            dim=-1,
        )


if __name__ == "__main__":
    render_interactively(
        __file__,
        control_two_agents=True,
        n_agents=5,
        joints=True,
    )
