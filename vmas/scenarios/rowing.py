#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import typing
from typing import List

import torch

from vmas import render_interactively
from vmas.simulator.core import Agent, Box, Landmark, World
from vmas.simulator.dynamics.forward import Forward
from vmas.simulator.joints import Joint
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.n_agents = kwargs.pop("n_agents", 2)
        self.rotatable_modules = kwargs.pop("rotatable_modules", False)
        self.module_mass = kwargs.pop("module_mass", 10)
        self.observe_shared = kwargs.pop("observe_shared", True)

        self.pos_shaping_factor = kwargs.pop("pos_shaping_factor", 1)

        ScenarioUtils.check_kwargs_consumed(kwargs)

        assert not self.n_agents % 2, "Rowing needs even agents"
        self.width = 0.2
        self.box_width = 0.05
        self.box_length = 0.15
        self.row_dist = 0.3
        self.goal_semidim = 2
        self.viewer_zoom = 1.5
        self.min_goal_dist = self.box_length

        # Make world
        world = World(
            batch_dim, device, substeps=5, torque_constraint_force=12, joint_force=300
        )

        # Goal
        goal = Landmark(
            name="goal",
            collide=False,
            color=Color.GREEN,
        )
        world.add_landmark(goal)
        self.goal = goal

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
        self.agent_joints = []
        while i < self.n_agents - 1:
            joint = Joint(
                world.agents[i],
                world.agents[i + 1],
                anchor_a=(0, 0),
                anchor_b=(0, 0),
                dist=self.width,
                rotate_a=False,
                rotate_b=False,
                collidable=False,
                width=0,
                mass=self.module_mass - 2,
            )
            self.agent_joints.append(joint)
            world.add_joint(joint)
            i += 2
        self.joint_joints = []
        for i in range(len(self.agent_joints) - 1):
            joint = Joint(
                self.agent_joints[i].landmark,
                self.agent_joints[i + 1].landmark,
                anchor_a=(0, 0),
                anchor_b=(0, 0),
                dist=self.row_dist,
                rotate_a=self.rotatable_modules,
                rotate_b=self.rotatable_modules,
                collidable=False,
                width=0,
                mass=1,
            )
            self.joint_joints.append(joint)
            world.add_joint(joint)

        self.on_goal = torch.zeros(world.batch_dim, dtype=torch.bool, device=device)
        return world

    def reset_world_at(self, env_index: int = None):

        ScenarioUtils.spawn_entities_randomly(
            [self.goal],
            self.world,
            env_index,
            0,
            (-self.goal_semidim, self.goal_semidim),
            (-self.goal_semidim, self.goal_semidim),
        )

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

        _, pos, _ = self._get_central_entity()
        if env_index is None:
            self.pos_shaping = (
                torch.linalg.vector_norm(
                    pos - self.goal.state.pos,
                    dim=-1,
                )
                * self.pos_shaping_factor
            )
        else:
            self.pos_shaping[env_index] = (
                torch.linalg.vector_norm(
                    pos[env_index] - self.goal.state.pos[env_index]
                )
                * self.pos_shaping_factor
            )

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            _, pos, _ = self._get_central_entity()
            goal_dist = torch.linalg.vector_norm(
                pos - self.goal.state.pos,
                dim=-1,
            )
            self.on_goal = goal_dist < self.min_goal_dist
            pos_shaping = goal_dist * self.pos_shaping_factor

            self.pos_rew = self.pos_shaping - pos_shaping
            self.pos_shaping = pos_shaping
        return self.pos_rew

    def _get_central_entity(self):
        if len(self.agent_joints) % 2:
            landmark = self.agent_joints[len(self.agent_joints) // 2].landmark
            pos = landmark.state.pos
            rot = landmark.state.rot
        else:
            landmark = self.joint_joints[len(self.joint_joints) // 2].landmark
            pos = landmark.state.pos
            rot = landmark.state.rot + torch.pi / 2
        return landmark, pos, rot

    def observation(self, agent: Agent):
        # get positions of all entities in this agent's reference frame
        if self.observe_shared:
            _, pos, rot = self._get_central_entity()
        else:
            pos = agent.state.pos
            rot = agent.state.rot - torch.pi / 2
        rot = torch.cat([torch.cos(rot), torch.sin(rot)], dim=-1)
        return torch.cat(
            [pos, rot],
            dim=-1,
        )

    def done(self):
        return self.on_goal

    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        from vmas.simulator import rendering

        geoms = []

        # Agent rotation
        entity = self.agent_joints[0].landmark
        color = entity.color
        line = rendering.Line(
            (0, 0),
            (0.15, 0),
            width=2,
        )
        xform = rendering.Transform()
        xform.set_rotation(entity.state.rot[env_index] + torch.pi / 2)
        xform.set_translation(*entity.state.pos[env_index])
        line.add_attr(xform)
        line.set_color(*color)
        geoms.append(line)

        return geoms


if __name__ == "__main__":
    render_interactively(
        __file__,
        control_two_agents=True,
        n_agents=2,
    )
