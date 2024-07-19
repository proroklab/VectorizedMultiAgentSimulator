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
from vmas.simulator.utils import Color, ScenarioUtils, X, Y

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom


def angle_to_vector(angle):
    return torch.cat([torch.cos(angle), torch.sin(angle)], dim=-1)


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.n_agents = kwargs.pop("n_agents", 2)
        self.rotatable_modules = kwargs.pop("rotatable_modules", False)
        self.agent_mass = kwargs.pop("agent_mass", 5)
        self.observe_shared = kwargs.pop("observe_shared", True)
        self.sparse_rewards = kwargs.pop("sparse_rewards", True)

        self.pos_shaping_factor = kwargs.pop("pos_shaping_factor", 0)
        self.rot_shaping_factor = kwargs.pop("rot_shaping_factor", 1)

        ScenarioUtils.check_kwargs_consumed(kwargs)

        assert not self.n_agents % 2, "Rowing needs even agents"
        self.width = 0.2
        self.box_width = 0.05
        self.box_length = 0.15
        self.row_dist = 0.3
        self.goal_semidim = 2
        self.viewer_zoom = 1.5
        self.min_goal_dist = self.box_length
        self.min_rot_dist = 0.95

        # Make world
        world = World(
            batch_dim,
            device,
            substeps=25,
            torque_constraint_force=100,
            joint_force=1000,
        )

        # Goal
        goal = Landmark(name="goal", collide=False, color=Color.GREEN)
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
                mass=self.agent_mass,
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
                mass=1,
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
        self.on_rot = torch.zeros(world.batch_dim, dtype=torch.bool, device=device)

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
        self.goal.set_rot(
            torch.zeros(
                (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                device=self.world.device,
                dtype=torch.float32,
            ).uniform_(
                -torch.pi,
                torch.pi,
            ),
            batch_index=env_index,
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

        _, pos, rot = self._get_central_entity()

        my_rot = angle_to_vector(rot)
        goal_rot = angle_to_vector(self.goal.state.rot)
        if env_index is None:
            self.pos_shaping = (
                torch.linalg.vector_norm(
                    pos - self.goal.state.pos,
                    dim=-1,
                )
                * self.pos_shaping_factor
            )
            self.rot_shaping = -((goal_rot * my_rot).sum(-1)) * self.rot_shaping_factor
        else:
            self.pos_shaping[env_index] = (
                torch.linalg.vector_norm(
                    pos[env_index] - self.goal.state.pos[env_index]
                )
                * self.pos_shaping_factor
            )
            self.rot_shaping[env_index] = (
                -((goal_rot[env_index] * my_rot[env_index]).sum(-1))
                * self.rot_shaping_factor
            )

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            _, pos, rot = self._get_central_entity()
            goal_dist = torch.linalg.vector_norm(
                pos - self.goal.state.pos,
                dim=-1,
            )
            self.on_goal = goal_dist < self.min_goal_dist
            pos_shaping = goal_dist * self.pos_shaping_factor

            if not self.sparse_rewards:
                self.pos_rew = self.pos_shaping - pos_shaping
            else:
                self.pos_rew = self.on_goal.float() * self.pos_shaping_factor
            self.pos_shaping = pos_shaping

            my_rot = angle_to_vector(rot)
            goal_rot = angle_to_vector(self.goal.state.rot)
            rot_dist = (goal_rot * my_rot).sum(-1)
            self.on_rot = rot_dist > self.min_rot_dist
            rot_shaping = -rot_dist * self.rot_shaping_factor

            if not self.sparse_rewards:
                self.rot_rew = self.rot_shaping - rot_shaping
            else:
                self.rot_rew = self.on_rot.float() * self.rot_shaping_factor
            self.rot_shaping = rot_shaping

        return self.pos_rew + self.rot_rew

    def _get_central_entity(self):
        if len(self.agent_joints) % 2:
            landmark = self.agent_joints[len(self.agent_joints) // 2].landmark
            pos = landmark.state.pos
            rot = landmark.state.rot + torch.pi / 2
        else:
            landmark = self.joint_joints[len(self.joint_joints) // 2].landmark
            pos = landmark.state.pos
            rot = landmark.state.rot + torch.pi
        return landmark, pos, rot

    def observation(self, agent: Agent):
        # get positions of all entities in this agent's reference frame
        entity, pos, rot = self._get_central_entity()
        if not self.observe_shared:
            # pos = agent.state.pos
            rot = agent.state.rot
        rot = angle_to_vector(rot)
        goal_rot = angle_to_vector(self.goal.state.rot)
        return torch.cat(
            [(goal_rot * rot).sum(-1, keepdim=True), rot, goal_rot],
            dim=-1,
        )

    def done(self):
        return self.on_rot

    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        from vmas.simulator import rendering

        geoms = []

        # Agent indices
        for i, entity in enumerate(self.world.agents):
            line = rendering.TextLine(
                text=str(i),
                font_size=15,
                x=(entity.state.pos[env_index, X] / (self.viewer_zoom**2))
                * self.viewer_size[X]
                / 2
                + self.viewer_size[X] / 2,
                y=(entity.state.pos[env_index, Y] / (self.viewer_zoom**2))
                * self.viewer_size[Y]
                / 2
                + self.viewer_size[Y] / 2,
            )
            xform = rendering.Transform()
            line.add_attr(xform)
            geoms.append(line)

        # Rotation
        for entity in [self.agent_joints[0].landmark, self.goal]:
            color = entity.color
            line = rendering.Line(
                (0, 0),
                (0.15, 0),
                width=2,
            )
            xform = rendering.Transform()
            xform.set_rotation(
                entity.state.rot[env_index]
                + (torch.pi / 2 if entity is not self.goal else 0)
            )
            xform.set_translation(*entity.state.pos[env_index])
            line.add_attr(xform)
            line.set_color(*color)
            geoms.append(line)

        return geoms


if __name__ == "__main__":
    render_interactively(
        __file__,
        control_two_agents=True,
        n_agents=4,
    )
