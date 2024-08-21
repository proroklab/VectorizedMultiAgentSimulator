#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import typing
from typing import List

import torch

from vmas import render_interactively
from vmas.simulator.core import Agent, Landmark, World
from vmas.simulator.dynamics.roatation import Rotation
from vmas.simulator.dynamics.static import Static
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom


def angle_to_vector(angle):
    return torch.cat([torch.cos(angle), torch.sin(angle)], dim=-1)


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.n_agents = kwargs.pop("n_agents", 2)
        self.agent_mass = kwargs.pop("agent_mass", 200)
        self.sparse_rewards = kwargs.pop("sparse_rewards", True)

        self.pos_shaping_factor = kwargs.pop("pos_shaping_factor", 0)
        self.rot_shaping_factor = kwargs.pop("rot_shaping_factor", 1)

        ScenarioUtils.check_kwargs_consumed(kwargs)

        assert not self.n_agents % 2, "Rowing needs even agents"

        self.viewer_zoom = 1.5
        self.radius = 0.1
        self.u_range = 1

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
                dynamics=Static(),
                collide=False,
                action_size=1,
                u_range=self.u_range,
            )
            world.add_agent(agent)

        self.left_agents = world.agents[: self.n_agents // 2]
        self.right_agents = world.agents[self.n_agents // 2 :]

        # Central entity
        self.central_entity = Agent(
            name="entity",
            collide=False,
            rotatable=True,
            dynamics=Rotation(),
            color=Color.BLACK,
            action_script=entity_script,
            mass=self.agent_mass * self.n_agents,
        )
        world.add_agent(self.central_entity)

        self.on_goal = torch.zeros(world.batch_dim, dtype=torch.bool, device=device)
        self.on_rot = torch.zeros(world.batch_dim, dtype=torch.bool, device=device)

        return world

    def reset_world_at(self, env_index: int = None):
        for i, agent in enumerate(self.left_agents):
            agent.set_pos(
                torch.tensor([1 + i * self.radius, 0], device=self.world.device),
                batch_index=env_index,
            )
        for i, agent in enumerate(self.right_agents):
            agent.set_pos(
                torch.tensor([-1 - i * self.radius, 0], device=self.world.device),
                batch_index=env_index,
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

        self.central_entity.set_rot(
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

        _, pos, rot = self._get_central_entity()

        my_rot = angle_to_vector(rot)
        goal_rot = angle_to_vector(self.goal.state.rot)
        if env_index is None:
            self.rot_shaping = -((goal_rot * my_rot).sum(-1)) * self.rot_shaping_factor
        else:

            self.rot_shaping[env_index] = (
                -((goal_rot[env_index] * my_rot[env_index]).sum(-1))
                * self.rot_shaping_factor
            )

    def process_action(self, agent: Agent):
        is_first = self.world.agents.index(agent) == 0
        if is_first:
            self.central_entity.torque = torch.zeros(
                self.world.batch_dim, 1, device=self.world.device, dtype=torch.float32
            )
        if agent is self.central_entity:
            self.central_entity.action.u = self.central_entity.torque
        else:
            self.central_entity.torque += (
                self.radius * agent.action.u * (1 if agent in self.left_agents else -1)
            )

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]
        is_last = self.world.agents.index(agent) == self.n_agents - 1

        if is_first:
            _, pos, rot = self._get_central_entity()

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
        if is_last:
            self.goal.state.rot = torch.where(
                self.on_rot.unsqueeze(-1),
                torch.zeros(
                    (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -torch.pi,
                    torch.pi,
                ),
                self.goal.state.rot,
            )

        return self.rot_rew

    def _get_central_entity(self):
        return (
            self.central_entity,
            self.central_entity.state.pos,
            self.central_entity.state.rot,
        )

    def observation(self, agent: Agent):
        # get positions of all entities in this agent's reference frame
        entity, pos, rot = self._get_central_entity()
        rot = angle_to_vector(rot)
        goal_rot = angle_to_vector(self.goal.state.rot)
        return torch.cat(
            [(goal_rot * rot).sum(-1, keepdim=True), rot, goal_rot],
            dim=-1,
        )

    # def done(self):
    #     return self.on_rot

    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        from vmas.simulator import rendering

        geoms = []

        # Agent actions
        for entity in self.left_agents + self.right_agents:
            color = entity.color
            line = rendering.Line(
                (0, 0),
                (
                    0,
                    entity.action.u[env_index] / 4
                    if entity.action.u is not None
                    else 0,
                ),
                width=2,
            )
            xform = rendering.Transform()
            xform.set_translation(*entity.state.pos[env_index])
            line.add_attr(xform)
            line.set_color(*color)
            geoms.append(line)

        # Rotation
        for entity in [self.central_entity, self.goal]:
            color = entity.color
            line = rendering.Line(
                (0, 0),
                (0.15, 0),
                width=2,
            )
            xform = rendering.Transform()
            xform.set_rotation(entity.state.rot[env_index])
            xform.set_translation(*entity.state.pos[env_index])
            line.add_attr(xform)
            line.set_color(*color)
            geoms.append(line)

        return geoms


def entity_script(agent: Agent, world: World):
    agent.action.u = torch.zeros(
        world.batch_dim, 1, device=world.device, dtype=torch.float32
    )
    return None


if __name__ == "__main__":
    render_interactively(
        __file__,
        control_two_agents=True,
        n_agents=8,
    )
