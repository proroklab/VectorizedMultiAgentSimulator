#  Copyright (c) 2022.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.


import torch
from torch import Tensor

from vmas import render_interactively
from vmas.simulator.core import Agent, Box, Landmark, Sphere, World, Line
from vmas.simulator.joints import Joint
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, Y, X


def get_line_angle_0_90(rot: Tensor):
    angle = torch.abs(rot) % torch.pi
    otehr_angle = torch.pi - angle
    return torch.minimum(angle, otehr_angle)


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.n_passages = kwargs.get("n_passages", 1)
        self.fixed_passage = kwargs.get("fixed_passage", True)
        self.joint_length = kwargs.get("joint_length", 0.5)

        assert 1 <= self.n_passages <= 20

        self.pos_shaping_factor = 1
        self.rot_shaping_factor = 1
        self.collision_reward = -0.1

        self.middle_angle = torch.pi / 2

        self.n_agents = 2

        self.agent_radius = 0.03333
        self.agent_spacing = 0.1
        self.passage_width = 0.2
        self.passage_length = 0.103

        # Make world
        world = World(
            batch_dim,
            device,
            x_semidim=1,
            y_semidim=1,
            substeps=7,
            joint_force=400,
            collision_force=1500,
        )
        # Add agents
        for i in range(self.n_agents):
            agent = Agent(
                name=f"agent {i}", shape=Sphere(self.agent_radius), u_multiplier=0.7
            )
            world.add_agent(agent)

        self.joint = Joint(
            world.agents[0],
            world.agents[1],
            anchor_a=(0, 0),
            anchor_b=(0, 0),
            dist=self.joint_length,
            rotate_a=True,
            rotate_b=True,
            collidable=True,
            width=0,
            mass=1,
        )

        self.goal = Landmark(
            name="joint_goal",
            shape=Line(length=self.joint_length),
            collide=False,
            color=Color.GREEN,
        )
        world.add_landmark(self.goal)

        def joint_collision_filter(e):
            try:
                return e.neighbour
            except AttributeError:
                return False

        self.joint.landmark.collision_filter = joint_collision_filter
        world.add_joint(self.joint)
        self.create_passage_map(world)

        return world

    def reset_world_at(self, env_index: int = None):
        joint_pos = torch.cat(
            [
                torch.zeros(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -1 + (self.agent_radius + self.joint_length / 2),
                    1 - (self.agent_radius + self.joint_length / 2),
                ),
                torch.zeros(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -1 + self.agent_radius,
                    -2 * self.agent_radius - self.passage_width / 2,
                ),
            ],
            dim=1,
        )
        goal_pos = torch.cat(
            [
                torch.zeros(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -1 + (self.agent_radius + self.joint_length / 2),
                    1 - (self.agent_radius + self.joint_length / 2),
                ),
                torch.zeros(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    2 * self.agent_radius + self.passage_width / 2,
                    1 - self.agent_radius,
                ),
            ],
            dim=1,
        )

        x_shift = torch.tensor(
            [
                [self.joint_length / 2, 0.0],
            ],
            device=self.world.device,
        )

        self.goal.set_pos(
            goal_pos,
            batch_index=env_index,
        )
        self.goal.set_rot(
            torch.zeros(
                (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                device=self.world.device,
                dtype=torch.float32,
            ).uniform_(
                0,
                0,
            ),
            batch_index=env_index,
        )

        for i, agent in enumerate(self.world.agents):
            if i == 0:
                agent.set_pos(
                    joint_pos - x_shift,
                    batch_index=env_index,
                )
            else:
                agent.set_pos(
                    joint_pos + x_shift,
                    batch_index=env_index,
                )
        self.spawn_passage_map(env_index)

        if env_index is None:
            self.joint.pos_shaping_pre = (
                torch.stack(
                    [
                        torch.linalg.vector_norm(
                            self.joint.landmark.state.pos - p.state.pos, dim=1
                        )
                        for p in self.passages
                        if not p.collide
                    ],
                    dim=1,
                ).min(dim=1)[0]
                * self.pos_shaping_factor
            )
            self.joint.pos_shaping_post = (
                torch.linalg.vector_norm(
                    self.joint.landmark.state.pos - self.goal.state.pos, dim=1
                )
                * self.pos_shaping_factor
            )

            self.joint.rot_shaping_pre = (
                torch.abs(
                    get_line_angle_0_90(self.joint.landmark.state.rot)
                    - self.middle_angle,
                ).squeeze(-1)
                * self.rot_shaping_factor
            )
            self.joint.rot_shaping_post = (
                torch.abs(
                    get_line_angle_0_90(self.joint.landmark.state.rot)
                    - get_line_angle_0_90(self.goal.state.rot),
                ).squeeze(-1)
                * self.rot_shaping_factor
            )

        else:
            self.joint.pos_shaping_pre[env_index] = (
                torch.stack(
                    [
                        torch.linalg.vector_norm(
                            self.joint.landmark.state.pos[env_index]
                            - p.state.pos[env_index],
                        ).unsqueeze(-1)
                        for p in self.passages
                        if not p.collide
                    ],
                    dim=1,
                ).min(dim=1)[0]
                * self.pos_shaping_factor
            )
            self.joint.pos_shaping_post[env_index] = (
                torch.linalg.vector_norm(
                    self.joint.landmark.state.pos[env_index]
                    - self.goal.state.pos[env_index],
                )
                * self.pos_shaping_factor
            )
            self.joint.rot_shaping_pre[env_index] = (
                torch.abs(
                    get_line_angle_0_90(self.joint.landmark.state.rot[env_index])
                    - self.middle_angle,
                ).squeeze(-1)
                * self.rot_shaping_factor
            )
            self.joint.rot_shaping_post[env_index] = (
                torch.abs(
                    get_line_angle_0_90(self.joint.landmark.state.rot[env_index])
                    - get_line_angle_0_90(self.goal.state.rot[env_index]),
                ).squeeze(-1)
                * self.rot_shaping_factor
            )

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            self.rew = torch.zeros(
                self.world.batch_dim, device=self.world.device, dtype=torch.float32
            )

            joint_passed = self.joint.landmark.state.pos[:, Y] > 0
            # all_passed = torch.all(
            #     torch.stack([a.state.pos[:, Y] for a in self.world.agents], dim=1) > 0,
            #     dim=1,
            # )

            # Pos shaping
            joint_dist_to_closest_pass = torch.stack(
                [
                    torch.linalg.vector_norm(
                        self.joint.landmark.state.pos - p.state.pos, dim=1
                    )
                    for p in self.passages
                    if not p.collide
                ],
                dim=1,
            ).min(dim=1)[0]
            joint_shaping = joint_dist_to_closest_pass * self.pos_shaping_factor
            self.rew[~joint_passed] += (self.joint.pos_shaping_pre - joint_shaping)[
                ~joint_passed
            ]
            self.joint.pos_shaping_pre = joint_shaping

            joint_dist_to_goal = torch.linalg.vector_norm(
                self.joint.landmark.state.pos - self.goal.state.pos,
                dim=1,
            )
            joint_shaping = joint_dist_to_goal * self.pos_shaping_factor
            self.rew[joint_passed] += (self.joint.pos_shaping_post - joint_shaping)[
                joint_passed
            ]
            self.joint.pos_shaping_post = joint_shaping

            # Rot shaping
            joint_dist_to_90_rot = torch.abs(
                get_line_angle_0_90(self.joint.landmark.state.rot) - self.middle_angle,
            ).squeeze(-1)
            joint_shaping = joint_dist_to_90_rot * self.rot_shaping_factor
            self.rew[~joint_passed] += (self.joint.rot_shaping_pre - joint_shaping)[
                ~joint_passed
            ]
            self.joint.rot_shaping_pre = joint_shaping

            joint_dist_to_goal_rot = torch.abs(
                get_line_angle_0_90(self.joint.landmark.state.rot)
                - get_line_angle_0_90(self.goal.state.rot),
            ).squeeze(-1)
            joint_shaping = joint_dist_to_goal_rot * self.rot_shaping_factor
            self.rew[joint_passed] += (self.joint.rot_shaping_post - joint_shaping)[
                joint_passed
            ]
            self.joint.rot_shaping_post = joint_shaping

            # Agent collisions
            for a in self.world.agents:
                for passage in self.passages:
                    if passage.collide:
                        self.rew[
                            self.world.is_overlapping(a, passage)
                        ] += self.collision_reward

            # Joint collisions
            for i, p in enumerate(self.passages):
                if p.collide and p.neighbour:
                    self.rew[
                        self.world.is_overlapping(p, self.joint.landmark)
                    ] += self.collision_reward

        return self.rew

    def observation(self, agent: Agent):
        # get positions of all entities in this agent's reference frame
        passage_obs = []
        for passage in self.passages:
            if not passage.collide:
                passage_obs.append(passage.state.pos - agent.state.pos)
        return torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
                self.joint.landmark.state.pos - self.goal.state.pos,
                # get_line_angle_0_90(self.joint.landmark.state.rot),
                *passage_obs,
            ],
            dim=-1,
        )

    def done(self):
        return torch.all(
            (
                torch.linalg.vector_norm(
                    self.joint.landmark.state.pos - self.goal.state.pos, dim=1
                )
                <= self.agent_radius
            )
            * (
                torch.abs(
                    get_line_angle_0_90(self.joint.landmark.state.rot)
                    - get_line_angle_0_90(self.goal.state.rot),
                )
                <= 0.01
            ),
            dim=1,
        )

    def create_passage_map(self, world: World):
        # Add landmarks
        self.passages = []
        n_boxes = int(
            (2 * world.x_semidim + 2 * self.agent_radius) // self.passage_length
        )

        def removed(i):
            return (
                (n_boxes // 2) - self.n_passages / 2
                <= i
                < (n_boxes // 2) + self.n_passages / 2
            )

        for i in range(n_boxes):
            passage = Landmark(
                name=f"passage {i}",
                collide=not removed(i),
                movable=False,
                shape=Box(length=self.passage_length, width=self.passage_width),
                color=Color.RED,
                collision_filter=lambda e: not isinstance(e.shape, Box),
            )
            passage.neighbour = removed(i - 1) or removed(i + 1)
            self.passages.append(passage)
            world.add_landmark(passage)

    def spawn_passage_map(self, env_index):
        if not self.fixed_passage:
            order = torch.randperm(len(self.passages)).tolist()
            self.passages = [self.passages[i] for i in order]
        for i, passage in enumerate(self.passages):
            if not passage.collide:
                passage.is_rendering[:] = False
            passage.set_pos(
                torch.tensor(
                    [
                        -1
                        - self.agent_radius
                        + self.passage_length / 2
                        + self.passage_length * i,
                        0.0,
                    ],
                    dtype=torch.float32,
                    device=self.world.device,
                ),
                batch_index=env_index,
            )

    def extra_render(self, env_index: int = 0):
        from vmas.simulator import rendering

        geoms = []
        # Perimeter
        for i in range(4):
            geom = Line(length=2 + self.agent_radius * 2).get_geometry()
            xform = rendering.Transform()
            geom.add_attr(xform)

            xform.set_translation(
                0.0
                if i % 2
                else (
                    self.world.x_semidim + self.agent_radius
                    if i == 0
                    else -self.world.x_semidim - self.agent_radius
                ),
                0.0
                if not i % 2
                else (
                    self.world.x_semidim + self.agent_radius
                    if i == 1
                    else -self.world.x_semidim - self.agent_radius
                ),
            )
            xform.set_rotation(torch.pi / 2 if not i % 2 else 0.0)
            color = Color.BLACK.value
            if isinstance(color, torch.Tensor) and len(color.shape) > 1:
                color = color[env_index]
            geom.set_color(*color)
            geoms.append(geom)

        # Agent goal circles
        color = self.goal.color
        if isinstance(color, torch.Tensor) and len(color.shape) > 1:
            color = color[env_index]
        goal_agent_1 = rendering.make_circle(self.agent_radius)
        xform = rendering.Transform()
        goal_agent_1.add_attr(xform)
        xform.set_translation(
            self.goal.state.pos[env_index][X] - self.joint_length / 2,
            self.goal.state.pos[env_index][Y],
        )

        goal_agent_1.set_color(*color)
        geoms.append(goal_agent_1)
        goal_agent_2 = rendering.make_circle(self.agent_radius)
        xform = rendering.Transform()
        goal_agent_2.add_attr(xform)
        xform.set_translation(
            self.goal.state.pos[env_index][X] + self.joint_length / 2,
            self.goal.state.pos[env_index][Y],
        )
        goal_agent_2.set_color(*color)
        geoms.append(goal_agent_2)

        return geoms


if __name__ == "__main__":
    render_interactively("joint", control_two_agents=True)
