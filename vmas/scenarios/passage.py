#  Copyright (c) 2022.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import torch

from vmas import render_interactively
from vmas.simulator.core import Agent, Box, Landmark, Sphere, World, Line
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.n_passages = kwargs.get("n_passages", 1)
        self.shared_reward = kwargs.get("shared_reward", False)

        assert self.n_passages >= 1 and self.n_passages <= 20

        self.shaping_factor = 100

        self.n_agents = 5
        self.agent_radius = 0.03333
        self.agent_spacing = 0.1
        self.passage_width = 0.2
        self.passage_length = 0.103

        # Make world
        world = World(batch_dim, device, x_semidim=1, y_semidim=1)
        # Add agents
        for i in range(self.n_agents):
            agent = Agent(
                name=f"agent {i}", shape=Sphere(self.agent_radius), u_multiplier=0.7
            )
            world.add_agent(agent)
            goal = Landmark(
                name=f"goal {i}",
                collide=False,
                shape=Sphere(radius=self.agent_radius),
                color=Color.LIGHT_GREEN,
            )
            agent.goal = goal
            world.add_landmark(goal)
        # Add landmarks
        for i in range(
            int((2 * world.x_semidim + 2 * self.agent_radius) // self.passage_length)
        ):
            removed = i < self.n_passages
            passage = Landmark(
                name=f"passage {i}",
                collide=not removed,
                movable=False,
                shape=Box(length=self.passage_length, width=self.passage_width),
                color=Color.RED,
                collision_filter=lambda e: not isinstance(e.shape, Box),
            )
            world.add_landmark(passage)

        return world

    def reset_world_at(self, env_index: int = None):
        central_agent_pos = torch.cat(
            [
                torch.zeros(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -1 + (3 * self.agent_radius + self.agent_spacing),
                    1 - (3 * self.agent_radius + self.agent_spacing),
                ),
                torch.zeros(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -1 + (3 * self.agent_radius + self.agent_spacing),
                    -(3 * self.agent_radius + self.agent_spacing)
                    - self.passage_width / 2,
                ),
            ],
            dim=1,
        )
        central_goal_pos = torch.cat(
            [
                torch.zeros(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -1 + (3 * self.agent_radius + self.agent_spacing),
                    1 - (3 * self.agent_radius + self.agent_spacing),
                ),
                torch.zeros(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    (3 * self.agent_radius + self.agent_spacing)
                    + self.passage_width / 2,
                    1 - (3 * self.agent_radius + self.agent_spacing),
                ),
            ],
            dim=1,
        )

        order = torch.randperm(self.n_agents).tolist()
        agents = [self.world.agents[i] for i in order]
        goals = [self.world.landmarks[i] for i in order]
        for i, goal in enumerate(goals):
            if i == self.n_agents - 1:
                goal.set_pos(
                    central_goal_pos,
                    batch_index=env_index,
                )
            else:
                goal.set_pos(
                    central_goal_pos
                    + torch.tensor(
                        [
                            [
                                0.0
                                if i % 2
                                else (
                                    self.agent_spacing
                                    if i == 0
                                    else -self.agent_spacing
                                ),
                                0.0
                                if not i % 2
                                else (
                                    self.agent_spacing
                                    if i == 1
                                    else -self.agent_spacing
                                ),
                            ],
                        ],
                        device=self.world.device,
                    ),
                    batch_index=env_index,
                )
        for i, agent in enumerate(agents):
            if i == self.n_agents - 1:
                agent.set_pos(
                    central_agent_pos,
                    batch_index=env_index,
                )
            else:
                agent.set_pos(
                    central_agent_pos
                    + torch.tensor(
                        [
                            [
                                0.0
                                if i % 2
                                else (
                                    self.agent_spacing
                                    if i == 0
                                    else -self.agent_spacing
                                ),
                                0.0
                                if not i % 2
                                else (
                                    self.agent_spacing
                                    if i == 1
                                    else -self.agent_spacing
                                ),
                            ],
                        ],
                        device=self.world.device,
                    ),
                    batch_index=env_index,
                )
            if env_index is None:
                agent.global_shaping = (
                    torch.linalg.vector_norm(
                        agent.state.pos - agent.goal.state.pos, dim=1
                    )
                    * self.shaping_factor
                )
            else:
                agent.global_shaping[env_index] = (
                    torch.linalg.vector_norm(
                        agent.state.pos[env_index] - agent.goal.state.pos[env_index]
                    )
                    * self.shaping_factor
                )

        order = torch.randperm(len(self.world.landmarks[self.n_agents :])).tolist()
        passages = [self.world.landmarks[self.n_agents :][i] for i in order]
        for i, passage in enumerate(passages):
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

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if self.shared_reward:
            if is_first:
                self.rew = torch.zeros(
                    self.world.batch_dim, device=self.world.device, dtype=torch.float32
                )
                for a in self.world.agents:
                    dist_to_goal = torch.linalg.vector_norm(
                        a.state.pos - a.goal.state.pos, dim=1
                    )
                    agent_shaping = dist_to_goal * self.shaping_factor
                    self.rew += a.global_shaping - agent_shaping
                    a.global_shaping = agent_shaping
        else:
            self.rew = torch.zeros(
                self.world.batch_dim, device=self.world.device, dtype=torch.float32
            )
            dist_to_goal = torch.linalg.vector_norm(
                agent.state.pos - agent.goal.state.pos, dim=1
            )
            agent_shaping = dist_to_goal * self.shaping_factor
            self.rew += agent.global_shaping - agent_shaping
            agent.global_shaping = agent_shaping

        if agent.collide:
            for a in self.world.agents:
                if a != agent:
                    self.rew[self.world.is_overlapping(a, agent)] -= 10
            for landmark in self.world.landmarks[self.n_agents :]:
                if landmark.collide:
                    self.rew[self.world.is_overlapping(agent, landmark)] -= 10

        return self.rew

    def observation(self, agent: Agent):
        # get positions of all entities in this agent's reference frame
        passage_obs = []
        passages = self.world.landmarks[self.n_agents :]
        for passage in passages:
            if not passage.collide:
                passage_obs.append(passage.state.pos - agent.state.pos)
        return torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
                agent.goal.state.pos - agent.state.pos,
                *passage_obs,
            ],
            dim=-1,
        )

    def done(self):
        return torch.all(
            torch.stack(
                [
                    torch.linalg.vector_norm(a.state.pos - a.goal.state.pos, dim=1)
                    <= a.shape.radius / 2
                    for a in self.world.agents
                ],
                dim=1,
            ),
            dim=1,
        )

    def extra_render(self, env_index: int = 0):
        from vmas.simulator import rendering

        geoms = []
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
        return geoms


if __name__ == "__main__":
    render_interactively(
        __file__, control_two_agents=True, n_passages=1, shared_reward=False
    )
