#  Copyright (c) 2022.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import torch

from vmas import render_interactively
from vmas.simulator.core import Agent, World, Landmark, Sphere, Line
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color
from vmas.simulator.velocity_controller import VelocityController


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.shared_reward = kwargs.get("shared_reward", True)
        self.dense_reward = kwargs.get("dense_reward", True)
        self.mirror_passage = kwargs.get("mirror_passage", True)
        self.viewer_size = (1600, 700)

        self.shaping_factor = 100

        cotnroller_params = [4, 1.25, 0.001]

        # Make world#
        world = World(batch_dim, device, drag=0.1, substeps=5, collision_force=500)

        self.reference_pos = torch.tensor([1, 1], device=world.device).repeat(
            world.batch_dim, 1
        )
        self.agent_radius = 0.175

        # Add agents
        blue_agent = Agent(
            name="blue agent",
            shape=Sphere(radius=self.agent_radius),
            u_range=1,
            f_range=2,
            mass=2,
            render_action=True,
        )
        blue_agent.controller = VelocityController(
            blue_agent, world, cotnroller_params, "standard"
        )
        blue_goal = Landmark(
            name="blue goal",
            collide=False,
            shape=Sphere(radius=self.agent_radius / 2),
            color=Color.BLUE,
        )
        world.add_agent(blue_agent)
        world.add_landmark(blue_goal)

        green_agent = Agent(
            name="green agent",
            color=Color.GREEN,
            shape=Sphere(radius=self.agent_radius),
            u_range=1,
            f_range=2,
            mass=2,
            render_action=True,
        )
        green_agent.controller = VelocityController(
            green_agent, world, cotnroller_params, "standard"
        )
        green_goal = Landmark(
            name="green goal",
            collide=False,
            shape=Sphere(radius=self.agent_radius / 2),
            color=Color.GREEN,
        )
        world.add_agent(green_agent)
        world.add_landmark(green_goal)

        self.spawn_map(world)

        return world

    def reset_world_at(self, env_index: int = None):
        self.world.agents[0].set_pos(
            torch.tensor(
                [-(self.scenario_length / 2 - self.agent_dist_from_wall), 0.0],
                dtype=torch.float32,
                device=self.world.device,
            ),
            batch_index=env_index,
        )
        self.world.agents[0].controller.reset(env_index)
        self.world.agents[0].goal = self.world.landmarks[0]
        self.world.landmarks[0].set_pos(
            torch.tensor(
                [(self.scenario_length / 2 - self.goal_dist_from_wall), 0.0],
                dtype=torch.float32,
                device=self.world.device,
            ),
            batch_index=env_index,
        )
        if env_index is None:
            self.world.landmarks[0].eaten = torch.full(
                (self.world.batch_dim,), False, device=self.world.device
            )
            self.world.landmarks[0].reset_render()
        else:
            self.world.landmarks[0].eaten[env_index] = False
            self.world.landmarks[0].is_rendering[env_index] = True
        self.world.agents[1].set_pos(
            torch.tensor(
                [self.scenario_length / 2 - self.agent_dist_from_wall, 0.0],
                dtype=torch.float32,
                device=self.world.device,
            ),
            batch_index=env_index,
        )
        self.world.agents[1].controller.reset(env_index)
        self.world.agents[1].goal = self.world.landmarks[1]
        self.world.landmarks[1].set_pos(
            torch.tensor(
                [-(self.scenario_length / 2 - self.goal_dist_from_wall), 0.0],
                dtype=torch.float32,
                device=self.world.device,
            ),
            batch_index=env_index,
        )
        if env_index is None:
            self.world.landmarks[1].eaten = torch.full(
                (self.world.batch_dim,), False, device=self.world.device
            )
            self.world.landmarks[1].reset_render()
        else:
            self.world.landmarks[1].eaten[env_index] = False
            self.world.landmarks[1].is_rendering[env_index] = True
        self.reset_map(env_index)
        for agent in self.world.agents:
            if env_index is None:
                agent.shaping = (
                    torch.linalg.vector_norm(
                        agent.state.pos - agent.goal.state.pos, dim=1
                    )
                    * self.shaping_factor
                )
            else:
                agent.shaping[env_index] = (
                    torch.linalg.vector_norm(
                        agent.state.pos[env_index] - agent.goal.state.pos[env_index]
                    )
                    * self.shaping_factor
                )

    def process_action(self, agent: Agent):
        vel_is_zero = torch.linalg.vector_norm(agent.action.u, dim=1) < 1e-3
        agent.controller.reset(vel_is_zero)
        agent.controller.process_force()

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]
        is_last = agent == self.world.agents[-1]

        blue_agent = self.world.agents[0]
        green_agent = self.world.agents[-1]

        rew = torch.zeros(
            self.world.batch_dim, device=self.world.device, dtype=torch.float32
        )
        self.final_rew = torch.zeros((self.world.batch_dim,), device=self.world.device)

        if is_first:
            self.blue_distance = torch.linalg.vector_norm(
                blue_agent.state.pos - blue_agent.goal.state.pos,
                dim=1,
            )
            self.green_distance = torch.linalg.vector_norm(
                green_agent.state.pos - green_agent.goal.state.pos,
                dim=1,
            )
            self.blue_on_goal = self.blue_distance < blue_agent.goal.shape.radius
            self.green_on_goal = self.green_distance < green_agent.goal.shape.radius

            if self.dense_reward:
                green_shaping = self.green_distance * self.shaping_factor
                self.green_rew = green_agent.shaping - green_shaping
                green_agent.shaping = green_shaping

                blue_shaping = self.blue_distance * self.shaping_factor
                self.blue_rew = blue_agent.shaping - blue_shaping
                blue_agent.shaping = blue_shaping

            self.final_rew[
                (self.blue_on_goal + blue_agent.goal.eaten)
                * (self.green_on_goal + green_agent.goal.eaten)
            ] = 200
        if self.shared_reward:
            if self.dense_reward:
                rew[~blue_agent.goal.eaten] += self.blue_rew[~blue_agent.goal.eaten]
                rew[~green_agent.goal.eaten] += self.green_rew[~green_agent.goal.eaten]
            else:
                rew[self.blue_on_goal * ~blue_agent.goal.eaten] += 1
                rew[self.green_on_goal * ~green_agent.goal.eaten] += 1
        else:
            if self.dense_reward:
                if agent == blue_agent:
                    rew[~blue_agent.goal.eaten] += self.blue_rew[~blue_agent.goal.eaten]
                else:
                    rew[~green_agent.goal.eaten] += self.green_rew[
                        ~green_agent.goal.eaten
                    ]
            else:
                if agent == blue_agent:
                    rew[self.blue_on_goal * ~blue_agent.goal.eaten] += 1
                else:
                    rew[self.green_on_goal * ~green_agent.goal.eaten] += 1

        if is_last:
            blue_agent.goal.eaten[self.blue_on_goal] = True
            green_agent.goal.eaten[self.green_on_goal] = True
            blue_agent.goal.is_rendering[self.blue_on_goal] = False
            green_agent.goal.is_rendering[self.green_on_goal] = False
            self._done = blue_agent.goal.eaten * green_agent.goal.eaten

        return rew + self.final_rew

    def observation(self, agent: Agent):
        return torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
                agent.state.pos,
            ],
            dim=-1,
        )

    def done(self):
        return self._done

    def info(self, agent: Agent):
        return {"final_rew": self.final_rew}

    def spawn_map(self, world: World):

        self.scenario_length = 5
        self.passage_length = self.agent_radius * 2 + 0.05
        self.corridor_width = self.passage_length
        self.small_ceiling_length = (self.scenario_length / 2) - (
            self.passage_length / 2
        )
        self.goal_dist_from_wall = 0.15
        self.agent_dist_from_wall = 0.5

        self.walls = []
        for i in range(2):
            landmark = Landmark(
                name=f"wall {i}",
                collide=True,
                shape=Line(length=self.corridor_width),
                color=Color.BLACK,
            )
            self.walls.append(landmark)
            world.add_landmark(landmark)
        self.small_ceilings_1 = []
        for i in range(2):
            landmark = Landmark(
                name=f"ceil 1 {i}",
                collide=True,
                shape=Line(length=self.small_ceiling_length),
                color=Color.BLACK,
            )
            self.small_ceilings_1.append(landmark)
            world.add_landmark(landmark)
        self.passage_1 = []
        for i in range(3):
            landmark = Landmark(
                name=f"ceil 2 {i}",
                collide=True,
                shape=Line(length=self.passage_length),
                color=Color.BLACK,
            )
            self.passage_1.append(landmark)
            world.add_landmark(landmark)

        if self.mirror_passage:
            self.small_ceilings_2 = []
            for i in range(2):
                landmark = Landmark(
                    name=f"ceil 12 {i}",
                    collide=True,
                    shape=Line(length=self.small_ceiling_length),
                    color=Color.BLACK,
                )
                self.small_ceilings_2.append(landmark)
                world.add_landmark(landmark)
            self.passage_2 = []
            for i in range(3):
                landmark = Landmark(
                    name=f"ceil 22 {i}",
                    collide=True,
                    shape=Line(length=self.passage_length),
                    color=Color.BLACK,
                )
                self.passage_2.append(landmark)
                world.add_landmark(landmark)
        else:
            # Add landmarks
            landmark = Landmark(
                name="floor",
                collide=True,
                shape=Line(length=self.scenario_length),
                color=Color.BLACK,
            )
            self.floor = landmark
            world.add_landmark(landmark)

    def reset_map(self, env_index):
        # Walls
        for i, landmark in enumerate(self.walls):
            landmark.set_pos(
                torch.tensor(
                    [
                        -self.scenario_length / 2
                        if i == 0
                        else self.scenario_length / 2,
                        0.0,
                    ],
                    dtype=torch.float32,
                    device=self.world.device,
                ),
                batch_index=env_index,
            )
            landmark.set_rot(
                torch.tensor(
                    [torch.pi / 2],
                    dtype=torch.float32,
                    device=self.world.device,
                ),
                batch_index=env_index,
            )

        # Ceiling
        small_ceiling_pos = self.small_ceiling_length / 2 - self.scenario_length / 2
        for i, landmark in enumerate(self.small_ceilings_1):
            landmark.set_pos(
                torch.tensor(
                    [
                        -small_ceiling_pos if i == 0 else small_ceiling_pos,
                        self.passage_length / 2,
                    ],
                    dtype=torch.float32,
                    device=self.world.device,
                ),
                batch_index=env_index,
            )

        # Asymmetric hole
        for i, landmark in enumerate(self.passage_1[:-1]):
            landmark.set_pos(
                torch.tensor(
                    [
                        -self.passage_length / 2 if i == 0 else self.passage_length / 2,
                        self.passage_length,
                    ],
                    dtype=torch.float32,
                    device=self.world.device,
                ),
                batch_index=env_index,
            )
            landmark.set_rot(
                torch.tensor(
                    [torch.pi / 2],
                    dtype=torch.float32,
                    device=self.world.device,
                ),
                batch_index=env_index,
            )
        self.passage_1[-1].set_pos(
            torch.tensor(
                [0, self.passage_length * (3 / 2)],
                dtype=torch.float32,
                device=self.world.device,
            ),
            batch_index=env_index,
        )

        if self.mirror_passage:
            # Ceiling
            for i, landmark in enumerate(self.small_ceilings_2):
                landmark.set_pos(
                    torch.tensor(
                        [
                            -small_ceiling_pos if i == 0 else small_ceiling_pos,
                            -self.passage_length / 2,
                        ],
                        dtype=torch.float32,
                        device=self.world.device,
                    ),
                    batch_index=env_index,
                )

            # Asymmetric hole
            for i, landmark in enumerate(self.passage_2[:-1]):
                landmark.set_pos(
                    torch.tensor(
                        [
                            -self.passage_length / 2
                            if i == 0
                            else self.passage_length / 2,
                            -self.passage_length,
                        ],
                        dtype=torch.float32,
                        device=self.world.device,
                    ),
                    batch_index=env_index,
                )
                landmark.set_rot(
                    torch.tensor(
                        [torch.pi / 2],
                        dtype=torch.float32,
                        device=self.world.device,
                    ),
                    batch_index=env_index,
                )
            self.passage_2[-1].set_pos(
                torch.tensor(
                    [0, -self.passage_length * (3 / 2)],
                    dtype=torch.float32,
                    device=self.world.device,
                ),
                batch_index=env_index,
            )
        else:
            # Floor
            self.floor.set_pos(
                torch.tensor(
                    [0, -self.passage_length / 2],
                    dtype=torch.float32,
                    device=self.world.device,
                ),
                batch_index=env_index,
            )


if __name__ == "__main__":
    render_interactively(
        __file__, control_two_agents=True, shared_reward=True, dense_reward=True
    )
