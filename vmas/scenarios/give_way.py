#  Copyright (c) 2022.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import torch

from vmas import render_interactively
from vmas.simulator.core import Agent, World, Landmark, Sphere, Line
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.shared_reward = kwargs.get("shared_reward", True)
        self.dense_reward = kwargs.get("dense_reward", True)

        self.shaping_factor = 100

        # Make world
        world = World(batch_dim, device)
        # Add agents
        blue_agent = Agent(name="blue agent", u_multiplier=0.6)
        blue_goal = Landmark(
            name="blue goal",
            collide=False,
            shape=Sphere(radius=0.015),
            color=Color.BLUE,
        )
        world.add_agent(blue_agent)
        world.add_landmark(blue_goal)

        green_agent = Agent(name="green agent", color=Color.GREEN, u_multiplier=0.6)
        green_goal = Landmark(
            name="green goal",
            collide=False,
            shape=Sphere(radius=0.015),
            color=Color.GREEN,
        )
        world.add_agent(green_agent)
        world.add_landmark(green_goal)

        # Add landmarks
        landmark = Landmark(
            name="floor",
            collide=True,
            shape=Line(length=2),
            color=Color.BLACK,
        )
        world.add_landmark(landmark)

        for i in range(2):
            landmark = Landmark(
                name=f"ceil 1 {i}",
                collide=True,
                shape=Line(length=0.95),
                color=Color.BLACK,
            )
            world.add_landmark(landmark)
        for i in range(2):
            landmark = Landmark(
                name=f"wall {i}",
                collide=True,
                shape=Line(length=0.13),
                color=Color.BLACK,
            )
            world.add_landmark(landmark)
        for i in range(3):
            landmark = Landmark(
                name=f"ceil 2 {i}",
                collide=True,
                shape=Line(length=0.11),
                color=Color.BLACK,
            )
            world.add_landmark(landmark)

        return world

    def reset_world_at(self, env_index: int = None):
        self.world.agents[0].set_pos(
            torch.tensor(
                [-0.85, 0.0],
                dtype=torch.float32,
                device=self.world.device,
            ),
            batch_index=env_index,
        )
        self.world.agents[0].goal = self.world.landmarks[0]
        self.world.landmarks[0].set_pos(
            torch.tensor(
                [0.96, 0.0],
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
                [0.85, 0.0],
                dtype=torch.float32,
                device=self.world.device,
            ),
            batch_index=env_index,
        )
        self.world.agents[1].goal = self.world.landmarks[1]
        self.world.landmarks[1].set_pos(
            torch.tensor(
                [-0.96, 0.0],
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

        # Floor
        self.world.landmarks[2].set_pos(
            torch.tensor(
                [0, -0.06],
                dtype=torch.float32,
                device=self.world.device,
            ),
            batch_index=env_index,
        )

        # Ceiling
        for i, landmark in enumerate(self.world.landmarks[3:5]):
            landmark.set_pos(
                torch.tensor(
                    [-0.525 if i == 0 else 0.525, 0.06],
                    dtype=torch.float32,
                    device=self.world.device,
                ),
                batch_index=env_index,
            )
        # Walls
        for i, landmark in enumerate(self.world.landmarks[5:7]):
            landmark.set_pos(
                torch.tensor(
                    [-1 if i == 0 else 1, 0.0],
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
        # Asymmetric hole
        for i, landmark in enumerate(self.world.landmarks[7:9]):
            landmark.set_pos(
                torch.tensor(
                    [-0.05 if i == 0 else 0.05, 0.11],
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
        self.world.landmarks[-1].set_pos(
            torch.tensor(
                [0, 0.16],
                dtype=torch.float32,
                device=self.world.device,
            ),
            batch_index=env_index,
        )
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

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]
        is_last = agent == self.world.agents[-1]

        blue_agent = self.world.agents[0]
        green_agent = self.world.agents[-1]

        rew = torch.zeros(
            self.world.batch_dim, device=self.world.device, dtype=torch.float32
        )

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

        if self.shared_reward:
            if self.dense_reward:
                rew += self.green_rew + self.blue_rew
            else:
                rew[self.blue_on_goal * ~blue_agent.goal.eaten] += 1
                rew[self.green_on_goal * ~green_agent.goal.eaten] += 1
        else:
            if self.dense_reward:
                if agent == blue_agent:
                    rew += self.blue_rew
                else:
                    rew += self.green_rew
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
        return rew

    def observation(self, agent: Agent):
        return torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
                torch.linalg.vector_norm(
                    agent.goal.state.pos - agent.state.pos, dim=1
                ).unsqueeze(-1),
            ],
            dim=-1,
        )

    def done(self):
        return self._done


if __name__ == "__main__":
    render_interactively("give_way", shared_reward=True, dense_reward=True)
