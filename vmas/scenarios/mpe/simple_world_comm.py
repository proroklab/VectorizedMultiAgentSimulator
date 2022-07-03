#  Copyright (c) 2022.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import torch

from vmas.simulator.core import World, Agent, Landmark, Sphere
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        world = World(
            batch_dim=batch_dim, device=device, x_semidim=1, y_semidim=1, dim_c=4
        )
        num_good_agents = kwargs.get("num_good_agents", 2)
        num_adversaries = kwargs.get("num_adversaries", 4)
        num_landmarks = kwargs.get("num_landmarks", 1)
        num_food = kwargs.get("num_food", 2)
        num_forests = kwargs.get("num_forests", 2)
        num_agents = num_good_agents + num_adversaries

        # Add agents
        for i in range(num_agents):
            adversary = True if i < num_adversaries else False
            agent = Agent(
                name=f"agent {i}",
                collide=True,
                shape=Sphere(radius=0.075 if adversary else 0.045),
                u_multiplier=3.0 if adversary else 4.0,
                max_speed=1.0 if adversary else 1.3,
                color=Color.RED if adversary else Color.GREEN,
                adversary=adversary,
                silent=True if i > 0 else False,
            )
            agent.leader = True if i == 0 else False
            world.add_agent(agent)
        # Add landmarks
        for i in range(num_landmarks):
            landmark = Landmark(
                name=f"landmark {i}",
                collide=True,
                shape=Sphere(radius=0.2),
            )
            landmark.boundary = False
            world.add_landmark(landmark)
        world.food = []
        for i in range(num_food):
            landmark = Landmark(
                name=f"food {i}",
                collide=False,
                shape=Sphere(radius=0.03),
            )
            landmark.boundary = False
            world.food.append(landmark)
            world.add_landmark(landmark)
        world.forests = []
        for i in range(num_forests):
            landmark = Landmark(
                name=f"forest {i}",
                collide=False,
                shape=Sphere(radius=0.3),
            )
            landmark.boundary = False
            world.forests.append(landmark)
            world.add_landmark(landmark)

        return world

    def reset_world_at(self, env_index: int = None):
        if env_index is None:
            # random properties for agents
            for i, agent in enumerate(self.world.agents):
                agent.color = (
                    torch.tensor(
                        [0.45, 0.95, 0.45],
                        device=self.world.device,
                        dtype=torch.float32,
                    )
                    if not agent.adversary
                    else torch.tensor(
                        [0.95, 0.45, 0.45],
                        device=self.world.device,
                        dtype=torch.float32,
                    )
                )
                agent.color -= (
                    torch.tensor(
                        [0.3, 0.3, 0.3], device=self.world.device, dtype=torch.float32
                    )
                    if agent.leader
                    else torch.tensor(
                        [0, 0, 0], device=self.world.device, dtype=torch.float32
                    )
                )
                # random properties for landmarks
            for i, landmark in enumerate(self.world.landmarks):
                landmark.color = torch.tensor(
                    [0.25, 0.25, 0.25], device=self.world.device, dtype=torch.float32
                )
            for i, landmark in enumerate(self.world.food):
                landmark.color = torch.tensor(
                    [0.15, 0.15, 0.65], device=self.world.device, dtype=torch.float32
                )
            for i, landmark in enumerate(self.world.forests):
                landmark.color = torch.tensor(
                    [0.6, 0.9, 0.6], device=self.world.device, dtype=torch.float32
                )

        for agent in self.world.agents:
            agent.set_pos(
                2
                * torch.rand(
                    self.world.dim_p, device=self.world.device, dtype=torch.float32
                )
                - 1,
                batch_index=env_index,
            )

        for landmark in self.world.landmarks:
            landmark.set_pos(
                1.8
                * torch.rand(
                    self.world.dim_p, device=self.world.device, dtype=torch.float32
                )
                - 0.9,
                batch_index=env_index,
            )

    def is_collision(self, agent1: Agent, agent2: Agent):
        delta_pos = agent1.state.pos - agent2.state.pos
        dist = torch.sqrt(torch.sum(torch.square(delta_pos), dim=-1))
        dist_min = agent1.shape.radius + agent2.shape.radius
        return dist < dist_min

    # return all agents that are not adversaries
    def good_agents(self):
        return [agent for agent in self.world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self):
        return [agent for agent in self.world.agents if agent.adversary]

    def reward(self, agent: Agent):
        # Agents are rewarded based on minimum agent distance to each landmark
        # boundary_reward = -10 if self.outside_boundary(agent) else 0
        main_reward = (
            self.adversary_reward(agent)
            if agent.adversary
            else self.agent_reward(agent)
        )
        return main_reward

    def agent_reward(self, agent: Agent):
        # Agents are rewarded based on minimum agent distance to each landmark
        rew = rew = torch.zeros(
            self.world.batch_dim, device=self.world.device, dtype=torch.float32
        )
        shape = False
        adversaries = self.adversaries()
        if shape:
            for adv in adversaries:
                rew += 0.1 * torch.sqrt(
                    torch.sum(torch.square(agent.state.pos - adv.state.pos), dim=-1)
                )

        if agent.collide:
            for a in adversaries:
                rew[self.is_collision(a, agent)] -= 5

        for food in self.world.food:
            rew[self.is_collision(agent, food)] += 2

        rew -= (
            0.05
            * torch.min(
                torch.stack(
                    [
                        torch.sqrt(
                            torch.sum(
                                torch.square(food.state.pos - agent.state.pos), dim=-1
                            )
                        )
                        for food in self.world.food
                    ],
                    dim=1,
                ),
                dim=-1,
            )[0]
        )

        return rew

    def adversary_reward(self, agent: Agent):
        # Agents are rewarded based on minimum agent distance to each landmark
        rew = rew = torch.zeros(
            self.world.batch_dim, device=self.world.device, dtype=torch.float32
        )
        shape = True
        agents = self.good_agents()
        adversaries = self.adversaries()
        if shape:
            rew -= (
                0.1
                * torch.min(
                    torch.stack(
                        [
                            torch.sqrt(
                                torch.sum(
                                    torch.square(a.state.pos - a.state.pos),
                                    dim=-1,
                                )
                            )
                            for a in agents
                        ],
                        dim=1,
                    ),
                    dim=-1,
                )[0]
            )
        if agent.collide:
            for ag in agents:
                for adv in adversaries:
                    rew[self.is_collision(ag, adv)] += 5
        return rew

    def observation(self, agent: Agent):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in self.world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.pos - agent.state.pos)

        in_forest = torch.full(
            (self.world.batch_dim, len(self.world.forests)),
            -1,
            device=self.world.device,
        )
        inf = torch.full(
            (self.world.batch_dim, len(self.world.forests)),
            False,
            device=self.world.device,
        )

        for i in range(len(self.world.forests)):
            index = self.is_collision(agent, self.world.forests[i])
            in_forest[index][:, i] = 1
            inf[index][:, i] = True

        food_pos = []
        for entity in self.world.food:
            if not entity.boundary:
                food_pos.append(entity.state.pos - agent.state.pos)

        other_pos = []
        other_vel = []
        for other in self.world.agents:
            if other is agent:
                continue
            # Shape batch x n_forests
            oth_f = torch.stack(
                [
                    self.is_collision(other, self.world.forests[i])
                    for i in range(len(self.world.forests))
                ],
                dim=1,
            )
            # without forest vis
            for i in range(len(self.world.forests)):
                other_info = torch.zeros(
                    self.world.batch_dim,
                    4,
                    device=self.world.device,
                    dtype=torch.float32,
                )
                index = torch.logical_and(inf[:, i], oth_f[:, i])
                other_info[index, :2] = other.state.pos[index] - agent.state.pos[index]
                if not other.adversary:
                    other_info[index, 2:] = other.state.vel[index]
                if agent.leader:
                    other_info[~index, :2] = (
                        other.state.pos[~index] - agent.state.pos[~index]
                    )
                    if not other.adversary:
                        other_info[~index, 2:] = other.state.vel[~index]
                other_pos.append(other_info[:, :2])
                other_vel.append(other_info[:, 2:])

        # to tell the pred when the prey are in the forest
        prey_forest = torch.full(
            (self.world.batch_dim, len(self.good_agents())),
            -1,
            device=self.world.device,
        )
        ga = self.good_agents()
        for i, a in enumerate(ga):
            index = torch.any(
                torch.stack(
                    [self.is_collision(a, f) for f in self.world.forests], dim=1
                ),
                dim=-1,
            )
            prey_forest[index][:, i] = 1

        # to tell leader when pred are in forest
        prey_forest = torch.full(
            (self.world.batch_dim, len(self.world.forests)),
            -1,
            device=self.world.device,
        )
        for i, f in enumerate(self.world.forests):
            index = torch.any(
                torch.stack([self.is_collision(a, f) for a in ga], dim=1), dim=-1
            )
            prey_forest[index, i] = 1

        comm = self.world.agents[0].state.c

        if agent.adversary and not agent.leader:
            return torch.cat(
                [
                    agent.state.vel,
                    agent.state.pos,
                    *entity_pos,
                    *other_pos,
                    *other_vel,
                    in_forest,
                    comm,
                ],
                dim=-1,
            )
        if agent.leader:

            return torch.cat(
                [
                    agent.state.vel,
                    agent.state.pos,
                    *entity_pos,
                    *other_pos,
                    *other_vel,
                    in_forest,
                    comm,
                ],
                dim=-1,
            )

        else:
            return torch.cat(
                [
                    agent.state.vel,
                    agent.state.pos,
                    *entity_pos,
                    *other_pos,
                    *other_vel,
                    in_forest,
                ],
                dim=-1,
            )
