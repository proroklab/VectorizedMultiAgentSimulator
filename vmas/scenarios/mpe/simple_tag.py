#  Copyright (c) 2022.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import torch
from vmas import render_interactively

from vmas.simulator.core import World, Agent, Landmark, Sphere
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        world = World(batch_dim=batch_dim, device=device, x_semidim=1, y_semidim=1)
        # set any world properties first
        num_good_agents = 1
        num_adversaries = 3
        num_agents = num_adversaries + num_good_agents
        num_landmarks = 2

        # Add agents
        for i in range(num_agents):
            adversary = True if i < num_adversaries else False
            agent = Agent(
                name=f"agent {i}",
                collide=True,
                shape=Sphere(radius=0.075 if adversary else 0.05),
                u_multiplier=3.0 if adversary else 4.0,
                max_speed=1.0 if adversary else 1.3,
                color=Color.RED if adversary else Color.GREEN,
                adversary=adversary,
            )
            world.add_agent(agent)
        # Add landmarks
        for i in range(num_landmarks):
            landmark = Landmark(
                name=f"landmark {i}",
                collide=True,
                shape=Sphere(radius=0.2),
                color=Color.BLACK,
            )
            world.add_landmark(landmark)

        return world

    def reset_world_at(self, env_index: int = None):
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
        main_reward = (
            self.adversary_reward(agent)
            if agent.adversary
            else self.agent_reward(agent)
        )
        return main_reward

    def agent_reward(self, agent: Agent):
        # Agents are negatively rewarded if caught by adversaries
        rew = torch.zeros(
            self.world.batch_dim, device=self.world.device, dtype=torch.float32
        )
        shape = False
        adversaries = self.adversaries()
        if (
            shape
        ):  # reward can optionally be shaped (increased reward for increased distance from adversary)
            for adv in adversaries:
                rew += 0.1 * torch.sqrt(
                    torch.sum(torch.square(agent.state.pos - adv.state.pos), dim=-1)
                )
        if agent.collide:
            for a in adversaries:
                rew[self.is_collision(a, agent)] -= 10

        return rew

    def adversary_reward(self, agent: Agent):
        # Adversaries are rewarded for collisions with agents
        rew = torch.zeros(
            self.world.batch_dim, device=self.world.device, dtype=torch.float32
        )
        shape = False
        agents = self.good_agents()
        adversaries = self.adversaries()
        if (
            shape
        ):  # reward can optionally be shaped (decreased reward for increased distance from agents)
            for adv in adversaries:
                rew -= (
                    0.1
                    * torch.min(
                        torch.stack(
                            [
                                torch.sqrt(
                                    torch.sum(
                                        torch.square(a.state.pos - adv.state.pos),
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
                    rew[self.is_collision(ag, adv)] += 10
        return rew

    def observation(self, agent: Agent):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in self.world.landmarks:
            entity_pos.append(entity.state.pos - agent.state.pos)

        other_pos = []
        other_vel = []
        for other in self.world.agents:
            if other is agent:
                continue
            other_pos.append(other.state.pos - agent.state.pos)
            if not other.adversary:
                other_vel.append(other.state.vel)
        return torch.cat(
            [agent.state.vel, agent.state.pos, *entity_pos, *other_pos, *other_vel],
            dim=-1,
        )


if __name__ == "__main__":
    render_interactively(__file__, control_two_agents=True)
