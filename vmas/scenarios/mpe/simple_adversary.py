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
        n_agents = kwargs.get("n_agents", 3)
        n_adversaries = kwargs.get("n_adversaries", 1)
        assert n_agents > n_adversaries

        world = World(
            batch_dim=batch_dim,
            device=device,
        )
        num_adversaries = n_adversaries
        num_landmarks = n_agents - 1

        # Add agents
        for i in range(n_agents):
            adversary = True if i < num_adversaries else False
            agent = Agent(
                name=f"agent {i}",
                collide=False,
                shape=Sphere(radius=0.15),
                color=Color.RED if adversary else Color.BLUE,
                adversary=adversary,
            )
            world.add_agent(agent)
        # Add landmarks
        for i in range(num_landmarks):
            landmark = Landmark(
                name=f"landmark {i}",
                collide=False,
                shape=Sphere(radius=0.08),
                color=Color.BLACK,
            )
            world.add_landmark(landmark)

        return world

    def reset_world_at(self, env_index: int = None):
        # set goal landmark
        if env_index is None:
            goal = self.world.landmarks[
                torch.randint(0, len(self.world.landmarks), (1,)).item()
            ]
            goal.color = Color.GREEN
            for agent in self.world.agents:
                agent.goal = goal

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
                2
                * torch.rand(
                    self.world.dim_p, device=self.world.device, dtype=torch.float32
                )
                - 1,
                batch_index=env_index,
            )

    # return all agents that are not adversaries
    def good_agents(self):
        return [agent for agent in self.world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self):
        return [agent for agent in self.world.agents if agent.adversary]

    def reward(self, agent: Agent):
        # Agents are rewarded based on minimum agent distance to each landmark
        return (
            self.adversary_reward(agent)
            if agent.adversary
            else self.agent_reward(agent)
        )

    def agent_reward(self, agent: Agent):
        # Rewarded based on how close any good agent is to the goal landmark, and how far the adversary is from it
        shaped_reward = True
        shaped_adv_reward = True

        # Calculate negative reward for adversary
        adversary_agents = self.adversaries()
        if shaped_adv_reward:  # distance-based adversary reward
            adv_rew = torch.sum(
                torch.stack(
                    [
                        torch.sqrt(
                            torch.sum(
                                torch.square(a.state.pos - a.goal.state.pos), dim=-1
                            )
                        )
                        for a in adversary_agents
                    ],
                    dim=1,
                ),
                dim=-1,
            )
        else:  # proximity-based adversary reward (binary)
            adv_rew = torch.zeros(
                self.world.batch_dim, device=self.world.device, dtype=torch.float32
            )
            for a in adversary_agents:
                is_too_close = (
                    torch.sqrt(
                        torch.sum(torch.square(a.state.pos - a.goal.state.pos), dim=-1)
                    )
                    < 2 * a.goal.size
                )
                adv_rew[is_too_close] -= 5

        # Calculate positive reward for agents
        good_agents = self.good_agents()
        if shaped_reward:  # distance-based agent reward
            pos_rew = -torch.min(
                torch.stack(
                    [
                        torch.sqrt(
                            torch.sum(
                                torch.square(a.state.pos - a.goal.state.pos), dim=-1
                            )
                        )
                        for a in good_agents
                    ],
                    dim=1,
                ),
                dim=-1,
            )[0]

        else:  # proximity-based agent reward (binary)
            pos_rew = torch.zeros(
                self.world.batch_dim, device=self.world.device, dtype=torch.float32
            )
            is_close_enough = (
                torch.min(
                    torch.stack(
                        [
                            torch.sqrt(
                                torch.sum(
                                    torch.square(a.state.pos - a.goal.state.pos), dim=-1
                                )
                            )
                            for a in good_agents
                        ],
                        dim=1,
                    ),
                    dim=-1,
                )
                < 2 * agent.goal.size
            )
            pos_rew[is_close_enough] += 5
            pos_rew -= torch.min(
                torch.stack(
                    [
                        torch.sqrt(
                            torch.sum(
                                torch.square(a.state.pos - a.goal.state.pos), dim=-1
                            )
                        )
                        for a in good_agents
                    ],
                    dim=1,
                ),
                dim=-1,
            )
        return pos_rew + adv_rew

    def adversary_reward(self, agent: Agent):
        # Rewarded based on proximity to the goal landmark
        shaped_reward = True
        if shaped_reward:  # distance-based reward
            return -torch.sqrt(
                torch.sum(torch.square(agent.state.pos - agent.goal.state.pos), dim=-1)
            )

        else:  # proximity-based reward (binary)
            adv_rew = torch.zeros(
                self.world.batch_dim, device=self.world.device, dtype=torch.float32
            )
            close_enough = (
                torch.sqrt(
                    torch.sum(
                        torch.square(agent.state.pos - agent.goal.state.pos), dim=-1
                    )
                )
                < 2 * agent.goal.size
            )
            adv_rew[close_enough] += 5
            return adv_rew

    def observation(self, agent: Agent):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in self.world.landmarks:
            entity_pos.append(entity.state.pos - agent.state.pos)
        # position of all other agents
        other_pos = []
        for other in self.world.agents:
            if other is agent:
                continue
            other_pos.append(other.state.pos - agent.state.pos)

        if not agent.adversary:
            return torch.cat(
                [agent.goal.state.pos - agent.state.pos, *entity_pos, *other_pos],
                dim=-1,
            )
        else:
            return torch.cat([*entity_pos, *other_pos], dim=-1)


if __name__ == "__main__":
    render_interactively(__file__, control_two_agents=True)
