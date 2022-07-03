"""
Scenario:
1 speaker, 2 listeners (one of which is an adversary). Good agents rewarded for proximity to goal, and distance from
adversary to goal. Adversary is rewarded for its distance to the goal.
"""

#  Copyright (c) 2022.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import torch

from vmas.simulator.core import World, Agent
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        dim_c = kwargs.get("dim_c", 4)
        assert dim_c > 0

        world = World(
            batch_dim=batch_dim,
            device=device,
            dim_c=dim_c,
        )
        # set any world properties first
        num_agents = 3
        num_adversaries = 1

        # Add agents
        for i in range(num_agents):
            adversary = True if i < num_adversaries else False
            speaker = True if i == 2 else False
            agent = Agent(
                name=("eve" if adversary else ("alice" if speaker else "bob")),
                collide=False,
                movable=False,
                color=Color.RED
                if adversary
                else (Color.GREEN if speaker else Color.BLUE),
                adversary=adversary,
                silent=False,
            )
            agent.speaker = speaker
            world.add_agent(agent)

        return world

    def reset_world_at(self, env_index: int = None):
        key = torch.randint(
            0, 2, (self.world.batch_dim, self.world.dim_c), device=self.world.device
        )
        secret = torch.randint(
            0, 2, (self.world.batch_dim, self.world.dim_c), device=self.world.device
        )

        if env_index is None:
            for agent in self.world.agents:
                agent.key = None if not agent.speaker else key
                agent.secret = secret
                agent.set_pos(
                    2
                    * torch.rand(
                        self.world.dim_p, device=self.world.device, dtype=torch.float32
                    )
                    - 1,
                    batch_index=env_index,
                )
        else:
            for agent in self.world.agents:
                if agent.speaker:
                    agent.key[env_index] = key[env_index]
                agent.secret[env_index] = secret[env_index]

    # return all agents that are not adversaries
    def good_listeners(self):
        return [
            agent
            for agent in self.world.agents
            if not agent.adversary and not agent.speaker
        ]

    # return all agents that are not adversaries
    def good_agents(self):
        return [agent for agent in self.world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self):
        return [agent for agent in self.world.agents if agent.adversary]

    def reward(self, agent: Agent):
        return (
            self.adversary_reward(agent)
            if agent.adversary
            else self.agent_reward(agent)
        )

    def agent_reward(self, agent: Agent):
        # Agents rewarded if Bob can reconstruct message, but adversary (Eve) cannot
        good_listeners = self.good_listeners()
        adversaries = self.adversaries()
        good_rew = torch.zeros(
            self.world.batch_dim, device=self.world.device, dtype=torch.float32
        )
        adv_rew = torch.zeros(
            self.world.batch_dim, device=self.world.device, dtype=torch.float32
        )
        for a in good_listeners:
            zero_comms = torch.all(
                a.state.c
                == torch.zeros(
                    self.world.batch_dim,
                    self.world.dim_c,
                    device=self.world.device,
                    dtype=torch.float32,
                ),
                dim=-1,
            )
            good_rew[~zero_comms] -= torch.sum(
                torch.square(a.state.c - agent.secret), dim=-1
            )[~zero_comms]
        for a in adversaries:
            zero_comms = torch.all(
                a.state.c
                == torch.zeros(
                    self.world.batch_dim,
                    self.world.dim_c,
                    device=self.world.device,
                    dtype=torch.float32,
                ),
                dim=-1,
            )
            adv_rew[~zero_comms] += torch.sum(
                torch.square(a.state.c - agent.secret), dim=-1
            )[~zero_comms]
        return adv_rew + good_rew

    def adversary_reward(self, agent: Agent):
        # Adversary (Eve) is rewarded if it can reconstruct original goal
        rew = torch.zeros(
            self.world.batch_dim, device=self.world.device, dtype=torch.float32
        )
        zero_comms = torch.all(
            agent.state.c
            == torch.zeros(
                self.world.batch_dim,
                self.world.dim_c,
                device=self.world.device,
                dtype=torch.float32,
            ),
            dim=-1,
        )
        rew[~zero_comms] -= torch.sum(
            torch.square(agent.state.c - agent.secret), dim=-1
        )[~zero_comms]
        return rew

    def observation(self, agent: Agent):
        # communication of all other agents
        comm = []
        for other in self.world.agents:
            if other is agent or not other.speaker:
                continue
            comm.append(other.state.c)

        key = self.world.agents[2].key
        secret = self.world.agents[0].secret

        # speaker
        if agent.speaker:
            return torch.cat(
                [
                    secret,
                    key,
                ],
                dim=-1,
            )
        # listener
        if not agent.speaker and not agent.adversary:
            return torch.cat([key, *comm], dim=-1)
        # adv
        if not agent.speaker and agent.adversary:
            return torch.cat([*comm], dim=-1)
