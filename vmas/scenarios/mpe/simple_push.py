#  Copyright (c) 2022.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import torch
from vmas import render_interactively

from vmas.simulator.core import World, Agent, Landmark
from vmas.simulator.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        world = World(
            batch_dim=batch_dim,
            device=device,
        )

        num_agents = 2
        num_adversaries = 1
        num_landmarks = 2

        # Add agents
        for i in range(num_agents):
            adversary = True if i < num_adversaries else False
            agent = Agent(
                name=f"agent {i}",
                collide=True,
                adversary=adversary,
            )
            world.add_agent(agent)
        # Add landmarks
        for i in range(num_landmarks):
            landmark = Landmark(
                name=f"landmark {i}",
                collide=False,
            )
            world.add_landmark(landmark)

        return world

    def reset_world_at(self, env_index: int = None):
        if env_index is None:
            # set goal landmark
            for i, landmark in enumerate(self.world.landmarks):
                landmark.color = torch.tensor(
                    [0.1, 0.1, 0.1], device=self.world.device, dtype=torch.float32
                )
                landmark.color[i + 1] += 0.8
                landmark.index = i
            # set goal landmark
            goal = self.world.landmarks[
                torch.randint(0, len(self.world.landmarks), (1,)).item()
            ]
            for i, agent in enumerate(self.world.agents):
                agent.color = torch.tensor(
                    [0.25, 0.25, 0.25], device=self.world.device, dtype=torch.float32
                )
                if agent.adversary:
                    agent.color = torch.tensor(
                        [0.75, 0.25, 0.25],
                        device=self.world.device,
                        dtype=torch.float32,
                    )
                else:
                    j = goal.index
                    agent.color[j + 1] += 0.5  # Agent color is similar to its goal
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

    def reward(self, agent: Agent):
        # Agents are rewarded based on minimum agent distance to each landmark
        return (
            self.adversary_reward(agent)
            if agent.adversary
            else self.agent_reward(agent)
        )

    def agent_reward(self, agent: Agent):
        # the distance to the goal
        return -torch.sqrt(
            torch.sum(torch.square(agent.state.pos - agent.goal.state.pos), dim=-1)
        )

    def adversary_reward(self, agent: Agent):
        # keep the nearest good agents away from the goal
        pos_rew = torch.min(
            torch.stack(
                [
                    torch.sqrt(
                        torch.sum(torch.square(a.state.pos - a.goal.state.pos), dim=-1)
                    )
                    for a in self.world.agents
                    if not a.adversary
                ],
                dim=1,
            ),
            dim=-1,
        )[0]
        neg_rew = -torch.sqrt(
            torch.sum(torch.square(agent.goal.state.pos - agent.state.pos), dim=-1)
        )

        return pos_rew + neg_rew

    def observation(self, agent: Agent):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in self.world.landmarks:  # world.entities:
            entity_pos.append(entity.state.pos - agent.state.pos)
        # entity colors
        entity_color = []
        for entity in self.world.landmarks:  # world.entities:
            entity_color.append(entity.color.repeat(self.world.batch_dim, 1))
        # communication of all other agents
        other_pos = []
        for other in self.world.agents:
            if other is agent:
                continue
            other_pos.append(other.state.pos - agent.state.pos)
        if not agent.adversary:
            return torch.cat(
                [
                    agent.state.vel,
                    agent.goal.state.pos - agent.state.pos,
                    agent.color.repeat(self.world.batch_dim, 1),
                    *entity_pos,
                    *entity_color,
                    *other_pos,
                ],
                dim=-1,
            )
        else:
            return torch.cat(
                [
                    agent.state.vel,
                    *entity_pos,
                    *other_pos,
                ],
                dim=-1,
            )


if __name__ == "__main__":
    render_interactively(__file__, control_two_agents=True)
