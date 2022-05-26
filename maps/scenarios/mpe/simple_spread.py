import torch

from maps.simulator.core import World, Agent, Landmark, Sphere
from maps.simulator.scenario import BaseScenario
from maps.simulator.utils import Color


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        world = World(batch_dim=batch_dim, device=device)
        # set any world properties first
        num_agents = 3
        num_landmarks = num_agents
        # Add agents
        for i in range(num_agents):
            agent = Agent(
                name=f"agent {i}",
                collide=True,
                shape=Sphere(radius=0.15),
                color=Color.BLUE,
            )
            world.add_agent(agent)
        # Add landmarks
        for i in range(num_landmarks):
            landmark = Landmark(
                name=f"landmark {i}",
                collide=False,
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
                2
                * torch.rand(
                    self.world.dim_p, device=self.world.device, dtype=torch.float32
                )
                - 1,
                batch_index=env_index,
            )

    def is_collision(self, agent1: Agent, agent2: Agent):
        delta_pos = agent1.state.pos - agent2.state.pos
        dist = torch.sqrt(torch.sum(torch.square(delta_pos), dim=-1))
        dist_min = agent1.shape.radius + agent2.shape.radius
        return dist < dist_min

    def reward(self, agent: Agent):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = torch.zeros(
            self.world.batch_dim, device=self.world.device, dtype=torch.float32
        )
        for l in self.world.landmarks:
            rew -= torch.min(
                torch.stack(
                    [
                        torch.sqrt(
                            torch.sum(torch.square(a.state.pos - l.state.pos), dim=-1)
                        )
                        for a in self.world.agents
                    ],
                    dim=1,
                ),
                dim=-1,
            )[0]
        if agent.collide:
            for a in self.world.agents:
                rew[self.is_collision(a, agent)] -= 1
        return rew

    def observation(self, agent: Agent):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in self.world.landmarks:  # world.entities:
            entity_pos.append(entity.state.pos - agent.state.pos)
        # communication of all other agents
        other_pos = []
        for other in self.world.agents:
            if other is agent:
                continue
            other_pos.append(other.state.pos - agent.state.pos)
        return torch.cat(
            [agent.state.vel, agent.state.pos, *entity_pos, *other_pos], dim=-1
        )
