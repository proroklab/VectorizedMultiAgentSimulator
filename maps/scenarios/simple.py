import torch

from maps.core import Agent, World, Landmark, Sphere, Box
from maps.scenario import BaseScenario
from maps.utils import Color


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device):
        world = World(batch_dim, device, damping=0.05)
        # add agents
        for i in range(5):
            agent = Agent(f"agent {i}", collide=True, shape=Sphere(radius=0.04))
            world.add_agent(agent)
        # add landmarks
        for i in range(5):
            landmark = Landmark(
                f"landmark {i}",
                collide=True,
                movable=False,
                shape=Box(),
                mass=5,
                color=Color.RED,
            )
            world.add_landmark(landmark)

        return world

    def reset_world_at(self, index: int = None):
        for i, agent in enumerate(self.world.agents):
            agent.set_pos(
                torch.tensor(
                    [-0.2 + 0.1 * i, 1.0],
                    dtype=torch.float64,
                    device=self.world.device,
                ),
                batch_index=index,
            )
        for i, landmark in enumerate(self.world.landmarks):
            landmark.set_pos(
                torch.tensor(
                    [0.2 if i % 2 else -0.2, 0.6 - 0.3 * i],
                    dtype=torch.float64,
                    device=self.world.device,
                ),
                batch_index=index,
            )
            landmark.set_rot(
                torch.tensor(
                    [torch.pi / 4 if i % 2 else -torch.pi / 4],
                    dtype=torch.float64,
                    device=self.world.device,
                ),
                batch_index=index,
            )

    def reward(self, agent):
        dist2 = torch.sum(
            torch.square(agent.state.pos - self.world.landmarks[0].state.pos), dim=-1
        )
        return -dist2

    def observation(self, agent):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in self.world.landmarks:
            entity_pos.append(entity.state.pos - agent.state.pos)
        return torch.cat([agent.state.vel, *entity_pos], dim=-1)
