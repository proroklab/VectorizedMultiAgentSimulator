import torch

from maps.core import Agent, World, Landmark, Sphere, Box
from maps.scenario import BaseScenario
from maps.utils import Color


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        n_agents = kwargs.get("n_agents", 5)

        # Make world
        world = World(batch_dim, device, dt=0.1, damping=0.05)
        # Add agents
        for i in range(n_agents):
            agent = Agent(name=f"agent {i}", shape=Sphere(radius=0.04))
            world.add_agent(agent)
        # Add landmarks
        for i in range(5):
            landmark = Landmark(
                name=f"landmark {i}",
                collide=True,
                movable=False,
                shape=Box(length=0.3, width=0.1),
                color=Color.RED,
            )
            world.add_landmark(landmark)

        return world

    def reset_world_at(self, env_index: int = None):
        for i, agent in enumerate(self.world.agents):
            agent.set_pos(
                torch.tensor(
                    [-0.2 + 0.1 * i, 1.0],
                    dtype=torch.float64,
                    device=self.world.device,
                ),
                batch_index=env_index,
            )
        for i, landmark in enumerate(self.world.landmarks):
            landmark.set_pos(
                torch.tensor(
                    [0.2 if i % 2 else -0.2, 0.6 - 0.3 * i],
                    dtype=torch.float64,
                    device=self.world.device,
                ),
                batch_index=env_index,
            )
            landmark.set_rot(
                torch.tensor(
                    [torch.pi / 4 if i % 2 else -torch.pi / 4],
                    dtype=torch.float64,
                    device=self.world.device,
                ),
                batch_index=env_index,
            )

    def reward(self, agent: Agent):
        dist2 = torch.sum(
            torch.square(agent.state.pos - self.world.landmarks[0].state.pos), dim=-1
        )
        return -dist2

    def observation(self, agent: Agent):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in self.world.landmarks:
            entity_pos.append(entity.state.pos - agent.state.pos)
        return torch.cat([agent.state.vel, *entity_pos], dim=-1)
