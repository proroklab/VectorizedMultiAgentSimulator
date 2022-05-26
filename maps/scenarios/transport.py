import torch

from maps.simulator.core import Agent, World, Landmark, Sphere, Box
from maps.simulator.scenario import BaseScenario
from maps.simulator.utils import Color


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        n_agents = kwargs.get("n_agents", 4)

        # Make world
        world = World(batch_dim, device)
        # Add agents
        for i in range(n_agents):
            agent = Agent(name=f"agent {i}", shape=Sphere(radius=0.04))
            world.add_agent(agent)
        # Add landmarks
        goal = Landmark(
            name=f"goal",
            collide=False,
            shape=Sphere(radius=0.03),
            color=Color.GREEN,
        )
        world.add_landmark(goal)
        package = Landmark(
            name=f"package",
            collide=True,
            movable=True,
            shape=Box(length=0.3, width=0.3),
            color=Color.RED,
        )
        world.add_landmark(package)

        return world

    def reset_world_at(self, env_index: int = None):
        for i, agent in enumerate(self.world.agents):
            x = -0.85 + 0.15 * (i + 1) if i < 2 else -0.85
            y = -0.85 if i < 2 else -0.85 + 0.15 * (i - 1)
            agent.set_pos(
                torch.tensor(
                    [x, y],
                    dtype=torch.float32,
                    device=self.world.device,
                ),
                batch_index=env_index,
            )

        package = self.world.landmarks[1]
        package.set_pos(
            torch.tensor(
                [-0.6, -0.6],
                dtype=torch.float32,
                device=self.world.device,
            ),
            batch_index=env_index,
        )
        goal = self.world.landmarks[0]
        goal.set_pos(
            torch.tensor(
                [0.8, 0.8],
                dtype=torch.float32,
                device=self.world.device,
            ),
            batch_index=env_index,
        )

    def reward(self, agent: Agent):
        dist2 = torch.sum(
            torch.square(agent.state.pos - self.world.landmarks[-1].state.pos), dim=-1
        )
        return -dist2

    def observation(self, agent: Agent):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in self.world.landmarks:
            entity_pos.append(entity.state.pos - agent.state.pos)
        return torch.cat([agent.state.vel, *entity_pos], dim=-1)
