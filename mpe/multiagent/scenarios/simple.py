import torch

from mpe.multiagent.core import Agent, World, Landmark, Sphere, Box
from mpe.multiagent.scenario import BaseScenario
from simulator.utils import Color


class Scenario(BaseScenario):
    def make_world(self):
        world = World(32, "cpu", damping=0.03)
        # add agents
        for i in range(5):
            agent = Agent(f"agent {i}", collide=True, shape=Sphere(radius=0.04))
            world.add_agent(agent)
        # add landmarks
        for i in range(5):
            landmark = Landmark(
                f"landmark {i}",
                collide=True,
                movable=True,
                shape=Box(),
                mass=5,
                color=Color.RED,
            )
            world.add_landmark(landmark)

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # # random properties for agents
        # for i, agent in enumerate(world.agents):
        #     agent.color = np.array([0.25, 0.25, 0.25])
        # # random properties for landmarks
        # for i, landmark in enumerate(world.landmarks):
        #     landmark.color = np.array([0.75, 0.75, 0.75])
        # world.landmarks[0].color = np.array([0.75, 0.25, 0.25])
        # set random initial states
        for i, agent in enumerate(world.agents):
            agent.state.pos = torch.tensor([-0.2 + 0.1 * i, 1.0]).repeat(
                world.batch_dim, 1
            )
            agent.state.vel = torch.zeros(world.dim_p).repeat(world.batch_dim, 1)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.pos = torch.tensor(
                [0.2 if i % 2 else -0.2, 0.6 - 0.3 * i]
            ).repeat(world.batch_dim, 1)
            landmark.state.vel = torch.zeros(world.dim_p).repeat(world.batch_dim, 1)
            landmark.state.rot = torch.tensor(
                [torch.pi / 4 if i % 2 else -torch.pi / 4]
            ).repeat(world.batch_dim)

    def reward(self, agent, world):
        dist2 = torch.sum(torch.square(agent.state.pos - world.landmarks[0].state.pos))
        return -dist2

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.pos - agent.state.pos)
        return torch.cat([agent.state.vel, *entity_pos], dim=-1)
