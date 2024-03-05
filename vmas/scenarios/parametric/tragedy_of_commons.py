#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import math
from typing import Dict, List

import torch
from vmas import render_interactively
from vmas.simulator.core import Agent, Landmark, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import AGENT_INFO_TYPE, Color, ScenarioUtils, TorchUtils


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.n_agents = kwargs.get("n_agents", 4)

        self.resource_growth_rate = torch.nn.Parameter(
            torch.tensor(
                [kwargs.get("resource_growth_rate", 0.01)],
                device=device,
                dtype=torch.float,
            )
        )
        self.reserve_consumption_rate = torch.nn.Parameter(
            torch.tensor(
                [kwargs.get("reserve_consumption_rate", 0.02)],
                device=device,
                dtype=torch.float,
            )
        )
        self.max_resources = torch.nn.Parameter(
            torch.tensor(
                [kwargs.get("max_resources", 10.0)],
                device=device,
                dtype=torch.float,
            )
        )

        self.initial_resources = kwargs.get("initial_resources", 2)
        self.initial_reserve = kwargs.get("initial_reserve", 1)
        self.agent_radius = kwargs.get("agent_radius", 0.035)
        self.resource_radius = kwargs.get("resource_radius", 0.05)
        self.world_semidim = kwargs.get("world_semidim", 1.0)

        self.eating_reward = kwargs.get("eating_reward", 1.0)
        self.reserve_empty_reward = kwargs.get("reserve_empty_reward", -1.0)
        self.energy_rew_coeff = kwargs.get("energy_rew_coeff", 1.0)

        self.min_distance_between_entities = (
            max(self.agent_radius, self.resource_radius) * 2 + 0.05
        )

        # Make world
        world = World(
            batch_dim,
            device,
            x_semidim=self.world_semidim,
            y_semidim=self.world_semidim,
        )
        # Add agents
        for i in range(self.n_agents):
            # Constraint: all agents have same action range and multiplier
            agent = Agent(
                name=f"agent_{i}",
                collide=False,
                shape=Sphere(radius=self.agent_radius),
            )
            world.add_agent(agent)
        # Add landmarks
        for i in range(self.max_resources.to(torch.int).item()):
            food = Landmark(
                name=f"resource_{i}",
                collide=False,
                shape=Sphere(radius=self.resource_radius),
                color=Color.GREEN,
            )
            world.add_landmark(food)

        self.rew_eating = torch.zeros(world.batch_dim, device=world.device)
        self.rew_running_out_of_food = torch.zeros(world.batch_dim, device=world.device)
        self.rew_energy = torch.zeros(world.batch_dim, device=world.device)
        self.t = 0

        return world

    def parameters(self) -> List:
        return [self.resource_growth_rate, self.reserve_consumption_rate]

    def to_log(self) -> Dict:
        return {}

    def reset_world_at(self, env_index: int = None):
        ScenarioUtils.spawn_entities_randomly(
            self.world.agents,
            self.world,
            env_index,
            self.min_distance_between_entities,
            (-self.world_semidim, 0.0),
            (-self.world_semidim, self.world_semidim),
        )
        ScenarioUtils.spawn_entities_randomly(
            self.world.landmarks,
            self.world,
            env_index,
            self.min_distance_between_entities,
            (0.0, self.world_semidim),
            (-self.world_semidim, self.world_semidim),
        )
        if env_index is None:
            self.current_resources = torch.full(
                (self.world.batch_dim,),
                self.initial_resources,
                device=self.world.device,
                dtype=torch.float,
            )
            self.current_reserve = torch.full(
                (self.world.batch_dim,),
                self.initial_reserve,
                device=self.world.device,
                dtype=torch.float,
            )
        else:
            self.current_resources = TorchUtils.where_from_index(
                env_index, self.initial_resources, self.current_resources
            )
            self.current_reserve = TorchUtils.where_from_index(
                env_index, self.initial_reserve, self.current_reserve
            )

        for i, landmark in enumerate(self.world.landmarks):
            render = i == 0

            if env_index is None:
                landmark._render = torch.full(
                    (self.world.batch_dim,), render, device=self.world.device
                )
            else:
                landmark._render = TorchUtils.where_from_index(
                    env_index, render, landmark._render
                )

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]
        is_last = agent == self.world.agents[-1]

        if is_first:
            if self.world.batch_dim == 1:
                import time

                self.t += 1
                print("Time", self.t)
                time.sleep(0.1)

            self.rew_eating = torch.zeros(
                self.world.batch_dim, device=self.world.device
            )
            self.rew_running_out_of_food = torch.zeros(
                self.world.batch_dim, device=self.world.device
            )
            self.rew_energy = torch.zeros(
                self.world.batch_dim, device=self.world.device
            )

            self.process_resources_consumption()

            for landmark in self.world.landmarks:
                consumed = landmark.anyone_on_food * landmark._render
                consumed_float = consumed.to(torch.float)

                self.current_reserve = self.current_reserve + consumed_float
                self.current_resources = self.current_resources - consumed_float

                # Reward for eating
                self.rew_eating = torch.where(
                    consumed,
                    self.eating_reward,
                    self.rew_eating,
                )

            # Reward for being out of reserve
            self.rew_running_out_of_food = torch.where(
                self.current_reserve == 0,
                self.reserve_empty_reward,
                self.rew_running_out_of_food,
            )

            # Reward for minimizing energy
            self.rew_energy = (
                (
                    -torch.stack(
                        [
                            torch.linalg.vector_norm(a.action.u, dim=-1)
                            / math.sqrt(self.world.dim_p * (a.u_range**2))
                            for a in self.world.agents
                        ],
                        dim=1,
                    ).sum(-1)
                )
                + 1
            ).clamp(max=0) * self.energy_rew_coeff

        if is_last:
            self.evolve_state()

        return self.rew_eating + self.rew_running_out_of_food + self.rew_energy

    def process_resources_consumption(self):
        for landmark in self.world.landmarks:
            landmark.how_many_on_food = torch.stack(
                [
                    torch.linalg.vector_norm(a.state.pos - landmark.state.pos, dim=-1)
                    < a.shape.radius + landmark.shape.radius
                    for a in self.world.agents
                ],
                dim=-1,
            ).sum(-1)
            landmark.anyone_on_food = landmark.how_many_on_food > 0

    def evolve_state(self):
        # Remove eaten resources
        for landmark in self.world.landmarks:
            landmark._render = torch.where(
                landmark.anyone_on_food, False, landmark._render
            )

        # Resources population growth --> Logistic growth model
        self.current_resources = self.current_resources + (
            self.resource_growth_rate
            * self.current_resources
            * (1 - self.current_resources / self.max_resources)
        )

        # Update resources rendering
        rendered_landmarks = torch.stack(
            [landmark._render for landmark in self.world.landmarks], dim=-1
        )
        n_rendered_landmarks_per_env = rendered_landmarks.to(torch.int).sum(-1)
        n_more_landmarks_to_render = torch.clamp(
            self.current_resources.to(torch.int) - n_rendered_landmarks_per_env, 0
        )
        indices = torch.randperm(len(self.world.landmarks))
        for i in indices:
            landmark = self.world.landmarks[i]
            new_render = ~landmark._render * (n_more_landmarks_to_render > 0)

            landmark._render = torch.where(new_render, True, landmark._render)
            n_more_landmarks_to_render = torch.where(
                new_render, n_more_landmarks_to_render - 1, n_more_landmarks_to_render
            )
        assert (n_more_landmarks_to_render == 0).all()

        # Reserve consumption model -> Linear decrease
        self.current_reserve = torch.clamp(
            self.current_reserve - self.reserve_consumption_rate, 0.0
        )

    def observation(self, agent: Agent):
        rel_pos = self.get_deterministic_resource(agent)
        return torch.cat(
            [
                self.current_reserve.unsqueeze(-1),
                self.current_resources.unsqueeze(-1),
                rel_pos,
            ],
            dim=-1,
        )

    def info(self, agent: Agent) -> AGENT_INFO_TYPE:
        return {
            "current_reserve": self.current_reserve,
            "current_resources": self.current_resources,
            "rew_eating": self.rew_eating,
            "rew_running_out_of_food": self.rew_running_out_of_food,
            "rew_energy": self.rew_energy,
        }

    def get_closest_resource(self, agent: Agent):
        landmark_rel_poses = []
        landmark_distances = []
        for landmark in self.world.landmarks:
            landmark_rel_pos = agent.state.pos - landmark.state.pos
            landmark_rel_poses.append(landmark_rel_pos)
            landmark_distance = torch.linalg.vector_norm(landmark_rel_pos, dim=-1)
            landmark_distance = torch.where(
                ~landmark._render, torch.inf, landmark_distance
            )
            landmark_distances.append(landmark_distance)
        landmark_rel_poses = torch.stack(landmark_rel_poses, dim=1)
        landmark_distances = torch.stack(landmark_distances, dim=1)
        min_dist_indices = landmark_distances.min(-1)[1]
        return landmark_rel_poses[torch.arange(self.world.batch_dim), min_dist_indices]

    def get_deterministic_resource(self, agent: Agent):
        landmark_rel_poses = []
        index = torch.zeros(
            self.world.batch_dim, dtype=torch.int, device=self.world.device
        )
        for i, landmark in enumerate(self.world.landmarks):
            landmark_rel_pos = agent.state.pos - landmark.state.pos
            landmark_rel_poses.append(landmark_rel_pos)
            index = torch.where(landmark._render, i, index)

        landmark_rel_poses = torch.stack(landmark_rel_poses, dim=1)

        return landmark_rel_poses[torch.arange(self.world.batch_dim), index]


if __name__ == "__main__":
    render_interactively(__file__, control_two_agents=True)
