#  Copyright (c) 2022-2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
from typing import Dict, Callable

import torch
from torch import Tensor

from vmas import render_interactively
from vmas.simulator.core import Agent, Landmark, Sphere, World, Entity
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import Color, X, Y, ScenarioUtils


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        n_agents = kwargs.get("n_agents", 4)
        n_obstacles = kwargs.get("n_obstacles", 5)
        self._min_dist_between_entities = kwargs.get("min_dist_between_entities", 0.15)

        # Make world
        world = World(batch_dim, device)
        # Add agents
        goal_entity_filter: Callable[[Entity], bool] = lambda e: e.name != "target"
        for i in range(n_agents):
            # Constraint: all agents have same action range and multiplier
            agent = Agent(
                name=f"agent_{i}",
                collide=True,
                sensors=[
                    Lidar(
                        world,
                        n_rays=12,
                        max_range=0.2,
                        entity_filter=goal_entity_filter,
                    )
                ],
            )
            world.add_agent(agent)

        # Add landmarks
        self.obstacles = []
        for i in range(n_obstacles):
            obstacle = Landmark(
                name=f"obstacle_{i}",
                collide=True,
                movable=False,
                shape=Sphere(radius=0.1),
                color=Color.RED,
            )
            world.add_landmark(obstacle)
            self.obstacles.append(obstacle)

        self._target = Landmark(
            name="target",
            collide=False,
            shape=Sphere(radius=0.03),
            color=Color.GREEN,
        )
        world.add_landmark(self._target)

        self.collision_rew = torch.zeros(batch_dim, device=device)
        self.velocity_rew = self.collision_rew.clone()
        self.separation_rew = self.collision_rew.clone()
        self.cohesion_rew = self.collision_rew.clone()

        return world

    def reset_world_at(self, env_index: int = None):
        ScenarioUtils.spawn_entities_randomly(
            self.obstacles + self.world.agents,
            self.world,
            env_index,
            self._min_dist_between_entities,
            x_bounds=(-1, 1),
            y_bounds=(-1, 1),
        )

    def reward(self, agent: Agent):
        # Avoid collisions with each other
        self.collision_rew[:] = 0
        for a in self.world.agents:
            if a != agent:
                self.collision_rew[self.world.is_overlapping(a, agent)] -= 1.0

        # stay close together (separation)
        agents_rel_pos = [agent.state.pos - a.state.pos for a in self.world.agents]
        agents_rel_dist = torch.linalg.norm(torch.stack(agents_rel_pos, dim=1), dim=2)
        agents_max_dist, _ = torch.max(agents_rel_dist, dim=1)
        self.separation_rew = -agents_max_dist

        # keep moving (reward velocity)
        self.velocity_rew = torch.linalg.norm(agent.state.vel, dim=1)

        # stay close to target (cohesion)
        dist_target = torch.linalg.norm(agent.state.pos - self._target.state.pos, dim=1)
        self.cohesion_rew = -dist_target

        return (
            self.collision_rew
            + self.velocity_rew
            + self.separation_rew
            + self.cohesion_rew
        )

    def observation(self, agent: Agent):
        return torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
                self._target.state.pos,
                agent.sensors[0].measure(),
            ],
            dim=-1,
        )

    def info(self, agent: Agent) -> Dict[str, Tensor]:

        info = {
            "collision_rew": self.collision_rew,
            "velocity_rew": self.velocity_rew,
            "separation_rew": self.separation_rew,
            "cohesion_rew": self.cohesion_rew,
        }

        return info


class HeuristicPolicy(BaseHeuristicPolicy):
    def compute_action(self, observation: torch.Tensor, u_range: float) -> torch.Tensor:
        assert self.continuous_actions

        # First calculate the closest point to a circle of radius circle_radius given the current position
        circle_origin = torch.zeros(1, 2)
        circle_radius = 0.3
        current_pos = observation[:, :2]
        v = current_pos - circle_origin
        closest_point_on_circ = (
            circle_origin + v / torch.linalg.norm(v, dim=1).unsqueeze(1) * circle_radius
        )

        # calculate the normal vector of the vector from the origin of the circle to that closest point
        # on the circle. Adding this scaled normal vector to the other vector gives us a target point we
        # try to reach, thus resulting in a circular motion.
        closest_point_on_circ_normal = torch.stack(
            [closest_point_on_circ[:, Y], -closest_point_on_circ[:, X]], dim=1
        )
        closest_point_on_circ_normal /= torch.linalg.norm(
            closest_point_on_circ_normal, dim=1
        ).unsqueeze(1)
        closest_point_on_circ_normal *= 0.1
        des_pos = closest_point_on_circ + closest_point_on_circ_normal

        # Move away from other agents and obstcles within visibility range
        lidar = observation[:, 6:18]
        object_visible = torch.any(lidar < 0.1, dim=1)
        _, object_dir_index = torch.min(lidar, dim=1)
        object_dir = object_dir_index / lidar.shape[1] * 2 * torch.pi
        object_vec = torch.stack([torch.cos(object_dir), torch.sin(object_dir)], dim=1)
        des_pos_object = current_pos - object_vec * 0.1
        des_pos[object_visible] = des_pos_object[object_visible]

        action = torch.clamp(
            (des_pos - current_pos) * 10,
            min=-u_range,
            max=u_range,
        )

        return action


if __name__ == "__main__":
    render_interactively(__file__, control_two_agents=True)
