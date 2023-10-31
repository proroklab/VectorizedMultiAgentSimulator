#  Copyright (c) 2022.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
from typing import Dict, Callable, List

import torch
from torch import Tensor
from vmas import render_interactively
from vmas.simulator.core import Agent, Landmark, Sphere, World, Entity
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import Color

class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        n_agents = kwargs.get("n_agents", 4)
        n_obstacles = kwargs.get("n_obstacles", 5)
        self._min_dist_between_entities = kwargs.get("min_dist_between_entities", 0.15)
        self._target_pos_range = kwargs.get("target_pos_range", 1.5)
        self._env_size_range = kwargs.get("env_size_range", 1.0)
        self._target_resampling_p = kwargs.get("target_resampling_p", 0.03)
        self._target_max_speed = kwargs.get("target_max_speed", 0.1)
        self._lidar_rays = kwargs.get("lidar_rays", 12)
        self._lidar_range = kwargs.get("lidar_range", 0.2)
        self._comm_range = kwargs.get("comm_range", 0.2)
        self._agent_dist_penalty = kwargs.get("agent_dist_penalty", 0.2)
        self._obstacle_dist_penalty = kwargs.get("obstacle_dist_penalty", 0.2)
        self._scalarise_rewards = kwargs.get("scalarise_rewards", False)
        self._scalarisation = kwargs.get("scalarisation", lambda n: torch.ones(n,5))
        self._diff_rewards = kwargs.get("diff_rewards", 0.9)

        self._current_scalarisation = torch.zeros(batch_dim, self._scalarisation(1).shape[-1])

        self._last_reward = {}

        # Make world
        world = World(batch_dim, device)
        # Add agents
        goal_entity_filter: Callable[
            [Entity], bool
        ] = lambda e: e.name != "target" and not e.name.startswith("agent")
        for i in range(n_agents):
            # Constraint: all agents have same action range and multiplier
            agent = Agent(
                name=f"agent_{i}",
                collide=True,
                mass=1.5,
                sensors=[
                    Lidar(
                        world,
                        n_rays=self._lidar_rays,
                        max_range=self._lidar_range,
                        entity_filter=goal_entity_filter,
                    )
                ],
            )
            world.add_agent(agent)

        self._target_agent = Agent(
            name="target",
            collide=False,
            mass=2.0,
            shape=Sphere(radius=0.03),
            max_speed=self._target_max_speed,
            color=Color.GREEN,
            action_script=self._target_agent_script,
        )
        world.add_agent(self._target_agent)

        # Add landmarks
        self.obstacles = []
        for i in range(n_obstacles):
            obstacle = Landmark(
                name=f"obstacle_{i}",
                mass=100.0,
                collide=True,
                movable=False,
                shape=Sphere(radius=0.1),
                color=Color.RED,
            )
            world.add_landmark(obstacle)
            self.obstacles.append(obstacle)

        return world

    def _target_agent_script(self, agent: Agent, world: World):
        new_target_pos = torch.empty(
            (agent._target_pos.shape[0], self.world.dim_p),
            device=self.world.device,
            dtype=torch.float32,
        ).uniform_(-self._target_pos_range, self._target_pos_range)

        target_resampling_mask = (
            torch.empty(
                agent._target_pos.shape[0],
                device=self.world.device,
                dtype=torch.float32,
            ).uniform_(0.0, 1.0)
            < self._target_resampling_p
        )
        agent._target_pos[target_resampling_mask] = new_target_pos[
            target_resampling_mask
        ]
        agent.action.u = torch.clamp(
            (agent._target_pos - agent.state.pos) * 2.0, -1.0, 1.0
        )

    def reset_world_at(self, env_index: int = None):
        self._target_agent._target_pos = torch.empty(
            (1, self.world.dim_p)
            if env_index is not None
            else (self.world.batch_dim, self.world.dim_p),
            device=self.world.device,
            dtype=torch.float32,
        ).uniform_(-self._target_pos_range, self._target_pos_range)

        if env_index is None:
            self._current_scalarisation = self._scalarisation(self.world.batch_dim)
        else:
            self._current_scalarisation[env_index] = self._scalarisation(1)[0]

        for agent in self.world.agents:
            self._last_reward[agent] = None

        occupied_positions = []
        for entity in self.obstacles + self.world.agents:
            pos = None
            while True:
                proposed_pos = torch.empty(
                    (1, self.world.dim_p)
                    if env_index is not None
                    else (self.world.batch_dim, self.world.dim_p),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(-self._target_pos_range, self._target_pos_range)
                if pos is None:
                    pos = proposed_pos
                if len(occupied_positions) == 0:
                    break

                overlaps = [
                    torch.linalg.norm(pos - o, dim=1) < self._min_dist_between_entities
                    for o in occupied_positions
                ]
                overlaps = torch.any(torch.stack(overlaps, dim=-1), dim=-1)
                if torch.any(overlaps, dim=0):
                    pos[overlaps] = proposed_pos[overlaps]
                else:
                    break

            occupied_positions.append(pos)
            entity.set_pos(pos, batch_index=env_index)

    def reward(self, agent: Agent):
        # Initial Calculation
        rel_pos = torch.zeros(self.world.batch_dim, len(self.world.agents)-2, 2, device=self.world.device)
        rel_vel = torch.zeros(self.world.batch_dim, len(self.world.agents)-2, 2, device=self.world.device)
        i = 0
        for other_agent in self.world.agents:
            if other_agent != agent and other_agent != self._target_agent:
                rel_pos[:,i,:] = other_agent.state.pos - agent.state.pos
                rel_vel[:,i,:] = other_agent.state.vel - agent.state.vel
                i += 1
        rel_pos_mag = rel_pos.norm(dim=-1)
        rel_vel_mag = rel_vel.norm(dim=-1)
        target_pos = self._target_agent.state.pos - agent.state.pos
        target_pos_mag = target_pos.norm(dim=-1)

        separation = -1 / rel_pos_mag
        separation_reward = separation.sum(dim=-1)

        cohesion = -(rel_pos_mag ** 2) * 3
        cohesion_reward = cohesion.sum(dim=-1)

        alignment = -rel_vel_mag * 5
        alignment_reward = alignment.sum(dim=-1)

        target = -(target_pos_mag ** 2) * 3
        target_reward = target

        # Collisions
        obstacle_reward = torch.zeros(self.world.batch_dim, device=self.world.device)
        agent_collisions = torch.zeros(self.world.batch_dim, device=self.world.device)
        for a in self.world.agents:
            if a != agent and a != self._target_agent:
                delta_p = a.state.pos - agent.state.pos
                distance = torch.linalg.norm(delta_p, dim=1)
                collisions = distance < self._agent_dist_penalty
                agent_collisions += collisions.type(torch.int)
                obstacle_reward[collisions] += -1

        obstacle_collisions = torch.zeros(self.world.batch_dim, device=self.world.device)
        for o in self.obstacles:
            delta_p = o.state.pos - agent.state.pos
            collisions = torch.linalg.norm(delta_p, dim=1) < self._obstacle_dist_penalty
            obstacle_reward[collisions] += -1
            obstacle_collisions += collisions.type(torch.int)

        # print("separation: {sep}\n cohesion: {coh}\n alignment: {ali}\n target: {tar}\n obstacle: {obs}\n".format(
        #     sep=separation_reward, coh=cohesion_reward, ali=alignment_reward, tar=target_reward, obs=obstacle_reward
        # ))

        reward = torch.stack([separation_reward,
                              cohesion_reward,
                              alignment_reward,
                              target_reward,
                              obstacle_reward
                              ], dim=-1)

        curr_reward = reward
        if self._last_reward.get(agent, None) is not None:
            reward = curr_reward - self._diff_rewards * self._last_reward[agent]
            reward[:,-1] = curr_reward[:,-1]
        else:
            reward = torch.zeros_like(reward)
        self._last_reward[agent] = curr_reward

        if self._scalarise_rewards:
            c_s, c_c, c_a, c_t, c_o = self._current_scalarisation.to(reward.device).unbind(dim=1)
            print(reward)
            reward = (
                + c_s * reward[:,0]
                + c_c * reward[:,1]
                + c_a * reward[:,2]
                + c_t * reward[:,3]
                + c_o * reward[:,4]
            )

        assert not torch.any(torch.isnan(reward))
        return reward

    def observation(self, agent: Agent):
        lidar = []
        if self._lidar_rays > 0:
            lidar = [agent.sensors[0]._max_range - agent.sensors[0].measure()]
        obs = torch.cat(
            [
                agent.state.pos - self._target_agent.state.pos,
                agent.state.vel - self._target_agent.state.vel,
                *lidar,
            ],
            dim=-1,
        )
        return obs

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        try:
            info = {
                "scalarisation": self._current_scalarisation,
            }
        # When reset is called before reward()
        except AttributeError:
            info = {}
        return info

    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        from vmas.simulator import rendering

        geoms: List[Geom] = []
        for agent in self.world.agents:
            if agent is self._target_agent:
                continue
            range_circle = rendering.make_circle(self._comm_range, filled=False)
            xform = rendering.Transform()
            xform.set_translation(*agent.state.pos[env_index])
            range_circle.add_attr(xform)
            range_circle.set_color(*Color.BLUE.value)
            geoms.append(range_circle)

        return geoms


class HeuristicPolicy(BaseHeuristicPolicy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.centralized = True

    def compute_action(self, observation: torch.Tensor, u_range: float) -> torch.Tensor:

        # Weights
        c_s = 1. # separation
        c_c = 1. # cohesion
        c_a = 1. # alignment
        c_t = 3. # target
        c_o = 1. # obstacle
        weight_cutoff = 2.

        # Initial calculations
        n_agents = observation.shape[1]
        pos = observation[..., :2]
        vel = observation[..., 2:4]
        target_pos = -observation[..., :2]
        disps = pos.unsqueeze(1) - pos.unsqueeze(2)
        dists = disps.norm(dim=-1)
        vel_disps = vel.unsqueeze(1) - vel.unsqueeze(2)
        vel_dists = vel_disps.norm(dim=-1)
        normalise_max = lambda x, p: x * p / torch.maximum(x.norm(dim=-1)[..., None], torch.tensor(p))
        normalise = lambda x, p: x * p / x.norm(dim=-1)[..., None]
        scaling = lambda x, x_int, y_int: (x_int - x) / ((x_int * x) + (x_int / y_int))

        # Separation
        separation_weight = 1 / dists[...,None] / n_agents
        separation_weight[torch.isinf(separation_weight)] = 0
        separation_weight = normalise_max(separation_weight, weight_cutoff)
        separation_dir = -normalise(disps, 1.)
        separation_dir[torch.isnan(separation_dir)] = 0
        separation_action = torch.sum(separation_dir * separation_weight, dim=2)

        # Cohesion
        cohesion_weight = dists[...,None] / n_agents
        cohesion_weight = normalise_max(cohesion_weight, weight_cutoff)
        cohesion_dir = normalise(disps, 1.)
        cohesion_dir[torch.isnan(cohesion_dir)] = 0
        cohesion_action = torch.sum(cohesion_dir * cohesion_weight, dim=2)

        # Alignment
        alignment_weight = vel_dists[...,None] / n_agents
        alignment_weight = normalise_max(alignment_weight, weight_cutoff)
        alignment_dir = normalise(vel_disps, 1.)
        alignment_dir[torch.isnan(alignment_dir)] = 0
        alignment_action = torch.sum(alignment_dir * alignment_weight, dim=2)

        # Target
        target_weight = (target_pos - pos).norm(dim=-1)[...,None]
        target_weight = normalise_max(target_weight, weight_cutoff)
        target_dir = normalise(target_pos-pos, 1.)
        target_dir[torch.isnan(target_dir)] = 0
        target_action = target_dir * target_weight

        # Move away from other agents and obstacles within visibility range
        lidar_range = 0.2
        lidar = lidar_range - observation[..., 8:]
        # object_visible = torch.any(lidar < lidar_range, dim=-1)
        object_dist, object_dir_index = torch.min(lidar, dim=-1)
        object_dir = object_dir_index / lidar.shape[-1] * 2 * torch.pi
        object_vec = torch.stack([torch.cos(object_dir), torch.sin(object_dir)], dim=-1)
        object_scaling = scaling(object_dist, x_int=lidar_range, y_int=weight_cutoff)
        object_action = -object_vec * object_scaling[..., None]

        action = c_s * separation_action + \
                 c_c * cohesion_action + \
                 c_a * alignment_action + \
                 c_t * target_action + \
                 c_o * object_action

        action = normalise_max(action, 1.)

        return action


if __name__ == "__main__":
    render_interactively("flocking2", n_agents=3, scalarise_rewards=True, scalarisation=lambda n: torch.tensor([0.,1.,0,0,0]).unsqueeze(0).expand(n,-1))