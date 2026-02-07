#  Copyright (c) ProrokLab.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import typing

import torch
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn import SoftmaxAggregation

from vmas import render_interactively
from vmas.simulator.core import Agent, Landmark, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import AGENT_INFO_TYPE, Color, ScenarioUtils

if typing.TYPE_CHECKING:
    pass


def agg_max(x, dim):
    return x.max(dim=dim, keepdim=True)[0]


def agg_min(x, dim):
    return x.min(dim=dim, keepdim=True)[0]


def agg_mean(x, dim):
    return x.mean(dim=dim, keepdim=True)


def agg_sum(x, dim):
    return x.sum(dim=dim, keepdim=True)


def agg_logsumexp(x, dim):
    return torch.logsumexp(x, dim=dim, keepdim=True)


class Square:
    def forward(self, x):
        return (x + 1e-7) ** 2

    def inverse(self, x):
        return (x.abs() + 1e-7).sqrt()


class PowerMeanAggregation(torch.nn.Module):
    def __init__(self, p: float = 1.0, learn: bool = False):
        super().__init__()

        self._init_p = p
        self.learn = learn

        self.p = Parameter(torch.empty(1)) if learn else p
        self.reset_parameters()

    def reset_parameters(self):
        if isinstance(self.p, Tensor):
            self.p.data.fill_(self._init_p)

    def forward(self, x, dim: int = -2) -> Tensor:

        p = self.p + 1e-2
        x = x.pow(p)
        x = x.mean(dim=dim, keepdim=True)
        x = x.pow(1.0 / p)

        return x


def tanh_squash(x, low, high):
    tanh_x = torch.tanh(x)
    scale = (high - low) / 2
    add = (high + low) / 2
    return tanh_x * scale + add


def tanh_unsquash(x, low, high):
    scale = (high - low) / 2
    add = (high + low) / 2
    return torch.atanh((x - add) / scale)


class PowerSumAggregation(torch.nn.Module):
    def __init__(self, t: float, low, high, device, learn: bool = True):
        super().__init__()

        self.low = torch.tensor(low, device=device)
        self.high = torch.tensor(high, device=device)
        self._init_inner_t = tanh_unsquash(t, self.low, self.high)
        if not learn:
            self._init_inner_t.requires_grad_(False)

        self.learn = learn
        self.dist = torch.distributions.Normal(loc=0, scale=1)

        self._inner_t = (
            Parameter(torch.empty(1, device=device)) if learn else self._init_inner_t
        )
        self.reset_parameters()

    def reset_parameters(self):
        if isinstance(self._inner_t, Tensor):
            self._inner_t.data.fill_(self._init_inner_t)

    @property
    def t(self):
        return tanh_squash(self._inner_t, self.low, self.high).clamp(
            min=self.low + 1e-1, max=self.high - 1e-1
        )

    def forward(self, x, dim: int = -2) -> Tensor:
        x = x.pow(self.t)
        x = x.sum(dim=dim, keepdim=True)

        return x


def get_aggregation_function(name, device):
    if name == "softmax":
        return SoftmaxAggregation(t=0, learn=True).to(device)
    elif name == "max":
        return agg_max
    elif name == "mean":
        return agg_mean
    elif name == "min":
        return agg_min
    elif name == "sum":
        return agg_sum
    elif name == "powersum":
        return PowerSumAggregation(t=1, learn=True, low=0.3, high=6, device=device)
    elif name == "powersum_04":
        return PowerSumAggregation(t=0.4, learn=False, low=0.3, high=6, device=device)
    elif name == "powersum_5":
        return PowerSumAggregation(t=5, learn=False, low=0.3, high=6, device=device)
    else:
        raise AssertionError


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.plot_grid = False
        self.n_agents = kwargs.pop("n_agents", 2)
        self.n_adversaries = kwargs.pop("n_adversaries", 0)

        self.n_escapers = kwargs.pop("n_escapers", 2)

        # Obstacles
        self.n_obstacles = kwargs.pop("n_obstacles", 2)
        self.obstacle_radius = kwargs.pop("obstacle_radius", 0.1)

        # World dimensions
        self.bound = kwargs.pop("bound", 1.0)

        self.chaser_radius = kwargs.pop("chaser_radius", 0.075)
        self.escaper_radius = kwargs.pop("escaper_radius", 0.05)

        self.use_lidar = kwargs.pop("use_lidar", False)
        self.lidar_range = kwargs.pop("lidar_range", 0.35)

        # Agent dynamics from simple_tag
        self.chaser_speed = kwargs.pop("chaser_speed", 1.0)
        self.chaser_u_multiplier = kwargs.pop("chaser_u_multiplier", 3.0)

        # Escaper AI parameters
        self.escaper_speed = kwargs.pop("escaper_speed", 1.8)
        self.escaper_activation_range = kwargs.pop("escaper_activation_range", 1.5)
        self.wall_avoidance_margin = kwargs.pop("wall_avoidance_margin", 0.5)
        self.wall_repulsion_strength = kwargs.pop("wall_repulsion_strength", 1.0)
        self.escaper_noise = kwargs.pop("escaper_noise", 0.00)
        self.escaper_repulsion_strength = kwargs.pop("escaper_repulsion_strength", 0.5)
        self.escaper_center_attraction_strength = kwargs.pop(
            "escaper_center_attraction_strength", 2.5
        )
        self.obstacle_repulsion_strength = kwargs.pop(
            "obstacle_repulsion_strength", 5.0
        )
        self.obstacle_avoidance_range = kwargs.pop("obstacle_avoidance_range", 0.5)

        # Smoothing parameters
        self.velocity_smoothing = kwargs.pop(
            "velocity_smoothing", 0.3
        )  # Momentum coefficient
        self.max_acceleration = kwargs.pop(
            "max_acceleration", 2.5
        )  # Limit acceleration for smoothness

        # Reward parameters
        self.scaling_coef_1 = kwargs.pop("scaling_coef_1", 0.1)
        self.scaling_coef_2 = kwargs.pop("scaling_coef_2", 1.0)
        self.gen_agg_type_inner = kwargs.pop("gen_agg_type_inner", "max")
        self.gen_agg_type_outer = kwargs.pop("gen_agg_type_outer", "min")

        self.inner_agg = get_aggregation_function(self.gen_agg_type_inner, device)
        self.outer_agg = get_aggregation_function(self.gen_agg_type_outer, device)

        ScenarioUtils.check_kwargs_consumed(kwargs)

        # Make world with smoother physics
        world = World(
            batch_dim,
            device,
            dt=0.05,  # Timestep size
            substeps=2,  # More substeps for stability
            drag=0.45,  # Increased drag for smoother movement (default: 0.25)
            linear_friction=0.0,  # Add some linear friction (default: 0.0)
            x_semidim=self.bound,
            y_semidim=self.bound,
        )

        # To store distances from previous step for progress reward
        self.previous_distances = torch.zeros(
            batch_dim, self.n_agents, self.n_escapers, device=device
        )

        self.blue_agents = []
        # Add agents
        for i in range(self.n_agents):
            agent = Agent(
                name=f"agent_{i}",
                collide=self.use_lidar,
                color=Color.BLUE,
                shape=Sphere(radius=self.chaser_radius),
                render_action=True,
                max_speed=self.chaser_speed,
                u_multiplier=self.chaser_u_multiplier,
                sensors=(
                    [
                        Lidar(
                            world,
                            n_rays=12,
                            max_range=self.lidar_range,
                            entity_filter=lambda e: isinstance(e, Agent),
                        ),
                    ]
                    if self.use_lidar
                    else None
                ),
            )
            # Initialize action counter for each agent
            agent.action_count = torch.zeros(batch_dim, device=device, dtype=torch.long)
            self.blue_agents.append(agent)
            world.add_agent(agent)

        self.escapers = []
        for i in range(self.n_escapers):
            # Add goals
            escaper = Landmark(
                name=f"goal_{i}",
                collide=False,
                movable=True,  # Make landmarks movable
                color=Color.RED,
                shape=Sphere(radius=self.escaper_radius),
            )
            self.escapers.append(escaper)
            world.add_landmark(escaper)

        # Add obstacles
        self.obstacles = []
        for i in range(self.n_obstacles):
            obstacle = Landmark(
                name=f"obstacle_{i}",
                collide=True,
                movable=False,
                color=Color.BLACK,
                shape=Sphere(radius=self.obstacle_radius),
            )
            self.obstacles.append(obstacle)
            world.add_landmark(obstacle)

        # Initialize previous velocities for smoothing
        self.escaper_prev_velocities = [
            torch.zeros(batch_dim, 2, device=device) for _ in range(self.n_escapers)
        ]

        return world

    def post_step(self):
        for agent in self.world.agents:
            # Lazy initialization of agent properties
            if not hasattr(agent, "is_capturing"):
                agent.is_capturing = torch.zeros(
                    self.world.batch_dim, device=self.world.device, dtype=torch.bool
                )
                agent.capture_pos = torch.zeros_like(agent.state.pos)
                agent.capture_pos_stored = torch.zeros(
                    self.world.batch_dim, device=self.world.device, dtype=torch.bool
                )
                agent.prev_pos = torch.zeros_like(agent.state.pos)

                # Debug: Check for large movements (potential teleportation)
                # if hasattr(agent, "prev_pos") and agent in self.blue_agents:
                #     movement = torch.linalg.vector_norm(agent.state.pos - agent.prev_pos, dim=-1)
                #     large_movement = movement > (self.max_agent_velocity * 2)  # Threshold for suspicious movement
                #     if torch.any(large_movement):
                #         print(f"WARNING: Large movement detected for {agent.name}")
                #         print(f"  Movement distance: {movement[large_movement]}")
                #         print(f"  Max allowed velocity: {self.max_agent_velocity}")
                #         print(f"  Previous pos: {agent.prev_pos[large_movement]}")
                #         print(f"  Current pos: {agent.state.pos[large_movement]}")
                #         print(f"  Current velocity: {agent.state.vel[large_movement]}")

                # Store current position for next step
                agent.prev_pos = agent.state.pos.clone().detach()

            if hasattr(agent, "is_capturing"):
                # If agent is capturing, force it back to its capture position
                # This happens after the physics step, overriding any movement
                agent.state.pos = torch.where(
                    agent.is_capturing.unsqueeze(-1),
                    agent.capture_pos,
                    agent.state.pos,
                )

                # Also zero out velocity to prevent drift in the next step
                agent.state.vel[agent.is_capturing] = 0.0

        # Update distances for the next step's reward calculation
        if hasattr(self, "current_distances"):
            self.previous_distances = self.current_distances.clone().detach()

    def parameters(self) -> typing.List:
        params = []
        if hasattr(self.inner_agg, "parameters"):
            params += self.inner_agg.parameters()
        if hasattr(self.outer_agg, "parameters"):
            params += self.outer_agg.parameters()
        return params

    def to_log(self) -> typing.Dict:
        result = {}
        if hasattr(self.inner_agg, "t"):
            result["inner_agg_t"] = self.inner_agg.t.item()
        if hasattr(self.outer_agg, "t"):
            result["outer_agg_t"] = self.outer_agg.t.item()
        return result

    def reset_world_at(self, env_index: int = None):
        # Reset action counters for agents
        for agent in self.blue_agents:
            if env_index is None:
                agent.action_count.fill_(0)
            else:
                agent.action_count[env_index] = 0

        # Spawn all entities randomly
        for entity in self.world.agents + self.world.landmarks:
            entity.set_pos(
                torch.zeros(
                    (
                        (1, self.world.dim_p)
                        if env_index is not None
                        else (self.world.batch_dim, self.world.dim_p)
                    ),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -self.bound,
                    self.bound,
                ),
                batch_index=env_index,
            )

        if env_index is None:
            self.escaper_distances = self._get_distance_to_escapers(env_index)
        else:
            self.escaper_distances[env_index] = self._get_distance_to_escapers(
                env_index
            )

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            self._update_escapers_velocity()
            self.current_distances = self._get_distance_to_escapers()

        agent_idx = self.world.agents.index(agent)
        distances_for_agent = self.current_distances[:, agent_idx, :]

        # Shaping reward: negative distance to closest escaper
        shaping_reward = -torch.min(distances_for_agent, dim=-1)[0]

        # Capture reward
        if is_first:
            capture_indicators = self.current_distances < (
                self.chaser_radius + self.escaper_radius
            )
            inner_agg_result = self.inner_agg(
                capture_indicators.float(), dim=-2
            ).squeeze(-2)
            self.outer_agg_result = self.outer_agg(inner_agg_result, dim=-1).squeeze(-1)

            # # Debug: Print when the aggregator reward is positive
            # if torch.any(self.outer_agg_result > 0):
            #     print(f"POSITIVE AGGREGATOR REWARD: {self.outer_agg_result}")

        # Collision penalty for hitting obstacles
        collision_reward = torch.zeros_like(shaping_reward)
        if self.n_obstacles > 0:
            for obstacle in self.obstacles:
                distance_to_obstacle = torch.linalg.vector_norm(
                    agent.state.pos - obstacle.state.pos, dim=-1
                )
                collision_occurred = distance_to_obstacle < (
                    self.chaser_radius + self.obstacle_radius
                )
                collision_reward -= (
                    collision_occurred.float() * 1.0
                )  # -1.0 penalty per collision

        total_reward = (
            self.scaling_coef_1 * shaping_reward
            + self.scaling_coef_2 * self.outer_agg_result
            + self.scaling_coef_1
            * collision_reward  # Scale collision penalty by scaling_coef_1
        )

        # Per-agent capture detection for freezing mechanism
        min_dist_to_escaper = torch.min(distances_for_agent, dim=-1)[0]
        currently_capturing = min_dist_to_escaper < (
            self.chaser_radius + self.escaper_radius
        )

        # Store position only on the very first frame the capture happens
        just_started_capturing = currently_capturing & ~agent.capture_pos_stored

        # Only update capture_pos if we haven't stored it yet
        agent.capture_pos = torch.where(
            just_started_capturing.unsqueeze(-1),
            agent.state.pos.detach(),
            agent.capture_pos,
        )

        # Update flags
        agent.capture_pos_stored = torch.where(
            just_started_capturing, True, agent.capture_pos_stored
        ).detach()
        agent.is_capturing = currently_capturing.detach()

        # Reset stored flag when no longer capturing
        agent.capture_pos_stored = torch.where(
            ~currently_capturing, False, agent.capture_pos_stored
        ).detach()

        return total_reward

    def _get_distance_to_escapers(self, env_index: typing.Optional[int] = None):
        if env_index is None:
            distances = torch.zeros(
                self.world.batch_dim,
                self.n_agents,
                self.n_escapers,
                device=self.world.device,
            )
            for i, agent in enumerate(self.world.agents):
                for j, escaper in enumerate(self.escapers):
                    distances[:, i, j] = torch.linalg.vector_norm(
                        agent.state.pos - escaper.state.pos,
                        dim=-1,
                    )
        else:
            distances = torch.zeros(
                self.n_agents, self.n_escapers, device=self.world.device
            )
            for i, agent in enumerate(self.world.agents):
                for j, escaper in enumerate(self.escapers):
                    distances[i, j] = torch.linalg.vector_norm(
                        agent.state.pos[env_index] - escaper.state.pos[env_index],
                        dim=-1,
                    )
        return distances

    def _get_distance_to_escapers_for_agent(self, agent: Agent):
        distances = torch.zeros(
            self.world.batch_dim,
            self.n_escapers,
            device=self.world.device,
        )
        for j, escaper in enumerate(self.escapers):
            distances[:, j] = torch.linalg.vector_norm(
                agent.state.pos - escaper.state.pos,
                dim=-1,
            )
        return distances

    def _update_escapers_velocity(self):
        for i, escaper in enumerate(self.escapers):
            # Smoother fleeing from all nearby agents (like tag_aggregator)
            flee_force = torch.zeros_like(escaper.state.pos)
            total_active_agents = torch.zeros(
                self.world.batch_dim, device=self.world.device
            )

            for agent in self.blue_agents:
                direction = escaper.state.pos - agent.state.pos
                distance = torch.linalg.vector_norm(direction, dim=-1)
                is_active = distance < self.escaper_activation_range
                total_active_agents += is_active.float()

                # Add force inversely proportional to distance for active agents
                force_magnitude = 1.0 / (distance + 1e-8)
                flee_force += (
                    (direction / (distance.unsqueeze(-1) + 1e-8))
                    * force_magnitude.unsqueeze(-1)
                    * is_active.unsqueeze(-1)
                )

            # Inactive if no agents are in range
            is_inactive = (total_active_agents == 0).unsqueeze(-1)

            # Smoother graded wall repulsion (from tag_aggregator)
            wall_avoidance_force = torch.zeros_like(escaper.state.pos)
            for dim_idx in range(self.world.dim_p):  # For x and y
                # Calculate repulsion from positive and negative walls
                dist_to_pos_wall = self.bound - escaper.state.pos[:, dim_idx]
                dist_to_neg_wall = self.bound + escaper.state.pos[:, dim_idx]

                # Repel from positive wall with quadratic falloff
                is_close_pos = dist_to_pos_wall < self.wall_avoidance_margin
                strength_pos = (
                    1.0 - (dist_to_pos_wall / self.wall_avoidance_margin)
                ) ** 2
                force_pos = -strength_pos * is_close_pos
                wall_avoidance_force[:, dim_idx] += force_pos

                # Repel from negative wall with quadratic falloff
                is_close_neg = dist_to_neg_wall < self.wall_avoidance_margin
                strength_neg = (
                    1.0 - (dist_to_neg_wall / self.wall_avoidance_margin)
                ) ** 2
                force_neg = strength_neg * is_close_neg
                wall_avoidance_force[:, dim_idx] += force_neg

            # Repulsion from other escapers (smoother version)
            escaper_repulsion_force = torch.zeros_like(escaper.state.pos)
            for other_escaper in self.escapers:
                if other_escaper != escaper:
                    direction = escaper.state.pos - other_escaper.state.pos
                    distance = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
                    # Smoother force calculation
                    force_magnitude = 1.0 / (distance + 1e-8)
                    escaper_repulsion_force += (
                        direction / (distance + 1e-8)
                    ) * force_magnitude

            # Attraction to the center of the map
            center_attraction_force = (
                -escaper.state.pos * self.escaper_center_attraction_strength
            )

            # Repulsion from obstacles
            obstacle_repulsion_force = torch.zeros_like(escaper.state.pos)
            if self.n_obstacles > 0:
                for obstacle in self.obstacles:
                    direction = escaper.state.pos - obstacle.state.pos
                    distance = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)

                    # Apply repulsion if within avoidance range
                    is_close = distance < self.obstacle_avoidance_range
                    force_magnitude = (self.obstacle_avoidance_range - distance) / (
                        distance + 1e-8
                    )
                    force_magnitude = torch.clamp(
                        force_magnitude, 0.0, 10.0
                    )  # Cap the force

                    repulsion = (
                        (direction / (distance + 1e-8)) * force_magnitude * is_close
                    )
                    obstacle_repulsion_force += repulsion

            # Combine forces to get desired velocity
            desired_vel = (
                flee_force
                + wall_avoidance_force * self.wall_repulsion_strength
                + escaper_repulsion_force * self.escaper_repulsion_strength
                + center_attraction_force
                + obstacle_repulsion_force * self.obstacle_repulsion_strength
            )

            # Add noise for unpredictability
            noise = torch.randn_like(desired_vel) * self.escaper_noise
            desired_vel += noise

            # Normalize and apply speed
            vel_magnitude = torch.linalg.vector_norm(desired_vel, dim=-1, keepdim=True)
            desired_vel = desired_vel / (vel_magnitude + 1e-8) * self.escaper_speed

            # Apply velocity smoothing with momentum
            prev_vel = self.escaper_prev_velocities[i]

            # Calculate acceleration (change in velocity)
            acceleration = desired_vel - prev_vel

            # Limit acceleration for smoother movement
            accel_magnitude = torch.linalg.vector_norm(
                acceleration, dim=-1, keepdim=True
            )
            max_accel = self.max_acceleration * self.world.dt  # Scale by timestep
            acceleration = torch.where(
                accel_magnitude > max_accel,
                acceleration / accel_magnitude * max_accel,
                acceleration,
            )

            # Apply momentum smoothing
            smoothed_vel = prev_vel * self.velocity_smoothing + acceleration

            # Check if escaper is tagged (pause movement if tagged)
            is_tagged = torch.zeros(
                self.world.batch_dim, device=self.world.device, dtype=torch.bool
            )
            for agent in self.blue_agents:
                dist_to_agent = torch.linalg.vector_norm(
                    escaper.state.pos - agent.state.pos, dim=-1
                )
                is_tagged |= dist_to_agent < (self.chaser_radius + self.escaper_radius)

            final_vel = torch.where(
                is_tagged.unsqueeze(-1), torch.zeros_like(smoothed_vel), smoothed_vel
            )
            # Override to be stationary if inactive
            final_vel = torch.where(is_inactive, torch.zeros_like(final_vel), final_vel)

            # Store current velocity for next step
            self.escaper_prev_velocities[i] = final_vel.clone().detach()

            # Apply velocity to escaper
            escaper.state.vel = final_vel.detach()

    def observation(self, agent: Agent):
        escaper_poses = []

        for escaper in self.escapers:
            escaper_poses.append(agent.state.pos - escaper.state.pos)

        return torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
            ]
            + escaper_poses
            + (
                [agent.sensors[0]._max_range - agent.sensors[0].measure()]
                if self.use_lidar
                else []
            ),  # maybe use rel pos directly
            dim=-1,
        )

    def process_action(self, agent: Agent):
        # Increment action counter for this agent
        agent.action_count += 1

        # If this is one of the first 3 actions, ignore it by setting action to zero
        if torch.any(agent.action_count <= 3):
            # Create a mask for environments where action_count <= 3
            ignore_mask = agent.action_count <= 3
            # Set action to zero for those environments
            if agent.action.u is not None:
                agent.action.u = torch.where(
                    ignore_mask.unsqueeze(-1),
                    torch.zeros_like(agent.action.u),
                    agent.action.u,
                )

    def info(self, agent: Agent) -> AGENT_INFO_TYPE:
        return {}


if __name__ == "__main__":
    render_interactively(
        __file__,
        control_two_agents=True,
    )
