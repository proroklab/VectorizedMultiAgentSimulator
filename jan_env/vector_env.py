import time
import gym
from ray.rllib.env.vector_env import VectorEnv
from ray.rllib.utils.typing import (
    EnvActionType,
    EnvConfigDict,
    EnvInfoDict,
    EnvObsType,
    EnvType,
    PartialTrainerConfigDict,
)
from typing import Callable, List, Optional, Tuple

from gym.utils import seeding
import torch
import math
import pygame

from scipy.spatial.transform import Rotation as R

X = 0
Y = 1

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)


def compute_dists_and_bearings(ps_a, ps_b):
    assert ps_a.ndim == 3
    assert ps_b.ndim == 3
    assert ps_a.shape[0] == ps_b.shape[0]
    assert ps_a.shape[2] == ps_b.shape[2]
    rel_pos = ps_a.unsqueeze(2).repeat(1, 1, ps_b.shape[1], 1) - ps_b.unsqueeze(
        1
    ).repeat(1, ps_a.shape[1], 1, 1)
    dists = torch.linalg.norm(rel_pos, dim=3)
    bearings = (torch.atan2(rel_pos[:, :, :, Y], rel_pos[:, :, :, X]) + math.pi) / (
        2 * math.pi
    )
    bearings[bearings >= 1.0] -= 1.0
    return dists, bearings


def compute_lidar(agent_ps, ps_observed, max_range, lidar_n_buckets):
    dists, bearings = compute_dists_and_bearings(agent_ps, ps_observed)
    bearing_indexes = (bearings * lidar_n_buckets).long()
    lidar = torch.zeros(agent_ps.shape[0], agent_ps.shape[1], lidar_n_buckets)
    vis_targets_idxs = (dists > 0) & (dists < max_range)
    for target_i in range(ps_observed.shape[1]):
        dists_to_visible = 1.0 - (
            dists[:, :, target_i][vis_targets_idxs[:, :, target_i]] / max_range
        )
        bearing_sel = bearing_indexes[vis_targets_idxs[:, :, target_i]][:, target_i]
        lidar[vis_targets_idxs[:, :, target_i]] = lidar[
            vis_targets_idxs[:, :, target_i]
        ].scatter_(1, bearing_sel.unsqueeze(1), dists_to_visible.unsqueeze(1))
    return lidar


def compute_target_vector(
    ps: torch.Tensor, target_ps: torch.Tensor, desired_dist_target
):
    """
    R: desired distance to target
    p_t: target position
    p: base position
    q = p - p_t
    U(q, R) = (||q||_2 - R)^2
    d/dq U(q, R) = q * (2 - (2 * R)/||q||_2)
    """
    assert ps.shape == target_ps.shape

    q = target_ps - ps
    return q * (
        2 - (2 * desired_dist_target / q.pow(2).sum(dim=-1).sqrt().unsqueeze(-1))
    )


class CoverageEnv(VectorEnv):
    def __init__(self, config):
        self.cfg = config
        n_agents = self.cfg["n_agents"]
        action_space = gym.spaces.Tuple(
            (
                gym.spaces.Box(
                    low=-float("inf"), high=float("inf"), shape=(2,), dtype=float
                ),
            )
            * n_agents
        )  # velocity world coordinates

        observation_space = gym.spaces.Dict(
            {
                "pos": gym.spaces.Box(-6.0, 6.0, shape=(n_agents, 2), dtype=float),
                "vel": gym.spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(n_agents, 2),
                    dtype=float,
                ),
                "targets": gym.spaces.Box(
                    -6.0, 6.0, shape=(self.cfg["n_targets"], 2), dtype=float
                ),
                "lidar": gym.spaces.Dict(
                    {
                        "robots": gym.spaces.Box(
                            0.0,
                            1.0,
                            shape=(n_agents, self.cfg["obs_vector_size"]),
                            dtype=float,
                        ),
                        "targets": gym.spaces.Box(
                            0.0,
                            1.0,
                            shape=(n_agents, self.cfg["obs_vector_size"]),
                            dtype=float,
                        ),
                        "perimeter": gym.spaces.Box(
                            0.0,
                            1.0,
                            shape=(n_agents, self.cfg["obs_vector_size"]),
                            dtype=float,
                        ),
                    }
                ),
            }
        )

        super().__init__(observation_space, action_space, self.cfg["num_envs"])

        self.device = torch.device(self.cfg["device"])
        self.vec_p_shape = (self.cfg["num_envs"], self.cfg["n_agents"], 2)

        w_min = torch.Tensor(self.cfg["world_perimeter"]).min(dim=0).values
        w_max = torch.Tensor(self.cfg["world_perimeter"]).max(dim=0).values
        self.world_dim = w_max - w_min

        # positions
        self.ps = self.create_state_tensor()
        self.target_ps = torch.zeros(
            (self.cfg["num_envs"], self.cfg["n_targets"], 2), dtype=torch.float32
        ).to(self.device)
        self.perimeter_ps = (
            torch.Tensor(self.cfg["world_perimeter"])
            .unsqueeze(0)
            .repeat(self.cfg["num_envs"], 1, 1)
        ).to(self.device)

        self.target_covered = torch.zeros(
            (self.cfg["num_envs"], self.cfg["n_targets"]), dtype=bool
        ).to(self.device)
        # measured velocities
        self.measured_vs = self.create_state_tensor()
        self.timesteps = torch.zeros(self.cfg["num_envs"], dtype=torch.int).to(
            self.device
        )
        self.vector_reset()

        pygame.init()
        size = (
            (torch.Tensor(self.world_dim) * self.cfg["render_px_per_m"])
            .type(torch.int)
            .tolist()
        )
        self.display = pygame.display.set_mode(size)

    def create_state_tensor(self):
        return torch.zeros(self.vec_p_shape, dtype=torch.float32).to(self.device)

    def compute_dists(self, ps):
        dists = torch.cdist(ps, ps)
        diags = (
            torch.eye(self.cfg["n_agents"]).unsqueeze(0).repeat(len(ps), 1, 1).bool()
        )
        dists[diags] = float("inf")
        return dists

    def rand(self, size, a: float, b: float):
        return (a - b) * torch.rand(size).to(self.device) + b

    def vector_reset(self) -> List[EnvObsType]:
        """Resets all sub-environments.
        Returns:
            obs (List[any]): List of observations from each environment.
        """
        return [self.reset_at(index) for index in range(self.cfg["num_envs"])]

    def sample_random_position(self, shape=(1,)):
        half_dim = self.world_dim / 2
        return torch.stack(
            [
                self.rand(shape, -half_dim[X], half_dim[X]),
                self.rand(shape, -half_dim[Y], half_dim[Y]),
            ],
            dim=-1,
        )

    def place_randomly(self, positions: torch.Tensor, object_radii: torch.Tensor):
        if len(positions) == len(object_radii):
            return positions
        # generate new pos until it is not colliding with other positions
        # if not the
        for _ in range(10):
            proposed_p = self.sample_random_position()[0]
            if len(positions) == 0:
                return self.place_randomly(proposed_p.unsqueeze(0), object_radii)
            ds = torch.linalg.norm(positions - proposed_p, dim=1)
            if (ds > object_radii[: len(ds)] + object_radii[len(ds)]).all():
                new_positions = torch.cat([positions, proposed_p.unsqueeze(0)], dim=0)
                return self.place_randomly(new_positions, object_radii)

        return positions

    def reset_at(self, index: Optional[int] = None) -> EnvObsType:
        """Resets a single environment.
        Args:
            index (Optional[int]): An optional sub-env index to reset.
        Returns:
            obs (obj): Observations from the reset sub environment.
        """

        radii = torch.Tensor(
            [1.0] * self.cfg["n_targets"]
            + [self.cfg["agent_radius"]] * self.cfg["n_agents"]
        )

        ps = torch.Tensor([])
        while len(ps) < len(radii):
            ps = self.place_randomly(ps, radii)

        self.target_ps[index] = ps[: self.cfg["n_targets"]]
        self.ps[index] = ps[self.cfg["n_targets"] :]
        self.measured_vs[index] = torch.zeros(self.cfg["n_agents"], 2)
        self.timesteps[index] = 0
        return self.compute_obs(index)[0]

    def compute_obs(self, index: int) -> List[EnvObsType]:
        own_ps = self.ps[index : index + 1] if index >= 0 else self.ps
        target_ps = self.target_ps[index : index + 1] if index >= 0 else self.target_ps
        perimeter_ps = (
            self.perimeter_ps[index : index + 1] if index >= 0 else self.perimeter_ps
        )
        lidar_agents = compute_lidar(
            own_ps, own_ps, self.cfg["visibility_range"], self.cfg["obs_vector_size"]
        )
        lidar_targets = compute_lidar(
            own_ps, target_ps, self.cfg["visibility_range"], self.cfg["obs_vector_size"]
        )
        lidar_perimeter = compute_lidar(
            own_ps,
            perimeter_ps,
            self.cfg["perimeter_visibility_range"],
            self.cfg["obs_vector_size"],
        )

        def get_obs_at_index(obs_index):
            return {
                "pos": own_ps[obs_index].cpu().tolist(),
                "vel": self.measured_vs[obs_index].cpu().tolist(),
                "targets": target_ps[obs_index].cpu().tolist(),
                "lidar": {
                    "robots": lidar_agents[obs_index].cpu().tolist(),
                    "targets": lidar_targets[obs_index].cpu().tolist(),
                    "perimeter": lidar_perimeter[obs_index].cpu().tolist(),
                },
            }

        if index >= 0:
            return [get_obs_at_index(0)]
        else:
            return [get_obs_at_index(i) for i in range(self.cfg["num_envs"])]

    def vector_step(
        self, actions: List[EnvActionType]
    ) -> Tuple[List[EnvObsType], List[float], List[bool], List[EnvInfoDict]]:
        """Performs a vectorized step on all sub environments using `actions`.
        Args:
            actions (List[any]): List of actions (one for each sub-env).
        Returns:
            obs (List[any]): New observations for each sub-env.
            rewards (List[any]): Reward values for each sub-env.
            dones (List[any]): Done values for each sub-env.
            infos (List[any]): Info values for each sub-env.
        """
        self.timesteps += 1

        assert len(actions) == self.cfg["num_envs"]
        # Step the agents while considering vel and acc constraints
        desired_vs = torch.clip(
            torch.Tensor(actions).to(self.device), -self.cfg["max_v"], self.cfg["max_v"]
        )
        assert not desired_vs.isnan().any()

        desired_as = (desired_vs - self.measured_vs) / self.cfg["dt"]
        possible_as = torch.clip(desired_as, -self.cfg["max_a"], self.cfg["max_a"])
        possible_vs = self.measured_vs + possible_as * self.cfg["dt"]

        previous_ps = self.ps.clone().to(self.device)

        # check if next position collisides with other agents or wall
        # have to update agent step by step to be able to attribute negative rewards to each agent
        rewards = torch.zeros(self.cfg["num_envs"], self.cfg["n_agents"])
        for i in range(self.cfg["n_agents"]):
            next_ps_agent = self.ps.clone()
            next_ps_agent[:, i] += possible_vs[:, i] * self.cfg["dt"]
            agents_ds = self.compute_dists(next_ps_agent)[:, i]
            agents_coll = torch.min(agents_ds, dim=1)[0] <= 2 * self.cfg["agent_radius"]

            d_targets = torch.cdist(
                next_ps_agent[:, i, :].unsqueeze(1), self.target_ps
            ).squeeze(1)
            targets_coll = (
                d_targets <= self.cfg["agent_radius"] + self.cfg["target_radius"]
            )
            collision = targets_coll.any(dim=1) | agents_coll
            # only update pos if there are no collisions
            self.ps[~collision, i] = next_ps_agent[~collision, i]
            # penalty when colliding
            rewards[collision, i] -= 10.0

        dim = torch.Tensor(self.world_dim) / 2
        for d in [X, Y]:
            rewards[self.ps[:, :, d] > dim[d]] -= 5.0
            rewards[self.ps[:, :, d] < -dim[d]] -= 5.0
            self.ps[:, :, d] = torch.clip(self.ps[:, :, d], -dim[d], dim[d])

        self.measured_vs = (self.ps - previous_ps) / self.cfg["dt"]

        dist_agents_targets = torch.cdist(self.ps, self.target_ps)
        agents_covering = dist_agents_targets < self.cfg["agent_target_covering_dist"]
        agents_per_target = agents_covering.sum(dim=1)
        covered_targets = agents_per_target == self.cfg["min_n_agents_per_target"]

        agents_covering_rew = (
            agents_covering
            & covered_targets.unsqueeze(1).repeat(1, self.cfg["n_agents"], 1)
            & ~self.target_covered.unsqueeze(1).repeat(1, self.cfg["n_agents"], 1)
        )
        for target_i in range(self.cfg["n_targets"]):
            # if we don't add this then agents will never be incentivized
            # to actually get to the point of covering a target
            rewards[agents_covering_rew[:, :, target_i]] += 50.0

        # compute combinatorial target vectors from all agents to all targets
        target_vecs = compute_target_vector(
            self.ps.unsqueeze(2).repeat(1, 1, self.cfg["n_targets"], 1),
            self.target_ps.unsqueeze(1).repeat(1, self.cfg["n_agents"], 1, 1),
            self.cfg["agent_target_covering_dist"],
        )

        measured_vs_norm = torch.linalg.norm(self.measured_vs, dim=-1).unsqueeze(-1)
        measured_vs_normed = self.measured_vs / measured_vs_norm
        measured_vs_normed[measured_vs_norm.repeat(1, 1, 2) <= 1e-3] = 0.0

        target_vecs_norm = torch.linalg.norm(target_vecs, dim=-1).unsqueeze(-1)
        target_vecs_normed = target_vecs / target_vecs_norm
        target_vecs_normed[target_vecs_norm.repeat(1, 1, 1, 2) <= 1e-3] = 0.0

        directional_reward = torch.bmm(
            measured_vs_normed.unsqueeze(2)
            .repeat(1, 1, self.cfg["n_targets"], 1)
            .view(-1, 1, 2),
            target_vecs_normed.view(-1, 2, 1),
        ).view(self.cfg["num_envs"], self.cfg["n_agents"], self.cfg["n_targets"])
        assert not directional_reward.isnan().any()

        # multiplier: increase reward for all if multiple agents cover a target
        agent_target_mult = (
            agents_covering.sum(dim=1)
            / max(1.0, self.cfg["min_n_agents_per_target"] - 1.0)
        ) + 1.0

        # normalize with velocity and covering multiplier
        dense_reward = (
            directional_reward
            * measured_vs_norm.repeat(1, 1, self.cfg["n_targets"])
            * agent_target_mult.unsqueeze(1).repeat(1, self.cfg["n_agents"], 1)
        )

        # now mask out the agents that can't see the targets
        dense_reward[~(dist_agents_targets < self.cfg["visibility_range"])] = 0.0
        assert not dense_reward.isnan().any()
        rewards += dense_reward.sum(dim=2)

        rewards /= 100.0

        self.target_covered[covered_targets] = True
        for i in range(self.cfg["num_envs"]):
            if self.target_covered[i].any():
                radii = torch.Tensor(
                    [self.cfg["agent_radius"]] * self.cfg["n_agents"]
                    + [1.0] * self.cfg["n_targets"]
                )

                new_target_ps = self.place_randomly(
                    torch.cat(
                        [self.ps[i], self.target_ps[i][~self.target_covered[i]]], dim=0
                    ),
                    radii,
                )
                if len(new_target_ps) == len(radii):
                    # successfully placed new target pos
                    self.target_ps[i] = new_target_ps[self.cfg["n_agents"] :]
                    self.target_covered[i] = False

        assert not rewards.isnan().any()

        obs = self.compute_obs(-1)
        timeout = self.timesteps >= self.cfg["max_time_steps"]
        dones = (timeout).tolist()
        covered_targets = (
            agents_covering_rew.sum(dim=2) / self.cfg["min_n_agents_per_target"]
        )
        infos = [
            {
                "rewards": {k: r for k, r in enumerate(env_rew)},
                "n_covered_targets": agent_cov,
            }
            for env_rew, agent_cov in zip(rewards.tolist(), covered_targets.tolist())
        ]
        return obs, torch.sum(rewards, dim=1).tolist(), dones, infos

    def get_unwrapped(self) -> List[EnvType]:
        return []

    def seed(self, seed):
        if seed is None:
            seed = 0
        torch.manual_seed(seed)
        return [seed]


class CoverageEnvRender(CoverageEnv):
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }
    reward_range = (-float("inf"), float("inf"))
    spec = None

    def __init__(self, config):
        super().__init__(config)

    def seed(self, seed=None):
        rng = torch.manual_seed(seed)
        initial = rng.initial_seed()
        return [initial]

    def reset(self):
        return self.reset_at(0)

    def step(self, actions):
        vector_actions = self.create_state_tensor()
        vector_actions[0] = torch.Tensor(actions)
        obs, r, done, info = self.vector_step(vector_actions)
        return obs[0], r[0], done[0], info[0]

    def close(self):
        pass

    def render(self, mode="rgb_array"):
        AGENT_COLOR = BLUE
        BACKGROUND_COLOR = WHITE
        TARGET_COLOR = GREEN
        PERIMETER_COLOR = RED

        index = 0

        def point_to_screen(point):
            return [
                int((p * f + world_dim / 2) * self.cfg["render_px_per_m"])
                for p, f, world_dim in zip(point, [-1, 1], self.world_dim)
            ]

        self.display.fill(BACKGROUND_COLOR)
        self.font = pygame.font.SysFont("Arial", 30)

        target_img = pygame.Surface(self.display.get_size(), pygame.SRCALPHA)
        for i, (target_p, target_covered) in enumerate(
            zip(self.target_ps[index], self.target_covered[index])
        ):
            for robot_p in self.ps[index]:
                if (
                    torch.linalg.norm(robot_p - target_p) > self.cfg["visibility_range"]
                ) or target_covered:
                    continue

                pygame.draw.line(
                    target_img,
                    TARGET_COLOR,
                    point_to_screen(robot_p),
                    point_to_screen(target_p),
                    4,
                )

            target_color = TARGET_COLOR
            if target_covered:
                target_color += (30,)
            pygame.draw.circle(
                target_img,
                target_color,
                point_to_screen(target_p),
                self.cfg["target_radius"] * self.cfg["render_px_per_m"],
            )
            pygame.draw.circle(
                target_img,
                target_color,
                point_to_screen(target_p),
                self.cfg["visibility_range"] * self.cfg["render_px_per_m"],
                width=2,
            )
            target_label = self.font.render(f"{i}", False, (0, 0, 0))
            self.display.blit(
                target_label,
                point_to_screen(target_p + self.cfg["target_radius"]),
            )

        self.display.blit(target_img, (0, 0))

        for i, agent_pos in enumerate(self.ps[index]):
            agent_img = pygame.Surface(self.display.get_size(), pygame.SRCALPHA)

            pygame.draw.circle(
                agent_img,
                PERIMETER_COLOR + (40,),
                point_to_screen(agent_pos),
                self.cfg["perimeter_visibility_range"] * self.cfg["render_px_per_m"],
                width=2,
            )
            pygame.draw.circle(
                agent_img,
                AGENT_COLOR + (50,),
                point_to_screen(agent_pos),
                self.cfg["visibility_range"] * self.cfg["render_px_per_m"],
            )
            pygame.draw.circle(
                agent_img,
                AGENT_COLOR,
                point_to_screen(agent_pos),
                self.cfg["agent_radius"] * self.cfg["render_px_per_m"],
            )
            robot_label = self.font.render(f"{i}", False, (0, 0, 0))
            self.display.blit(
                robot_label,
                point_to_screen(agent_pos + self.cfg["agent_radius"]),
            )

            for other_agent_pos in self.ps[index]:
                if (
                    torch.linalg.norm(agent_pos - other_agent_pos)
                    > self.cfg["visibility_range"]
                ):
                    continue

                pygame.draw.line(
                    agent_img,
                    AGENT_COLOR,
                    point_to_screen(agent_pos),
                    point_to_screen(other_agent_pos),
                    4,
                )

            self.display.blit(agent_img, (0, 0))

        perimeter_img = pygame.Surface(self.display.get_size(), pygame.SRCALPHA)
        for perimeter_pos in torch.tensor(self.cfg["world_perimeter"]):
            for i, agent_pos in enumerate(self.ps[index]):
                if (
                    torch.linalg.norm(agent_pos - perimeter_pos)
                    > self.cfg["perimeter_visibility_range"]
                ):
                    continue

                pygame.draw.line(
                    perimeter_img,
                    PERIMETER_COLOR,
                    point_to_screen(agent_pos),
                    point_to_screen(perimeter_pos),
                    4,
                )

            pygame.draw.circle(
                perimeter_img,
                PERIMETER_COLOR,
                point_to_screen(perimeter_pos),
                0.1 * self.cfg["render_px_per_m"],
            )
        self.display.blit(perimeter_img, (0, 0))

        if mode == "human":
            pygame.display.update()
        elif mode == "rgb_array":
            return pygame.surfarray.array3d(self.display)

    def try_render_at(self, index: Optional[int] = None) -> None:
        """Renders a single environment.
        Args:
            index (Optional[int]): An optional sub-env index to render.
        """
        return self.render(mode="rgb_array")


if __name__ == "__main__":
    env = CoverageEnvRender(
        {
            "world_perimeter": [
                [-2.0, -3.0],
                # [0.0, -3.0],
                # [2.0, -3.0],
                # [2.0, 1.0],
                # [2.0, -1.0],
                [2.0, 3.0],
                # [0.0, 3.0],
                # [-2.0, 3.0],
                # [-2.0, 1.0],
                # [-2.0, -1.0],
            ],
            "dt": 0.05,
            "n_agents": 1,
            "n_targets": 1,
            "min_n_agents_per_target": 1,
            "placement_keepout_border": 0.5,
            "border_penalty_range": 0.1,
            "max_time_steps": 1000,
            "grid_px_per_m": 40,
            "agent_radius": 0.2,
            "target_radius": 0.1,
            "desired_dist_target": 0.8,
            "agent_target_covering_dist": 0.9,
            "visibility_range": 1.5,
            "perimeter_visibility_range": 0.0,
            "obs_vector_size": 8,
            "render_px_per_m": 160,
            "reward_mode": "cooperative",
            "render_target_vecs": True,
            "max_v": 0.5,
            "max_a": float("inf"),
            "num_envs": 4,
            "device": "cpu",
        }
    )
    import time

    torch.manual_seed(1)
    env.vector_reset()
    # env.reset()
    returns = torch.zeros((env.cfg["n_agents"]))
    selected_agent = 0
    rew = 0
    while True:

        a = torch.zeros((env.cfg["n_agents"], 2))
        for event in pygame.event.get():
            if event.type == pygame.KEYUP:
                env.reset()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                selected_agent += 1
                if selected_agent >= env.cfg["n_agents"]:
                    selected_agent = 0
            elif event.type == pygame.MOUSEMOTION:
                v = (
                    torch.clip(torch.Tensor([-event.rel[0], event.rel[1]]), -20, 20)
                    / 20
                )
                a[selected_agent] = v

        # env.ps[0, 0, X] = 1.0
        env.render(mode="human")

        obs, r, done, info = env.step(a)
        # print(obs)
        rew += r
        for key, agent_reward in info["rewards"].items():
            returns[key] += agent_reward
        # print(info)
        print(returns)
        # if done:
        #    env.reset()
        #    returns = torch.zeros((env.cfg["n_agents"]))
