#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import torch

from vmas import make_env


def test_vectorized_lidar(n_envs=12, n_steps=15):
    def get_obs(env):
        rollout_obs = []
        for _ in range(n_steps):
            obs, _, _, _ = env.step(env.get_random_actions())
            obs = torch.stack(obs, dim=-1)
            rollout_obs.append(obs)
        return torch.stack(rollout_obs, dim=-1)

    env_vec_lidar = make_env(
        scenario="pollock", num_envs=n_envs, seed=0, lidar=True, vectorized_lidar=True
    )
    obs_vec_lidar = get_obs(env_vec_lidar)
    env_non_vec_lidar = make_env(
        scenario="pollock", num_envs=n_envs, seed=0, lidar=True, vectorized_lidar=False
    )
    obs_non_vec_lidar = get_obs(env_non_vec_lidar)

    assert torch.allclose(obs_vec_lidar, obs_non_vec_lidar)
