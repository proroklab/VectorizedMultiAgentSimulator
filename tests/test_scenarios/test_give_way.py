#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import torch

from vmas import make_env


class TestGiveWay:
    def setup_env(self, n_envs, **kwargs) -> None:
        self.continuous_actions = True

        self.env = make_env(
            scenario="give_way",
            num_envs=n_envs,
            device="cpu",
            continuous_actions=self.continuous_actions,
            # Environment specific variables
            **kwargs,
        )
        self.env.seed(0)

    def test_heuristic(self, n_envs=4):
        self.setup_env(mirror_passage=False, n_envs=n_envs)
        all_done = torch.full((n_envs,), False)
        obs = self.env.reset()
        u_range = self.env.agents[0].u_range
        total_rew = torch.zeros((n_envs,))
        while not (total_rew > 17).all():
            obs_agent = obs[0]
            if (obs[1][:, :1] < 0).all():
                action_1 = torch.tensor([u_range / 2, -u_range]).repeat(n_envs, 1)
            else:
                action_1 = torch.tensor([u_range / 2, u_range]).repeat(n_envs, 1)
            action_2 = torch.tensor([-u_range / 3, 0]).repeat(n_envs, 1)
            obs, rews, dones, _ = self.env.step([action_1, action_2])
            for rew in rews:
                total_rew += rew
            if dones.any():
                # Done envs should have exactly sum of rewards equal to num_agents
                all_done += dones
                for env_index, done in enumerate(dones):
                    if done:
                        self.env.reset_at(env_index)
