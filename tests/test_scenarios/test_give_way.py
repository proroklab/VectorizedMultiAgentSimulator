#  Copyright (c) 2022.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import unittest

import torch

from vmas import make_env


class TestGiveWay(unittest.TestCase):
    def setup_env(self, **kwargs) -> None:
        super().setUp()

        self.continuous_actions = True
        self.n_envs = 15
        self.env = make_env(
            scenario_name="give_way",
            num_envs=self.n_envs,
            device="cpu",
            continuous_actions=self.continuous_actions,
            # Environment specific variables
            **kwargs,
        )
        self.env.seed(0)

    def test_heuristic(self):
        self.setup_env(mirror_passage=False)
        all_done = torch.full((self.n_envs,), False)
        obs = self.env.reset()
        u_range = self.env.agents[0].u_range
        total_rew = torch.zeros((self.n_envs,))
        while not (total_rew > 17).all():
            obs_agent = obs[0]
            if (obs[1][:, :1] < 0).all():
                action_1 = torch.tensor([u_range / 2, -u_range]).repeat(self.n_envs, 1)
            else:
                action_1 = torch.tensor([u_range / 2, u_range]).repeat(self.n_envs, 1)
            action_2 = torch.tensor([-u_range / 3, 0]).repeat(self.n_envs, 1)
            obs, rews, dones, _ = self.env.step([action_1, action_2])
            for rew in rews:
                total_rew += rew
            if dones.any():
                # Done envs should have exactly sum of rewards equal to num_agents
                all_done += dones
                for env_index, done in enumerate(dones):
                    if done:
                        self.env.reset_at(env_index)
