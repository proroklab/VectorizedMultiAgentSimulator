#  Copyright (c) 2022.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import unittest

import torch

from vmas import make_env


class TestGiveWay(unittest.TestCase):
    def setup_env(self, **kwargs) -> None:
        super().setUp()
        self.shared_reward = kwargs.get("shared_reward", False)

        self.continuous_actions = True
        self.n_envs = 25
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
        self.setup_env(n_agents=1)
        all_done = torch.full((self.n_envs,), False)
        obs = self.env.reset()
        while not all_done.all():
            obs_agent = obs[0]
            if (obs[1][:, :1] < 0).all():
                action_1 = torch.tensor([0.5, -1]).repeat(self.n_envs, 1)
            else:
                action_1 = torch.tensor([0.5, 1]).repeat(self.n_envs, 1)
            action_2 = torch.tensor([-0.4, 0]).repeat(self.n_envs, 1)
            obs, rews, dones, _ = self.env.step([action_1, action_2])
            if dones.any():
                # Done envs should have exactly sum of rewards equal to num_agents
                all_done += dones
                for env_index, done in enumerate(dones):
                    if done:
                        self.env.reset_at(env_index)
