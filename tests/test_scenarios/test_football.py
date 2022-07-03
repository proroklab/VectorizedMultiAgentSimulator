#  Copyright (c) 2022.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import unittest

import torch
from tqdm import tqdm

from vmas import make_env


class TestFootball(unittest.TestCase):
    def setup_env(self, **kwargs) -> None:
        super().setUp()

        self.continuous_actions = True
        self.n_envs = 32
        self.env = make_env(
            scenario_name="football",
            num_envs=self.n_envs,
            device="cpu",
            continuous_actions=True,
            # Environment specific variables
            **kwargs,
        )
        self.env.seed(0)

    def test_ai_vs_random(self):
        n_agents = 3
        self.setup_env(
            n_red_agents=n_agents,
            n_blue_agents=n_agents,
            ai_red_agents=True,
            ai_blue_agents=False,
            dense_reward_ratio=0,
        )
        all_done = torch.full((self.n_envs,), False)
        obs = self.env.reset()
        total_rew = torch.zeros(self.env.num_envs, n_agents)
        with tqdm(total=self.n_envs) as pbar:
            while not all_done.all():
                pbar.update(all_done.sum().item() - pbar.n)
                actions = []
                for i in range(n_agents):
                    actions.append(torch.rand(self.n_envs, 2))

                obs, rews, dones, _ = self.env.step(actions)
                for i in range(n_agents):
                    total_rew[:, i] += rews[i]
                if dones.any():
                    # Done envs should have exactly sum of rewards equal to num_agents
                    actual_rew = -1 * n_agents
                    self.assertTrue(
                        torch.equal(
                            total_rew[dones].sum(-1).to(torch.long),
                            torch.full((dones.sum(),), actual_rew),
                        )
                    )
                    total_rew[dones] = 0
                    all_done += dones
                    for env_index, done in enumerate(dones):
                        if done:
                            self.env.reset_at(env_index)
