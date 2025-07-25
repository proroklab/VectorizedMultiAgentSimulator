#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import sys

import pytest
import torch
from tqdm import tqdm

from vmas import make_env


class TestFootball:
    def setup_env(self, n_envs, **kwargs) -> None:
        self.continuous_actions = True

        self.env = make_env(
            scenario="football",
            num_envs=n_envs,
            device="cpu",
            continuous_actions=True,
            # Environment specific variables
            **kwargs,
        )
        self.env.seed(0)

    @pytest.mark.skipif(
        sys.platform.startswith("win32"), reason="Test does not work on windows"
    )
    def test_ai_vs_random(self, n_envs=4, n_agents=3, scoring_reward=1):
        self.setup_env(
            n_red_agents=n_agents,
            n_blue_agents=n_agents,
            ai_red_agents=True,
            ai_blue_agents=False,
            dense_reward=False,
            n_envs=n_envs,
            scoring_reward=scoring_reward,
        )
        all_done = torch.full((n_envs,), False)
        obs = self.env.reset()
        total_rew = torch.zeros(self.env.num_envs, n_agents)
        with tqdm(total=n_envs) as pbar:
            while not all_done.all():
                pbar.update(all_done.sum().item() - pbar.n)
                actions = []
                for _ in range(n_agents):
                    actions.append(torch.rand(n_envs, 2))

                obs, rews, dones, _ = self.env.step(actions)
                for i in range(n_agents):
                    total_rew[:, i] += rews[i]
                if dones.any():
                    # Done envs should have exactly sum of rewards equal to num_agents
                    actual_rew = -scoring_reward * n_agents
                    assert torch.equal(
                        total_rew[dones].sum(-1).to(torch.long),
                        torch.full((dones.sum(),), actual_rew, dtype=torch.long),
                    )
                    total_rew[dones] = 0
                    all_done += dones
                    for env_index, done in enumerate(dones):
                        if done:
                            self.env.reset_at(env_index)
