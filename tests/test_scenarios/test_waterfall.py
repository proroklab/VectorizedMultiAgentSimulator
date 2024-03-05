#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import torch

from vmas import make_env


class TestWaterfall:
    def setUp(self, n_envs, n_agents) -> None:
        self.continuous_actions = True

        self.env = make_env(
            scenario="waterfall",
            num_envs=n_envs,
            device="cpu",
            continuous_actions=self.continuous_actions,
            # Environment specific variables
            n_agents=n_agents,
        )
        self.env.seed(0)

    def test_heuristic(self, n_agents=5, n_envs=4, n_steps=50):
        self.setUp(n_envs=n_envs, n_agents=n_agents)
        obs = self.env.reset()
        for _ in range(n_steps):
            actions = []
            for i in range(n_agents):
                obs_agent = obs[i]
                action_agent = torch.clamp(
                    obs_agent[:, -2:],
                    min=-self.env.agents[i].u_range,
                    max=self.env.agents[i].u_range,
                )
                actions.append(action_agent)
            obs, new_rews, _, _ = self.env.step(actions)
