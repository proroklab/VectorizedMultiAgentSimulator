#  Copyright (c) 2022.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import unittest

import torch

from vmas import make_env


class TestWaterfall(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.n_agents = 5

        self.continuous_actions = True
        self.n_envs = 19
        self.env = make_env(
            scenario_name="waterfall",
            num_envs=self.n_envs,
            device="cpu",
            continuous_actions=self.continuous_actions,
            # Environment specific variables
            n_agents=self.n_agents,
        )
        self.env.seed(0)

    def test_heuristic(self):
        obs = self.env.reset()
        rews = None
        for _ in range(200):
            actions = []
            for i in range(self.n_agents):
                obs_agent = obs[i]
                action_agent = torch.clamp(
                    obs_agent[:, -2:],
                    min=-self.env.agents[i].u_range,
                    max=self.env.agents[i].u_range,
                )
                actions.append(action_agent)
            obs, new_rews, _, _ = self.env.step(actions)
            if rews is not None:
                for i in range(self.n_agents):
                    self.assertTrue((new_rews[i] >= rews[i]).all())
                rews = new_rews
