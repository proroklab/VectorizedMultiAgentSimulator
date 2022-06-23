#  Copyright (c) 2022. Matteo Bettini
#  All rights reserved.
import unittest

import torch

from maps import make_env
from maps.simulator.utils import Y


class TestBalance(unittest.TestCase):
    def setup_env(
        self,
        **kwargs,
    ) -> None:
        super().setUp()
        self.n_agents = kwargs.get("n_agents", 4)

        self.continuous_actions = True
        self.n_envs = 25
        self.env = make_env(
            scenario_name="balance",
            num_envs=self.n_envs,
            device="cpu",
            continuous_actions=self.continuous_actions,
            rllib_wrapped=False,
            # Environment specific variables
            **kwargs,
        )
        self.env.seed(0)

    def test_heuristic(self):
        for n_agents in [2, 5, 6, 10]:
            self.setup_env(n_agents=n_agents, random_package_pos_on_line=False)

            obs = self.env.reset()
            rews = None

            for _ in range(100):
                actions = []
                for i in range(n_agents):
                    obs_agent = obs[i]

                    dist_package_goal = obs_agent[:, 8:10]
                    y_distance_is_0 = dist_package_goal[:, Y] >= 0

                    action_agent = torch.clamp(
                        torch.stack(
                            [
                                torch.zeros(self.n_envs),
                                -dist_package_goal[:, Y]
                                + torch.randn(
                                    self.n_envs,
                                )
                                * 0.3,
                            ],
                            dim=1,
                        ),
                        min=-self.env.agents[i].u_range,
                        max=self.env.agents[i].u_range,
                    )
                    action_agent[:, Y][y_distance_is_0] = 0

                    actions.append(action_agent)

                obs, new_rews, dones, _ = self.env.step(actions)

                if rews is not None:
                    for i in range(self.n_agents):
                        self.assertTrue((new_rews[i] >= rews[i]).all())
                    rews = new_rews
