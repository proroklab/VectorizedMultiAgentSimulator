#  Copyright (c) 2022.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import unittest

import torch

from vmas import make_env


class TestReverseTransport(unittest.TestCase):
    def setup_env(self, **kwargs) -> None:
        super().setUp()
        self.n_agents = kwargs.get("n_agents", 4)
        self.package_width = kwargs.get("package_width", 0.6)
        self.package_length = kwargs.get("package_length", 0.6)
        self.package_mass = kwargs.get("package_mass", 50)

        self.continuous_actions = True
        self.n_envs = 32
        self.env = make_env(
            scenario_name="reverse_transport",
            num_envs=self.n_envs,
            device="cpu",
            continuous_actions=self.continuous_actions,
            # Environment specific variables
            **kwargs,
        )
        self.env.seed(0)

    def test_heuristic(self):
        for n_agents in [4, 5]:
            self.setup_env(n_agents=n_agents)
            obs = self.env.reset()
            all_done = torch.full((self.n_envs,), False)
            rew = torch.full((self.n_envs,), float("-inf"))
            while not all_done.all():
                actions = []
                for i in range(n_agents):
                    obs_agent = obs[i]
                    action_agent = torch.clamp(
                        -obs_agent[:, -2:],
                        min=-self.env.agents[i].u_range,
                        max=self.env.agents[i].u_range,
                    )
                    actions.append(action_agent)
                obs, new_rews, dones, _ = self.env.step(actions)
                rew = new_rews[0]
                if dones.any():
                    all_done += dones
                    rew[dones] = float("-inf")
                    for env_index, done in enumerate(dones):
                        if done:
                            self.env.reset_at(env_index)
