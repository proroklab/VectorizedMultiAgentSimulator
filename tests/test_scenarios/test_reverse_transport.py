#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import pytest
import torch

from vmas import make_env


class TestReverseTransport:
    def setup_env(self, n_envs, **kwargs) -> None:
        self.n_agents = kwargs.get("n_agents", 4)
        self.package_width = kwargs.get("package_width", 0.6)
        self.package_length = kwargs.get("package_length", 0.6)
        self.package_mass = kwargs.get("package_mass", 50)

        self.continuous_actions = True

        self.env = make_env(
            scenario="reverse_transport",
            num_envs=n_envs,
            device="cpu",
            continuous_actions=self.continuous_actions,
            # Environment specific variables
            **kwargs,
        )
        self.env.seed(0)

    @pytest.mark.parametrize("n_agents", [5])
    def test_heuristic(self, n_agents, n_envs=4):
        self.setup_env(n_agents=n_agents, n_envs=n_envs)
        obs = self.env.reset()
        all_done = torch.full((n_envs,), False)

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

            if dones.any():
                all_done += dones

                for env_index, done in enumerate(dones):
                    if done:
                        self.env.reset_at(env_index)
