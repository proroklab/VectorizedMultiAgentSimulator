#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import pytest

from vmas import make_env
from vmas.scenarios import wheel


class TestWheel:
    def setup_env(
        self,
        n_envs,
        n_agents,
        **kwargs,
    ) -> None:
        self.desired_velocity = kwargs.get("desired_velocity", 0.1)

        self.continuous_actions = True
        self.n_envs = 15
        self.env = make_env(
            scenario="wheel",
            num_envs=n_envs,
            device="cpu",
            continuous_actions=self.continuous_actions,
            # Environment specific variables
            n_agents=n_agents,
            **kwargs,
        )
        self.env.seed(0)

    @pytest.mark.parametrize("n_agents", [2, 10])
    def test_heuristic(self, n_agents, n_steps=50, n_envs=4):
        line_length = 2
        self.setup_env(n_agents=n_agents, line_length=line_length, n_envs=n_envs)
        policy = wheel.HeuristicPolicy(self.continuous_actions)

        obs = self.env.reset()

        for _ in range(n_steps):
            actions = []

            for i in range(n_agents):
                obs_agent = obs[i]

                action_agent = policy.compute_action(
                    obs_agent, self.env.agents[i].u_range
                )
                actions.append(action_agent)

            obs, new_rews, dones, _ = self.env.step(actions)
