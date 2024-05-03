#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import pytest
import torch

from vmas import make_env
from vmas.scenarios.navigation import HeuristicPolicy


class TestNavigation:
    def setUp(self, n_envs, n_agents) -> None:
        self.continuous_actions = True

        self.env = make_env(
            scenario="navigation",
            num_envs=n_envs,
            device="cpu",
            continuous_actions=self.continuous_actions,
            # Environment specific variables
            n_agents=n_agents,
        )
        self.env.seed(0)

    @pytest.mark.parametrize("n_agents", [1])
    def test_heuristic(
        self,
        n_agents,
        n_envs=5,
    ):
        self.setUp(n_envs=n_envs, n_agents=n_agents)

        policy = HeuristicPolicy(
            continuous_action=self.continuous_actions, clf_epsilon=0.4, clf_slack=100.0
        )

        obs = self.env.reset()
        all_done = torch.zeros(n_envs, dtype=torch.bool)

        while not all_done.all():
            actions = []
            for i in range(n_agents):
                obs_agent = obs[i]

                action_agent = policy.compute_action(
                    obs_agent, self.env.agents[i].action.u_range_tensor
                )

                actions.append(action_agent)

            obs, new_rews, dones, _ = self.env.step(actions)
            if dones.any():
                all_done += dones

                for env_index, done in enumerate(dones):
                    if done:
                        self.env.reset_at(env_index)
