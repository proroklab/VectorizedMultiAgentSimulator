#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import pytest
import torch

from vmas import make_env
from vmas.scenarios import balance


class TestBalance:
    def setup_env(
        self,
        n_envs,
        **kwargs,
    ) -> None:
        self.n_agents = kwargs.get("n_agents", 4)

        self.continuous_actions = True
        self.env = make_env(
            scenario="balance",
            num_envs=n_envs,
            device="cpu",
            continuous_actions=self.continuous_actions,
            # Environment specific variables
            **kwargs,
        )
        self.env.seed(0)

    @pytest.mark.parametrize("n_agents", [2, 5])
    def test_heuristic(self, n_agents, n_steps=50, n_envs=4):
        self.setup_env(
            n_agents=n_agents, random_package_pos_on_line=False, n_envs=n_envs
        )
        policy = balance.HeuristicPolicy(self.continuous_actions)

        obs = self.env.reset()

        prev_package_dist_to_goal = obs[0][:, 8:10]

        for _ in range(n_steps):
            actions = []
            for i in range(n_agents):
                obs_agent = obs[i]
                package_dist_to_goal = obs_agent[:, 8:10]

                action_agent = policy.compute_action(
                    obs_agent, self.env.agents[i].u_range
                )

                actions.append(action_agent)

            obs, new_rews, dones, _ = self.env.step(actions)

            assert (
                torch.linalg.vector_norm(package_dist_to_goal, dim=-1)
                <= torch.linalg.vector_norm(prev_package_dist_to_goal, dim=-1)
            ).all()
            prev_package_dist_to_goal = package_dist_to_goal
