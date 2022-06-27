#  Copyright (c) 2022. Jan Blumenkamp
#  All rights reserved.
import unittest

import torch

from maps import make_env


class TestDispersion(unittest.TestCase):
    def setup_env(self, n_agents: int) -> None:
        super().setUp()
        self.n_agents = n_agents

        self.continuous_actions = True
        self.n_envs = 25
        self.env = make_env(
            scenario_name="flocking",
            num_envs=self.n_envs,
            device="cpu",
            continuous_actions=self.continuous_actions,
            rllib_wrapped=False,
            # Environment specific variables
            n_agents=self.n_agents,
        )
        self.env.seed(0)

    def test_heuristic(self):
        for n_agents in [5]:
            self.setup_env(
                n_agents=n_agents,
            )
            all_done = torch.full((self.n_envs,), False)
            obs = self.env.reset()
            total_rew = torch.zeros(self.env.num_envs, n_agents)
            while not all_done.all():
                actions = []
                for i in range(n_agents):
                    action_agent = torch.zeros(25, 2)
                    actions.append(action_agent)

                obs, rews, dones, _ = self.env.step(actions)
                self.env.render()
                for i in range(n_agents):
                    total_rew[:, i] += rews[i]
                if dones.any():
                    # Done envs should have exactly sum of rewards equal to num_agents
                    self.assertTrue(
                        torch.equal(
                            total_rew[dones].sum(-1).to(torch.long),
                            torch.full((dones.sum(),), n_agents),
                        )
                    )
                    total_rew[dones] = 0
                    all_done += dones
                    for env_index, done in enumerate(dones):
                        if done:
                            self.env.reset_at(env_index)
