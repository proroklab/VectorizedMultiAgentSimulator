#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import pytest
import torch

from vmas import make_env


class TestDispersion:
    def setup_env(
        self, n_agents: int, share_reward: bool, penalise_by_time: bool, n_envs
    ) -> None:
        self.n_agents = n_agents
        self.share_reward = share_reward
        self.penalise_by_time = penalise_by_time

        self.continuous_actions = True
        self.env = make_env(
            scenario="dispersion",
            num_envs=n_envs,
            device="cpu",
            continuous_actions=self.continuous_actions,
            # Environment specific variables
            n_agents=self.n_agents,
            share_reward=self.share_reward,
            penalise_by_time=self.penalise_by_time,
        )
        self.env.seed(0)

    @pytest.mark.parametrize("n_agents", [1, 5, 10])
    def test_heuristic(self, n_agents, n_envs=4):
        self.setup_env(
            n_agents=n_agents,
            share_reward=False,
            penalise_by_time=False,
            n_envs=n_envs,
        )
        all_done = torch.full((n_envs,), False)
        obs = self.env.reset()
        total_rew = torch.zeros(self.env.num_envs, n_agents)
        while not all_done.all():
            actions = []
            idx = 0
            for i in range(n_agents):
                obs_agent = obs[i]
                obs_idx = 4 + idx
                action_agent = torch.clamp(
                    obs_agent[:, obs_idx : obs_idx + 2],
                    min=-self.env.agents[i].u_range,
                    max=self.env.agents[i].u_range,
                )
                idx += 3
                actions.append(action_agent)

            obs, rews, dones, _ = self.env.step(actions)
            for i in range(n_agents):
                total_rew[:, i] += rews[i]
            if dones.any():
                # Done envs should have exactly sum of rewards equal to num_agents
                assert torch.equal(
                    total_rew[dones].sum(-1).to(torch.long),
                    torch.full((dones.sum(),), n_agents),
                )
                total_rew[dones] = 0
                all_done += dones
                for env_index, done in enumerate(dones):
                    if done:
                        self.env.reset_at(env_index)

    @pytest.mark.parametrize("n_agents", [1, 5, 10, 20])
    def test_heuristic_share_reward(self, n_agents, n_envs=4):
        self.setup_env(
            n_agents=n_agents,
            share_reward=True,
            penalise_by_time=False,
            n_envs=n_envs,
        )
        all_done = torch.full((n_envs,), False)
        obs = self.env.reset()
        total_rew = torch.zeros(self.env.num_envs, n_agents)
        while not all_done.all():
            actions = []
            idx = 0
            for i in range(n_agents):
                obs_agent = obs[i]
                obs_idx = 4 + idx
                action_agent = torch.clamp(
                    obs_agent[:, obs_idx : obs_idx + 2],
                    min=-self.env.agents[i].u_range,
                    max=self.env.agents[i].u_range,
                )
                idx += 3
                actions.append(action_agent)

            obs, rews, dones, _ = self.env.step(actions)
            for i in range(n_agents):
                total_rew[:, i] += rews[i]
            if dones.any():
                # Done envs should have exactly sum of rewards equal to num_agents
                assert torch.equal(
                    total_rew[dones],
                    torch.full((dones.sum(), n_agents), n_agents).to(torch.float),
                )
                total_rew[dones] = 0
                all_done += dones
                for env_index, done in enumerate(dones):
                    if done:
                        self.env.reset_at(env_index)
