#  Copyright (c) 2022.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import unittest

import torch

from vmas import make_env
from vmas.scenarios.dropout import DEFAULT_ENERGY_COEFF


class TestDropout(unittest.TestCase):
    def setup_env(
        self,
        n_agents: int,
        energy_coeff: float = DEFAULT_ENERGY_COEFF,
        num_envs: int = 23,
    ) -> None:
        super().setUp()
        self.n_agents = n_agents
        self.energy_coeff = energy_coeff

        self.continuous_actions = True
        self.n_envs = num_envs
        self.env = make_env(
            scenario_name="dropout",
            num_envs=self.n_envs,
            device="cpu",
            continuous_actions=self.continuous_actions,
            # Environment specific variables
            n_agents=self.n_agents,
            energy_coeff=self.energy_coeff,
        )
        self.env.seed(0)

    # Test that one agent can always reach the goal no matter the conditions
    def test_heuristic(self):
        for n_agents in [1, 5, 10, 20]:
            self.setup_env(n_agents=n_agents, num_envs=1)

            obs = self.env.reset()
            total_rew = torch.zeros(self.env.num_envs)

            current_min = float("inf")
            best_i = None
            for i in range(n_agents):
                obs_agent = obs[i]
                if (
                    torch.linalg.vector_norm(obs_agent[:, -3:-1], dim=1)[0]
                    < current_min
                ):
                    current_min = torch.linalg.vector_norm(obs_agent[:, -3:-1], dim=1)[
                        0
                    ]
                    best_i = i

            done = False
            while not done:
                obs_agent = obs[best_i]
                action_agent = torch.clamp(
                    obs_agent[:, -3:-1],
                    min=-self.env.agents[best_i].u_range,
                    max=self.env.agents[best_i].u_range,
                )

                actions = []
                other_agents_action = torch.zeros(
                    self.env.num_envs, self.env.world.dim_p
                )
                for j in range(self.n_agents):
                    if best_i != j:
                        actions.append(other_agents_action)
                    else:
                        actions.append(action_agent)

                obs, new_rews, dones, _ = self.env.step(actions)
                for j in range(self.n_agents):
                    self.assertTrue(torch.equal(new_rews[0], new_rews[j]))
                total_rew += new_rews[0]
                self.assertTrue((total_rew[dones] > 0).all())
                done = dones.any()

    # Test that one agent can always reach the goal no matter the conditions
    def test_one_random_agent_can_do_it(self):
        rewards = []
        for n_agents in [1, 5, 10]:
            self.setup_env(n_agents=n_agents)
            for i in range(self.n_agents):
                obs = self.env.reset()
                total_rew = torch.zeros(self.env.num_envs)
                for _ in range(50):
                    obs_agent = obs[i]
                    action_agent = torch.clamp(
                        obs_agent[:, -3:-1],
                        min=-self.env.agents[i].u_range,
                        max=self.env.agents[i].u_range,
                    )

                    actions = []
                    other_agents_action = torch.zeros(
                        self.env.num_envs, self.env.world.dim_p
                    )
                    for j in range(self.n_agents):
                        if i != j:
                            actions.append(other_agents_action)
                        else:
                            actions.append(action_agent)

                    obs, new_rews, dones, _ = self.env.step(actions)
                    for j in range(self.n_agents):
                        self.assertTrue(torch.equal(new_rews[0], new_rews[j]))
                    total_rew += new_rews[0]
                    self.assertTrue((total_rew[dones] > 0).all())
                    for env_index, done in enumerate(dones):
                        if done:
                            self.env.reset_at(env_index)
                    if dones.any():
                        rewards.append(total_rew[dones].clone())
                    total_rew[dones] = 0
        return sum([rew.mean().item() for rew in rewards]) / len(rewards)

    def test_all_agents_cannot_do_it(self):
        # Test that all agents together cannot reach the goal no matter the conditions (to be sure we do 5+ agents)
        self.assertLess(self.all_agents(DEFAULT_ENERGY_COEFF), 0)
        # Test that all agents together can reach the goal with no energy penalty
        self.assertGreater(self.all_agents(0), 0)

    def all_agents(self, energy_coeff: float):
        rewards = []
        for n_agents in [5, 10, 50]:
            self.setup_env(n_agents=n_agents, energy_coeff=energy_coeff)
            obs = self.env.reset()
            total_rew = torch.zeros(self.env.num_envs)
            for _ in range(50):
                actions = []
                for i in range(self.n_agents):
                    obs_i = obs[i]
                    action_i = torch.clamp(
                        obs_i[:, -3:-1],
                        min=-self.env.agents[i].u_range,
                        max=self.env.agents[i].u_range,
                    )
                    actions.append(action_i)

                obs, new_rews, dones, _ = self.env.step(actions)
                for j in range(self.n_agents):
                    self.assertTrue(torch.equal(new_rews[0], new_rews[j]))
                total_rew += new_rews[0]
                for env_index, done in enumerate(dones):
                    if done:
                        self.env.reset_at(env_index)
                if dones.any():
                    rewards.append(total_rew[dones].clone())
                total_rew[dones] = 0
        return sum([rew.mean().item() for rew in rewards]) / len(rewards)
