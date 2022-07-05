#  Copyright (c) 2022. Jan Blumenkamp
#  All rights reserved.
import unittest

import torch

from vmas import make_env


class TestFlocking(unittest.TestCase):
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
            max_steps=100,
            # Environment specific variables
            n_agents=self.n_agents,
        )
        self.env.seed(0)

    def test_heuristic(self):
        for n_agents in [1, 10, 20]:
            self.setup_env(
                n_agents=n_agents,
            )
            all_done = torch.full((self.n_envs,), False)
            obs = self.env.reset()
            total_rew = torch.zeros(self.env.num_envs, n_agents)
            while not all_done.all():
                actions = []
                for i in range(n_agents):
                    agent_phase_shift = 2 * torch.pi / n_agents * i
                    angular_v_rad_per_step = 1 / 15
                    agent_dist_to_target = 0.5
                    angle = self.env.steps * angular_v_rad_per_step + agent_phase_shift
                    desired_pos = (
                        torch.stack([torch.cos(angle), torch.sin(angle)], dim=1)
                        * agent_dist_to_target
                    )
                    delta_pos = desired_pos - obs[i][:, :2]
                    action = torch.clamp(
                        delta_pos * 2,
                        min=-self.env.agents[i].u_range,
                        max=self.env.agents[i].u_range,
                    )
                    actions.append(action)

                obs, rews, dones, _ = self.env.step(actions)
                for i in range(n_agents):
                    total_rew[:, i] += rews[i]
                if dones.any():
                    self.assertTrue((total_rew[dones] < -5.0).all())
                    self.assertTrue((total_rew[dones] > -200.0).all())
                    total_rew[dones] = 0
                    all_done += dones
                    for env_index, done in enumerate(dones):
                        if done:
                            self.env.reset_at(env_index)
