#  Copyright (c) 2022.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import unittest

import torch
from vmas import make_env
from vmas.scenarios import transport


class TestTransport(unittest.TestCase):
    def setup_env(self, **kwargs) -> None:
        super().setUp()
        self.n_agents = kwargs.get("n_agents", 4)
        self.n_packages = kwargs.get("n_packages", 1)
        self.package_width = kwargs.get("package_width", 0.15)
        self.package_length = kwargs.get("package_length", 0.15)
        self.package_mass = kwargs.get("package_mass", 50)

        self.continuous_actions = True
        self.n_envs = 25
        self.env = make_env(
            scenario_name="transport",
            num_envs=self.n_envs,
            device="cpu",
            continuous_actions=self.continuous_actions,
            # Environment specific variables
            **kwargs
        )
        self.env.seed(0)

    def test_not_passing_through_packages(self):
        self.setup_env(n_agents=1)

        for _ in range(10):
            obs = self.env.reset()
            for _ in range(100):
                obs_agent = obs[0]
                self.assertTrue(
                    (
                        torch.linalg.vector_norm(obs_agent[:, 6:8], dim=1)
                        > self.env.agents[0].shape.radius
                    ).all()
                )
                action_agent = torch.clamp(
                    obs_agent[:, 6:8],
                    min=-self.env.agents[0].u_range,
                    max=self.env.agents[0].u_range,
                )
                action_agent /= torch.linalg.vector_norm(action_agent, dim=1).unsqueeze(
                    -1
                )
                action_agent *= self.env.agents[0].u_range

                obs, rews, dones, _ = self.env.step([action_agent])

    def test_heuristic(self):

        for n_agents in [4]:
            self.setup_env(n_agents=n_agents, random_package_pos_on_line=False)
            policy = transport.HeuristicPolicy(self.continuous_actions)

            obs = self.env.reset()
            rews = None

            for _ in range(100):
                actions = []
                for i in range(n_agents):
                    obs_agent = obs[i]

                    action_agent = policy.compute_action(
                        obs_agent, self.env.agents[i].u_range
                    )

                    actions.append(action_agent)

                obs, new_rews, dones, _ = self.env.step(actions)

                if rews is not None:
                    for i in range(self.n_agents):
                        self.assertTrue((new_rews[i] >= rews[i]).all())
                    rews = new_rews
