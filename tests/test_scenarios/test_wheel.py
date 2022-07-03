#  Copyright (c) 2022.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import unittest

from vmas import make_env
from vmas.scenarios import wheel


class TestWheel(unittest.TestCase):
    def setup_env(
        self,
        **kwargs,
    ) -> None:
        super().setUp()
        self.n_agents = kwargs.get("n_agents", 4)
        self.desired_velocity = kwargs.get("desired_velocity", 0.1)

        self.continuous_actions = True
        self.n_envs = 25
        self.env = make_env(
            scenario_name="wheel",
            num_envs=self.n_envs,
            device="cpu",
            continuous_actions=self.continuous_actions,
            # Environment specific variables
            **kwargs,
        )
        self.env.seed(0)

    def test_heuristic(self):
        line_length = 2
        for n_agents in [1, 3, 5, 10]:
            self.setup_env(n_agents=n_agents, line_length=line_length)
            policy = wheel.HeuristicPolicy(self.continuous_actions)

            obs = self.env.reset()
            rews = None
            for _ in range(300):
                actions = []
                idx = 0
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
