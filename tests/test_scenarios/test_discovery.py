#  Copyright (c) 2022.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import unittest

from vmas import make_env
from vmas.scenarios import discovery


class TestDiscovery(unittest.TestCase):
    def setup_env(
        self,
        **kwargs,
    ) -> None:
        super().setUp()

        self.n_envs = 25
        self.env = make_env(
            scenario_name="discovery",
            num_envs=self.n_envs,
            device="cpu",
            # Environment specific variables
            **kwargs,
        )
        self.env.seed(0)

    def test_heuristic(self):

        for n_agents in [5, 10]:
            self.setup_env(n_agents=n_agents, random_package_pos_on_line=False)
            policy = discovery.HeuristicPolicy(True)

            obs = self.env.reset()
            rews = None

            for _ in range(500):
                actions = []
                for i in range(n_agents):
                    obs_agent = obs[i]

                    action_agent = policy.compute_action(
                        obs_agent, self.env.agents[i].u_range
                    )

                    actions.append(action_agent)

                obs, new_rews, dones, _ = self.env.step(actions)

                if rews is not None:
                    for i in range(n_agents):
                        self.assertTrue((new_rews[i] >= rews[i]).all())
                    rews = new_rews
