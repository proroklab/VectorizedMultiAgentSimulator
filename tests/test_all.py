#  Copyright (c) 2022. Matteo Bettini
#  All rights reserved.
import unittest

from maps import make_env


class TestAll(unittest.TestCase):
    def test_all(self):
        scenario_name = "waterfall"

        # Scenario specific variables
        n_agents = 4

        num_envs = 32
        continuous_actions = False
        device = "cpu"  # or cuda or any other torch device
        wrapped = False
        n_steps = 100

        env = make_env(
            scenario_name=scenario_name,
            num_envs=num_envs,
            device=device,
            continuous_actions=continuous_actions,
            rllib_wrapped=wrapped,
            # Environment specific variables
            n_agents=n_agents,
        )
