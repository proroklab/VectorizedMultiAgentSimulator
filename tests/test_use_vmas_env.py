#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.


import pytest
from vmas import debug_scenarios, mpe_scenarios, scenarios

from vmas.examples.use_vmas_env import use_vmas_env


@pytest.mark.parametrize("scenario", scenarios + mpe_scenarios + debug_scenarios)
@pytest.mark.parametrize("continuous_actions", [True, False])
def test_use_vmas_env(scenario, continuous_actions, num_envs=10, n_steps=10):
    use_vmas_env(
        render=False,
        random_action=True,
        device="cpu",
        scenario_name=scenario,
        continuous_actions=continuous_actions,
        num_envs=num_envs,
        n_steps=n_steps,
    )


def test_render(scenario="waterfall", continuous_actions=True, num_envs=10, n_steps=10):
    use_vmas_env(
        render=True,
        save_render=True,
        visualize_when_rgb=False,
        random_action=True,
        device="cpu",
        scenario_name=scenario,
        continuous_actions=continuous_actions,
        num_envs=num_envs,
        n_steps=n_steps,
    )
