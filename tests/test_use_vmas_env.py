#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import os

from pathlib import Path

import pytest

from vmas.examples.use_vmas_env import use_vmas_env


def scenario_names():
    scenarios = []
    scenarios_folder = Path(__file__).parent.parent / "vmas" / "scenarios"
    for _, _, filenames in os.walk(scenarios_folder):
        scenarios += filenames
    scenarios = [
        scenario.split(".")[0]
        for scenario in scenarios
        if scenario.endswith(".py") and not scenario.startswith("__")
    ]
    return scenarios


def test_all_scenarios_included():
    from vmas import debug_scenarios, mpe_scenarios, scenarios

    assert sorted(scenario_names()) == sorted(
        scenarios + mpe_scenarios + debug_scenarios
    )


@pytest.mark.parametrize("scenario", scenario_names())
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
