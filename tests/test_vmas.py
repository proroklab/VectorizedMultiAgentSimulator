#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import os
import sys
from pathlib import Path

import pytest
import torch

import vmas
from vmas import make_env
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
def test_use_vmas_env(
    scenario, continuous_actions, dict_spaces=True, num_envs=10, n_steps=10
):
    render = True
    if sys.platform.startswith("win32"):
        # Windows on github servers has issues with pyglet
        render = False
    use_vmas_env(
        render=render,
        save_render=False,
        visualize_render=False,
        random_action=True,
        device="cpu",
        scenario_name=scenario,
        continuous_actions=continuous_actions,
        num_envs=num_envs,
        n_steps=n_steps,
        dict_spaces=dict_spaces,
    )


@pytest.mark.parametrize("scenario", scenario_names())
def test_multi_discrete_actions(scenario, num_envs=10, n_steps=10):
    env = make_env(
        scenario=scenario,
        num_envs=num_envs,
        seed=0,
        multidiscrete_actions=True,
        continuous_actions=False,
    )
    for _ in range(n_steps):
        env.step(env.get_random_actions())


@pytest.mark.parametrize("scenario", scenario_names())
def test_non_dict_spaces_actions(scenario, num_envs=10, n_steps=10):
    env = make_env(
        scenario=scenario,
        num_envs=num_envs,
        seed=0,
        continuous_actions=True,
        dict_spaces=False,
    )
    for _ in range(n_steps):
        env.step(env.get_random_actions())


@pytest.mark.parametrize("scenario", scenario_names())
def test_partial_reset(scenario, num_envs=10, n_steps=10):
    env = make_env(
        scenario=scenario,
        num_envs=num_envs,
        seed=0,
    )
    env_index = 0
    for _ in range(n_steps):
        env.step(env.get_random_actions())
        env.reset_at(env_index)
        env_index += 1
        if env_index >= num_envs:
            env_index = 0


@pytest.mark.parametrize("scenario", scenario_names())
def test_global_reset(scenario, num_envs=10, n_steps=10):
    env = make_env(
        scenario=scenario,
        num_envs=num_envs,
        seed=0,
    )
    for step in range(n_steps):
        env.step(env.get_random_actions())
        if step == n_steps // 2:
            env.reset()


@pytest.mark.parametrize("scenario", vmas.scenarios + vmas.mpe_scenarios)
def test_vmas_differentiable(scenario, n_steps=10, n_envs=10):
    if (
        scenario == "football"
        or scenario == "simple_crypto"
        or scenario == "road_traffic"
    ):
        pytest.skip()
    env = make_env(
        scenario=scenario,
        num_envs=n_envs,
        continuous_actions=True,
        seed=0,
        grad_enabled=True,
    )

    for step in range(n_steps):
        actions = []
        for agent in env.agents:
            action = env.get_random_action(agent)
            action.requires_grad_(True)
            if step == 0:
                first_action = action
            actions.append(action)
        obs, rews, dones, info = env.step(actions)

    loss = obs[-1].mean() + rews[-1].mean()
    grad = torch.autograd.grad(loss, first_action)
