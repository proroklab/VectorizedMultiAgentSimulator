#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import math
import random
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
    for path in scenarios_folder.glob("**/*.py"):
        if path.is_file() and not path.name.startswith("__"):
            scenarios.append(path.stem)
    return scenarios


def random_nvecs(count, l_min=2, l_max=6, n_min=2, n_max=6, seed=0):
    random.seed(seed)
    return [
        [random.randint(n_min, n_max) for _ in range(random.randint(l_min, l_max))]
        for _ in range(count)
    ]


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
@pytest.mark.parametrize("multidiscrete_actions", [True, False])
def test_discrete_action_nvec(scenario, multidiscrete_actions, num_envs=10, n_steps=5):
    env = make_env(
        scenario=scenario,
        num_envs=num_envs,
        seed=0,
        multidiscrete_actions=multidiscrete_actions,
        continuous_actions=False,
    )
    if (
        type(env.scenario).process_action
        is not vmas.simulator.scenario.BaseScenario.process_action
    ):
        pytest.skip("Scenario uses a custom process_action method.")

    random.seed(0)
    for agent in env.world.agents:
        agent.discrete_action_nvec = [
            random.randint(2, 6) for _ in range(agent.action_size)
        ]
    env.action_space = env.get_action_space()

    def to_multidiscrete(action, nvec):
        action_multi = []
        for i in range(len(nvec)):
            n = math.prod(nvec[i + 1 :])
            action_multi.append(action // n)
            action = action % n
        return torch.stack(action_multi, dim=-1)

    def full_nvec(agent, world):
        return list(agent.discrete_action_nvec) + (
            [world.dim_c] if not agent.silent and world.dim_c != 0 else []
        )

    for _ in range(n_steps):
        actions = env.get_random_actions()

        # Check that generated actions are in the action space
        for a_batch, s in zip(actions, env.action_space.spaces):
            for a in a_batch:
                assert a.numpy() in s

        env.step(actions)

        if not multidiscrete_actions:
            actions = [
                to_multidiscrete(a.squeeze(-1), full_nvec(agent, env.world))
                for a, agent in zip(actions, env.world.policy_agents)
            ]

        # Check that discrete action to continuous control mapping is correct.
        for i_a, agent in enumerate(env.world.policy_agents):
            for i, n in enumerate(agent.discrete_action_nvec):
                a = actions[i_a][:, i]
                u = agent.action.u[:, i]
                U = agent.action.u_range_tensor[i]
                k = agent.action.u_multiplier_tensor[i]
                for aj, uj in zip(a, u):
                    assert aj in range(
                        n
                    ), f"discrete action {aj} not in [0,{n-1}] (n={n}, U={U}, k={k})"
                    if n % 2 != 0:
                        assert (
                            aj != 0 or uj == 0
                        ), f"discrete action {aj} maps to control {uj} (n={n}), U={U}, k={k})"
                        assert (aj < 1 or aj > n // 2) or torch.isclose(
                            uj / k, (2 * U * (aj - 1)) / (n - 1) - U
                        ), f"discrete action {aj} maps to control {uj} (n={n}, U={U}, k={k})"
                        assert (aj <= n // 2) or torch.isclose(
                            uj / k, 2 * U * (aj / (n - 1)) - U
                        ), f"discrete action {aj} maps to control {uj} (n={n}), U={U}, k={k})"
                    else:
                        assert torch.isclose(
                            uj / k, 2 * U * (aj / (n - 1)) - U
                        ), f"discrete action {aj} maps to control {uj} (n={n}), U={U}, k={k})"


@pytest.mark.parametrize(
    "nvecs", list(zip(random_nvecs(10, seed=0), random_nvecs(10, seed=42)))
)
def test_discrete_action_nvec_discrete_to_multi(
    nvecs, scenario="transport", num_envs=10, n_steps=5
):
    kwargs = {
        "scenario": scenario,
        "num_envs": num_envs,
        "seed": 0,
        "continuous_actions": False,
    }
    env = make_env(**kwargs, multidiscrete_actions=False)
    env_multi = make_env(**kwargs, multidiscrete_actions=True)
    if (
        type(env.scenario).process_action
        is not vmas.simulator.scenario.BaseScenario.process_action
    ):
        pytest.skip("Scenario uses a custom process_action method.")

    def set_nvec(agent, nvec):
        agent.action_size = len(nvec)
        agent.discrete_action_nvec = nvec
        agent.action.action_size = agent.action_size

    random.seed(0)
    for agent, agent_multi, nvec in zip(
        env.world.policy_agents, env_multi.world.policy_agents, nvecs
    ):
        set_nvec(agent, nvec)
        set_nvec(agent_multi, nvec)
    env.action_space = env.get_action_space()
    env_multi.action_space = env.get_action_space()

    def full_nvec(agent, world):
        return list(agent.discrete_action_nvec) + (
            [world.dim_c] if not agent.silent and world.dim_c != 0 else []
        )

    def full_action_size(agent, world):
        return len(full_nvec(agent, world))

    for _ in range(n_steps):
        actions_multi = env_multi.get_random_actions()
        prodss = [
            [
                math.prod(full_nvec(agent, env.world)[i + 1 :])
                for i in range(full_action_size(agent, env.world))
            ]
            for agent in env.world.policy_agents
        ]
        # Compute the expected mapping from multi-discrete to discrete
        actions = [
            (a_multi * torch.tensor(prods)).sum(dim=1)
            for a_multi, prods in zip(actions_multi, prodss)
        ]

        env_multi.step(actions_multi)
        env.step(actions)

        # Check that both discrete and multi-discrete actions result in the
        # same control value
        for agent, agent_multi, action, action_multi in zip(
            env.world.policy_agents,
            env_multi.world.policy_agents,
            actions,
            actions_multi,
        ):
            U = agent.action.u_range_tensor
            k = agent.action.u_multiplier_tensor
            for u, u_multi, a, a_multi in zip(
                agent.action.u, agent_multi.action.u, action, action_multi
            ):
                assert torch.allclose(
                    u, u_multi
                ), f"{u} != {u_multi} (nvec={agent.discrete_action_nvec}, a={a}, a_multi={a_multi}, U={U}, k={k})"


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


def test_seeding():
    env = make_env(scenario="balance", num_envs=2, seed=0)
    env.seed(0)
    random_obs = env.reset()[0][0, 0]
    env.seed(0)
    assert random_obs == env.reset()[0][0, 0]
    env.seed(0)
    torch.manual_seed(1)
    assert random_obs == env.reset()[0][0, 0]

    torch.manual_seed(0)
    random_obs = torch.randn(1)
    torch.manual_seed(0)
    env.seed(1)
    env.reset()
    assert random_obs == torch.randn(1)
