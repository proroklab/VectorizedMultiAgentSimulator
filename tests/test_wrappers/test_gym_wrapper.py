#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import gym
import numpy as np
import pytest
from torch import Tensor

from vmas import make_env
from vmas.simulator.environment import Environment


TEST_SCENARIOS = [
    "balance",
    "discovery",
    "give_way",
    "joint_passage",
    "navigation",
    "passage",
    "transport",
    "waterfall",
    "simple_world_comm",
]


def _check_obs_type(obss, obs_shapes, dict_space, return_numpy):
    if dict_space:
        assert isinstance(
            obss, dict
        ), f"Expected dictionary of observations, got {type(obss)}"
        for k, obs in obss.items():
            obs_shape = obs_shapes[k]
            assert (
                obs.shape == obs_shape
            ), f"Expected shape {obs_shape}, got {obs.shape}"
            if return_numpy:
                assert isinstance(
                    obs, np.ndarray
                ), f"Expected numpy array, got {type(obs)}"
            else:
                assert isinstance(
                    obs, Tensor
                ), f"Expected torch tensor, got {type(obs)}"
    else:
        assert isinstance(
            obss, list
        ), f"Expected list of observations, got {type(obss)}"
        for obs, shape in zip(obss, obs_shapes):
            assert obs.shape == shape, f"Expected shape {shape}, got {obs.shape}"
            if return_numpy:
                assert isinstance(
                    obs, np.ndarray
                ), f"Expected numpy array, got {type(obs)}"
            else:
                assert isinstance(
                    obs, Tensor
                ), f"Expected torch tensor, got {type(obs)}"


@pytest.mark.parametrize("scenario", TEST_SCENARIOS)
@pytest.mark.parametrize("return_numpy", [True, False])
@pytest.mark.parametrize("continuous_actions", [True, False])
@pytest.mark.parametrize("dict_space", [True, False])
def test_gym_wrapper(
    scenario, return_numpy, continuous_actions, dict_space, max_steps=10
):
    env = make_env(
        scenario=scenario,
        num_envs=1,
        device="cpu",
        continuous_actions=continuous_actions,
        dict_spaces=dict_space,
        wrapper="gym",
        wrapper_kwargs={"return_numpy": return_numpy},
        max_steps=max_steps,
    )

    assert (
        len(env.observation_space) == env.unwrapped.n_agents
    ), "Expected one observation per agent"
    assert (
        len(env.action_space) == env.unwrapped.n_agents
    ), "Expected one action per agent"
    if dict_space:
        assert isinstance(
            env.observation_space, gym.spaces.Dict
        ), "Expected Dict observation space"
        assert isinstance(
            env.action_space, gym.spaces.Dict
        ), "Expected Dict action space"
        obs_shapes = {
            k: obs_space.shape for k, obs_space in env.observation_space.spaces.items()
        }
    else:
        assert isinstance(
            env.observation_space, gym.spaces.Tuple
        ), "Expected Tuple observation space"
        assert isinstance(
            env.action_space, gym.spaces.Tuple
        ), "Expected Tuple action space"
        obs_shapes = [obs_space.shape for obs_space in env.observation_space.spaces]

    assert isinstance(
        env.unwrapped, Environment
    ), "The unwrapped attribute of the Gym wrapper should be a VMAS Environment"

    obss = env.reset()
    _check_obs_type(obss, obs_shapes, dict_space, return_numpy=return_numpy)

    for _ in range(max_steps):
        actions = [
            env.unwrapped.get_random_action(agent).numpy()
            for agent in env.unwrapped.agents
        ]
        obss, rews, done, info = env.step(actions)
        _check_obs_type(obss, obs_shapes, dict_space, return_numpy=return_numpy)

        assert len(rews) == env.unwrapped.n_agents, "Expected one reward per agent"
        if not dict_space:
            assert isinstance(
                rews, list
            ), f"Expected list of rewards but got {type(rews)}"

            rew_values = rews
        else:
            assert isinstance(
                rews, dict
            ), f"Expected dictionary of rewards but got {type(rews)}"
            rew_values = list(rews.values())
        assert all(
            isinstance(rew, float) for rew in rew_values
        ), f"Expected float rewards but got {type(rew_values[0])}"

        assert isinstance(done, bool), f"Expected bool for done but got {type(done)}"

        assert isinstance(
            info, dict
        ), f"Expected info to be a dictionary but got {type(info)}"

    assert done, "Expected done to be True after 100 steps"
