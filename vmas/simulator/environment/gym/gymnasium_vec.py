#  Copyright (c) ProrokLab.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
import importlib

import warnings
from typing import Optional

import numpy as np

from vmas.simulator.environment.environment import Environment
from vmas.simulator.environment.gym.base import BaseGymWrapper


if (
    importlib.util.find_spec("gymnasium") is not None
    and importlib.util.find_spec("shimmy") is not None
):
    import gymnasium as gym
    from gymnasium.vector.utils import batch_space
    from shimmy.openai_gym_compatibility import _convert_space
else:
    raise ImportError(
        "Gymnasium or shimmy is not installed. Please install it with `pip install gymnasium shimmy`."
    )


class GymnasiumVectorizedWrapper(gym.Env, BaseGymWrapper):
    metadata = Environment.metadata

    def __init__(
        self,
        env: Environment,
        return_numpy: bool = True,
        render_mode: str = "human",
    ):
        super().__init__(env, return_numpy=return_numpy, vectorized=True)
        self._num_envs = self._env.num_envs
        assert (
            self._env.terminated_truncated
        ), "GymnasiumWrapper is only compatible with termination and truncation flags. Please set `terminated_truncated=True` in the VMAS environment."
        self.single_observation_space = _convert_space(self._env.observation_space)
        self.single_action_space = _convert_space(self._env.action_space)
        self.observation_space = batch_space(
            self.single_observation_space, n=self._num_envs
        )
        self.action_space = batch_space(self.single_action_space, n=self._num_envs)
        self.render_mode = render_mode
        warnings.warn(
            "The Gymnasium Vector wrapper currently does not have auto-resets or support partial resets."
            "We warn you that by using this class, individual environments will not be reset when they are done and you"
            "will only have access to global resets. We strongly suggest using the VMAS API unless your scenario does not implement"
            "the `done` function and thus all sub-environments are done at the same time."
        )

    @property
    def unwrapped(self) -> Environment:
        return self._env

    def step(self, action):
        action = self._action_list_to_tensor(action)
        obs, rews, terminated, truncated, info = self._env.step(action)
        env_data = self._convert_env_data(
            obs=obs, rews=rews, info=info, terminated=terminated, truncated=truncated
        )
        return (
            env_data.obs,
            env_data.rews,
            env_data.terminated,
            env_data.truncated,
            env_data.info,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        if seed is not None:
            self._env.seed(seed)
        obs, info = self._env.reset(return_info=True)
        env_data = self._convert_env_data(obs=obs, info=info)
        return env_data.obs, env_data.info

    def render(
        self,
        agent_index_focus: Optional[int] = None,
        visualize_when_rgb: bool = False,
        **kwargs,
    ) -> Optional[np.ndarray]:
        return self._env.render(
            mode=self.render_mode,
            agent_index_focus=agent_index_focus,
            visualize_when_rgb=visualize_when_rgb,
            **kwargs,
        )
