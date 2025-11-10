#  Copyright (c) ProrokLab.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Optional

import gym
import numpy as np

from vmas.simulator.environment.environment import Environment
from vmas.simulator.environment.gym.base import BaseGymWrapper


class GymWrapper(gym.Env, BaseGymWrapper):
    metadata = Environment.metadata

    def __init__(
        self,
        env: Environment,
        return_numpy: bool = True,
    ):
        super().__init__(env, return_numpy=return_numpy, vectorized=False)
        assert (
            env.num_envs == 1
        ), f"GymEnv wrapper is not vectorised, got env.num_envs: {env.num_envs}"

        assert (
            not self._env.terminated_truncated
        ), "GymWrapper is not compatible with termination and truncation flags. Please set `terminated_truncated=False` in the VMAS environment."
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    @property
    def unwrapped(self) -> Environment:
        return self._env

    def step(self, action):
        action = self._action_list_to_tensor(action)
        obs, rews, done, info = self._env.step(action)
        env_data = self._convert_env_data(
            obs=obs,
            rews=rews,
            info=info,
            done=done,
        )
        return env_data.obs, env_data.rews, env_data.done, env_data.info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        if seed is not None:
            self._env.seed(seed)
        obs = self._env.reset_at(index=0)
        env_data = self._convert_env_data(obs=obs)
        return env_data.obs

    def render(
        self,
        mode="human",
        agent_index_focus: Optional[int] = None,
        visualize_when_rgb: bool = False,
        **kwargs,
    ) -> Optional[np.ndarray]:
        return self._env.render(
            mode=mode,
            env_index=0,
            agent_index_focus=agent_index_focus,
            visualize_when_rgb=visualize_when_rgb,
            **kwargs,
        )
