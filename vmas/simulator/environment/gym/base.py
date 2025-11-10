#  Copyright (c) ProrokLab.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from collections import namedtuple
from typing import List, Optional

import numpy as np
import torch

from vmas.simulator.environment import Environment

from vmas.simulator.utils import extract_nested_with_index, TorchUtils


EnvData = namedtuple(
    "EnvData", ["obs", "rews", "terminated", "truncated", "done", "info"]
)


class BaseGymWrapper(ABC):
    def __init__(self, env: Environment, return_numpy: bool, vectorized: bool):
        self._env = env
        self.return_numpy = return_numpy
        self.dict_spaces = env.dict_spaces
        self.vectorized = vectorized

    @property
    def env(self):
        return self._env

    def _maybe_to_numpy(self, tensor):
        return TorchUtils.to_numpy(tensor) if self.return_numpy else tensor

    def _convert_output(self, data, item: bool = False):
        if not self.vectorized:
            data = extract_nested_with_index(data, index=0)
            if item:
                return data.item()
        return self._maybe_to_numpy(data)

    def _compress_infos(self, infos):
        if isinstance(infos, dict):
            return infos
        elif isinstance(infos, list):
            return {self._env.agents[i].name: info for i, info in enumerate(infos)}
        else:
            raise ValueError(
                f"Expected list or dictionary for infos but got {type(infos)}"
            )

    def _convert_env_data(
        self, obs=None, rews=None, info=None, terminated=None, truncated=None, done=None
    ):
        if self.dict_spaces:
            for agent in obs.keys():
                if obs is not None:
                    obs[agent] = self._convert_output(obs[agent])
                if info is not None:
                    info[agent] = self._convert_output(info[agent])
                if rews is not None:
                    rews[agent] = self._convert_output(rews[agent], item=True)
        else:
            for i in range(self._env.n_agents):
                if obs is not None:
                    obs[i] = self._convert_output(obs[i])
                if info is not None:
                    info[i] = self._convert_output(info[i])
                if rews is not None:
                    rews[i] = self._convert_output(rews[i], item=True)
        terminated = (
            self._convert_output(terminated, item=True)
            if terminated is not None
            else None
        )
        truncated = (
            self._convert_output(truncated, item=True)
            if truncated is not None
            else None
        )
        done = self._convert_output(done, item=True) if done is not None else None
        info = self._compress_infos(info) if info is not None else None
        return EnvData(
            obs=obs,
            rews=rews,
            terminated=terminated,
            truncated=truncated,
            done=done,
            info=info,
        )

    def _action_list_to_tensor(self, list_in: List) -> List:
        assert (
            len(list_in) == self._env.n_agents
        ), f"Expecting actions for {self._env.n_agents} agents, got {len(list_in)} actions"

        dtype = torch.float32 if self._env.continuous_actions else torch.long

        return [
            torch.tensor(act, device=self._env.device, dtype=dtype).reshape(
                self._env.num_envs, self._env.get_agent_action_size(agent)
            )
            if not isinstance(act, torch.Tensor)
            else act.to(dtype=dtype, device=self._env.device).reshape(
                self._env.num_envs, self._env.get_agent_action_size(agent)
            )
            for agent, act in zip(self._env.agents, list_in)
        ]

    @abstractmethod
    def step(self, action):
        raise NotImplementedError

    @abstractmethod
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        raise NotImplementedError

    @abstractmethod
    def render(
        self,
        agent_index_focus: Optional[int] = None,
        visualize_when_rgb: bool = False,
        **kwargs,
    ) -> Optional[np.ndarray]:
        raise NotImplementedError
