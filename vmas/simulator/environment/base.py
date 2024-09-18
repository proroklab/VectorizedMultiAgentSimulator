from abc import ABC, abstractmethod
from collections import namedtuple
from typing import List, Optional

import numpy as np
import torch

from vmas.simulator.utils import TorchUtils, extract_nested_with_index
from vmas.simulator.environment import Environment


EnvData = namedtuple(
    "EnvData", ["obs", "rews", "terminated", "truncated", "done", "info"]
)


class VMASBaseWrapper(ABC):
    def __init__(self, env: Environment, return_numpy: bool, vectorized: bool):
        self._env = env
        self.return_numpy = return_numpy
        self.dict_spaces = env.dict_spaces
        self.vectorized = vectorized

    @property
    def env(self):
        return self._env

    def _ensure_tensor_type(self, tensor):
        return TorchUtils.to_numpy(tensor) if self.return_numpy else tensor

    def _convert_single_obs(self, obs):
        if not self.vectorized:
            obs = extract_nested_with_index(obs, index=0)
        return self._ensure_tensor_type(obs)

    def _convert_single_info(self, info):
        if not self.vectorized:
            info = extract_nested_with_index(info, index=0)
        return self._ensure_tensor_type(info)

    def _convert_single_rew(self, rews):
        return (
            rews[0].cpu().item()
            if not self.vectorized
            else self._ensure_tensor_type(rews)
        )

    def _convert_single_termination(self, terminated):
        return (
            terminated[0].cpu().item()
            if not self.vectorized
            else self._ensure_tensor_type(terminated)
        )

    def _compress_infos(self, infos):
        info = {}
        for i, info in enumerate(infos):
            for k, v in info.items():
                info[f"agent_{i}/{k}"] = v
        return info

    def _convert_env_data(
        self, obs=None, rews=None, info=None, terminated=None, truncated=None, done=None
    ):
        if self.dict_spaces:
            for agent in obs.keys():
                if obs is not None:
                    obs[agent] = self._convert_single_obs(obs[agent])
                if info is not None:
                    info[agent] = self._convert_single_info(info[agent])
                if rews is not None:
                    rews[agent] = self._convert_single_rew(rews[agent])
        else:
            for i in range(self._env.n_agents):
                if obs is not None:
                    obs[i] = self._convert_single_obs(obs[i])
                if info is not None:
                    info[i] = self._convert_single_info(info[i])
                if rews is not None:
                    rews[i] = self._convert_single_rew(rews[i])
        terminated = (
            self._convert_single_termination(terminated)
            if terminated is not None
            else None
        )
        truncated = (
            self._convert_single_termination(truncated)
            if truncated is not None
            else None
        )
        done = self._convert_single_termination(done) if done is not None else None
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
        actions = []
        for agent in self._env.agents:
            actions.append(
                torch.zeros(
                    self._env.num_envs,
                    self._env.get_agent_action_size(agent),
                    device=self._env.device,
                    dtype=torch.float32,
                )
            )

        for i in range(self._env.n_agents):
            act = torch.tensor(list_in[i], dtype=torch.float32, device=self._env.device)
            if not self.vectorized:
                if len(act.shape) == 0:
                    assert (
                        self._env.get_agent_action_size(self._env.agents[i]) == 1
                    ), f"Action of agent {i} is supposed to be an scalar int"
                else:
                    assert len(act.shape) == 1 and act.shape[
                        0
                    ] == self._env.get_agent_action_size(self._env.agents[i]), (
                        f"Action of agent {i} hase wrong shape: "
                        f"expected {self._env.get_agent_action_size(self._env.agents[i])}, got {act.shape[0]}"
                    )
                actions[i][0] = act
            else:
                assert (
                    act.shape[0] == self._env.num_envs
                ), f"Action of agent {i} is supposed to be a vector of shape ({self._num_envs}, ...)"
                if len(act.shape) == 1:
                    assert (
                        self._env.get_agent_action_size(self._env.agents[i]) == 1
                    ), f"Action of agent {i} is supposed to be an vector of shape ({self.n_num_envs},)."
                else:
                    assert len(act.shape) == 2 and act.shape[
                        1
                    ] == self._env.get_agent_action_size(self._env.agents[i]), (
                        f"Action of agent {i} hase wrong shape: "
                        f"expected {self._env.get_agent_action_size(self._env.agents[i])}, got {act.shape[0]}"
                    )
                actions[i] = act
        return actions

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
