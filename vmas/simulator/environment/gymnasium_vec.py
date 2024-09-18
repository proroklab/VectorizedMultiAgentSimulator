#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
from typing import List, Optional

import gymnasium as gym
import numpy as np
import torch

from vmas.simulator.environment.environment import Environment
from vmas.simulator.environment.gymnasium import _convert_space


def _get_vectorized_space(space: gym.Space, num_envs: int) -> gym.Space:
    """Convert singleton gymnasium space to a vectorized gymnasium space.

    Args:
        space: the singleton gymnasium space
        num_envs: the number of environments
    Returns:
        The vectorized gymnasium space
    """
    if isinstance(space, gym.spaces.Discrete):
        return gym.spaces.MultiDiscrete(nvec=np.broadcast_to(space.n, (num_envs,)))
    elif isinstance(space, gym.spaces.Box):
        return gym.spaces.Box(
            low=np.broadcast_to(space.low, (num_envs, *space.low.shape)),
            high=np.broadcast_to(space.high, (num_envs, *space.high.shape)),
            shape=(num_envs, *space.shape),
            dtype=space.dtype,
        )
    elif isinstance(space, gym.spaces.MultiDiscrete):
        return gym.spaces.MultiDiscrete(
            nvec=np.broadcast_to(space.nvec, (num_envs, *space.shape))
        )
    elif isinstance(space, gym.spaces.MultiBinary):
        return gym.spaces.MultiBinary(
            n=np.broadcast_to(np.array(space.n, dtype=int), (num_envs,))
        )
    elif isinstance(space, gym.spaces.Tuple):
        return gym.spaces.Tuple(
            spaces=tuple(
                map(lambda s: _get_vectorized_space(s, num_envs), space.spaces)
            )
        )
    elif isinstance(space, gym.spaces.Dict):
        return gym.spaces.Dict(
            spaces={
                k: _get_vectorized_space(v, num_envs) for k, v in space.spaces.items()
            }
        )
    elif isinstance(space, gym.spaces.Graph):
        return gym.spaces.Graph(
            node_space=_get_vectorized_space(space.node_space, num_envs),
            edge_space=_get_vectorized_space(space.edge_space, num_envs),
        )
    else:
        raise NotImplementedError(
            f"Cannot convert space of type {space}. Please upgrade your code to gymnasium."
        )


class GymnasiumVectorizedWrapper(gym.Env):
    metadata = Environment.metadata

    def __init__(
        self,
        env: Environment,
        return_numpy: bool = True,
        render_mode: str = "human",
        **kwargs,
    ):
        self._env = env
        self._num_envs = self._env.num_envs
        assert self._env.terminated_truncated, "GymnasiumWrapper is only compatible with termination and truncation flags. Please set `terminated_truncated=True` in the VMAS environment."
        self.single_observation_space = _convert_space(self._env.observation_space)
        self.single_action_space = _convert_space(self._env.action_space)
        self.observation_space = _get_vectorized_space(
            self.single_observation_space, self._num_envs
        )
        self.action_space = _get_vectorized_space(
            self.single_action_space, self._num_envs
        )

        self.return_numpy = return_numpy
        self.render_mode = render_mode

    def unwrapped(self) -> Environment:
        return self._env

    def _ensure_tensor_type(self, tensor):
        return tensor.detach().cpu().numpy() if self.return_numpy else tensor

    @property
    def env(self):
        return self._env

    def _compress_infos(self, infos):
        info = {}
        for i, info in enumerate(infos):
            for k, v in info.items():
                info[f"agent_{i}/{k}"] = v
        return info

    def step(self, action):
        action = self._action_list_to_tensor(action)
        obs, rews, terminated, truncated, info = self._env.step(action)
        if self._env.dict_spaces:
            for agent in obs.keys():
                obs[agent] = self._ensure_tensor_type(obs[agent])
                info[agent] = info[agent]
                rews[agent] = self._ensure_tensor_type(rews[agent])
        else:
            for i in range(self._env.n_agents):
                obs[i] = self._ensure_tensor_type(obs[i])
                info[i] = info[i]
                rews[i] = self._ensure_tensor_type(rews[i])
        terminated = self._ensure_tensor_type(terminated)
        truncated = self._ensure_tensor_type(truncated)
        return obs, rews, terminated, truncated, self._compress_infos(info)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        if seed is not None:
            self._env.seed(seed)
        obs, infos = self._env.reset_at(index=0, return_info=True)

        if self._env.dict_spaces:
            for agent in obs.keys():
                obs[agent] = self._ensure_tensor_type(obs[agent])
        else:
            for i in range(self._env.n_agents):
                obs[i] = self._ensure_tensor_type(obs[i])
        return obs, self._compress_infos(infos)

    def render(
        self,
        agent_index_focus: Optional[int] = None,
        visualize_when_rgb: bool = False,
        **kwargs,
    ) -> Optional[np.ndarray]:
        return self._env.render(
            mode=self.render_mode,
            env_index=0,
            agent_index_focus=agent_index_focus,
            visualize_when_rgb=visualize_when_rgb,
            **kwargs,
        )

    def _action_list_to_tensor(self, list_in: List) -> List:
        assert (
            len(list_in) == self._env.n_agents
        ), f"Expecting actions for {self._env.n_agents} agents, got {len(list_in)} actions"
        actions = []
        for agent in self._env.agents:
            actions.append(
                torch.zeros(
                    self._num_envs,
                    self._env.get_agent_action_size(agent),
                    device=self._env.device,
                    dtype=torch.float32,
                )
            )

        for i in range(self._env.n_agents):
            act = torch.tensor(list_in[i], dtype=torch.float32, device=self._env.device)
            assert (
                act.shape[0] == self._num_envs
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
