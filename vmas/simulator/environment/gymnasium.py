#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
from typing import List, Optional
import importlib

import numpy as np
import torch

from vmas.simulator.environment.environment import Environment
from vmas.simulator.utils import extract_nested_with_index


if importlib.util.find_spec("gymnasium") is not None:
    import gymnasium as gym
    from shimmy.openai_gym_compatibility import _convert_space
else:
    raise ImportError(
        "Gymnasium is not installed. Please install it with `pip install gymnasium`."
    )


class GymnasiumWrapper(gym.Env):
    metadata = Environment.metadata

    def __init__(
        self,
        env: Environment,
        return_numpy: bool = True,
        render_mode: str = "human",
        **kwargs,
    ):
        assert (
            env.num_envs == 1
        ), "GymnasiumEnv wrapper only supports singleton VMAS environment! For vectorized environments, use vectorized wrapper with `wrapper=gymnasium_vec`."

        self._env = env
        assert self._env.terminated_truncated, "GymnasiumWrapper is only compatible with termination and truncation flags. Please set `terminated_truncated=True` in the VMAS environment."
        self.observation_space = _convert_space(self._env.observation_space)
        self.action_space = _convert_space(self._env.action_space)
        self.return_numpy = return_numpy
        self.render_mode = render_mode

    def unwrapped(self) -> Environment:
        return self._env

    def _ensure_obs_type(self, obs):
        return obs.detach().cpu().numpy() if self.return_numpy else obs

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
        terminated = terminated[0].item()
        truncated = truncated[0].item()
        if self._env.dict_spaces:
            for agent in obs.keys():
                obs[agent] = self._ensure_obs_type(
                    extract_nested_with_index(obs[agent], index=0)
                )
                info[agent] = extract_nested_with_index(info[agent], index=0)
                rews[agent] = rews[agent][0].item()
        else:
            for i in range(self._env.n_agents):
                obs[i] = self._ensure_obs_type(
                    extract_nested_with_index(obs[i], index=0)
                )
                info[i] = extract_nested_with_index(info[i], index=0)
                rews[i] = rews[i][0].item()
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
                obs[agent] = self._ensure_obs_type(
                    extract_nested_with_index(obs[agent], index=0)
                )
        else:
            for i in range(self._env.n_agents):
                obs[i] = self._ensure_obs_type(
                    extract_nested_with_index(obs[i], index=0)
                )
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
                    1,
                    self._env.get_agent_action_size(agent),
                    device=self._env.device,
                    dtype=torch.float32,
                )
            )

        for i in range(self._env.n_agents):
            act = torch.tensor(list_in[i], dtype=torch.float32, device=self._env.device)
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
        return actions
