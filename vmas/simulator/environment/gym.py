#  Copyright (c) 2022-2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
from typing import List, Optional

import gym
import numpy as np
import torch
from vmas.simulator.environment.environment import Environment


class GymWrapper(gym.Env):
    metadata = Environment.metadata

    def __init__(
        self,
        env: Environment,
    ):
        assert (
            env.num_envs == 1
        ), f"GymEnv wrapper is not vectorised, got env.num_envs: {env.num_envs}"

        self._env = env
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def unwrapped(self) -> Environment:
        return self._env

    @property
    def env(self):
        return self._env

    def step(self, action):
        action = self._action_list_to_tensor(action)
        obs, rews, done, info = self._env.step(action)
        done = done[0].item()
        for i in range(self._env.n_agents):
            obs[i] = obs[i][0]
            rews[i] = rews[i][0].item()
        return obs, rews, done, info

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
        for i in range(self._env.n_agents):
            obs[i] = obs[i][0]
        return obs

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
