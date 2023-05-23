#  Copyright (c) 2022-2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
from numpy import ndarray
from ray import rllib
from ray.rllib.utils.typing import EnvActionType, EnvInfoDict, EnvObsType
from torch import Tensor
from vmas.simulator.environment.environment import Environment
from vmas.simulator.utils import (
    OBS_TYPE,
    REWARD_TYPE,
    INFO_TYPE,
    TorchUtils,
)


class VectorEnvWrapper(rllib.VectorEnv):
    """
    Vector environment wrapper for rllib
    """

    def __init__(
        self,
        env: Environment,
    ):
        self._env = env
        super().__init__(
            observation_space=self._env.observation_space,
            action_space=self._env.action_space,
            num_envs=self._env.num_envs,
        )

    @property
    def env(self):
        return self._env

    def vector_reset(self) -> List[EnvObsType]:
        obs = TorchUtils.to_numpy(self._env.reset())
        return self._read_data(obs)[0]

    def reset_at(self, index: Optional[int] = None) -> EnvObsType:
        assert index is not None
        obs = self._env.reset_at(index)
        return self._read_data(obs, env_index=index)[0]

    def vector_step(
        self, actions: List[EnvActionType]
    ) -> Tuple[List[EnvObsType], List[float], List[bool], List[EnvInfoDict]]:
        # saved_actions = actions
        actions = self._action_list_to_tensor(actions)
        obs, rews, dones, infos = TorchUtils.to_numpy(self._env.step(actions))

        obs, infos, rews = self._read_data(obs, infos, rews)

        # print("\nStep results in wrapped environment")
        # print(
        #     f"Actions len (num_envs): {len(saved_actions)}, len actions[0] (n_agents): {len(saved_actions[0])},"
        #     f" actions[0][0] (action agent 0 env 0): {saved_actions[0][0]}"
        # )
        # print(
        #     f"Obs len (num_envs): {len(obs_list)}, len obs[0] (n_agents): {len(obs_list[0])},"
        #     f" obs[0][0] (obs agent 0 env 0): {obs_list[0][0]}"
        # )
        # print(
        #     f"Total rews len (num_envs): {len(total_rews)}, total_rews[0] (total rew env 0): {total_rews[0]}"
        # )
        # print(f"Dones len (num_envs): {len(dones)}, dones[0] (done env 0): {dones[0]}")
        # print(
        #     f"Total infos len (num_envs): {len(total_infos)}, total_infos[0] (infos env 0): {total_infos[0]}"
        # )
        return obs, rews, dones, infos

    def seed(self, seed=None):
        return self._env.seed(seed)

    def try_render_at(
        self,
        index: Optional[int] = None,
        mode="human",
        agent_index_focus: Optional[int] = None,
        visualize_when_rgb: bool = False,
        **kwargs,
    ) -> Optional[np.ndarray]:
        """
        Render function for environment using pyglet

        On servers use mode="rgb_array" and set
        ```
        export DISPLAY=':99.0'
        Xvfb :99 -screen 0 1400x900x24 > /dev/null 2>&1 &
        ```

        :param mode: One of human or rgb_array
        :param index: Index of the environment to render
        :param agent_index_focus: If specified the camera will stay on the agent with this index.
                                  If None, the camera will stay in the center and zoom out to contain all agents
        :param visualize_when_rgb: Also run human visualization when mode=="rgb_array"
        :return: Rgb array or None, depending on the mode
        """
        if index is None:
            index = 0
        return self._env.render(
            mode=mode,
            env_index=index,
            agent_index_focus=agent_index_focus,
            visualize_when_rgb=visualize_when_rgb,
            **kwargs,
        )

    def get_sub_environments(self) -> List[Environment]:
        return [self._env]

    def _action_list_to_tensor(self, list_in: List) -> List:
        if len(list_in) == self.num_envs:
            actions = []
            for agent in self._env.agents:
                actions.append(
                    torch.zeros(
                        self.num_envs,
                        self._env.get_agent_action_size(agent),
                        device=self._env.device,
                        dtype=torch.float32,
                    )
                )
            for j in range(self.num_envs):
                assert (
                    len(list_in[j]) == self._env.n_agents
                ), f"Expecting actions for {self._env.n_agents} agents, got {len(list_in[j])} actions"
                for i in range(self._env.n_agents):
                    act = torch.tensor(
                        list_in[j][i], dtype=torch.float32, device=self._env.device
                    )
                    if len(act.shape) == 0:
                        assert (
                            self._env.get_agent_action_size(self._env.agents[i]) == 1
                        ), f"Action of agent {i} in env {j} is supposed to be an scalar int"
                    else:
                        assert len(act.shape) == 1 and act.shape[
                            0
                        ] == self._env.get_agent_action_size(self._env.agents[i]), (
                            f"Action of agent {i} in env {j} hase wrong shape: "
                            f"expected {self._env.get_agent_action_size(self._env.agents[i])}, got {act.shape[0]}"
                        )
                    actions[i][j] = act
            return actions
        else:
            assert False, "Input action is not in correct format"

    def _read_data(
        self,
        obs: Optional[OBS_TYPE],
        info: Optional[INFO_TYPE] = None,
        reward: Optional[REWARD_TYPE] = None,
        env_index: Optional[int] = None,
    ):
        if env_index is None:
            obs_list = []
            if info:
                info_list = []
            if reward:
                rew_list = []

            for env_index in range(self.num_envs):
                (
                    observations_processed,
                    info_processed,
                    reward_processed,
                ) = self._get_data_at_env_index(env_index, obs, info, reward)
                obs_list.append(observations_processed)
                if info:
                    info_list.append(info_processed)
                if reward:
                    rew_list.append(reward_processed)

            return obs_list, info_list if info else None, rew_list if reward else None
        else:
            return self._get_data_at_env_index(env_index, obs, info, reward)

    def _get_data_at_env_index(
        self,
        env_index: int,
        obs: Optional[OBS_TYPE],
        info: Optional[INFO_TYPE] = None,
        reward: Optional[REWARD_TYPE] = None,
    ):
        assert len(obs) == self._env.n_agents
        total_rew = 0.0
        if info:
            new_info = {"rewards": {}}
        if isinstance(obs, Dict):
            new_obs = {}
            for agent_index, agent in enumerate(self._env.agents):
                new_obs[agent.name] = self._get_agent_data_at_env_index(
                    env_index, obs[agent.name]
                )
                if info:
                    new_info[agent.name] = self._get_agent_data_at_env_index(
                        env_index, info[agent.name]
                    )
                if reward:
                    agent_rew = self._get_agent_data_at_env_index(
                        env_index, reward[agent.name]
                    )
                    new_info["rewards"].update({agent_index: agent_rew})
                    total_rew += agent_rew

        elif isinstance(obs, List):
            new_obs = []
            for agent_index, agent in enumerate(self._env.agents):
                new_obs.append(
                    self._get_agent_data_at_env_index(env_index, obs[agent_index])
                )
                if info:
                    new_info[agent.name] = self._get_agent_data_at_env_index(
                        env_index, info[agent_index]
                    )
                if reward:
                    agent_rew = self._get_agent_data_at_env_index(
                        env_index, reward[agent_index]
                    )
                    new_info["rewards"].update({agent_index: agent_rew})
                    total_rew += agent_rew

        else:
            raise ValueError(f"Unsupported obs type {obs}")

        return (
            new_obs,
            new_info if info else None,
            total_rew / self._env.n_agents if reward else None,
        )

    def _get_agent_data_at_env_index(
        self,
        env_index: int,
        agent_data,
    ):
        if isinstance(agent_data, (ndarray, Tensor)):
            assert agent_data.shape[0] == self._env.num_envs
            if len(agent_data.shape) == 1 or (
                len(agent_data.shape) == 2 and agent_data.shape[1] == 1
            ):
                return agent_data[env_index].item()
            elif isinstance(agent_data, Tensor):
                return agent_data[env_index].cpu().detach().numpy()
            else:
                return agent_data[env_index]
        elif isinstance(agent_data, Dict):
            return {
                key: self._get_agent_data_at_env_index(env_index, value)
                for key, value in agent_data.items()
            }
        else:
            raise ValueError(f"Unsupported data type {agent_data}")
