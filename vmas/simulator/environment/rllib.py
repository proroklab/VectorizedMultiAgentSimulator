#  Copyright (c) 2022-2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
from typing import List, Optional, Tuple

import numpy as np
import torch
from ray import rllib
from ray.rllib.utils.typing import EnvActionType, EnvInfoDict, EnvObsType
from vmas.simulator.environment.environment import Environment


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
        obs = self._env.reset()
        return self._tensor_to_list(obs, self.num_envs)

    def reset_at(self, index: Optional[int] = None) -> EnvObsType:
        assert index is not None
        obs = self._env.reset_at(index)

        for i in range(self._env.n_agents):
            obs[i] = obs[i][index].unsqueeze(0)

        return self._tensor_to_list(obs, 1)[0]

    def vector_step(
        self, actions: List[EnvActionType]
    ) -> Tuple[List[EnvObsType], List[float], List[bool], List[EnvInfoDict]]:
        # saved_actions = actions
        actions = self._action_list_to_tensor(actions)
        obs, rews, dones, infos = self._env.step(actions)

        dones = dones.tolist()

        total_rews = []
        total_infos = []
        obs_list = []
        for j in range(self.num_envs):
            obs_list.append([])
            env_infos = {"rewards": {}}
            total_env_rew = 0.0
            for i, agent in enumerate(self._env.agents):
                obs_list[j].append(obs[i][j].cpu().numpy())
                total_env_rew += np.float32(rews[i][j].item())
                env_infos["rewards"].update({i: np.float32(rews[i][j].item())})
                env_infos.update(
                    {agent.name: {k: v[j].tolist() for k, v in infos[i].items()}}
                )
            total_infos.append(env_infos)
            total_rews.append(total_env_rew / self._env.n_agents)  # Average reward

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
        return obs_list, total_rews, dones, total_infos

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

    def _tensor_to_list(self, list_in: List, num_envs: int) -> List:
        assert (
            len(list_in) == self._env.n_agents
        ), f"Tensor used in output of env must be of len {self._env.n_agents}, got {len(list_in)}"
        assert list_in[0].shape[0] == num_envs, (
            f"Input tensor for each agent should have"
            f" vector dim {num_envs}, but got {list_in[0].shape[0]}"
        )
        list_out = []
        for j in range(num_envs):
            list_per_env = []
            for i in range(self._env.n_agents):
                list_per_env.append(list_in[i][j].cpu().numpy())
            list_out.append(list_per_env)
        return list_out
