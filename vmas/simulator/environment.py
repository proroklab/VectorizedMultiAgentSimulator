#  Copyright (c) 2022.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
from enum import Enum
from typing import List, Optional, Tuple

import gym
import numpy as np
import torch
from gym import spaces
from ray import rllib
from ray.rllib.utils.typing import EnvActionType, EnvInfoDict, EnvObsType
from torch import Tensor

from vmas.simulator.core import Agent, TorchVectorizedObject
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import VIEWER_MIN_SIZE
from vmas.simulator.utils import X, Y, ALPHABET


# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class Environment(TorchVectorizedObject):
    metadata = {
        "render.modes": ["human", "rgb_array"],
        "runtime.vectorized": True,
    }

    def __init__(
        self,
        scenario: BaseScenario,
        num_envs: int = 32,
        device: str = "cpu",
        max_steps: int = None,
        continuous_actions: bool = True,
        **kwargs,
    ):

        self.scenario = scenario
        self.num_envs = num_envs
        TorchVectorizedObject.__init__(self, num_envs, torch.device(device))
        self.world = self.scenario.env_make_world(self.num_envs, self.device, **kwargs)

        self.agents = self.world.policy_agents
        self.n_agents = len(self.agents)
        self.max_steps = max_steps
        self.continuous_actions = continuous_actions

        self.reset()

        # configure spaces
        self.action_space = gym.spaces.Tuple(
            (
                (
                    spaces.Box(
                        low=np.array(
                            [-agent.u_range] * self.world.dim_p
                            + [0.0] * (self.world.dim_c if not agent.silent else 0)
                        ),
                        high=np.array(
                            [agent.u_range] * self.world.dim_p
                            + [1.0] * (self.world.dim_c if not agent.silent else 0)
                        ),
                        shape=(self.get_agent_action_size(agent),),
                        dtype=float,
                    )
                    if self.continuous_actions
                    else (
                        spaces.Discrete(self.world.dim_p * 2 + 1)
                        if self.world.dim_c == 0 or agent.silent
                        else spaces.MultiDiscrete(
                            [self.world.dim_p * 2 + 1, self.world.dim_c]
                        )
                    )
                )
                for agent in self.agents
            )
        )
        self.observation_space = gym.spaces.Tuple(
            (
                (
                    spaces.Box(
                        low=-float("inf"),
                        high=float("inf"),
                        shape=(len(self.scenario.observation(agent)[0]),),
                        dtype=float,
                    )
                )
                for agent in self.agents
            )
        )

        # rendering
        self.render_geoms_xform = None
        self.render_geoms = None
        self.viewer = None
        self.headless = None
        self.visible_display = None

    def reset(self, seed: int = None):
        """
        Resets the environment in a vectorized way
        Returns observations for all envs and agents
        """
        # Seed will be none at first call of this fun from constructor
        if seed is not None:
            self.seed(seed)
        # reset world
        self.scenario.env_reset_world_at(env_index=None)
        self.steps = torch.zeros(self.num_envs, device=self.device)
        # record observations for each agent
        obs = []
        for agent in self.agents:
            obs.append(self.scenario.observation(agent))
        return obs

    def reset_at(self, index: int = None):
        """
        Resets the environment at index
        Returns observations for all agents in that environment
        """
        self._check_batch_index(index)
        self.scenario.env_reset_world_at(index)
        self.steps[index] = 0
        obs = []
        for agent in self.agents:
            obs.append(self.scenario.observation(agent)[index].unsqueeze(0))
        return obs

    def seed(self, seed=None):
        if seed is None:
            seed = 0
        torch.manual_seed(seed)
        self.scenario.seed()
        return [seed]

    def step(self, actions: List):
        """Performs a vectorized step on all sub environments using `actions`.
        Args:
            actions: Is a list on len 'self.n_agents' of which each element is a torch.Tensor of shape
            '(self.num_envs, action_size_of_agent)'.
        Returns:
            obs: List on len 'self.n_agents' of which each element is a torch.Tensor
                 of shape '(self.num_envs, obs_size_of_agent)'
            rewards: List on len 'self.n_agents' of which each element is a torch.Tensor of shape '(self.num_envs)'
            dones: Tensor of len 'self.num_envs' of which each element is a bool
            infos : List on len 'self.n_agents' of which each element is a dictionary for which each key is a metric
                    and the value is a tensor of shape '(self.num_envs, metric_size_per_agent)'
        """
        assert (
            len(actions) == self.n_agents
        ), f"Expecting actions for {self.n_agents}, got {len(actions)} actions"
        for i in range(len(actions)):
            if not isinstance(actions[i], Tensor):
                actions[i] = torch.tensor(
                    actions[i], dtype=torch.float32, device=self.device
                )
            if len(actions[i].shape) == 1:
                actions[i].unsqueeze_(-1)
            assert (
                actions[i].shape[0] == self.num_envs
            ), f"Actions used in input of env must be of len {self.num_envs}, got {actions[i].shape[0]}"
            assert actions[i].shape[1] == self.get_agent_action_size(self.agents[i]), (
                f"Action for agent {self.agents[i].name} has shape {actions[i].shape[1]},"
                f" but should have shape {self.get_agent_action_size(self.agents[i])}"
            )

        # set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(actions[i], agent)
        # advance world state
        self.world.step()

        obs = []
        rewards = []
        infos = []
        for agent in self.agents:
            rewards.append(self.scenario.reward(agent).clone())
            obs.append(self.scenario.observation(agent).clone())
            # A dictionary per agent
            infos.append(self.scenario.info(agent))

        dones = self.scenario.done().clone()

        self.steps += 1
        if self.max_steps is not None:
            dones += self.steps >= self.max_steps

        # print("\nStep results in unwrapped environment")
        # print(
        #     f"Actions len (n_agents): {len(actions)}, "
        #     f"actions[0] shape (num_envs, agent 0 action shape): {actions[0].shape}, "
        #     f"actions[0][0] (action agent 0 env 0): {actions[0][0]}"
        # )
        # print(
        #     f"Obs len (n_agents): {len(obs)}, "
        #     f"obs[0] shape (num_envs, agent 0 obs shape): {obs[0].shape}, obs[0][0] (obs agent 0 env 0): {obs[0][0]}"
        # )
        # print(
        #     f"Rewards len (n_agents): {len(rewards)}, rewards[0] shape (num_envs, 1): {rewards[0].shape}, "
        #     f"rewards[0][0] (agent 0 env 0): {rewards[0][0]}"
        # )
        # print(f"Dones len (n_envs): {len(dones)}, dones[0] (done env 0): {dones[0]}")
        # print(f"Info len (n_agents): {len(infos)}, info[0] (infos agent 0): {infos[0]}")
        return obs, rewards, dones, infos

    def get_agent_action_size(self, agent: Agent):
        return (
            self.world.dim_p + (self.world.dim_c if not agent.silent else 0)
            if self.continuous_actions
            else 1 + (1 if not agent.silent else 0)
        )

    def _check_discrete_action(self, action: Tensor, low: int, high: int, type: str):
        assert torch.all(
            torch.any(
                torch.arange(low, high, device=self.device).repeat(self.num_envs)
                == action,
                dim=-1,
            )
        ), f"Discrete {type} actions are out of bounds, allowed int range [{low},{high})"

    # set env action for a particular agent
    def _set_action(self, action, agent):
        action = action.clone().to(self.device)
        agent.action.u = torch.zeros(
            self.batch_dim, self.world.dim_p, device=self.device, dtype=torch.float32
        )

        assert action.shape[1] == self.get_agent_action_size(agent), (
            f"Agent {agent.name} has wrong action size, got {action.shape[1]}, "
            f"expected {self.get_agent_action_size(agent)}"
        )

        if agent.movable:
            if self.continuous_actions:
                physical_action = action[:, : self.world.dim_p]
                assert not torch.any(
                    torch.abs(physical_action) > agent.u_range
                ), f"Physical actions of agent {agent.name} are out of its range {agent.u_range}"

                agent.action.u = physical_action.to(torch.float32)
            else:
                physical_action = action[:, :1]

                self._check_discrete_action(
                    physical_action,
                    low=0,
                    high=self.world.dim_p * 2 + 1,
                    type="physical",
                )

                arr1 = physical_action == 1
                arr2 = physical_action == 2
                arr3 = physical_action == 3
                arr4 = physical_action == 4

                disc_action_value = agent.u_range

                agent.action.u[:, X] -= disc_action_value * arr1.squeeze(-1)
                agent.action.u[:, X] += disc_action_value * arr2.squeeze(-1)
                agent.action.u[:, Y] -= disc_action_value * arr3.squeeze(-1)
                agent.action.u[:, Y] += disc_action_value * arr4.squeeze(-1)

            agent.action.u *= agent.u_multiplier
        if self.world.dim_c > 0 and not agent.silent:
            if not self.continuous_actions:
                comm_action = action[:, 1:]
                self._check_discrete_action(
                    comm_action, 0, self.world.dim_c, "communication"
                )
                comm_action = comm_action.long()
                agent.action.c = torch.zeros(
                    self.num_envs,
                    self.world.dim_c,
                    device=self.device,
                    dtype=torch.float32,
                )
                # Discrete to one-hot
                agent.action.c.scatter_(1, comm_action, 1)
            else:
                comm_action = action[:, self.world.dim_p :]
                assert not torch.any(comm_action > 1) and not torch.any(
                    comm_action < 0
                ), "Comm actions are out of range [0,1]"
                agent.action.c = comm_action

    def render(
        self,
        mode="human",
        env_index=0,
        agent_index_focus: int = None,
        visualize_when_rgb: bool = False,
    ):
        """
        Render function for environment using pyglet

        On servers use mode="rgb_array" and set
        ```
        export DISPLAY=':99.0'
        Xvfb :99 -screen 0 1400x900x24 > /dev/null 2>&1 &
        ```

        :param mode: One of human or rgb_array
        :param env_index: Index of the environment to render
        :param agent_index_focus: If specified the camera will stay on the agent with this index.
                                  If None, the camera will stay in the center and zoom out to contain all agents
        :param visualize_when_rgb: Also run human visualization when mode=="rgb_array"
        :return: Rgb array or None, depending on the mode
        """
        self._check_batch_index(env_index)
        assert (
            mode in self.metadata["render.modes"]
        ), f"Invalid mode {mode} received, allowed modes: {self.metadata['render.modes']}"
        if agent_index_focus is not None:
            assert 0 <= agent_index_focus < self.n_agents, (
                f"Agent focus in rendering should be a valid agent index"
                f" between 0 and {self.n_agents}, got {agent_index_focus}"
            )
        shared_viewer = agent_index_focus is None
        aspect_ratio = self.scenario.viewer_size[X] / self.scenario.viewer_size[Y]

        headless = mode == "rgb_array" and not visualize_when_rgb
        # First time rendering
        if self.visible_display is None:
            self.visible_display = not headless
            self.headless = headless
        # All other times headless should be the same
        else:
            assert self.visible_display is not headless

        # First time rendering
        if self.viewer is None:
            try:
                import pyglet
            except ImportError:
                raise ImportError(
                    "Cannot import pyg;et: you can install pyglet directly via 'pip install pyglet'."
                )

            try:
                pyglet.lib.load_library("EGL")
            except ImportError:
                self.headless = False
            pyglet.options["headless"] = self.headless

            self._init_rendering()

        # Render comm messages
        if self.world.dim_c > 0:
            idx = 0
            for agent in self.world.agents:
                if agent.silent:
                    continue
                assert (
                    agent.state.c is not None
                ), "Agent has no comm state but it should"
                if self.continuous_actions:
                    word = (
                        "["
                        + ",".join([f"{comm:.2f}" for comm in agent.state.c[env_index]])
                        + "]"
                    )
                else:
                    word = ALPHABET[torch.argmax(agent.state.c[env_index]).item()]

                message = agent.name + " sends " + word + "   "
                self.viewer.text_lines[idx].set_text(message)
                idx += 1

        if aspect_ratio < 1:
            cam_range = torch.tensor(
                [VIEWER_MIN_SIZE, VIEWER_MIN_SIZE / aspect_ratio], device=self.device
            )
        else:
            cam_range = torch.tensor(
                [VIEWER_MIN_SIZE * aspect_ratio, VIEWER_MIN_SIZE], device=self.device
            )

        if shared_viewer:
            # zoom out to fit everyone
            all_poses = torch.stack(
                [agent.state.pos[env_index] for agent in self.world.agents], dim=0
            )
            viewer_size_fit = (
                torch.stack(
                    [
                        torch.max(torch.abs(all_poses[:, X])),
                        torch.max(torch.abs(all_poses[:, Y])),
                    ]
                )
                + VIEWER_MIN_SIZE
                - 1
            )

            viewer_size = torch.maximum(
                viewer_size_fit / cam_range, torch.tensor(1, device=self.device)
            )
            cam_range *= torch.max(viewer_size)
            self.viewer.set_bounds(
                -cam_range[X],
                cam_range[X],
                -cam_range[Y],
                cam_range[Y],
            )
        else:
            # update bounds to center around agent
            pos = self.agents[agent_index_focus].state.pos[env_index]
            self.viewer.set_bounds(
                pos[X] - cam_range[X],
                pos[X] + cam_range[X],
                pos[Y] - cam_range[Y],
                pos[Y] + cam_range[Y],
            )

        for geom in self.scenario.extra_render(env_index):
            self.viewer.add_onetime(geom)

        for joint in self.world.joints:
            [
                self.viewer.add_onetime(geom)
                for geom in joint.render(env_index=env_index)
            ]

        for entity in self.world.entities:
            [
                self.viewer.add_onetime(geom)
                for geom in entity.render(env_index=env_index)
            ]

        # render to display or array
        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def _init_rendering(self):
        from vmas.simulator import rendering

        self.viewer = rendering.Viewer(
            *self.scenario.viewer_size, visible=self.visible_display
        )

        idx = 0
        if self.world.dim_c > 0:
            self.viewer.text_lines = []
            for agent in self.world.agents:
                if not agent.silent:
                    text_line = rendering.TextLine(self.viewer.window, idx)
                    self.viewer.text_lines.append(text_line)
                    idx += 1


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
                total_env_rew += rews[i][j].item()
                env_infos["rewards"].update({i: rews[i][j].item()})
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
        obs = self._env.reset_at(index=0)
        for i in range(self._env.n_agents):
            obs[i] = obs[i]
        return obs

    def render(
        self,
        mode="human",
        agent_index_focus: Optional[int] = None,
        visualize_when_rgb: bool = False,
    ) -> Optional[np.ndarray]:

        return self._env.render(
            mode=mode,
            env_index=0,
            agent_index_focus=agent_index_focus,
            visualize_when_rgb=visualize_when_rgb,
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


class Wrapper(Enum):
    RLLIB = 0
    GYM = 1

    def get_env(self, env: Environment):
        if self is self.RLLIB:
            return VectorEnvWrapper(env)
        elif self is self.GYM:
            return GymWrapper(env)
