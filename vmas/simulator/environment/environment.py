#  Copyright (c) ProrokLab.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
import contextlib
import math
import random
from ctypes import byref
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from gym import spaces
from torch import Tensor

import vmas.simulator.utils
from vmas.simulator.core import Agent, TorchVectorizedObject
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import (
    AGENT_OBS_TYPE,
    ALPHABET,
    DEVICE_TYPING,
    override,
    TorchUtils,
    X,
    Y,
)


@contextlib.contextmanager
def local_seed(vmas_random_state):
    torch_state = torch.random.get_rng_state()
    np_state = np.random.get_state()
    py_state = random.getstate()

    torch.random.set_rng_state(vmas_random_state[0])
    np.random.set_state(vmas_random_state[1])
    random.setstate(vmas_random_state[2])
    yield
    vmas_random_state[0] = torch.random.get_rng_state()
    vmas_random_state[1] = np.random.get_state()
    vmas_random_state[2] = random.getstate()

    torch.random.set_rng_state(torch_state)
    np.random.set_state(np_state)
    random.setstate(py_state)


class Environment(TorchVectorizedObject):
    """
    The VMAS environment
    """

    metadata = {
        "render.modes": ["human", "rgb_array"],
        "runtime.vectorized": True,
    }
    vmas_random_state = [
        torch.random.get_rng_state(),
        np.random.get_state(),
        random.getstate(),
    ]

    @local_seed(vmas_random_state)
    def __init__(
        self,
        scenario: BaseScenario,
        num_envs: int = 32,
        device: DEVICE_TYPING = "cpu",
        max_steps: Optional[int] = None,
        continuous_actions: bool = True,
        seed: Optional[int] = None,
        dict_spaces: bool = False,
        multidiscrete_actions: bool = False,
        clamp_actions: bool = False,
        grad_enabled: bool = False,
        terminated_truncated: bool = False,
        **kwargs,
    ):
        if multidiscrete_actions:
            assert (
                not continuous_actions
            ), "When asking for multidiscrete_actions, make sure continuous_actions=False"

        self.scenario = scenario
        self.num_envs = num_envs
        TorchVectorizedObject.__init__(self, num_envs, torch.device(device))
        self.world = self.scenario.env_make_world(self.num_envs, self.device, **kwargs)

        self.agents = self.world.policy_agents
        self.n_agents = len(self.agents)
        self.max_steps = max_steps
        self.continuous_actions = continuous_actions
        self.dict_spaces = dict_spaces
        self.clamp_action = clamp_actions
        self.grad_enabled = grad_enabled
        self.terminated_truncated = terminated_truncated

        observations = self._reset(seed=seed)

        # configure spaces
        self.multidiscrete_actions = multidiscrete_actions
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space(observations)

        # rendering
        self.viewer = None
        self.headless = None
        self.visible_display = None
        self.text_lines = None

    @local_seed(vmas_random_state)
    def reset(
        self,
        seed: Optional[int] = None,
        return_observations: bool = True,
        return_info: bool = False,
        return_dones: bool = False,
    ):
        """
        Resets the environment in a vectorized way
        Returns observations for all envs and agents
        """
        return self._reset(
            seed=seed,
            return_observations=return_observations,
            return_info=return_info,
            return_dones=return_dones,
        )

    @local_seed(vmas_random_state)
    def reset_at(
        self,
        index: int,
        return_observations: bool = True,
        return_info: bool = False,
        return_dones: bool = False,
    ):
        """
        Resets the environment at index
        Returns observations for all agents in that environment
        """
        return self._reset_at(
            index=index,
            return_observations=return_observations,
            return_info=return_info,
            return_dones=return_dones,
        )

    @local_seed(vmas_random_state)
    def get_from_scenario(
        self,
        get_observations: bool,
        get_rewards: bool,
        get_infos: bool,
        get_dones: bool,
        dict_agent_names: Optional[bool] = None,
    ):
        """
        Get the environment data from the scenario

        Args:
            get_observations (bool): whether to return the observations
            get_rewards (bool): whether to return the rewards
            get_infos (bool): whether to return the infos
            get_dones (bool): whether to return the dones
            dict_agent_names (bool, optional): whether to return the information in a dictionary with agent names as keys
                or in a list

        Returns:
            The agents' data

        """
        return self._get_from_scenario(
            get_observations=get_observations,
            get_rewards=get_rewards,
            get_infos=get_infos,
            get_dones=get_dones,
            dict_agent_names=dict_agent_names,
        )

    @local_seed(vmas_random_state)
    def seed(self, seed=None):
        """
        Sets the seed for the environment
        Args:
            seed (int, optional): Seed for the environment. Defaults to None.

        """
        return self._seed(seed=seed)

    @local_seed(vmas_random_state)
    def done(self):
        """
        Get the done flags for the scenario.

        Returns:
            Either terminated, truncated (if self.terminated_truncated==True) or terminated + truncated (if self.terminated_truncated==False)

        """
        return self._done()

    def _reset(
        self,
        seed: Optional[int] = None,
        return_observations: bool = True,
        return_info: bool = False,
        return_dones: bool = False,
    ):
        """
        Resets the environment in a vectorized way
        Returns observations for all envs and agents
        """

        if seed is not None:
            self._seed(seed)
        # reset world
        self.scenario.env_reset_world_at(env_index=None)
        self.steps = torch.zeros(self.num_envs, device=self.device)

        result = self._get_from_scenario(
            get_observations=return_observations,
            get_infos=return_info,
            get_rewards=False,
            get_dones=return_dones,
        )
        return result[0] if result and len(result) == 1 else result

    def _reset_at(
        self,
        index: int,
        return_observations: bool = True,
        return_info: bool = False,
        return_dones: bool = False,
    ):
        """
        Resets the environment at index
        Returns observations for all agents in that environment
        """
        self._check_batch_index(index)
        self.scenario.env_reset_world_at(index)
        self.steps[index] = 0

        result = self._get_from_scenario(
            get_observations=return_observations,
            get_infos=return_info,
            get_rewards=False,
            get_dones=return_dones,
        )

        return result[0] if result and len(result) == 1 else result

    def _get_from_scenario(
        self,
        get_observations: bool,
        get_rewards: bool,
        get_infos: bool,
        get_dones: bool,
        dict_agent_names: Optional[bool] = None,
    ):
        if not get_infos and not get_dones and not get_rewards and not get_observations:
            return
        if dict_agent_names is None:
            dict_agent_names = self.dict_spaces

        obs = rewards = infos = terminated = truncated = dones = None

        if get_observations:
            obs = {} if dict_agent_names else []
        if get_rewards:
            rewards = {} if dict_agent_names else []
        if get_infos:
            infos = {} if dict_agent_names else []

        if get_rewards:
            for agent in self.agents:
                reward = self.scenario.reward(agent).clone()
                if dict_agent_names:
                    rewards.update({agent.name: reward})
                else:
                    rewards.append(reward)
        if get_observations:
            for agent in self.agents:
                observation = TorchUtils.recursive_clone(
                    self.scenario.observation(agent)
                )
                if dict_agent_names:
                    obs.update({agent.name: observation})
                else:
                    obs.append(observation)
        if get_infos:
            for agent in self.agents:
                info = TorchUtils.recursive_clone(self.scenario.info(agent))
                if dict_agent_names:
                    infos.update({agent.name: info})
                else:
                    infos.append(info)

        if self.terminated_truncated:
            if get_dones:
                terminated, truncated = self._done()
            result = [obs, rewards, terminated, truncated, infos]
        else:
            if get_dones:
                dones = self._done()
            result = [obs, rewards, dones, infos]

        return [data for data in result if data is not None]

    def _seed(self, seed=None):
        """
        Sets the seed for the environment
        Args:
            seed (int, optional): Seed for the environment. Defaults to None.

        """
        if seed is None:
            seed = 0
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        return [seed]

    @local_seed(vmas_random_state)
    def step(self, actions: Union[List, Dict]):
        """Performs a vectorized step on all sub environments using `actions`.

        Args:
            actions: Is a list on len 'self.n_agents' of which each element is a torch.Tensor of shape '(self.num_envs, action_size_of_agent)'.

        Returns:
            obs: List on len 'self.n_agents' of which each element is a torch.Tensor of shape '(self.num_envs, obs_size_of_agent)'
            rewards: List on len 'self.n_agents' of which each element is a torch.Tensor of shape '(self.num_envs)'
            dones: Tensor of len 'self.num_envs' of which each element is a bool
            infos: List on len 'self.n_agents' of which each element is a dictionary for which each key is a metric and the value is a tensor of shape '(self.num_envs, metric_size_per_agent)'

        Examples:
            >>> import vmas
            >>> env = vmas.make_env(
            ...     scenario="waterfall",  # can be scenario name or BaseScenario class
            ...     num_envs=32,
            ...     device="cpu",  # Or "cuda" for GPU
            ...     continuous_actions=True,
            ...     max_steps=None,  # Defines the horizon. None is infinite horizon.
            ...     seed=None,  # Seed of the environment
            ...     n_agents=3,  # Additional arguments you want to pass to the scenario
            ... )
            >>> obs = env.reset()
            >>> for _ in range(10):
            ...     obs, rews, dones, info = env.step(env.get_random_actions())

        """
        if isinstance(actions, Dict):
            actions_dict = actions
            actions = []
            for agent in self.agents:
                try:
                    actions.append(actions_dict[agent.name])
                except KeyError:
                    raise AssertionError(
                        f"Agent '{agent.name}' not contained in action dict"
                    )
            assert (
                len(actions_dict) == self.n_agents
            ), f"Expecting actions for {self.n_agents}, got {len(actions_dict)} actions"

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
        # Scenarios can define a custom action processor. This step takes care also of scripted agents automatically
        for agent in self.world.agents:
            self.scenario.env_process_action(agent)

        # advance world state
        self.scenario.pre_step()
        self.world.step()
        self.scenario.post_step()

        self.steps += 1

        return self._get_from_scenario(
            get_observations=True,
            get_infos=True,
            get_rewards=True,
            get_dones=True,
        )

    def _done(self):
        """
        Get the done flags for the scenario.

        Returns:
            Either terminated, truncated (if self.terminated_truncated==True) or terminated + truncated (if self.terminated_truncated==False)

        """
        terminated = self.scenario.done().clone()

        if self.max_steps is not None:
            truncated = self.steps >= self.max_steps
        else:
            truncated = None

        if self.terminated_truncated:
            if truncated is None:
                truncated = torch.zeros_like(terminated)
            return terminated, truncated
        else:
            if truncated is None:
                return terminated
            return terminated + truncated

    def get_action_space(self):
        if not self.dict_spaces:
            return spaces.Tuple(
                [self.get_agent_action_space(agent) for agent in self.agents]
            )
        else:
            return spaces.Dict(
                {
                    agent.name: self.get_agent_action_space(agent)
                    for agent in self.agents
                }
            )

    def get_observation_space(self, observations: Union[List, Dict]):
        if not self.dict_spaces:
            return spaces.Tuple(
                [
                    self.get_agent_observation_space(agent, observations[i])
                    for i, agent in enumerate(self.agents)
                ]
            )
        else:
            return spaces.Dict(
                {
                    agent.name: self.get_agent_observation_space(
                        agent, observations[agent.name]
                    )
                    for agent in self.agents
                }
            )

    def get_agent_action_size(self, agent: Agent):
        if self.continuous_actions:
            return agent.action.action_size + (
                self.world.dim_c if not agent.silent else 0
            )
        elif self.multidiscrete_actions:
            return agent.action_size + (
                1 if not agent.silent and self.world.dim_c != 0 else 0
            )
        else:
            return 1

    def get_agent_action_space(self, agent: Agent):
        if self.continuous_actions:
            return spaces.Box(
                low=np.array(
                    (-agent.action.u_range_tensor).tolist()
                    + [0] * (self.world.dim_c if not agent.silent else 0),
                    dtype=np.float32,
                ),
                high=np.array(
                    agent.action.u_range_tensor.tolist()
                    + [1] * (self.world.dim_c if not agent.silent else 0),
                    dtype=np.float32,
                ),
                shape=(self.get_agent_action_size(agent),),
                dtype=np.float32,
            )
        elif self.multidiscrete_actions:
            actions = agent.discrete_action_nvec + (
                [self.world.dim_c] if not agent.silent and self.world.dim_c != 0 else []
            )
            return spaces.MultiDiscrete(actions)
        else:
            return spaces.Discrete(
                math.prod(agent.discrete_action_nvec)
                * (
                    self.world.dim_c
                    if not agent.silent and self.world.dim_c != 0
                    else 1
                )
            )

    def get_agent_observation_space(self, agent: Agent, obs: AGENT_OBS_TYPE):
        if isinstance(obs, Tensor):
            return spaces.Box(
                low=-np.float32("inf"),
                high=np.float32("inf"),
                shape=obs.shape[1:],
                dtype=np.float32,
            )
        elif isinstance(obs, Dict):
            return spaces.Dict(
                {
                    key: self.get_agent_observation_space(agent, value)
                    for key, value in obs.items()
                }
            )
        else:
            raise NotImplementedError(
                f"Invalid type of observation {obs} for agent {agent.name}"
            )

    @local_seed(vmas_random_state)
    def get_random_action(self, agent: Agent) -> torch.Tensor:
        """Returns a random action for the given agent.

        Args:
            agent (Agent): The agent to get the action for

        Returns:
            torch.tensor: the random actions tensor with shape ``(agent.batch_dim, agent.action_size)``

        """
        if self.continuous_actions:
            actions = []
            for action_index in range(agent.action_size):
                actions.append(
                    torch.zeros(
                        agent.batch_dim,
                        device=agent.device,
                        dtype=torch.float32,
                    ).uniform_(
                        -agent.action.u_range_tensor[action_index],
                        agent.action.u_range_tensor[action_index],
                    )
                )
            if self.world.dim_c != 0 and not agent.silent:
                # If the agent needs to communicate
                for _ in range(self.world.dim_c):
                    actions.append(
                        torch.zeros(
                            agent.batch_dim,
                            device=agent.device,
                            dtype=torch.float32,
                        ).uniform_(
                            0,
                            1,
                        )
                    )
            action = torch.stack(actions, dim=-1)
        else:
            action_space = self.get_agent_action_space(agent)
            if self.multidiscrete_actions:
                actions = [
                    torch.randint(
                        low=0,
                        high=action_space.nvec[action_index],
                        size=(agent.batch_dim,),
                        device=agent.device,
                    )
                    for action_index in range(action_space.shape[0])
                ]
                action = torch.stack(actions, dim=-1)
            else:
                action = torch.randint(
                    low=0,
                    high=action_space.n,
                    size=(agent.batch_dim,),
                    device=agent.device,
                )
        return action

    def get_random_actions(self) -> Sequence[torch.Tensor]:
        """Returns random actions for all agents that you can feed to :meth:`step`

        Returns:
            Sequence[torch.tensor]: the random actions for the agents

        Examples:
            >>> import vmas
            >>> env = vmas.make_env(
            ...     scenario="waterfall",  # can be scenario name or BaseScenario class
            ...     num_envs=32,
            ...     device="cpu",  # Or "cuda" for GPU
            ...     continuous_actions=True,
            ...     max_steps=None,  # Defines the horizon. None is infinite horizon.
            ...     seed=None,  # Seed of the environment
            ...     n_agents=3,  # Additional arguments you want to pass to the scenario
            ... )
            >>> obs = env.reset()
            >>> for _ in range(10):
            ...     obs, rews, dones, info = env.step(env.get_random_actions())

        """
        return [self.get_random_action(agent) for agent in self.agents]

    def _check_discrete_action(self, action: Tensor, low: int, high: int, type: str):
        assert torch.all(
            (action >= torch.tensor(low, device=self.device))
            * (action < torch.tensor(high, device=self.device))
        ), f"Discrete {type} actions are out of bounds, allowed int range [{low},{high})"

    # set env action for a particular agent
    def _set_action(self, action, agent):
        action = action.clone()
        if not self.grad_enabled:
            action = action.detach()
        action = action.to(self.device)
        assert not action.isnan().any()
        agent.action.u = torch.zeros(
            self.batch_dim,
            agent.action_size,
            device=self.device,
            dtype=torch.float32,
        )

        assert action.shape[1] == self.get_agent_action_size(agent), (
            f"Agent {agent.name} has wrong action size, got {action.shape[1]}, "
            f"expected {self.get_agent_action_size(agent)}"
        )
        if self.clamp_action and self.continuous_actions:
            physical_action = action[..., : agent.action_size]
            a_range = agent.action.u_range_tensor.unsqueeze(0).expand(
                physical_action.shape
            )
            physical_action = physical_action.clamp(-a_range, a_range)

            if self.world.dim_c > 0 and not agent.silent:  # If comms
                comm_action = action[..., agent.action_size :]
                action = torch.cat([physical_action, comm_action.clamp(0, 1)], dim=-1)
            else:
                action = physical_action

        action_index = 0

        if self.continuous_actions:
            physical_action = action[:, action_index : action_index + agent.action_size]
            action_index += self.world.dim_p
            assert not torch.any(
                torch.abs(physical_action) > agent.action.u_range_tensor
            ), f"Physical actions of agent {agent.name} are out of its range {agent.u_range}"

            agent.action.u = physical_action.to(torch.float32)

        else:
            if not self.multidiscrete_actions:
                # This bit of code translates the discrete action (taken from a space that
                # is the cartesian product of all action spaces) into a multi discrete action.
                # This is done by iteratively taking the modulo of the action and dividing by the
                # number of actions in the current action space, which treats the action as if
                # it was the "flat index" of the multi-discrete actions. E.g. if we have
                # nvec = [3,2], action 0 corresponds to the actions [0,0],
                # action 1 corresponds to the action [0,1], action 2 corresponds
                # to the action [1,0], action 3 corresponds to the action [1,1], etc.
                flat_action = action.squeeze(-1)
                actions = []
                nvec = list(agent.discrete_action_nvec) + (
                    [self.world.dim_c]
                    if not agent.silent and self.world.dim_c != 0
                    else []
                )
                for i in range(len(nvec)):
                    n = math.prod(nvec[i + 1 :])
                    actions.append(flat_action // n)
                    flat_action = flat_action % n
                action = torch.stack(actions, dim=-1)

            # Now we have an action with shape [n_envs, action_size+comms_actions]
            for n in agent.discrete_action_nvec:
                physical_action = action[:, action_index]
                self._check_discrete_action(
                    physical_action.unsqueeze(-1),
                    low=0,
                    high=n,
                    type="physical",
                )
                u_max = agent.action.u_range_tensor[action_index]
                # For odd n we want the first action to always map to u=0, so
                # we swap 0 values with the middle value, and shift the first
                # half of the remaining values by -1.
                if n % 2 != 0:
                    stay = physical_action == 0
                    decrement = (physical_action > 0) & (physical_action <= n // 2)
                    physical_action[stay] = n // 2
                    physical_action[decrement] -= 1
                # We know u must be in [-u_max, u_max], and we know action is
                # in [0, n-1]. Conversion steps: [0, n-1] -> [0, 1] -> [0, 2*u_max] -> [-u_max, u_max]
                # E.g. action 0 -> -u_max, action n-1 -> u_max, action 1 -> -u_max + 2*u_max/(n-1)
                agent.action.u[:, action_index] = (physical_action / (n - 1)) * (
                    2 * u_max
                ) - u_max

                action_index += 1

        agent.action.u *= agent.action.u_multiplier_tensor

        if agent.action.u_noise > 0:
            noise = (
                torch.randn(
                    *agent.action.u.shape,
                    device=self.device,
                    dtype=torch.float32,
                )
                * agent.u_noise
            )
            agent.action.u += noise
        if self.world.dim_c > 0 and not agent.silent:
            if not self.continuous_actions:
                comm_action = action[:, action_index:]
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
                comm_action = action[:, action_index:]
                assert not torch.any(comm_action > 1) and not torch.any(
                    comm_action < 0
                ), "Comm actions are out of range [0,1]"
                agent.action.c = comm_action
            if agent.c_noise > 0:
                noise = (
                    torch.randn(
                        *agent.action.c.shape,
                        device=self.device,
                        dtype=torch.float32,
                    )
                    * agent.c_noise
                )
                agent.action.c += noise

    @local_seed(vmas_random_state)
    def render(
        self,
        mode="human",
        env_index=0,
        agent_index_focus: int = None,
        visualize_when_rgb: bool = False,
        plot_position_function: Callable = None,
        plot_position_function_precision: float = 0.01,
        plot_position_function_range: Optional[
            Union[
                float,
                Tuple[float, float],
                Tuple[Tuple[float, float], Tuple[float, float]],
            ]
        ] = None,
        plot_position_function_cmap_range: Optional[Tuple[float, float]] = None,
        plot_position_function_cmap_alpha: Optional[float] = 1.0,
        plot_position_function_cmap_name: Optional[str] = "viridis",
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
        :param agent_index_focus: If specified the camera will stay on the agent with this index. If None, the camera will stay in the center and zoom out to contain all agents
        :param visualize_when_rgb: Also run human visualization when mode=="rgb_array"
        :param plot_position_function: A function to plot under the rendering.
        The function takes a numpy array with shape (n_points, 2), which represents a set of x,y values to evaluate f over and plot it
        It should output either an array with shape (n_points, 1) which will be plotted as a colormap
        or an array with shape (n_points, 4), which will be plotted as RGBA values
        :param plot_position_function_precision: The precision to use for plotting the function
        :param plot_position_function_range: The position range to plot the function in.
        If float, the range for x and y is (-function_range, function_range)
        If Tuple[float, float], the range for x is (-function_range[0], function_range[0]) and y is (-function_range[1], function_range[1])
        If Tuple[Tuple[float, float], Tuple[float, float]], the first tuple is the x range and the second tuple is the y range
        :param plot_position_function_cmap_range: The range of the cmap in case plot_position_function outputs a single value
        :param plot_position_function_cmap_alpha: The alpha of the cmap in case plot_position_function outputs a single value
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
                # Try to use EGL
                pyglet.lib.load_library("EGL")

                # Only if we have GPUs
                from pyglet.libs.egl import egl, eglext

                num_devices = egl.EGLint()
                eglext.eglQueryDevicesEXT(0, None, byref(num_devices))
                assert num_devices.value > 0

            except (ImportError, AssertionError):
                self.headless = False
            pyglet.options["headless"] = self.headless

            self._init_rendering()

        if self.scenario.viewer_zoom <= 0:
            raise ValueError("Scenario viewer zoom must be > 0")
        zoom = self.scenario.viewer_zoom

        if aspect_ratio < 1:
            cam_range = torch.tensor([zoom, zoom / aspect_ratio], device=self.device)
        else:
            cam_range = torch.tensor([zoom * aspect_ratio, zoom], device=self.device)

        if shared_viewer:
            # zoom out to fit everyone
            all_poses = torch.stack(
                [agent.state.pos[env_index] for agent in self.world.agents],
                dim=0,
            )
            max_agent_radius = max(
                [agent.shape.circumscribed_radius() for agent in self.world.agents]
            )
            viewer_size_fit = (
                torch.stack(
                    [
                        torch.max(
                            torch.abs(all_poses[:, X] - self.scenario.render_origin[X])
                        ),
                        torch.max(
                            torch.abs(all_poses[:, Y] - self.scenario.render_origin[Y])
                        ),
                    ]
                )
                + 2 * max_agent_radius
            )

            viewer_size = torch.maximum(
                viewer_size_fit / cam_range,
                torch.tensor(zoom, device=self.device),
            )
            cam_range *= torch.max(viewer_size)
            self.viewer.set_bounds(
                -cam_range[X] + self.scenario.render_origin[X],
                cam_range[X] + self.scenario.render_origin[X],
                -cam_range[Y] + self.scenario.render_origin[Y],
                cam_range[Y] + self.scenario.render_origin[Y],
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

        # Render
        if self.scenario.visualize_semidims:
            self.plot_boundary()

        self._set_agent_comm_messages(env_index)

        if plot_position_function is not None:
            self.viewer.add_onetime(
                self.plot_function(
                    plot_position_function,
                    precision=plot_position_function_precision,
                    plot_range=plot_position_function_range,
                    cmap_range=plot_position_function_cmap_range,
                    cmap_alpha=plot_position_function_cmap_alpha,
                    cmap_name=plot_position_function_cmap_name,
                )
            )

        from vmas.simulator.rendering import Grid

        if self.scenario.plot_grid:
            grid = Grid(spacing=self.scenario.grid_spacing)
            grid.set_color(*vmas.simulator.utils.Color.BLACK.value, alpha=0.3)
            self.viewer.add_onetime(grid)

        self.viewer.add_onetime_list(self.scenario.extra_render(env_index))

        for entity in self.world.entities:
            self.viewer.add_onetime_list(entity.render(env_index=env_index))

        # render to display or array
        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def plot_boundary(self):
        # include boundaries in the rendering if the environment is dimension-limited
        if self.world.x_semidim is not None or self.world.y_semidim is not None:
            from vmas.simulator.rendering import Line
            from vmas.simulator.utils import Color

            # set a big value for the cases where the environment is dimension-limited only in one coordinate
            infinite_value = 100

            x_semi = (
                self.world.x_semidim
                if self.world.x_semidim is not None
                else infinite_value
            )
            y_semi = (
                self.world.y_semidim
                if self.world.y_semidim is not None
                else infinite_value
            )

            # set the color for the boundary line
            color = Color.GRAY.value

            # Define boundary points based on whether world semidims are provided
            if (
                self.world.x_semidim is not None and self.world.y_semidim is not None
            ) or self.world.y_semidim is not None:
                boundary_points = [
                    (-x_semi, y_semi),
                    (x_semi, y_semi),
                    (x_semi, -y_semi),
                    (-x_semi, -y_semi),
                ]
            else:
                boundary_points = [
                    (-x_semi, y_semi),
                    (-x_semi, -y_semi),
                    (x_semi, y_semi),
                    (x_semi, -y_semi),
                ]

            # Create lines by connecting points
            for i in range(
                0,
                len(boundary_points),
                1
                if (
                    self.world.x_semidim is not None
                    and self.world.y_semidim is not None
                )
                else 2,
            ):
                start = boundary_points[i]
                end = boundary_points[(i + 1) % len(boundary_points)]
                line = Line(start, end, width=0.7)
                line.set_color(*color)
                self.viewer.add_onetime(line)

    def plot_function(
        self, f, precision, plot_range, cmap_range, cmap_alpha, cmap_name
    ):
        from vmas.simulator.rendering import render_function_util

        if plot_range is None:
            assert self.viewer.bounds is not None, "Set viewer bounds before plotting"
            x_min, x_max, y_min, y_max = self.viewer.bounds.tolist()
            plot_range = (
                [x_min - precision, x_max - precision],
                [
                    y_min - precision,
                    y_max + precision,
                ],
            )

        geom = render_function_util(
            f=f,
            precision=precision,
            plot_range=plot_range,
            cmap_range=cmap_range,
            cmap_alpha=cmap_alpha,
            cmap_name=cmap_name,
        )
        return geom

    def _init_rendering(self):
        from vmas.simulator import rendering

        self.viewer = rendering.Viewer(
            *self.scenario.viewer_size, visible=self.visible_display
        )

        self.text_lines = []
        idx = 0
        if self.world.dim_c > 0:
            for agent in self.world.agents:
                if not agent.silent:
                    text_line = rendering.TextLine(y=idx * 40)
                    self.viewer.geoms.append(text_line)
                    self.text_lines.append(text_line)
                    idx += 1

    def _set_agent_comm_messages(self, env_index: int):
        # Render comm messages
        if self.world.dim_c > 0:
            idx = 0
            for agent in self.world.agents:
                if not agent.silent:
                    assert (
                        agent.state.c is not None
                    ), "Agent has no comm state but it should"
                    if self.continuous_actions:
                        word = (
                            "["
                            + ",".join(
                                [f"{comm:.2f}" for comm in agent.state.c[env_index]]
                            )
                            + "]"
                        )
                    else:
                        word = ALPHABET[torch.argmax(agent.state.c[env_index]).item()]

                    message = agent.name + " sends " + word + "   "
                    self.text_lines[idx].set_text(message)
                    idx += 1

    @override(TorchVectorizedObject)
    def to(self, device: DEVICE_TYPING):
        device = torch.device(device)
        self.scenario.to(device)
        super().to(device)
