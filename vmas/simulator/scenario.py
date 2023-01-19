#  Copyright (c) 2022-2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import typing
from abc import ABC, abstractmethod
from typing import Dict, List

import torch
from torch import Tensor

from vmas.simulator.core import World, Agent
from vmas.simulator.utils import INITIAL_VIEWER_SIZE, VIEWER_MIN_ZOOM

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom


class BaseScenario(ABC):
    def __init__(self):
        """Do not override"""
        self._world = None
        # This is the viewer size and can be set in the `make_world' function
        self.viewer_size = INITIAL_VIEWER_SIZE
        # This is the zoom level of the rendering
        self.viewer_zoom = VIEWER_MIN_ZOOM
        # Whether to plot a grid in the scenario background
        self.plot_grid = False
        # The distance between lines in the background grid
        self.grid_spacing = 0.1

    @property
    def world(self):
        """Do not override"""
        assert (
            self._world is not None
        ), "You first need to set `self._world` in the `make_world` method"
        return self._world

    def to(self, device: torch.device):
        """Do not override"""
        for attr, value in self.__dict__.items():
            if isinstance(value, Tensor):
                self.__dict__[attr] = value.to(device)
        self.world.to(device)

    def env_make_world(self, batch_dim: int, device: torch.device, **kwargs) -> World:
        """Do not override"""
        self._world = self.make_world(batch_dim, device, **kwargs)
        return self._world

    def env_reset_world_at(self, env_index: typing.Optional[int]):
        """Do not override"""
        self.world.reset(env_index)
        self.reset_world_at(env_index)

    def env_process_action(self, agent: Agent):
        """Do not override"""
        if agent.action_script is not None:
            agent.action_callback(self.world)
        self.process_action(agent)

    @abstractmethod
    def make_world(self, batch_dim: int, device: torch.device, **kwargs) -> World:
        """
        This function needs to be implemented when creating a scenario
        In this function the user should instantiate the world and insert agents and landmarks in it

        Args:
        :param batch_dim: the number of environments to step parallely
        :param device: the torch device to use
        :param kwargs: named arguments passed during environment creation
        :return world: returns the instantiated world which is automatically set in 'self.world'
        Examples:
            >>> from vmas.simulator.core import Agent, World, Landmark, Sphere, Box
            >>> from vmas.simulator.scenario import BaseScenario
            >>> from vmas.simulator.utils import Color
            >>> class Scenario(BaseScenario):
            >>>     def make_world(self, batch_dim: int, device: torch.device, **kwargs):
            ...         # Pass any kwargs you desire when creating the environment
            ...         n_agents = kwargs.get("n_agents", 5)
            ...
            ...         # Create world
            ...         world = World(batch_dim, device, dt=0.1, drag=0.25, dim_c=0)
            ...         # Add agents
            ...         for i in range(n_agents):
            ...             agent = Agent(
            ...                 name=f"agent {i}",
            ...                 collide=True,
            ...                 mass=1.0,
            ...                 shape=Sphere(radius=0.04),
            ...                 max_speed=None,
            ...                 color=Color.BLUE,
            ...                 u_range=1.0,
            ...             )
            ...             world.add_agent(agent)
            ...         # Add landmarks
            ...         for i in range(5):
            ...             landmark = Landmark(
            ...                 name=f"landmark {i}",
            ...                 collide=True,
            ...                 movable=False,
            ...                 shape=Box(length=0.3,width=0.1),
            ...                 color=Color.RED,
            ...             )
            ...             world.add_landmark(landmark)
            ...         return world
        """
        raise NotImplementedError()

    @abstractmethod
    def reset_world_at(self, env_index: int = None):
        """
        Resets the world at the specified env_index.
        When a None index is passed, the world should make a vectorized (batched) reset.
        The 'entity.set_x()' methodes already have this logic integrated and will perform
        batched operations when index is None

        Implementors can access the world at 'self.world'

        To increase performance, torch tensors created with the device set, like:
        torch.tensor(..., device=self.world.device)

        :param env_index: index of the environment to reset. If None a vectorized reset should be performed
        Examples:
            >>> from vmas.simulator.core import Agent, World, Landmark, Sphere, Box
            >>> from vmas.simulator.scenario import BaseScenario
            >>> from vmas.simulator.utils import Color
            >>> class Scenario(BaseScenario):
            >>>     def reset_world_at(self, env_index: int = None)
            ...        for i, agent in enumerate(self.world.agents):
            ...            agent.set_pos(
            ...                torch.tensor(
            ...                     [-0.2 + 0.1 * i, 1.0],
            ...                     dtype=torch.float32,
            ...                     device=self.world.device,
            ...                ),
            ...                 batch_index=env_index,
            ...            )
            ...        for i, landmark in enumerate(self.world.landmarks):
            ...            landmark.set_pos(
            ...                torch.tensor(
            ...                     [0.2 if i % 2 else -0.2, 0.6 - 0.3 * i],
            ...                     dtype=torch.float32,
            ...                     device=self.world.device,
            ...                ),
            ...                 batch_index=env_index,
            ...            )
            ...            landmark.set_rot(
            ...                torch.tensor(
            ...                     [torch.pi / 4 if i % 2 else -torch.pi / 4],
            ...                     dtype=torch.float32,
            ...                     device=self.world.device,
            ...                ),
            ...                 batch_index=env_index,
            ...            )
        """
        raise NotImplementedError()

    @abstractmethod
    def observation(self, agent: Agent) -> Tensor:
        """
        This function computes the observations for 'agent' in a vectorized way
        The returned tensor should contain the observations for 'agent' in all envs and should have
        shape (n_envs, n_agent_obs)

        Implementors can access the world at 'self.world'

        To increase performance, tensors created should have the device set, like:
        torch.tensor(..., device=self.world.device)

        :param agent: Agent batch to compute observation of
        :return observation: Tensor of shape (n_envs, n_agent_obs)
        Examples:
            >>> from vmas.simulator.core import Agent, World, Landmark, Sphere, Box
            >>> from vmas.simulator.scenario import BaseScenario
            >>> from vmas.simulator.utils import Color
            >>> def observation(self, agent: Agent):
            ...      # get positions of all entities in this agent's reference frame
            ...      entity_pos = []
            ...      for entity in self.world.landmarks:
            ...          entity_pos.append(entity.state.pos - agent.state.pos)
            ...      return torch.cat([agent.state.vel, *entity_pos], dim=-1)
        """

        raise NotImplementedError()

    @abstractmethod
    def reward(self, agent: Agent) -> Tensor:
        """
        This function computes the reward for 'agent' in a vectorized way
        The returned tensor should contain the reward for 'agent' in all envs and should have
        shape (n_envs)

        Implementors can access the world at 'self.world'

        To increase performance, tensors created should have the device set, like:
        torch.tensor(..., device=self.world.device)

        :param agent: Agent batch to compute reward of
        :return observation: Tensor of shape (n_envs)
        Examples:
            >>> from vmas.simulator.core import Agent, World, Landmark, Sphere, Box
            >>> from vmas.simulator.scenario import BaseScenario
            >>> from vmas.simulator.utils import Color
            >>> def observation(self, agent: Agent):
            ...      # reward every agent proportionally to distance from first landmark
            ...      dist2 = torch.sum(
            ...          torch.square(agent.state.pos - self.world.landmarks[0].state.pos), dim=-1
            ...      )
            ...      return -dist2
        """
        raise NotImplementedError()

    def done(self):
        """
        This function computes the done flag for each env in a vectorized way
        The returned tensor should contain the 'done' for all envs and should have
        shape (n_envs)

        Implementors can access the world at 'self.world'

        To increase performance, tensors created should have the device set, like:
        torch.tensor(..., device=self.world.device)

        :return dones: Bool tensor of shape (n_envs)
        """
        return torch.tensor([False], device=self.world.device).repeat(
            self.world.batch_dim
        )

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        """
        This function computes the info dict for 'agent' in a vectorized way
        The returned dict should have a key for each info of interest and the corresponding value should
        be a tensor of shape (n_envs, info_size)

        Implementors can access the world at 'self.world'

        To increase performance, tensors created should have the device set, like:
        torch.tensor(..., device=self.world.device)

        :param agent: Agent batch to compute info of
        :return: info: A dict with a key for each info of interest, and a tensor value  of shape (n_envs, info_size)
        """
        return {}

    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        """
        This function facilitates additional user/scenario-level rendering for a specific environment index.
        The returned list is a list of geometries. It is the user's responsibility to set attributes such as color,
        position and rotation.

        :param env_index: index of the environment to render.
        :return: A list of geometries to render for the current time step.
        """
        return []

    def process_action(self, agent: Agent):
        """
        This function can be overridden to process the agent actions before the simulation step.
        It has access to the world through the `self.world` attribute

        It can also be used for any other type of computation that has to happen after
         the input actions have been set but before the simulation step

        :param agent: the agent whose actions have to be processed.
        """
        pass
