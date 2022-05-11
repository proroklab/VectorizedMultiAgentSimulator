from abc import ABC, abstractmethod
from typing import Dict

import torch
from torch import Tensor

from maps.core import World, Agent


class BaseScenario(ABC):
    def __init__(self):
        self._world = None

    @property
    def world(self):
        assert (
            self._world is not None
        ), "You first need to set `self._world` in the `make_world` method"
        return self._world

    def reset_world(self):
        self.reset_world_at()

    def seed(self, seed: int = None):
        if seed is None:
            seed = 0
        torch.manual_seed(seed)
        self.world.seed()
        return [seed]

    def env_make_world(self, batch_dim: int, device: torch.device) -> World:
        self._world = self.make_world(batch_dim, device)
        self.reset_world()
        return self._world

    @abstractmethod
    def make_world(self, batch_dim: int, device: torch.device) -> World:
        """
        This function needs to be implemented when creating a scenario

        In this function the user should instantiate the world and insert agents and landmarks in it

        Args:
        :param batch_dim:
        :param device:
        :return world

         Examples:
            >>> from maps.core import Agent, World, Landmark, Sphere, Box
            >>> from maps.scenario import BaseScenario
            >>> from maps.utils import Color
            >>> class Scenario(BaseScenario):
            >>>     def make_world(self, batch_dim: int, device: torch.device):
            ...
            ...
            ...
            ...

        """
        raise NotImplementedError()

    @abstractmethod
    def reset_world_at(self, env_index: int = None):
        """

        :param env_index:
        :type env_index:
        """
        raise NotImplementedError()

    @abstractmethod
    def observation(self, agent: Agent) -> Tensor:
        """

        :param agent:
        :type agent:
        """
        raise NotImplementedError()

    @abstractmethod
    def reward(self, agent: Agent) -> Tensor:
        """

        :param agent:
        :type agent:
        """
        raise NotImplementedError()

    def done(self):
        """

        :return:
        :rtype:
        """
        return torch.tensor([False], device=self.world.device).repeat(
            self.world.batch_dim
        )

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        """

        :param agent:
        :type agent:
        :return:
        :rtype:
        """
        return {}
