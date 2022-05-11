from abc import ABC, abstractmethod
from typing import Dict

import torch
from torch import Tensor

from mpe.multiagent.core import Agent, World


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

    # create elements of the world
    @abstractmethod
    def make_world(self, batch_dim: int, device: torch.device) -> World:
        """
        Creates the world and sets it in `self._world`
        """
        raise NotImplementedError()

    @abstractmethod
    def reset_world_at(self, env_index: int = None):
        raise NotImplementedError()

    @abstractmethod
    def observation(self, agent: Agent) -> Tensor:
        raise NotImplementedError()

    @abstractmethod
    def reward(self, agent: Agent) -> Tensor:
        raise NotImplementedError()

    def done(self):
        return torch.tensor([False], device=self.world.device).repeat(
            self.world.batch_dim
        )

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        return {}
