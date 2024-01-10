#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import abc
from abc import ABC
from typing import Union

from torch import Tensor


class Dynamics(ABC):
    def __init__(
        self,
    ):
        self._agent = None

    def reset(self, index: Union[Tensor, int] = None):
        pass

    @property
    def agent(self):
        if self._agent is None:
            raise ValueError(
                "You need to add the dynamics to an agent during construction before accessing its properties"
            )
        return self._agent

    @agent.setter
    def agent(self, value):
        if self._agent is not None:
            raise ValueError("Agent in dynamics has already been set")
        if value.action_size < self.needed_action_size:
            raise ValueError(
                f"Agent action size {value.action_size} is less than the required dynamics action size {self.needed_action_size}"
            )
        self._agent = value

    @property
    @abc.abstractmethod
    def needed_action_size(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def process_action(self):
        raise NotImplementedError
