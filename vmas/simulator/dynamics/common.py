#  Copyright (c) ProrokLab.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
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
        return

    def zero_grad(self):
        return

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
        self._agent = value

    def check_and_process_action(self):
        action = self.agent.action.u
        if action.shape[1] < self.needed_action_size:
            raise ValueError(
                f"Agent action size {action.shape[1]} is less than the required dynamics action size {self.needed_action_size}"
            )
        self.process_action()

    @property
    @abc.abstractmethod
    def needed_action_size(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def process_action(self):
        raise NotImplementedError
