#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import abc
from abc import ABC
from typing import Union

import torch
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

    @classmethod
    def __init_subclass__(cls) -> None:
        if (cls.needed_action_size is Dynamics.needed_action_size) == (
            cls.action_nvec is Dynamics.action_nvec
        ):
            raise TypeError(
                "Dynamics subclasses must override exactly one of needed_action_size or action_nvec."
            )
        super().__init_subclass__()

    @property
    def needed_action_size(self) -> int:
        """The size of the action vector needed by the dynamics.

        If not overridden, defaults to the size of the action_nvec."""
        return len(self.action_nvec)

    @property
    def action_nvec(self) -> list:
        """The number of possible values for the discrete version of each action.

        If not overridden, defaults to a list of 3s of length action_size,
        which results in the mapping [0, 1, 2] -> [-u_range, 0, u_range] for
        each action."""
        return [3] * self.needed_action_size

    @abc.abstractmethod
    def process_action(self):
        raise NotImplementedError
