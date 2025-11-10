#  Copyright (c) ProrokLab.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
from abc import ABC, abstractmethod

import torch


class BaseHeuristicPolicy(ABC):
    def __init__(self, continuous_action: bool):
        self.continuous_actions = continuous_action

    @abstractmethod
    def compute_action(self, observation: torch.Tensor, u_range: float) -> torch.Tensor:
        raise NotImplementedError


class RandomPolicy(BaseHeuristicPolicy):
    def compute_action(self, observation: torch.Tensor, u_range: float) -> torch.Tensor:
        n_envs = observation.shape[0]
        return torch.clamp(torch.randn(n_envs, 2), -u_range, u_range)
