#  Copyright (c) ProrokLab.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
import torch

from vmas.simulator.dynamics.common import Dynamics
from vmas.simulator.utils import TorchUtils, X


class Forward(Dynamics):
    @property
    def needed_action_size(self) -> int:
        return 1

    def process_action(self):
        force = torch.zeros(
            self.agent.batch_dim, 2, device=self.agent.device, dtype=torch.float
        )
        force[:, X] = self.agent.action.u[:, 0]
        self.agent.state.force = TorchUtils.rotate_vector(force, self.agent.state.rot)
