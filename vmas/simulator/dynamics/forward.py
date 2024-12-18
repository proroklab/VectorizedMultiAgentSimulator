#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
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
