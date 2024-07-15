#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

from vmas.simulator.dynamics.common import Dynamics


class Rotation(Dynamics):
    @property
    def needed_action_size(self) -> int:
        return 1

    def process_action(self):
        self.agent.state.torque = self.agent.action.u[:, 0]
