#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

from vmas.simulator.dynamics.common import Dynamics


class Holonomic(Dynamics):
    @property
    def needed_action_size(self) -> int:
        return 2

    def process_action(self):
        self.agent.state.force = self.agent.action.u[:, : self.needed_action_size]
