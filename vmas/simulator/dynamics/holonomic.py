#  Copyright (c) ProrokLab.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

from vmas.simulator.dynamics.common import Dynamics


class Holonomic(Dynamics):
    @property
    def needed_action_size(self) -> int:
        return 2

    def process_action(self):
        self.agent.state.force = self.agent.action.u[:, : self.needed_action_size]
