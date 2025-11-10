#  Copyright (c) ProrokLab.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

from vmas.simulator.dynamics.common import Dynamics


class Static(Dynamics):
    @property
    def needed_action_size(self) -> int:
        return 0

    def process_action(self):
        pass
