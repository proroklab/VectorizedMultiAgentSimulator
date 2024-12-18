#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

from vmas.simulator.dynamics.common import Dynamics


class Static(Dynamics):
    @property
    def needed_action_size(self) -> int:
        return 0

    def process_action(self):
        pass
