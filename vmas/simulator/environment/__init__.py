#  Copyright (c) 2022.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
from enum import Enum

from vmas.simulator.environment.environment import Environment


class Wrapper(Enum):
    RLLIB = 0
    GYM = 1

    def get_env(self, env: Environment):
        if self is self.RLLIB:
            from vmas.simulator.environment.rllib import VectorEnvWrapper

            return VectorEnvWrapper(env)
        elif self is self.GYM:
            from vmas.simulator.environment.gym import GymWrapper

            return GymWrapper(env)
