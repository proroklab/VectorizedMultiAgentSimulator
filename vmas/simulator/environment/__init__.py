#  Copyright (c) ProrokLab.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
from enum import Enum

from vmas.simulator.environment.environment import Environment


class Wrapper(Enum):
    RLLIB = 0
    GYM = 1
    GYMNASIUM = 2
    GYMNASIUM_VEC = 3

    def get_env(self, env: Environment, **kwargs):
        if self is self.RLLIB:
            from vmas.simulator.environment.rllib import VectorEnvWrapper

            return VectorEnvWrapper(env, **kwargs)
        elif self is self.GYM:
            from vmas.simulator.environment.gym import GymWrapper

            return GymWrapper(env, **kwargs)
        elif self is self.GYMNASIUM:
            from vmas.simulator.environment.gym.gymnasium import GymnasiumWrapper

            return GymnasiumWrapper(env, **kwargs)
        elif self is self.GYMNASIUM_VEC:
            from vmas.simulator.environment.gym.gymnasium_vec import (
                GymnasiumVectorizedWrapper,
            )

            return GymnasiumVectorizedWrapper(env, **kwargs)
