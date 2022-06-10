#  Copyright (c) 2022. Matteo Bettini
#  All rights reserved.

from maps.make_env import make_env
from maps.simulator.utils import _init_pyglet_device

_init_pyglet_device()

__all__ = [
    "make_env",
]
