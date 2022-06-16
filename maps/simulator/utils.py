#  Copyright (c) 2022. Matteo Bettini
#  All rights reserved.
import os
from enum import Enum

X = 0
Y = 1
Z = 2
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
VIEWER_MIN_SIZE = 1.2
INITIAL_VIEWER_SIZE = (700, 700)
LINE_MIN_DIST = 4 / 6e2


class Color(Enum):
    RED = (0.75, 0.25, 0.25)
    GREEN = (0.25, 0.75, 0.25)
    BLUE = (0.25, 0.25, 0.75)
    LIGHT_GREEN = (0.45, 0.95, 0.45)
    WHITE = (0.75, 0.75, 0.75)
    GRAY = (0.25, 0.25, 0.25)
    BLACK = (0.15, 0.15, 0.15)


class SensorType(Enum):
    LIDAR = 1


def override(cls):
    """Decorator for documenting method overrides."""

    def check_override(method):
        if method.__name__ not in dir(cls):
            raise NameError("{} does not override any method of {}".format(method, cls))
        return method

    return check_override


def _init_pyglet_device():
    available_devices = os.getenv("CUDA_VISIBLE_DEVICES")
    if available_devices is not None and len(available_devices) > 0:
        os.environ["PYGLET_HEADLESS_DEVICE"] = (
            available_devices.split(",")[0]
            if len(available_devices) > 1
            else available_devices
        )
