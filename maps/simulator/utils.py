#  Copyright (c) 2022. Matteo Bettini
#  All rights reserved.

from enum import Enum

X = 0
Y = 1
Z = 2


class Color(Enum):
    RED = (0.75, 0.25, 0.25)
    GREEN = (0.25, 0.75, 0.25)
    BLUE = (0.25, 0.25, 0.75)
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


def create_fake_screen():
    import pyvirtualdisplay

    display = pyvirtualdisplay.Display(visible=False, size=(1400, 900))
    display.start()
    return display
