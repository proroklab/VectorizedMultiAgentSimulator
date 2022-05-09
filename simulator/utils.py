from enum import Enum

X = 0
Y = 1
Z = 2


class Color(Enum):
    RED = (0.75, 0.25, 0.25)
    GREEN = (0, 1, 0)
    BLUE = (0, 0, 1)
    WHITE = (1, 1, 1)
    BLACK = (0, 0, 0)
    GRAY = (0.25, 0.25, 0.25)


class SensorType(Enum):
    LIDAR = 1
