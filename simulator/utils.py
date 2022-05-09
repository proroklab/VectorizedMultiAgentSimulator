from enum import Enum

X = 0
Y = 1
Z = 2


class Colors(Enum):
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GRAY = (100, 100, 100)


class SensorType(Enum):
    LIDAR = 1
