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


def override(cls):
    """Decorator for documenting method overrides.
    Args:
        cls (type): The superclass that provides the overridden method. If this
            cls does not actually have the method, an error is raised.
    Examples:
        >>> from ray.rllib.policy import Policy
        >>> class TorchPolicy(Policy): # doctest: +SKIP
        ...     ...
        ...     # Indicates that `TorchPolicy.loss()` overrides the parent
        ...     # Policy class' own `loss method. Leads to an error if Policy
        ...     # does not have a `loss` method.
        ...     @override(Policy) # doctest: +SKIP
        ...     def loss(self, model, action_dist, train_batch): # doctest: +SKIP
        ...         ... # doctest: +SKIP
    """

    def check_override(method):
        if method.__name__ not in dir(cls):
            raise NameError("{} does not override any method of {}".format(method, cls))
        return method

    return check_override
