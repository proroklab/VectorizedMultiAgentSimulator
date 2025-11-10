#  Copyright (c) ProrokLab.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
import typing
from abc import ABC, abstractmethod
from typing import Callable, List, Sequence, Tuple, Union

import torch
from torch import Tensor

from vmas.simulator.dynamics.common import Dynamics
from vmas.simulator.dynamics.holonomic import Holonomic
from vmas.simulator.joints import Joint
from vmas.simulator.physics import (
    _get_closest_box_box,
    _get_closest_line_box,
    _get_closest_point_box,
    _get_closest_point_line,
    _get_closest_points_line_line,
    _get_inner_point_box,
)
from vmas.simulator.sensors import Sensor
from vmas.simulator.utils import (
    ANGULAR_FRICTION,
    COLLISION_FORCE,
    Color,
    DRAG,
    JOINT_FORCE,
    LINE_MIN_DIST,
    LINEAR_FRICTION,
    Observable,
    override,
    TorchUtils,
    TORQUE_CONSTRAINT_FORCE,
    X,
    Y,
)

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom


class TorchVectorizedObject(object):
    def __init__(self, batch_dim: int = None, device: torch.device = None):
        # batch dim
        self._batch_dim = batch_dim
        # device
        self._device = device

    @property
    def batch_dim(self):
        return self._batch_dim

    @batch_dim.setter
    def batch_dim(self, batch_dim: int):
        assert self._batch_dim is None, "You can set batch dim only once"
        self._batch_dim = batch_dim

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device: torch.device):
        self._device = device

    def _check_batch_index(self, batch_index: int):
        if batch_index is not None:
            assert (
                0 <= batch_index < self.batch_dim
            ), f"Index must be between 0 and {self.batch_dim}, got {batch_index}"

    def to(self, device: torch.device):
        self.device = device
        for attr, value in self.__dict__.items():
            if isinstance(value, Tensor):
                self.__dict__[attr] = value.to(device)


class Shape(ABC):
    @abstractmethod
    def moment_of_inertia(self, mass: float):
        raise NotImplementedError

    @abstractmethod
    def get_delta_from_anchor(self, anchor: Tuple[float, float]) -> Tuple[float, float]:
        raise NotImplementedError

    @abstractmethod
    def get_geometry(self):
        raise NotImplementedError

    @abstractmethod
    def circumscribed_radius(self):
        raise NotImplementedError


class Box(Shape):
    def __init__(self, length: float = 0.3, width: float = 0.1, hollow: bool = False):
        super().__init__()
        assert length > 0, f"Length must be > 0, got {length}"
        assert width > 0, f"Width must be > 0, got {length}"
        self._length = length
        self._width = width
        self.hollow = hollow

    @property
    def length(self):
        return self._length

    @property
    def width(self):
        return self._width

    def get_delta_from_anchor(self, anchor: Tuple[float, float]) -> Tuple[float, float]:
        return anchor[X] * self.length / 2, anchor[Y] * self.width / 2

    def moment_of_inertia(self, mass: float):
        return (1 / 12) * mass * (self.length**2 + self.width**2)

    def circumscribed_radius(self):
        return math.sqrt((self.length / 2) ** 2 + (self.width / 2) ** 2)

    def get_geometry(self) -> "Geom":
        from vmas.simulator import rendering

        l, r, t, b = (
            -self.length / 2,
            self.length / 2,
            self.width / 2,
            -self.width / 2,
        )
        return rendering.make_polygon([(l, b), (l, t), (r, t), (r, b)])


class Sphere(Shape):
    def __init__(self, radius: float = 0.05):
        super().__init__()
        assert radius > 0, f"Radius must be > 0, got {radius}"
        self._radius = radius

    @property
    def radius(self):
        return self._radius

    def get_delta_from_anchor(self, anchor: Tuple[float, float]) -> Tuple[float, float]:
        delta = torch.tensor([anchor[X] * self.radius, anchor[Y] * self.radius]).to(
            torch.float32
        )
        delta_norm = torch.linalg.vector_norm(delta)
        if delta_norm > self.radius:
            delta /= delta_norm * self.radius
        return tuple(delta.tolist())

    def moment_of_inertia(self, mass: float):
        return (1 / 2) * mass * self.radius**2

    def circumscribed_radius(self):
        return self.radius

    def get_geometry(self) -> "Geom":
        from vmas.simulator import rendering

        return rendering.make_circle(self.radius)


class Line(Shape):
    def __init__(self, length: float = 0.5):
        super().__init__()
        assert length > 0, f"Length must be > 0, got {length}"
        self._length = length
        self._width = 2

    @property
    def length(self):
        return self._length

    @property
    def width(self):
        return self._width

    def moment_of_inertia(self, mass: float):
        return (1 / 12) * mass * (self.length**2)

    def circumscribed_radius(self):
        return self.length / 2

    def get_delta_from_anchor(self, anchor: Tuple[float, float]) -> Tuple[float, float]:
        return anchor[X] * self.length / 2, 0.0

    def get_geometry(self) -> "Geom":
        from vmas.simulator import rendering

        return rendering.Line(
            (-self.length / 2, 0),
            (self.length / 2, 0),
            width=self.width,
        )


class EntityState(TorchVectorizedObject):
    def __init__(self):
        super().__init__()
        # physical position
        self._pos = None
        # physical velocity
        self._vel = None
        # physical rotation -- from -pi to pi
        self._rot = None
        # angular velocity
        self._ang_vel = None

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, pos: Tensor):
        assert (
            self._batch_dim is not None and self._device is not None
        ), "First add an entity to the world before setting its state"
        assert (
            pos.shape[0] == self._batch_dim
        ), f"Internal state must match batch dim, got {pos.shape[0]}, expected {self._batch_dim}"
        if self._vel is not None:
            assert (
                pos.shape == self._vel.shape
            ), f"Position shape must match velocity shape, got {pos.shape} expected {self._vel.shape}"

        self._pos = pos.to(self._device)

    @property
    def vel(self):
        return self._vel

    @vel.setter
    def vel(self, vel: Tensor):
        assert (
            self._batch_dim is not None and self._device is not None
        ), "First add an entity to the world before setting its state"
        assert (
            vel.shape[0] == self._batch_dim
        ), f"Internal state must match batch dim, got {vel.shape[0]}, expected {self._batch_dim}"
        if self._pos is not None:
            assert (
                vel.shape == self._pos.shape
            ), f"Velocity shape must match position shape, got {vel.shape} expected {self._pos.shape}"

        self._vel = vel.to(self._device)

    @property
    def ang_vel(self):
        return self._ang_vel

    @ang_vel.setter
    def ang_vel(self, ang_vel: Tensor):
        assert (
            self._batch_dim is not None and self._device is not None
        ), "First add an entity to the world before setting its state"
        assert (
            ang_vel.shape[0] == self._batch_dim
        ), f"Internal state must match batch dim, got {ang_vel.shape[0]}, expected {self._batch_dim}"

        self._ang_vel = ang_vel.to(self._device)

    @property
    def rot(self):
        return self._rot

    @rot.setter
    def rot(self, rot: Tensor):
        assert (
            self._batch_dim is not None and self._device is not None
        ), "First add an entity to the world before setting its state"
        assert (
            rot.shape[0] == self._batch_dim
        ), f"Internal state must match batch dim, got {rot.shape[0]}, expected {self._batch_dim}"

        self._rot = rot.to(self._device)

    def _reset(self, env_index: typing.Optional[int]):
        for attr_name in ["pos", "rot", "vel", "ang_vel"]:
            attr = self.__getattribute__(attr_name)
            if attr is not None:
                if env_index is None:
                    self.__setattr__(attr_name, torch.zeros_like(attr))
                else:
                    self.__setattr__(
                        attr_name,
                        TorchUtils.where_from_index(env_index, 0, attr),
                    )

    def zero_grad(self):
        for attr_name in ["pos", "rot", "vel", "ang_vel"]:
            attr = self.__getattribute__(attr_name)
            if attr is not None:
                self.__setattr__(attr_name, attr.detach())

    def _spawn(self, dim_c: int, dim_p: int):
        self.pos = torch.zeros(
            self.batch_dim, dim_p, device=self.device, dtype=torch.float32
        )
        self.vel = torch.zeros(
            self.batch_dim, dim_p, device=self.device, dtype=torch.float32
        )
        self.rot = torch.zeros(
            self.batch_dim, 1, device=self.device, dtype=torch.float32
        )
        self.ang_vel = torch.zeros(
            self.batch_dim, 1, device=self.device, dtype=torch.float32
        )


class AgentState(EntityState):
    def __init__(
        self,
    ):
        super().__init__()
        # communication utterance
        self._c = None
        # Agent force from actions
        self._force = None
        # Agent torque from actions
        self._torque = None

    @property
    def c(self):
        return self._c

    @c.setter
    def c(self, c: Tensor):
        assert (
            self._batch_dim is not None and self._device is not None
        ), "First add an entity to the world before setting its state"
        assert (
            c.shape[0] == self._batch_dim
        ), f"Internal state must match batch dim, got {c.shape[0]}, expected {self._batch_dim}"

        self._c = c.to(self._device)

    @property
    def force(self):
        return self._force

    @force.setter
    def force(self, value):
        assert (
            self._batch_dim is not None and self._device is not None
        ), "First add an entity to the world before setting its state"
        assert (
            value.shape[0] == self._batch_dim
        ), f"Internal state must match batch dim, got {value.shape[0]}, expected {self._batch_dim}"

        self._force = value.to(self._device)

    @property
    def torque(self):
        return self._torque

    @torque.setter
    def torque(self, value):
        assert (
            self._batch_dim is not None and self._device is not None
        ), "First add an entity to the world before setting its state"
        assert (
            value.shape[0] == self._batch_dim
        ), f"Internal state must match batch dim, got {value.shape[0]}, expected {self._batch_dim}"

        self._torque = value.to(self._device)

    @override(EntityState)
    def _reset(self, env_index: typing.Optional[int]):
        for attr_name in ["c", "force", "torque"]:
            attr = self.__getattribute__(attr_name)
            if attr is not None:
                if env_index is None:
                    self.__setattr__(attr_name, torch.zeros_like(attr))
                else:
                    self.__setattr__(
                        attr_name,
                        TorchUtils.where_from_index(env_index, 0, attr),
                    )
        super()._reset(env_index)

    @override(EntityState)
    def zero_grad(self):
        for attr_name in ["c", "force", "torque"]:
            attr = self.__getattribute__(attr_name)
            if attr is not None:
                self.__setattr__(attr_name, attr.detach())
        super().zero_grad()

    @override(EntityState)
    def _spawn(self, dim_c: int, dim_p: int):
        if dim_c > 0:
            self.c = torch.zeros(
                self.batch_dim, dim_c, device=self.device, dtype=torch.float32
            )
        self.force = torch.zeros(
            self.batch_dim, dim_p, device=self.device, dtype=torch.float32
        )
        self.torque = torch.zeros(
            self.batch_dim, 1, device=self.device, dtype=torch.float32
        )
        super()._spawn(dim_c, dim_p)


# action of an agent
class Action(TorchVectorizedObject):
    def __init__(
        self,
        u_range: Union[float, Sequence[float]],
        u_multiplier: Union[float, Sequence[float]],
        u_noise: Union[float, Sequence[float]],
        action_size: int,
    ):
        super().__init__()
        # physical motor noise amount
        self._u_noise = u_noise
        # control range
        self._u_range = u_range
        # agent action is a force multiplied by this amount
        self._u_multiplier = u_multiplier
        # Number of actions
        self.action_size = action_size

        # physical action
        self._u = None
        # communication_action
        self._c = None

        self._u_range_tensor = None
        self._u_multiplier_tensor = None
        self._u_noise_tensor = None

        self._check_action_init()

    def _check_action_init(self):
        for attr in (self.u_multiplier, self.u_range, self.u_noise):
            if isinstance(attr, List):
                assert len(attr) == self.action_size, (
                    "Action attributes u_... must be either a float or a list of floats"
                    " (one per action) all with same length"
                )

    @property
    def u(self):
        return self._u

    @u.setter
    def u(self, u: Tensor):
        assert (
            self._batch_dim is not None and self._device is not None
        ), "First add an agent to the world before setting its action"
        assert (
            u.shape[0] == self._batch_dim
        ), f"Action must match batch dim, got {u.shape[0]}, expected {self._batch_dim}"

        self._u = u.to(self._device)

    @property
    def c(self):
        return self._c

    @c.setter
    def c(self, c: Tensor):
        assert (
            self._batch_dim is not None and self._device is not None
        ), "First add an agent to the world before setting its action"
        assert (
            c.shape[0] == self._batch_dim
        ), f"Action must match batch dim, got {c.shape[0]}, expected {self._batch_dim}"

        self._c = c.to(self._device)

    @property
    def u_range(self):
        return self._u_range

    @property
    def u_multiplier(self):
        return self._u_multiplier

    @property
    def u_noise(self):
        return self._u_noise

    @property
    def u_range_tensor(self):
        if self._u_range_tensor is None:
            self._u_range_tensor = self._to_tensor(self.u_range)
        return self._u_range_tensor

    @property
    def u_multiplier_tensor(self):
        if self._u_multiplier_tensor is None:
            self._u_multiplier_tensor = self._to_tensor(self.u_multiplier)
        return self._u_multiplier_tensor

    @property
    def u_noise_tensor(self):
        if self._u_noise_tensor is None:
            self._u_noise_tensor = self._to_tensor(self.u_noise)
        return self._u_noise_tensor

    def _to_tensor(self, value):
        return torch.tensor(
            value if isinstance(value, Sequence) else [value] * self.action_size,
            device=self.device,
            dtype=torch.float,
        )

    def _reset(self, env_index: typing.Optional[int]):
        for attr_name in ["u", "c"]:
            attr = self.__getattribute__(attr_name)
            if attr is not None:
                if env_index is None:
                    self.__setattr__(attr_name, torch.zeros_like(attr))
                else:
                    self.__setattr__(
                        attr_name,
                        TorchUtils.where_from_index(env_index, 0, attr),
                    )

    def zero_grad(self):
        for attr_name in ["u", "c"]:
            attr = self.__getattribute__(attr_name)
            if attr is not None:
                self.__setattr__(attr_name, attr.detach())


# properties and state of physical world entity
class Entity(TorchVectorizedObject, Observable, ABC):
    def __init__(
        self,
        name: str,
        movable: bool = False,
        rotatable: bool = False,
        collide: bool = True,
        density: float = 25.0,  # Unused for now
        mass: float = 1.0,
        shape: Shape = None,
        v_range: float = None,
        max_speed: float = None,
        color=Color.GRAY,
        is_joint: bool = False,
        drag: float = None,
        linear_friction: float = None,
        angular_friction: float = None,
        gravity: typing.Union[float, Tensor] = None,
        collision_filter: Callable[[Entity], bool] = lambda _: True,
    ):
        if shape is None:
            shape = Sphere()

        TorchVectorizedObject.__init__(self)
        Observable.__init__(self)
        # name
        self._name = name
        # entity can move / be pushed
        self._movable = movable
        # entity can rotate
        self._rotatable = rotatable
        # entity collides with others
        self._collide = collide
        # material density (affects mass)
        self._density = density
        # mass
        self._mass = mass
        # max speed
        self._max_speed = max_speed
        self._v_range = v_range
        # color
        self._color = color
        # shape
        self._shape = shape
        # is joint
        self._is_joint = is_joint
        # collision filter
        self._collision_filter = collision_filter
        # state
        self._state = EntityState()
        # drag
        self._drag = drag
        # friction
        self._linear_friction = linear_friction
        self._angular_friction = angular_friction
        # gravity
        if isinstance(gravity, Tensor):
            self._gravity = gravity
        else:
            self._gravity = (
                torch.tensor(gravity, device=self.device, dtype=torch.float32)
                if gravity is not None
                else gravity
            )
        # entity goal
        self._goal = None
        # Render the entity
        self._render = None

    @TorchVectorizedObject.batch_dim.setter
    def batch_dim(self, batch_dim: int):
        TorchVectorizedObject.batch_dim.fset(self, batch_dim)
        self._state.batch_dim = batch_dim

    @property
    def is_rendering(self):
        if self._render is None:
            self.reset_render()
        return self._render

    def reset_render(self):
        self._render = torch.full((self.batch_dim,), True, device=self.device)

    def collides(self, entity: Entity):
        if not self.collide:
            return False
        return self._collision_filter(entity)

    @property
    def is_joint(self):
        return self._is_joint

    @property
    def mass(self):
        return self._mass

    @mass.setter
    def mass(self, mass: float):
        self._mass = mass

    @property
    def moment_of_inertia(self):
        return self.shape.moment_of_inertia(self.mass)

    @property
    def state(self):
        return self._state

    @property
    def movable(self):
        return self._movable

    @property
    def collide(self):
        return self._collide

    @property
    def shape(self):
        return self._shape

    @property
    def max_speed(self):
        return self._max_speed

    @property
    def v_range(self):
        return self._v_range

    @property
    def name(self):
        return self._name

    @property
    def rotatable(self):
        return self._rotatable

    @property
    def color(self):
        if isinstance(self._color, Color):
            return self._color.value
        return self._color

    @color.setter
    def color(self, color):
        self._color = color

    @property
    def goal(self):
        return self._goal

    @property
    def drag(self):
        return self._drag

    @property
    def linear_friction(self):
        return self._linear_friction

    @linear_friction.setter
    def linear_friction(self, value):
        self._linear_friction = value

    @property
    def gravity(self):
        return self._gravity

    @gravity.setter
    def gravity(self, value):
        self._gravity = value

    @property
    def angular_friction(self):
        return self._angular_friction

    @goal.setter
    def goal(self, goal: Entity):
        self._goal = goal

    @property
    def collision_filter(self):
        return self._collision_filter

    @collision_filter.setter
    def collision_filter(self, collision_filter: Callable[[Entity], bool]):
        self._collision_filter = collision_filter

    def _spawn(self, dim_c: int, dim_p: int):
        self.state._spawn(dim_c, dim_p)

    def _reset(self, env_index: int):
        self.state._reset(env_index)

    def zero_grad(self):
        self.state.zero_grad()

    def set_pos(self, pos: Tensor, batch_index: int):
        self._set_state_property(EntityState.pos, self.state, pos, batch_index)

    def set_vel(self, vel: Tensor, batch_index: int):
        self._set_state_property(EntityState.vel, self.state, vel, batch_index)

    def set_rot(self, rot: Tensor, batch_index: int):
        self._set_state_property(EntityState.rot, self.state, rot, batch_index)

    def set_ang_vel(self, ang_vel: Tensor, batch_index: int):
        self._set_state_property(EntityState.ang_vel, self.state, ang_vel, batch_index)

    def _set_state_property(
        self, prop, entity: EntityState, new: Tensor, batch_index: int
    ):
        assert (
            self.batch_dim is not None
        ), f"Tried to set property of {self.name} without adding it to the world"
        self._check_batch_index(batch_index)
        new = new.to(self.device)
        if batch_index is None:
            if len(new.shape) > 1 and new.shape[0] == self.batch_dim:
                prop.fset(entity, new)
            else:
                prop.fset(entity, new.repeat(self.batch_dim, 1))
        else:
            value = prop.fget(entity)
            value[batch_index] = new
        self.notify_observers()

    @override(TorchVectorizedObject)
    def to(self, device: torch.device):
        super().to(device)
        self.state.to(device)

    def render(self, env_index: int = 0) -> "List[Geom]":
        from vmas.simulator import rendering

        if not self.is_rendering[env_index]:
            return []
        geom = self.shape.get_geometry()
        xform = rendering.Transform()
        geom.add_attr(xform)

        xform.set_translation(*self.state.pos[env_index])
        xform.set_rotation(self.state.rot[env_index])

        color = self.color
        if isinstance(color, torch.Tensor) and len(color.shape) > 1:
            color = color[env_index]
        geom.set_color(*color)

        return [geom]


# properties of landmark entities
class Landmark(Entity):
    def __init__(
        self,
        name: str,
        shape: Shape = None,
        movable: bool = False,
        rotatable: bool = False,
        collide: bool = True,
        density: float = 25.0,  # Unused for now
        mass: float = 1.0,
        v_range: float = None,
        max_speed: float = None,
        color=Color.GRAY,
        is_joint: bool = False,
        drag: float = None,
        linear_friction: float = None,
        angular_friction: float = None,
        gravity: float = None,
        collision_filter: Callable[[Entity], bool] = lambda _: True,
    ):
        super().__init__(
            name,
            movable,
            rotatable,
            collide,
            density,  # Unused for now
            mass,
            shape,
            v_range,
            max_speed,
            color,
            is_joint,
            drag,
            linear_friction,
            angular_friction,
            gravity,
            collision_filter,
        )


# properties of agent entities
class Agent(Entity):
    def __init__(
        self,
        name: str,
        shape: Shape = None,
        movable: bool = True,
        rotatable: bool = True,
        collide: bool = True,
        density: float = 25.0,  # Unused for now
        mass: float = 1.0,
        f_range: float = None,
        max_f: float = None,
        t_range: float = None,
        max_t: float = None,
        v_range: float = None,
        max_speed: float = None,
        color=Color.BLUE,
        alpha: float = 0.5,
        obs_range: float = None,
        obs_noise: float = None,
        u_noise: Union[float, Sequence[float]] = 0.0,
        u_range: Union[float, Sequence[float]] = 1.0,
        u_multiplier: Union[float, Sequence[float]] = 1.0,
        action_script: Callable[[Agent, World], None] = None,
        sensors: List[Sensor] = None,
        c_noise: float = 0.0,
        silent: bool = True,
        adversary: bool = False,
        drag: float = None,
        linear_friction: float = None,
        angular_friction: float = None,
        gravity: float = None,
        collision_filter: Callable[[Entity], bool] = lambda _: True,
        render_action: bool = False,
        dynamics: Dynamics = None,  # Defaults to holonomic
        action_size: int = None,  # Defaults to what required by the dynamics
        discrete_action_nvec: List[
            int
        ] = None,  # Defaults to 3-way discretization if discrete actions are chosen (stay, decrement, increment)
    ):
        super().__init__(
            name,
            movable,
            rotatable,
            collide,
            density,  # Unused for now
            mass,
            shape,
            v_range,
            max_speed,
            color,
            is_joint=False,
            drag=drag,
            linear_friction=linear_friction,
            angular_friction=angular_friction,
            gravity=gravity,
            collision_filter=collision_filter,
        )
        if obs_range == 0.0:
            assert sensors is None, f"Blind agent cannot have sensors, got {sensors}"

        if action_size is not None and discrete_action_nvec is not None:
            if action_size != len(discrete_action_nvec):
                raise ValueError(
                    f"action_size {action_size} is inconsistent with discrete_action_nvec {discrete_action_nvec}"
                )
        if discrete_action_nvec is not None:
            if not all(n > 1 for n in discrete_action_nvec):
                raise ValueError(
                    f"All values in discrete_action_nvec must be greater than 1, got {discrete_action_nvec}"
                )

        # cannot observe the world
        self._obs_range = obs_range
        # observation noise
        self._obs_noise = obs_noise
        # force constraints
        self._f_range = f_range
        self._max_f = max_f
        # torque constraints
        self._t_range = t_range
        self._max_t = max_t
        # script behavior to execute
        self._action_script = action_script
        # agents sensors
        self._sensors = []
        if sensors is not None:
            [self.add_sensor(sensor) for sensor in sensors]
        # non differentiable communication noise
        self._c_noise = c_noise
        # cannot send communication signals
        self._silent = silent
        # render the agent action force
        self._render_action = render_action
        # is adversary
        self._adversary = adversary
        # Render alpha
        self._alpha = alpha

        # Dynamics
        self.dynamics = dynamics if dynamics is not None else Holonomic()
        # Action
        if action_size is not None:
            self.action_size = action_size
        elif discrete_action_nvec is not None:
            self.action_size = len(discrete_action_nvec)
        else:
            self.action_size = self.dynamics.needed_action_size
        if discrete_action_nvec is None:
            self.discrete_action_nvec = [3] * self.action_size
        else:
            self.discrete_action_nvec = discrete_action_nvec
        self.dynamics.agent = self
        self._action = Action(
            u_range=u_range,
            u_multiplier=u_multiplier,
            u_noise=u_noise,
            action_size=self.action_size,
        )

        # state
        self._state = AgentState()

    def add_sensor(self, sensor: Sensor):
        sensor.agent = self
        self._sensors.append(sensor)

    @Entity.batch_dim.setter
    def batch_dim(self, batch_dim: int):
        Entity.batch_dim.fset(self, batch_dim)
        self._action.batch_dim = batch_dim

    @property
    def action_script(self) -> Callable[[Agent, World], None]:
        return self._action_script

    def action_callback(self, world: World):
        self._action_script(self, world)
        if self._silent or world.dim_c == 0:
            assert (
                self._action.c is None
            ), f"Agent {self.name} should not communicate but action script communicates"
        assert (
            self._action.u is not None
        ), f"Action script of {self.name} should set u action"
        assert (
            self._action.u.shape[1] == self.action_size
        ), f"Scripted action of agent {self.name} has wrong shape"

        assert (
            (self._action.u / self.action.u_multiplier_tensor).abs()
            <= self.action.u_range_tensor
        ).all(), f"Scripted physical action of {self.name} is out of range"

    @property
    def u_range(self):
        return self.action.u_range

    @property
    def obs_noise(self):
        return self._obs_noise if self._obs_noise is not None else 0

    @property
    def action(self) -> Action:
        return self._action

    @property
    def u_multiplier(self):
        return self.action.u_multiplier

    @property
    def max_f(self):
        return self._max_f

    @property
    def f_range(self):
        return self._f_range

    @property
    def max_t(self):
        return self._max_t

    @property
    def t_range(self):
        return self._t_range

    @property
    def silent(self):
        return self._silent

    @property
    def sensors(self) -> List[Sensor]:
        return self._sensors

    @property
    def u_noise(self):
        return self.action.u_noise

    @property
    def c_noise(self):
        return self._c_noise

    @property
    def adversary(self):
        return self._adversary

    @override(Entity)
    def _spawn(self, dim_c: int, dim_p: int):
        if dim_c == 0:
            assert (
                self.silent
            ), f"Agent {self.name} must be silent when world has no communication"
        if self.silent:
            dim_c = 0
        super()._spawn(dim_c, dim_p)

    @override(Entity)
    def _reset(self, env_index: int):
        self.action._reset(env_index)
        self.dynamics.reset(env_index)
        super()._reset(env_index)

    def zero_grad(self):
        self.action.zero_grad()
        self.dynamics.zero_grad()
        super().zero_grad()

    @override(Entity)
    def to(self, device: torch.device):
        super().to(device)
        self.action.to(device)
        for sensor in self.sensors:
            sensor.to(device)

    @override(Entity)
    def render(self, env_index: int = 0) -> "List[Geom]":
        from vmas.simulator import rendering

        geoms = super().render(env_index)
        if len(geoms) == 0:
            return geoms
        for geom in geoms:
            geom.set_color(*self.color, alpha=self._alpha)
        if self._sensors is not None:
            for sensor in self._sensors:
                geoms += sensor.render(env_index=env_index)
        if self._render_action and self.state.force is not None:
            velocity = rendering.Line(
                self.state.pos[env_index],
                self.state.pos[env_index]
                + self.state.force[env_index] * 10 * self.shape.circumscribed_radius(),
                width=2,
            )
            velocity.set_color(*self.color)
            geoms.append(velocity)

        return geoms


# Multi-agent world
class World(TorchVectorizedObject):
    def __init__(
        self,
        batch_dim: int,
        device: torch.device,
        dt: float = 0.1,
        substeps: int = 1,  # if you use joints, higher this value to gain simulation stability
        drag: float = DRAG,
        linear_friction: float = LINEAR_FRICTION,
        angular_friction: float = ANGULAR_FRICTION,
        x_semidim: float = None,
        y_semidim: float = None,
        dim_c: int = 0,
        collision_force: float = COLLISION_FORCE,
        joint_force: float = JOINT_FORCE,
        torque_constraint_force: float = TORQUE_CONSTRAINT_FORCE,
        contact_margin: float = 1e-3,
        gravity: Tuple[float, float] = (0.0, 0.0),
    ):
        assert batch_dim > 0, f"Batch dim must be greater than 0, got {batch_dim}"

        super().__init__(batch_dim, device)
        # list of agents and entities (can change at execution-time!)
        self._agents = []
        self._landmarks = []
        # world dims: no boundaries if none
        self._x_semidim = x_semidim
        self._y_semidim = y_semidim
        # position dimensionality
        self._dim_p = 2
        # communication channel dimensionality
        self._dim_c = dim_c
        # simulation timestep
        self._dt = dt
        self._substeps = substeps
        self._sub_dt = self._dt / self._substeps
        # drag coefficient
        self._drag = drag
        # gravity
        self._gravity = torch.tensor(gravity, device=self.device, dtype=torch.float32)
        # friction coefficients
        self._linear_friction = linear_friction
        self._angular_friction = angular_friction
        # constraint response parameters
        self._collision_force = collision_force
        self._joint_force = joint_force
        self._contact_margin = contact_margin
        self._torque_constraint_force = torque_constraint_force
        # joints
        self._joints = {}
        # Pairs of collidable shapes
        self._collidable_pairs = [
            {Sphere, Sphere},
            {Sphere, Box},
            {Sphere, Line},
            {Line, Line},
            {Line, Box},
            {Box, Box},
        ]
        # Map to save entity indexes
        self.entity_index_map = {}

    def add_agent(self, agent: Agent):
        """Only way to add agents to the world"""
        agent.batch_dim = self._batch_dim
        agent.to(self._device)
        agent._spawn(dim_c=self._dim_c, dim_p=self.dim_p)
        self._agents.append(agent)

    def add_landmark(self, landmark: Landmark):
        """Only way to add landmarks to the world"""
        landmark.batch_dim = self._batch_dim
        landmark.to(self._device)
        landmark._spawn(dim_c=self.dim_c, dim_p=self.dim_p)
        self._landmarks.append(landmark)

    def add_joint(self, joint: Joint):
        assert self._substeps > 1, "For joints, world substeps needs to be more than 1"
        if joint.landmark is not None:
            self.add_landmark(joint.landmark)
        for constraint in joint.joint_constraints:
            self._joints.update(
                {
                    frozenset(
                        {constraint.entity_a.name, constraint.entity_b.name}
                    ): constraint
                }
            )

    def reset(self, env_index: int):
        for e in self.entities:
            e._reset(env_index)

    def zero_grad(self):
        for e in self.entities:
            e.zero_grad()

    @property
    def agents(self) -> List[Agent]:
        return self._agents

    @property
    def landmarks(self) -> List[Landmark]:
        return self._landmarks

    @property
    def x_semidim(self):
        return self._x_semidim

    @property
    def dt(self):
        return self._dt

    @property
    def y_semidim(self):
        return self._y_semidim

    @property
    def dim_p(self):
        return self._dim_p

    @property
    def dim_c(self):
        return self._dim_c

    @property
    def joints(self):
        return self._joints.values()

    # return all entities in the world
    @property
    def entities(self) -> List[Entity]:
        return self._landmarks + self._agents

    # return all agents controllable by external policies
    @property
    def policy_agents(self) -> List[Agent]:
        return [agent for agent in self._agents if agent.action_script is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self) -> List[Agent]:
        return [agent for agent in self._agents if agent.action_script is not None]

    def _cast_ray_to_box(
        self,
        box: Entity,
        ray_origin: Tensor,
        ray_direction: Tensor,
        max_range: float,
    ):
        """
        Inspired from https://tavianator.com/2011/ray_box.html
        Computes distance of ray originating from pos at angle to a box and sets distance to
        max_range if there is no intersection.
        """
        assert ray_origin.ndim == 2 and ray_direction.ndim == 1
        assert ray_origin.shape[0] == ray_direction.shape[0]
        assert isinstance(box.shape, Box)

        pos_origin = ray_origin - box.state.pos
        pos_aabb = TorchUtils.rotate_vector(pos_origin, -box.state.rot)
        ray_dir_world = torch.stack(
            [torch.cos(ray_direction), torch.sin(ray_direction)], dim=-1
        )
        ray_dir_aabb = TorchUtils.rotate_vector(ray_dir_world, -box.state.rot)

        tx1 = (-box.shape.length / 2 - pos_aabb[:, X]) / ray_dir_aabb[:, X]
        tx2 = (box.shape.length / 2 - pos_aabb[:, X]) / ray_dir_aabb[:, X]
        tx = torch.stack([tx1, tx2], dim=-1)
        tmin, _ = torch.min(tx, dim=-1)
        tmax, _ = torch.max(tx, dim=-1)

        ty1 = (-box.shape.width / 2 - pos_aabb[:, Y]) / ray_dir_aabb[:, Y]
        ty2 = (box.shape.width / 2 - pos_aabb[:, Y]) / ray_dir_aabb[:, Y]
        ty = torch.stack([ty1, ty2], dim=-1)
        tymin, _ = torch.min(ty, dim=-1)
        tymax, _ = torch.max(ty, dim=-1)
        tmin, _ = torch.max(torch.stack([tmin, tymin], dim=-1), dim=-1)
        tmax, _ = torch.min(torch.stack([tmax, tymax], dim=-1), dim=-1)

        intersect_aabb = tmin.unsqueeze(1) * ray_dir_aabb + pos_aabb
        intersect_world = (
            TorchUtils.rotate_vector(intersect_aabb, box.state.rot) + box.state.pos
        )

        collision = (tmax >= tmin) & (tmin > 0.0)
        dist = torch.linalg.norm(ray_origin - intersect_world, dim=1)
        dist[~collision] = max_range
        return dist

    def _cast_rays_to_box(
        self,
        box_pos,
        box_rot,
        box_length,
        box_width,
        ray_origin: Tensor,
        ray_direction: Tensor,
        max_range: float,
    ):
        """
        Inspired from https://tavianator.com/2011/ray_box.html
        Computes distance of ray originating from pos at angle to a box and sets distance to
        max_range if there is no intersection.
        """
        batch_size = ray_origin.shape[:-1]
        assert batch_size[0] == self.batch_dim
        assert ray_origin.shape[-1] == 2  # ray_origin is [*batch_size, 2]
        assert (
            ray_direction.shape[:-1] == batch_size
        )  # ray_direction is [*batch_size, n_angles]
        assert box_pos.shape[:-2] == batch_size
        assert box_pos.shape[-1] == 2
        assert box_rot.shape[:-1] == batch_size
        assert box_width.shape[:-1] == batch_size
        assert box_length.shape[:-1] == batch_size

        num_angles = ray_direction.shape[-1]
        n_boxes = box_pos.shape[-2]

        # Expand input to [*batch_size, n_boxes, num_angles, 2]
        ray_origin = (
            ray_origin.unsqueeze(-2)
            .unsqueeze(-2)
            .expand(*batch_size, n_boxes, num_angles, 2)
        )
        box_pos_expanded = box_pos.unsqueeze(-2).expand(
            *batch_size, n_boxes, num_angles, 2
        )
        # Expand input to [*batch_size, n_boxes, num_angles]
        ray_direction = ray_direction.unsqueeze(-2).expand(
            *batch_size, n_boxes, num_angles
        )
        box_rot_expanded = box_rot.unsqueeze(-1).expand(
            *batch_size, n_boxes, num_angles
        )
        box_width_expanded = box_width.unsqueeze(-1).expand(
            *batch_size, n_boxes, num_angles
        )
        box_length_expanded = box_length.unsqueeze(-1).expand(
            *batch_size, n_boxes, num_angles
        )

        # Compute pos_origin and pos_aabb
        pos_origin = ray_origin - box_pos_expanded
        pos_aabb = TorchUtils.rotate_vector(pos_origin, -box_rot_expanded)

        # Calculate ray_dir_world
        ray_dir_world = torch.stack(
            [torch.cos(ray_direction), torch.sin(ray_direction)], dim=-1
        )

        # Calculate ray_dir_aabb
        ray_dir_aabb = TorchUtils.rotate_vector(ray_dir_world, -box_rot_expanded)

        # Calculate tx, ty, tmin, and tmax
        tx1 = (-box_length_expanded / 2 - pos_aabb[..., X]) / ray_dir_aabb[..., X]
        tx2 = (box_length_expanded / 2 - pos_aabb[..., X]) / ray_dir_aabb[..., X]
        tx = torch.stack([tx1, tx2], dim=-1)
        tmin, _ = torch.min(tx, dim=-1)
        tmax, _ = torch.max(tx, dim=-1)

        ty1 = (-box_width_expanded / 2 - pos_aabb[..., Y]) / ray_dir_aabb[..., Y]
        ty2 = (box_width_expanded / 2 - pos_aabb[..., Y]) / ray_dir_aabb[..., Y]
        ty = torch.stack([ty1, ty2], dim=-1)
        tymin, _ = torch.min(ty, dim=-1)
        tymax, _ = torch.max(ty, dim=-1)
        tmin, _ = torch.max(torch.stack([tmin, tymin], dim=-1), dim=-1)
        tmax, _ = torch.min(torch.stack([tmax, tymax], dim=-1), dim=-1)

        # Compute intersect_aabb and intersect_world
        intersect_aabb = tmin.unsqueeze(-1) * ray_dir_aabb + pos_aabb
        intersect_world = (
            TorchUtils.rotate_vector(intersect_aabb, box_rot_expanded)
            + box_pos_expanded
        )

        # Calculate collision and distances
        collision = (tmax >= tmin) & (tmin > 0.0)
        dist = torch.linalg.norm(ray_origin - intersect_world, dim=-1)
        dist[~collision] = max_range
        return dist

    def _cast_ray_to_sphere(
        self,
        sphere: Entity,
        ray_origin: Tensor,
        ray_direction: Tensor,
        max_range: float,
    ):
        ray_dir_world = torch.stack(
            [torch.cos(ray_direction), torch.sin(ray_direction)], dim=-1
        )
        test_point_pos = sphere.state.pos
        line_rot = ray_direction
        line_length = max_range
        line_pos = ray_origin + ray_dir_world * (line_length / 2)

        closest_point = _get_closest_point_line(
            line_pos,
            line_rot.unsqueeze(-1),
            line_length,
            test_point_pos,
            limit_to_line_length=False,
        )

        d = test_point_pos - closest_point
        d_norm = torch.linalg.vector_norm(d, dim=1)
        ray_intersects = d_norm < sphere.shape.radius
        a = sphere.shape.radius**2 - d_norm**2
        m = torch.sqrt(torch.where(a > 0, a, 1e-8))

        u = test_point_pos - ray_origin
        u1 = closest_point - ray_origin

        # Dot product of u and u1
        u_dot_ray = (u * ray_dir_world).sum(-1)
        sphere_is_in_front = u_dot_ray > 0.0
        dist = torch.linalg.vector_norm(u1, dim=1) - m
        dist[~(ray_intersects & sphere_is_in_front)] = max_range

        return dist

    def _cast_rays_to_sphere(
        self,
        sphere_pos,
        sphere_radius,
        ray_origin: Tensor,
        ray_direction: Tensor,
        max_range: float,
    ):
        batch_size = ray_origin.shape[:-1]
        assert batch_size[0] == self.batch_dim
        assert ray_origin.shape[-1] == 2  # ray_origin is [*batch_size, 2]
        assert (
            ray_direction.shape[:-1] == batch_size
        )  # ray_direction is [*batch_size, n_angles]
        assert sphere_pos.shape[:-2] == batch_size
        assert sphere_pos.shape[-1] == 2
        assert sphere_radius.shape[:-1] == batch_size

        num_angles = ray_direction.shape[-1]
        n_spheres = sphere_pos.shape[-2]

        # Expand input to [*batch_size, n_spheres, num_angles, 2]
        ray_origin = (
            ray_origin.unsqueeze(-2)
            .unsqueeze(-2)
            .expand(*batch_size, n_spheres, num_angles, 2)
        )
        sphere_pos_expanded = sphere_pos.unsqueeze(-2).expand(
            *batch_size, n_spheres, num_angles, 2
        )
        # Expand input to [*batch_size, n_spheres, num_angles]
        ray_direction = ray_direction.unsqueeze(-2).expand(
            *batch_size, n_spheres, num_angles
        )
        sphere_radius_expanded = sphere_radius.unsqueeze(-1).expand(
            *batch_size, n_spheres, num_angles
        )

        # Calculate ray_dir_world
        ray_dir_world = torch.stack(
            [torch.cos(ray_direction), torch.sin(ray_direction)], dim=-1
        )

        line_rot = ray_direction.unsqueeze(-1)

        # line_length remains scalar and will be broadcasted as needed
        line_length = max_range

        # Calculate line_pos
        line_pos = ray_origin + ray_dir_world * (line_length / 2)

        # Call the updated _get_closest_point_line function
        closest_point = _get_closest_point_line(
            line_pos,
            line_rot,
            line_length,
            sphere_pos_expanded,
            limit_to_line_length=False,
        )

        # Calculate distances and other metrics
        d = sphere_pos_expanded - closest_point
        d_norm = torch.linalg.vector_norm(d, dim=-1)
        ray_intersects = d_norm < sphere_radius_expanded
        a = sphere_radius_expanded**2 - d_norm**2
        m = torch.sqrt(torch.where(a > 0, a, 1e-8))

        u = sphere_pos_expanded - ray_origin
        u1 = closest_point - ray_origin

        # Dot product of u and u1
        u_dot_ray = (u * ray_dir_world).sum(-1)
        sphere_is_in_front = u_dot_ray > 0.0
        dist = torch.linalg.vector_norm(u1, dim=-1) - m
        dist[~(ray_intersects & sphere_is_in_front)] = max_range

        return dist

    def _cast_ray_to_line(
        self,
        line: Entity,
        ray_origin: Tensor,
        ray_direction: Tensor,
        max_range: float,
    ):
        """
        Inspired by https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect/565282#565282
        Computes distance of ray originating from pos at angle to a line and sets distance to
        max_range if there is no intersection.
        """
        assert ray_origin.ndim == 2 and ray_direction.ndim == 1
        assert ray_origin.shape[0] == ray_direction.shape[0]
        assert isinstance(line.shape, Line)

        p = line.state.pos
        r = (
            torch.stack(
                [
                    torch.cos(line.state.rot.squeeze(1)),
                    torch.sin(line.state.rot.squeeze(1)),
                ],
                dim=-1,
            )
            * line.shape.length
        )

        q = ray_origin
        s = torch.stack(
            [
                torch.cos(ray_direction),
                torch.sin(ray_direction),
            ],
            dim=-1,
        )

        rxs = TorchUtils.cross(r, s)
        t = TorchUtils.cross(q - p, s / rxs)
        u = TorchUtils.cross(q - p, r / rxs)
        d = torch.linalg.norm(u * s, dim=-1)

        perpendicular = rxs == 0.0
        above_line = t > 0.5
        below_line = t < -0.5
        behind_line = u < 0.0
        d[perpendicular.squeeze(-1)] = max_range
        d[above_line.squeeze(-1)] = max_range
        d[below_line.squeeze(-1)] = max_range
        d[behind_line.squeeze(-1)] = max_range
        return d

    def _cast_rays_to_line(
        self,
        line_pos,
        line_rot,
        line_length,
        ray_origin: Tensor,
        ray_direction: Tensor,
        max_range: float,
    ):
        """
        Inspired by https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect/565282#565282
        Computes distance of ray originating from pos at angle to a line and sets distance to
        max_range if there is no intersection.
        """
        batch_size = ray_origin.shape[:-1]
        assert batch_size[0] == self.batch_dim
        assert ray_origin.shape[-1] == 2  # ray_origin is [*batch_size, 2]
        assert (
            ray_direction.shape[:-1] == batch_size
        )  # ray_direction is [*batch_size, n_angles]
        assert line_pos.shape[:-2] == batch_size
        assert line_pos.shape[-1] == 2
        assert line_rot.shape[:-1] == batch_size
        assert line_length.shape[:-1] == batch_size

        num_angles = ray_direction.shape[-1]
        n_lines = line_pos.shape[-2]

        # Expand input to [*batch_size, n_lines, num_angles, 2]
        ray_origin = (
            ray_origin.unsqueeze(-2)
            .unsqueeze(-2)
            .expand(*batch_size, n_lines, num_angles, 2)
        )
        line_pos_expanded = line_pos.unsqueeze(-2).expand(
            *batch_size, n_lines, num_angles, 2
        )
        # Expand input to [*batch_size, n_lines, num_angles]
        ray_direction = ray_direction.unsqueeze(-2).expand(
            *batch_size, n_lines, num_angles
        )
        line_rot_expanded = line_rot.unsqueeze(-1).expand(
            *batch_size, n_lines, num_angles
        )
        line_length_expanded = line_length.unsqueeze(-1).expand(
            *batch_size, n_lines, num_angles
        )

        # Expand line state variables
        r = torch.stack(
            [
                torch.cos(line_rot_expanded),
                torch.sin(line_rot_expanded),
            ],
            dim=-1,
        ) * line_length_expanded.unsqueeze(-1)

        # Calculate q and s
        q = ray_origin
        s = torch.stack(
            [
                torch.cos(ray_direction),
                torch.sin(ray_direction),
            ],
            dim=-1,
        )

        # Calculate rxs, t, u, and d
        rxs = TorchUtils.cross(r, s)
        t = TorchUtils.cross(q - line_pos_expanded, s / rxs)
        u = TorchUtils.cross(q - line_pos_expanded, r / rxs)
        d = torch.linalg.norm(u * s, dim=-1)

        # Handle edge cases
        perpendicular = rxs == 0.0
        above_line = t > 0.5
        below_line = t < -0.5
        behind_line = u < 0.0
        d[perpendicular.squeeze(-1)] = max_range
        d[above_line.squeeze(-1)] = max_range
        d[below_line.squeeze(-1)] = max_range
        d[behind_line.squeeze(-1)] = max_range
        return d

    def cast_ray(
        self,
        entity: Entity,
        angles: Tensor,
        max_range: float,
        entity_filter: Callable[[Entity], bool] = lambda _: False,
    ):
        pos = entity.state.pos

        assert pos.ndim == 2 and angles.ndim == 1
        assert pos.shape[0] == angles.shape[0]

        # Initialize with full max_range to avoid dists being empty when all entities are filtered
        dists = [
            torch.full((self.batch_dim,), fill_value=max_range, device=self.device)
        ]
        for e in self.entities:
            if entity is e or not entity_filter(e):
                continue
            assert e.collides(entity) and entity.collides(
                e
            ), "Rays are only casted among collidables"
            if isinstance(e.shape, Box):
                d = self._cast_ray_to_box(e, pos, angles, max_range)
            elif isinstance(e.shape, Sphere):
                d = self._cast_ray_to_sphere(e, pos, angles, max_range)
            elif isinstance(e.shape, Line):
                d = self._cast_ray_to_line(e, pos, angles, max_range)
            else:
                raise RuntimeError(f"Shape {e.shape} currently not handled by cast_ray")
            dists.append(d)
        dist, _ = torch.min(torch.stack(dists, dim=-1), dim=-1)
        return dist

    def cast_rays(
        self,
        entity: Entity,
        angles: Tensor,
        max_range: float,
        entity_filter: Callable[[Entity], bool] = lambda _: False,
    ):
        pos = entity.state.pos

        # Initialize with full max_range to avoid dists being empty when all entities are filtered
        dists = torch.full_like(
            angles, fill_value=max_range, device=self.device
        ).unsqueeze(-1)
        boxes = []
        spheres = []
        lines = []
        for e in self.entities:
            if entity is e or not entity_filter(e):
                continue
            assert e.collides(entity) and entity.collides(
                e
            ), "Rays are only casted among collidables"
            if isinstance(e.shape, Box):
                boxes.append(e)
            elif isinstance(e.shape, Sphere):
                spheres.append(e)
            elif isinstance(e.shape, Line):
                lines.append(e)
            else:
                raise RuntimeError(f"Shape {e.shape} currently not handled by cast_ray")

        # Boxes
        if len(boxes):
            pos_box = []
            rot_box = []
            length_box = []
            width_box = []
            for box in boxes:
                pos_box.append(box.state.pos)
                rot_box.append(box.state.rot)
                length_box.append(torch.tensor(box.shape.length, device=self.device))
                width_box.append(torch.tensor(box.shape.width, device=self.device))
            pos_box = torch.stack(pos_box, dim=-2)
            rot_box = torch.stack(rot_box, dim=-2)
            length_box = (
                torch.stack(
                    length_box,
                    dim=-1,
                )
                .unsqueeze(0)
                .expand(self.batch_dim, -1)
            )
            width_box = (
                torch.stack(
                    width_box,
                    dim=-1,
                )
                .unsqueeze(0)
                .expand(self.batch_dim, -1)
            )
            dist_boxes = self._cast_rays_to_box(
                pos_box,
                rot_box.squeeze(-1),
                length_box,
                width_box,
                pos,
                angles,
                max_range,
            )
            dists = torch.cat([dists, dist_boxes.transpose(-1, -2)], dim=-1)
        # Spheres
        if len(spheres):
            pos_s = []
            radius_s = []
            for s in spheres:
                pos_s.append(s.state.pos)
                radius_s.append(torch.tensor(s.shape.radius, device=self.device))
            pos_s = torch.stack(pos_s, dim=-2)
            radius_s = (
                torch.stack(
                    radius_s,
                    dim=-1,
                )
                .unsqueeze(0)
                .expand(self.batch_dim, -1)
            )
            dist_spheres = self._cast_rays_to_sphere(
                pos_s,
                radius_s,
                pos,
                angles,
                max_range,
            )
            dists = torch.cat([dists, dist_spheres.transpose(-1, -2)], dim=-1)
        # Lines
        if len(lines):
            pos_l = []
            rot_l = []
            length_l = []
            for line in lines:
                pos_l.append(line.state.pos)
                rot_l.append(line.state.rot)
                length_l.append(torch.tensor(line.shape.length, device=self.device))
            pos_l = torch.stack(pos_l, dim=-2)
            rot_l = torch.stack(rot_l, dim=-2)
            length_l = (
                torch.stack(
                    length_l,
                    dim=-1,
                )
                .unsqueeze(0)
                .expand(self.batch_dim, -1)
            )
            dist_lines = self._cast_rays_to_line(
                pos_l,
                rot_l.squeeze(-1),
                length_l,
                pos,
                angles,
                max_range,
            )
            dists = torch.cat([dists, dist_lines.transpose(-1, -2)], dim=-1)

        dist, _ = torch.min(dists, dim=-1)
        return dist

    def get_distance_from_point(
        self, entity: Entity, test_point_pos, env_index: int = None
    ):
        self._check_batch_index(env_index)

        if isinstance(entity.shape, Sphere):
            delta_pos = entity.state.pos - test_point_pos
            dist = torch.linalg.vector_norm(delta_pos, dim=-1)
            return_value = dist - entity.shape.radius
        elif isinstance(entity.shape, Box):
            closest_point = _get_closest_point_box(
                entity.state.pos,
                entity.state.rot,
                entity.shape.width,
                entity.shape.length,
                test_point_pos,
            )
            distance = torch.linalg.vector_norm(test_point_pos - closest_point, dim=-1)
            return_value = distance - LINE_MIN_DIST
        elif isinstance(entity.shape, Line):
            closest_point = _get_closest_point_line(
                entity.state.pos,
                entity.state.rot,
                entity.shape.length,
                test_point_pos,
            )
            distance = torch.linalg.vector_norm(test_point_pos - closest_point, dim=-1)
            return_value = distance - LINE_MIN_DIST
        else:
            raise RuntimeError("Distance not computable for given entity")
        if env_index is not None:
            return_value = return_value[env_index]
        return return_value

    def get_distance(self, entity_a: Entity, entity_b: Entity, env_index: int = None):
        a_shape = entity_a.shape
        b_shape = entity_b.shape

        if isinstance(a_shape, Sphere) and isinstance(b_shape, Sphere):
            dist = self.get_distance_from_point(entity_a, entity_b.state.pos, env_index)
            return_value = dist - b_shape.radius
        elif (
            isinstance(entity_a.shape, Box)
            and isinstance(entity_b.shape, Sphere)
            or isinstance(entity_b.shape, Box)
            and isinstance(entity_a.shape, Sphere)
        ):
            box, sphere = (
                (entity_a, entity_b)
                if isinstance(entity_b.shape, Sphere)
                else (entity_b, entity_a)
            )
            dist = self.get_distance_from_point(box, sphere.state.pos, env_index)
            return_value = dist - sphere.shape.radius
            is_overlapping = self.is_overlapping(entity_a, entity_b)
            return_value[is_overlapping] = -1
        elif (
            isinstance(entity_a.shape, Line)
            and isinstance(entity_b.shape, Sphere)
            or isinstance(entity_b.shape, Line)
            and isinstance(entity_a.shape, Sphere)
        ):
            line, sphere = (
                (entity_a, entity_b)
                if isinstance(entity_b.shape, Sphere)
                else (entity_b, entity_a)
            )
            dist = self.get_distance_from_point(line, sphere.state.pos, env_index)
            return_value = dist - sphere.shape.radius
        elif isinstance(entity_a.shape, Line) and isinstance(entity_b.shape, Line):
            point_a, point_b = _get_closest_points_line_line(
                entity_a.state.pos,
                entity_a.state.rot,
                entity_a.shape.length,
                entity_b.state.pos,
                entity_b.state.rot,
                entity_b.shape.length,
            )
            dist = torch.linalg.vector_norm(point_a - point_b, dim=1)
            return_value = dist - LINE_MIN_DIST
        elif (
            isinstance(entity_a.shape, Box)
            and isinstance(entity_b.shape, Line)
            or isinstance(entity_b.shape, Box)
            and isinstance(entity_a.shape, Line)
        ):
            box, line = (
                (entity_a, entity_b)
                if isinstance(entity_b.shape, Line)
                else (entity_b, entity_a)
            )
            point_box, point_line = _get_closest_line_box(
                box.state.pos,
                box.state.rot,
                box.shape.width,
                box.shape.length,
                line.state.pos,
                line.state.rot,
                line.shape.length,
            )
            dist = torch.linalg.vector_norm(point_box - point_line, dim=1)
            return_value = dist - LINE_MIN_DIST
        elif isinstance(entity_a.shape, Box) and isinstance(entity_b.shape, Box):
            point_a, point_b = _get_closest_box_box(
                entity_a.state.pos,
                entity_a.state.rot,
                entity_a.shape.width,
                entity_a.shape.length,
                entity_b.state.pos,
                entity_b.state.rot,
                entity_b.shape.width,
                entity_b.shape.length,
            )
            dist = torch.linalg.vector_norm(point_a - point_b, dim=-1)
            return_value = dist - LINE_MIN_DIST
        else:
            raise RuntimeError("Distance not computable for given entities")
        return return_value

    def is_overlapping(self, entity_a: Entity, entity_b: Entity, env_index: int = None):
        a_shape = entity_a.shape
        b_shape = entity_b.shape
        self._check_batch_index(env_index)

        # Sphere sphere, sphere line, line line, line box, box box
        if (
            (isinstance(a_shape, Sphere) and isinstance(b_shape, Sphere))
            or (
                (
                    isinstance(entity_a.shape, Line)
                    and isinstance(entity_b.shape, Sphere)
                    or isinstance(entity_b.shape, Line)
                    and isinstance(entity_a.shape, Sphere)
                )
            )
            or (isinstance(entity_a.shape, Line) and isinstance(entity_b.shape, Line))
            or (
                isinstance(entity_a.shape, Box)
                and isinstance(entity_b.shape, Line)
                or isinstance(entity_b.shape, Box)
                and isinstance(entity_a.shape, Line)
            )
            or (isinstance(entity_a.shape, Box) and isinstance(entity_b.shape, Box))
        ):
            return self.get_distance(entity_a, entity_b, env_index) < 0
        elif (
            isinstance(entity_a.shape, Box)
            and isinstance(entity_b.shape, Sphere)
            or isinstance(entity_b.shape, Box)
            and isinstance(entity_a.shape, Sphere)
        ):
            box, sphere = (
                (entity_a, entity_b)
                if isinstance(entity_b.shape, Sphere)
                else (entity_b, entity_a)
            )
            closest_point = _get_closest_point_box(
                box.state.pos,
                box.state.rot,
                box.shape.width,
                box.shape.length,
                sphere.state.pos,
            )

            distance_sphere_closest_point = torch.linalg.vector_norm(
                sphere.state.pos - closest_point, dim=-1
            )
            distance_sphere_box = torch.linalg.vector_norm(
                sphere.state.pos - box.state.pos, dim=-1
            )
            distance_closest_point_box = torch.linalg.vector_norm(
                box.state.pos - closest_point, dim=-1
            )
            dist_min = sphere.shape.radius + LINE_MIN_DIST
            return_value = (distance_sphere_box < distance_closest_point_box) + (
                distance_sphere_closest_point < dist_min
            )
        else:
            raise RuntimeError("Overlap not computable for give entities")
        if env_index is not None:
            return_value = return_value[env_index]
        return return_value

    # update state of the world
    def step(self):
        self.entity_index_map = {e: i for i, e in enumerate(self.entities)}

        for substep in range(self._substeps):
            self.forces_dict = {
                e: torch.zeros(
                    self._batch_dim,
                    self._dim_p,
                    device=self.device,
                    dtype=torch.float32,
                )
                for e in self.entities
            }
            self.torques_dict = {
                e: torch.zeros(
                    self._batch_dim,
                    1,
                    device=self.device,
                    dtype=torch.float32,
                )
                for e in self.entities
            }

            for entity in self.entities:
                if isinstance(entity, Agent):
                    # apply agent force controls
                    self._apply_action_force(entity)
                    # apply agent torque controls
                    self._apply_action_torque(entity)
                # apply friction
                self._apply_friction_force(entity)
                # apply gravity
                self._apply_gravity(entity)

            self._apply_vectorized_enviornment_force()

            for entity in self.entities:
                # integrate physical state
                self._integrate_state(entity, substep)

        # update non-differentiable comm state
        if self._dim_c > 0:
            for agent in self._agents:
                self._update_comm_state(agent)

    # gather agent action forces
    def _apply_action_force(self, agent: Agent):
        if agent.movable:
            if agent.max_f is not None:
                agent.state.force = TorchUtils.clamp_with_norm(
                    agent.state.force, agent.max_f
                )
            if agent.f_range is not None:
                agent.state.force = torch.clamp(
                    agent.state.force, -agent.f_range, agent.f_range
                )
            self.forces_dict[agent] = self.forces_dict[agent] + agent.state.force

    def _apply_action_torque(self, agent: Agent):
        if agent.rotatable:
            if agent.max_t is not None:
                agent.state.torque = TorchUtils.clamp_with_norm(
                    agent.state.torque, agent.max_t
                )
            if agent.t_range is not None:
                agent.state.torque = torch.clamp(
                    agent.state.torque, -agent.t_range, agent.t_range
                )

            self.torques_dict[agent] = self.torques_dict[agent] + agent.state.torque

    def _apply_gravity(self, entity: Entity):
        if entity.movable:
            if not (self._gravity == 0.0).all():
                self.forces_dict[entity] = (
                    self.forces_dict[entity] + entity.mass * self._gravity
                )
            if entity.gravity is not None:
                self.forces_dict[entity] = (
                    self.forces_dict[entity] + entity.mass * entity.gravity
                )

    def _apply_friction_force(self, entity: Entity):
        def get_friction_force(vel, coeff, force, mass):
            speed = torch.linalg.vector_norm(vel, dim=-1)
            static = speed == 0
            static_exp = static.unsqueeze(-1).expand(vel.shape)

            if not isinstance(coeff, Tensor):
                coeff = torch.full_like(force, coeff, device=self.device)
            coeff = coeff.expand(force.shape)

            friction_force_constant = coeff * mass

            friction_force = -(
                vel / torch.where(static, 1e-8, speed).unsqueeze(-1)
            ) * torch.minimum(
                friction_force_constant, (vel.abs() / self._sub_dt) * mass
            )
            friction_force = torch.where(static_exp, 0.0, friction_force)

            return friction_force

        if entity.linear_friction is not None:
            self.forces_dict[entity] = self.forces_dict[entity] + get_friction_force(
                entity.state.vel,
                entity.linear_friction,
                self.forces_dict[entity],
                entity.mass,
            )
        elif self._linear_friction > 0:
            self.forces_dict[entity] = self.forces_dict[entity] + get_friction_force(
                entity.state.vel,
                self._linear_friction,
                self.forces_dict[entity],
                entity.mass,
            )
        if entity.angular_friction is not None:
            self.torques_dict[entity] = self.torques_dict[entity] + get_friction_force(
                entity.state.ang_vel,
                entity.angular_friction,
                self.torques_dict[entity],
                entity.moment_of_inertia,
            )
        elif self._angular_friction > 0:
            self.torques_dict[entity] = self.torques_dict[entity] + get_friction_force(
                entity.state.ang_vel,
                self._angular_friction,
                self.torques_dict[entity],
                entity.moment_of_inertia,
            )

    def _apply_vectorized_enviornment_force(self):
        s_s = []
        l_s = []
        b_s = []
        l_l = []
        b_l = []
        b_b = []
        joints = []
        for a, entity_a in enumerate(self.entities):
            for b, entity_b in enumerate(self.entities):
                if b <= a:
                    continue
                joint = self._joints.get(
                    frozenset({entity_a.name, entity_b.name}), None
                )
                if joint is not None:
                    joints.append(joint)
                    if joint.dist == 0:
                        continue
                if not self.collides(entity_a, entity_b):
                    continue
                if isinstance(entity_a.shape, Sphere) and isinstance(
                    entity_b.shape, Sphere
                ):
                    s_s.append((entity_a, entity_b))
                elif (
                    isinstance(entity_a.shape, Line)
                    and isinstance(entity_b.shape, Sphere)
                    or isinstance(entity_b.shape, Line)
                    and isinstance(entity_a.shape, Sphere)
                ):
                    line, sphere = (
                        (entity_a, entity_b)
                        if isinstance(entity_b.shape, Sphere)
                        else (entity_b, entity_a)
                    )
                    l_s.append((line, sphere))
                elif isinstance(entity_a.shape, Line) and isinstance(
                    entity_b.shape, Line
                ):
                    l_l.append((entity_a, entity_b))
                elif (
                    isinstance(entity_a.shape, Box)
                    and isinstance(entity_b.shape, Sphere)
                    or isinstance(entity_b.shape, Box)
                    and isinstance(entity_a.shape, Sphere)
                ):
                    box, sphere = (
                        (entity_a, entity_b)
                        if isinstance(entity_b.shape, Sphere)
                        else (entity_b, entity_a)
                    )
                    b_s.append((box, sphere))
                elif (
                    isinstance(entity_a.shape, Box)
                    and isinstance(entity_b.shape, Line)
                    or isinstance(entity_b.shape, Box)
                    and isinstance(entity_a.shape, Line)
                ):
                    box, line = (
                        (entity_a, entity_b)
                        if isinstance(entity_b.shape, Line)
                        else (entity_b, entity_a)
                    )
                    b_l.append((box, line))
                elif isinstance(entity_a.shape, Box) and isinstance(
                    entity_b.shape, Box
                ):
                    b_b.append((entity_a, entity_b))
                else:
                    raise AssertionError()
        # Joints
        self._vectorized_joint_constraints(joints)

        # Sphere and sphere
        self._sphere_sphere_vectorized_collision(s_s)
        # Line and sphere
        self._sphere_line_vectorized_collision(l_s)
        # Line and line
        self._line_line_vectorized_collision(l_l)
        # Box and sphere
        self._box_sphere_vectorized_collision(b_s)
        # Box and line
        self._box_line_vectorized_collision(b_l)
        # Box and box
        self._box_box_vectorized_collision(b_b)

    def update_env_forces(self, entity_a, f_a, t_a, entity_b, f_b, t_b):
        if entity_a.movable:
            self.forces_dict[entity_a] = self.forces_dict[entity_a] + f_a
        if entity_a.rotatable:
            self.torques_dict[entity_a] = self.torques_dict[entity_a] + t_a
        if entity_b.movable:
            self.forces_dict[entity_b] = self.forces_dict[entity_b] + f_b
        if entity_b.rotatable:
            self.torques_dict[entity_b] = self.torques_dict[entity_b] + t_b

    def _vectorized_joint_constraints(self, joints):
        if len(joints):
            pos_a = []
            pos_b = []
            pos_joint_a = []
            pos_joint_b = []
            dist = []
            rotate = []
            rot_a = []
            rot_b = []
            joint_rot = []
            for joint in joints:
                entity_a = joint.entity_a
                entity_b = joint.entity_b
                pos_joint_a.append(joint.pos_point(entity_a))
                pos_joint_b.append(joint.pos_point(entity_b))
                pos_a.append(entity_a.state.pos)
                pos_b.append(entity_b.state.pos)
                dist.append(torch.tensor(joint.dist, device=self.device))
                rotate.append(torch.tensor(joint.rotate, device=self.device))
                rot_a.append(entity_a.state.rot)
                rot_b.append(entity_b.state.rot)
                joint_rot.append(
                    torch.tensor(joint.fixed_rotation, device=self.device)
                    .unsqueeze(-1)
                    .expand(self.batch_dim, 1)
                    if isinstance(joint.fixed_rotation, float)
                    else joint.fixed_rotation
                )
            pos_a = torch.stack(pos_a, dim=-2)
            pos_b = torch.stack(pos_b, dim=-2)
            pos_joint_a = torch.stack(pos_joint_a, dim=-2)
            pos_joint_b = torch.stack(pos_joint_b, dim=-2)
            rot_a = torch.stack(rot_a, dim=-2)
            rot_b = torch.stack(rot_b, dim=-2)
            dist = (
                torch.stack(
                    dist,
                    dim=-1,
                )
                .unsqueeze(0)
                .expand(self.batch_dim, -1)
            )
            rotate_prior = torch.stack(
                rotate,
                dim=-1,
            )
            rotate = rotate_prior.unsqueeze(0).expand(self.batch_dim, -1).unsqueeze(-1)
            joint_rot = torch.stack(joint_rot, dim=-2)

            (force_a_attractive, force_b_attractive,) = self._get_constraint_forces(
                pos_joint_a,
                pos_joint_b,
                dist_min=dist,
                attractive=True,
                force_multiplier=self._joint_force,
            )
            force_a_repulsive, force_b_repulsive = self._get_constraint_forces(
                pos_joint_a,
                pos_joint_b,
                dist_min=dist,
                attractive=False,
                force_multiplier=self._joint_force,
            )
            force_a = force_a_attractive + force_a_repulsive
            force_b = force_b_attractive + force_b_repulsive
            r_a = pos_joint_a - pos_a
            r_b = pos_joint_b - pos_b

            torque_a_rotate = TorchUtils.compute_torque(force_a, r_a)
            torque_b_rotate = TorchUtils.compute_torque(force_b, r_b)

            torque_a_fixed, torque_b_fixed = self._get_constraint_torques(
                rot_a, rot_b + joint_rot, force_multiplier=self._torque_constraint_force
            )

            torque_a = torch.where(
                rotate, torque_a_rotate, torque_a_rotate + torque_a_fixed
            )
            torque_b = torch.where(
                rotate, torque_b_rotate, torque_b_rotate + torque_b_fixed
            )

            for i, joint in enumerate(joints):
                self.update_env_forces(
                    joint.entity_a,
                    force_a[:, i],
                    torque_a[:, i],
                    joint.entity_b,
                    force_b[:, i],
                    torque_b[:, i],
                )

    def _sphere_sphere_vectorized_collision(self, s_s):
        if len(s_s):
            pos_s_a = []
            pos_s_b = []
            radius_s_a = []
            radius_s_b = []
            for s_a, s_b in s_s:
                pos_s_a.append(s_a.state.pos)
                pos_s_b.append(s_b.state.pos)
                radius_s_a.append(torch.tensor(s_a.shape.radius, device=self.device))
                radius_s_b.append(torch.tensor(s_b.shape.radius, device=self.device))

            pos_s_a = torch.stack(pos_s_a, dim=-2)
            pos_s_b = torch.stack(pos_s_b, dim=-2)
            radius_s_a = (
                torch.stack(
                    radius_s_a,
                    dim=-1,
                )
                .unsqueeze(0)
                .expand(self.batch_dim, -1)
            )
            radius_s_b = (
                torch.stack(
                    radius_s_b,
                    dim=-1,
                )
                .unsqueeze(0)
                .expand(self.batch_dim, -1)
            )
            force_a, force_b = self._get_constraint_forces(
                pos_s_a,
                pos_s_b,
                dist_min=radius_s_a + radius_s_b,
                force_multiplier=self._collision_force,
            )

            for i, (entity_a, entity_b) in enumerate(s_s):
                self.update_env_forces(
                    entity_a,
                    force_a[:, i],
                    0,
                    entity_b,
                    force_b[:, i],
                    0,
                )

    def _sphere_line_vectorized_collision(self, l_s):
        if len(l_s):
            pos_l = []
            pos_s = []
            rot_l = []
            radius_s = []
            length_l = []
            for line, sphere in l_s:
                pos_l.append(line.state.pos)
                pos_s.append(sphere.state.pos)
                rot_l.append(line.state.rot)
                radius_s.append(torch.tensor(sphere.shape.radius, device=self.device))
                length_l.append(torch.tensor(line.shape.length, device=self.device))
            pos_l = torch.stack(pos_l, dim=-2)
            pos_s = torch.stack(pos_s, dim=-2)
            rot_l = torch.stack(rot_l, dim=-2)
            radius_s = (
                torch.stack(
                    radius_s,
                    dim=-1,
                )
                .unsqueeze(0)
                .expand(self.batch_dim, -1)
            )
            length_l = (
                torch.stack(
                    length_l,
                    dim=-1,
                )
                .unsqueeze(0)
                .expand(self.batch_dim, -1)
            )

            closest_point = _get_closest_point_line(pos_l, rot_l, length_l, pos_s)
            force_sphere, force_line = self._get_constraint_forces(
                pos_s,
                closest_point,
                dist_min=radius_s + LINE_MIN_DIST,
                force_multiplier=self._collision_force,
            )
            r = closest_point - pos_l
            torque_line = TorchUtils.compute_torque(force_line, r)

            for i, (entity_a, entity_b) in enumerate(l_s):
                self.update_env_forces(
                    entity_a,
                    force_line[:, i],
                    torque_line[:, i],
                    entity_b,
                    force_sphere[:, i],
                    0,
                )

    def _line_line_vectorized_collision(self, l_l):
        if len(l_l):
            pos_l_a = []
            pos_l_b = []
            rot_l_a = []
            rot_l_b = []
            length_l_a = []
            length_l_b = []
            for l_a, l_b in l_l:
                pos_l_a.append(l_a.state.pos)
                pos_l_b.append(l_b.state.pos)
                rot_l_a.append(l_a.state.rot)
                rot_l_b.append(l_b.state.rot)
                length_l_a.append(torch.tensor(l_a.shape.length, device=self.device))
                length_l_b.append(torch.tensor(l_b.shape.length, device=self.device))
            pos_l_a = torch.stack(pos_l_a, dim=-2)
            pos_l_b = torch.stack(pos_l_b, dim=-2)
            rot_l_a = torch.stack(rot_l_a, dim=-2)
            rot_l_b = torch.stack(rot_l_b, dim=-2)
            length_l_a = (
                torch.stack(
                    length_l_a,
                    dim=-1,
                )
                .unsqueeze(0)
                .expand(self.batch_dim, -1)
            )
            length_l_b = (
                torch.stack(
                    length_l_b,
                    dim=-1,
                )
                .unsqueeze(0)
                .expand(self.batch_dim, -1)
            )

            point_a, point_b = _get_closest_points_line_line(
                pos_l_a,
                rot_l_a,
                length_l_a,
                pos_l_b,
                rot_l_b,
                length_l_b,
            )
            force_a, force_b = self._get_constraint_forces(
                point_a,
                point_b,
                dist_min=LINE_MIN_DIST,
                force_multiplier=self._collision_force,
            )
            r_a = point_a - pos_l_a
            r_b = point_b - pos_l_b

            torque_a = TorchUtils.compute_torque(force_a, r_a)
            torque_b = TorchUtils.compute_torque(force_b, r_b)
            for i, (entity_a, entity_b) in enumerate(l_l):
                self.update_env_forces(
                    entity_a,
                    force_a[:, i],
                    torque_a[:, i],
                    entity_b,
                    force_b[:, i],
                    torque_b[:, i],
                )

    def _box_sphere_vectorized_collision(self, b_s):
        if len(b_s):
            pos_box = []
            pos_sphere = []
            rot_box = []
            length_box = []
            width_box = []
            not_hollow_box = []
            radius_sphere = []
            for box, sphere in b_s:
                pos_box.append(box.state.pos)
                pos_sphere.append(sphere.state.pos)
                rot_box.append(box.state.rot)
                length_box.append(torch.tensor(box.shape.length, device=self.device))
                width_box.append(torch.tensor(box.shape.width, device=self.device))
                not_hollow_box.append(
                    torch.tensor(not box.shape.hollow, device=self.device)
                )
                radius_sphere.append(
                    torch.tensor(sphere.shape.radius, device=self.device)
                )
            pos_box = torch.stack(pos_box, dim=-2)
            pos_sphere = torch.stack(pos_sphere, dim=-2)
            rot_box = torch.stack(rot_box, dim=-2)
            length_box = (
                torch.stack(
                    length_box,
                    dim=-1,
                )
                .unsqueeze(0)
                .expand(self.batch_dim, -1)
            )
            width_box = (
                torch.stack(
                    width_box,
                    dim=-1,
                )
                .unsqueeze(0)
                .expand(self.batch_dim, -1)
            )
            not_hollow_box_prior = torch.stack(
                not_hollow_box,
                dim=-1,
            )
            not_hollow_box = not_hollow_box_prior.unsqueeze(0).expand(
                self.batch_dim, -1
            )
            radius_sphere = (
                torch.stack(
                    radius_sphere,
                    dim=-1,
                )
                .unsqueeze(0)
                .expand(self.batch_dim, -1)
            )

            closest_point_box = _get_closest_point_box(
                pos_box,
                rot_box,
                width_box,
                length_box,
                pos_sphere,
            )

            inner_point_box = closest_point_box
            d = torch.zeros_like(radius_sphere, device=self.device, dtype=torch.float)
            if not_hollow_box_prior.any():
                inner_point_box_hollow, d_hollow = _get_inner_point_box(
                    pos_sphere, closest_point_box, pos_box
                )
                cond = not_hollow_box.unsqueeze(-1).expand(inner_point_box.shape)
                inner_point_box = torch.where(
                    cond, inner_point_box_hollow, inner_point_box
                )
                d = torch.where(not_hollow_box, d_hollow, d)

            force_sphere, force_box = self._get_constraint_forces(
                pos_sphere,
                inner_point_box,
                dist_min=radius_sphere + LINE_MIN_DIST + d,
                force_multiplier=self._collision_force,
            )
            r = closest_point_box - pos_box
            torque_box = TorchUtils.compute_torque(force_box, r)

            for i, (entity_a, entity_b) in enumerate(b_s):
                self.update_env_forces(
                    entity_a,
                    force_box[:, i],
                    torque_box[:, i],
                    entity_b,
                    force_sphere[:, i],
                    0,
                )

    def _box_line_vectorized_collision(self, b_l):
        if len(b_l):
            pos_box = []
            pos_line = []
            rot_box = []
            rot_line = []
            length_box = []
            width_box = []
            not_hollow_box = []
            length_line = []
            for box, line in b_l:
                pos_box.append(box.state.pos)
                pos_line.append(line.state.pos)
                rot_box.append(box.state.rot)
                rot_line.append(line.state.rot)
                length_box.append(torch.tensor(box.shape.length, device=self.device))
                width_box.append(torch.tensor(box.shape.width, device=self.device))
                not_hollow_box.append(
                    torch.tensor(not box.shape.hollow, device=self.device)
                )
                length_line.append(torch.tensor(line.shape.length, device=self.device))
            pos_box = torch.stack(pos_box, dim=-2)
            pos_line = torch.stack(pos_line, dim=-2)
            rot_box = torch.stack(rot_box, dim=-2)
            rot_line = torch.stack(rot_line, dim=-2)
            length_box = (
                torch.stack(
                    length_box,
                    dim=-1,
                )
                .unsqueeze(0)
                .expand(self.batch_dim, -1)
            )
            width_box = (
                torch.stack(
                    width_box,
                    dim=-1,
                )
                .unsqueeze(0)
                .expand(self.batch_dim, -1)
            )
            not_hollow_box_prior = torch.stack(
                not_hollow_box,
                dim=-1,
            )
            not_hollow_box = not_hollow_box_prior.unsqueeze(0).expand(
                self.batch_dim, -1
            )
            length_line = (
                torch.stack(
                    length_line,
                    dim=-1,
                )
                .unsqueeze(0)
                .expand(self.batch_dim, -1)
            )

            point_box, point_line = _get_closest_line_box(
                pos_box,
                rot_box,
                width_box,
                length_box,
                pos_line,
                rot_line,
                length_line,
            )

            inner_point_box = point_box
            d = torch.zeros_like(length_line, device=self.device, dtype=torch.float)
            if not_hollow_box_prior.any():
                inner_point_box_hollow, d_hollow = _get_inner_point_box(
                    point_line, point_box, pos_box
                )
                cond = not_hollow_box.unsqueeze(-1).expand(inner_point_box.shape)
                inner_point_box = torch.where(
                    cond, inner_point_box_hollow, inner_point_box
                )
                d = torch.where(not_hollow_box, d_hollow, d)

            force_box, force_line = self._get_constraint_forces(
                inner_point_box,
                point_line,
                dist_min=LINE_MIN_DIST + d,
                force_multiplier=self._collision_force,
            )
            r_box = point_box - pos_box
            r_line = point_line - pos_line

            torque_box = TorchUtils.compute_torque(force_box, r_box)
            torque_line = TorchUtils.compute_torque(force_line, r_line)

            for i, (entity_a, entity_b) in enumerate(b_l):
                self.update_env_forces(
                    entity_a,
                    force_box[:, i],
                    torque_box[:, i],
                    entity_b,
                    force_line[:, i],
                    torque_line[:, i],
                )

    def _box_box_vectorized_collision(self, b_b):
        if len(b_b):
            pos_box = []
            pos_box2 = []
            rot_box = []
            rot_box2 = []
            length_box = []
            width_box = []
            not_hollow_box = []
            length_box2 = []
            width_box2 = []
            not_hollow_box2 = []
            for box, box2 in b_b:
                pos_box.append(box.state.pos)
                rot_box.append(box.state.rot)
                length_box.append(torch.tensor(box.shape.length, device=self.device))
                width_box.append(torch.tensor(box.shape.width, device=self.device))
                not_hollow_box.append(
                    torch.tensor(not box.shape.hollow, device=self.device)
                )
                pos_box2.append(box2.state.pos)
                rot_box2.append(box2.state.rot)
                length_box2.append(torch.tensor(box2.shape.length, device=self.device))
                width_box2.append(torch.tensor(box2.shape.width, device=self.device))
                not_hollow_box2.append(
                    torch.tensor(not box2.shape.hollow, device=self.device)
                )

            pos_box = torch.stack(pos_box, dim=-2)
            rot_box = torch.stack(rot_box, dim=-2)
            length_box = (
                torch.stack(
                    length_box,
                    dim=-1,
                )
                .unsqueeze(0)
                .expand(self.batch_dim, -1)
            )
            width_box = (
                torch.stack(
                    width_box,
                    dim=-1,
                )
                .unsqueeze(0)
                .expand(self.batch_dim, -1)
            )
            not_hollow_box_prior = torch.stack(
                not_hollow_box,
                dim=-1,
            )
            not_hollow_box = not_hollow_box_prior.unsqueeze(0).expand(
                self.batch_dim, -1
            )
            pos_box2 = torch.stack(pos_box2, dim=-2)
            rot_box2 = torch.stack(rot_box2, dim=-2)
            length_box2 = (
                torch.stack(
                    length_box2,
                    dim=-1,
                )
                .unsqueeze(0)
                .expand(self.batch_dim, -1)
            )
            width_box2 = (
                torch.stack(
                    width_box2,
                    dim=-1,
                )
                .unsqueeze(0)
                .expand(self.batch_dim, -1)
            )
            not_hollow_box2_prior = torch.stack(
                not_hollow_box2,
                dim=-1,
            )
            not_hollow_box2 = not_hollow_box2_prior.unsqueeze(0).expand(
                self.batch_dim, -1
            )

            point_a, point_b = _get_closest_box_box(
                pos_box,
                rot_box,
                width_box,
                length_box,
                pos_box2,
                rot_box2,
                width_box2,
                length_box2,
            )

            inner_point_a = point_a
            d_a = torch.zeros_like(length_box, device=self.device, dtype=torch.float)
            if not_hollow_box_prior.any():
                inner_point_box_hollow, d_hollow = _get_inner_point_box(
                    point_b, point_a, pos_box
                )
                cond = not_hollow_box.unsqueeze(-1).expand(inner_point_a.shape)
                inner_point_a = torch.where(cond, inner_point_box_hollow, inner_point_a)
                d_a = torch.where(not_hollow_box, d_hollow, d_a)

            inner_point_b = point_b
            d_b = torch.zeros_like(length_box2, device=self.device, dtype=torch.float)
            if not_hollow_box2_prior.any():
                inner_point_box2_hollow, d_hollow2 = _get_inner_point_box(
                    point_a, point_b, pos_box2
                )
                cond = not_hollow_box2.unsqueeze(-1).expand(inner_point_b.shape)
                inner_point_b = torch.where(
                    cond, inner_point_box2_hollow, inner_point_b
                )
                d_b = torch.where(not_hollow_box2, d_hollow2, d_b)

            force_a, force_b = self._get_constraint_forces(
                inner_point_a,
                inner_point_b,
                dist_min=d_a + d_b + LINE_MIN_DIST,
                force_multiplier=self._collision_force,
            )
            r_a = point_a - pos_box
            r_b = point_b - pos_box2
            torque_a = TorchUtils.compute_torque(force_a, r_a)
            torque_b = TorchUtils.compute_torque(force_b, r_b)

            for i, (entity_a, entity_b) in enumerate(b_b):
                self.update_env_forces(
                    entity_a,
                    force_a[:, i],
                    torque_a[:, i],
                    entity_b,
                    force_b[:, i],
                    torque_b[:, i],
                )

    def collides(self, a: Entity, b: Entity) -> bool:
        if (not a.collides(b)) or (not b.collides(a)) or a is b:
            return False
        a_shape = a.shape
        b_shape = b.shape
        if not a.movable and not a.rotatable and not b.movable and not b.rotatable:
            return False
        if not {a_shape.__class__, b_shape.__class__} in self._collidable_pairs:
            return False
        if not (
            torch.linalg.vector_norm(a.state.pos - b.state.pos, dim=-1)
            <= a.shape.circumscribed_radius() + b.shape.circumscribed_radius()
        ).any():
            return False

        return True

    def _get_constraint_forces(
        self,
        pos_a: Tensor,
        pos_b: Tensor,
        dist_min,
        force_multiplier: float,
        attractive: bool = False,
    ) -> Tensor:
        min_dist = 1e-6
        delta_pos = pos_a - pos_b
        dist = torch.linalg.vector_norm(delta_pos, dim=-1)
        sign = -1 if attractive else 1

        # softmax penetration
        k = self._contact_margin
        penetration = (
            torch.logaddexp(
                torch.tensor(0.0, dtype=torch.float32, device=self.device),
                (dist_min - dist) * sign / k,
            )
            * k
        )
        force = (
            sign
            * force_multiplier
            * delta_pos
            / torch.where(dist > 0, dist, 1e-8).unsqueeze(-1)
            * penetration.unsqueeze(-1)
        )
        force = torch.where((dist < min_dist).unsqueeze(-1), 0.0, force)
        if not attractive:
            force = torch.where((dist > dist_min).unsqueeze(-1), 0.0, force)
        else:
            force = torch.where((dist < dist_min).unsqueeze(-1), 0.0, force)
        return force, -force

    def _get_constraint_torques(
        self,
        rot_a: Tensor,
        rot_b: Tensor,
        force_multiplier: float = TORQUE_CONSTRAINT_FORCE,
    ) -> Tensor:
        min_delta_rot = 1e-9
        delta_rot = rot_a - rot_b
        abs_delta_rot = torch.linalg.vector_norm(delta_rot, dim=-1).unsqueeze(-1)

        # softmax penetration
        k = 1
        penetration = k * (torch.exp(abs_delta_rot / k) - 1)

        torque = force_multiplier * delta_rot.sign() * penetration
        torque = torch.where((abs_delta_rot < min_delta_rot), 0.0, torque)

        return -torque, torque

    # integrate physical state
    # uses semi-implicit euler with sub-stepping
    def _integrate_state(self, entity: Entity, substep: int):
        if entity.movable:
            # Compute translation
            if substep == 0:
                if entity.drag is not None:
                    entity.state.vel = entity.state.vel * (1 - entity.drag)
                else:
                    entity.state.vel = entity.state.vel * (1 - self._drag)
            accel = self.forces_dict[entity] / entity.mass
            entity.state.vel = entity.state.vel + accel * self._sub_dt
            if entity.max_speed is not None:
                entity.state.vel = TorchUtils.clamp_with_norm(
                    entity.state.vel, entity.max_speed
                )
            if entity.v_range is not None:
                entity.state.vel = entity.state.vel.clamp(
                    -entity.v_range, entity.v_range
                )
            new_pos = entity.state.pos + entity.state.vel * self._sub_dt
            entity.state.pos = torch.stack(
                [
                    (
                        new_pos[..., X].clamp(-self._x_semidim, self._x_semidim)
                        if self._x_semidim is not None
                        else new_pos[..., X]
                    ),
                    (
                        new_pos[..., Y].clamp(-self._y_semidim, self._y_semidim)
                        if self._y_semidim is not None
                        else new_pos[..., Y]
                    ),
                ],
                dim=-1,
            )

        if entity.rotatable:
            # Compute rotation
            if substep == 0:
                if entity.drag is not None:
                    entity.state.ang_vel = entity.state.ang_vel * (1 - entity.drag)
                else:
                    entity.state.ang_vel = entity.state.ang_vel * (1 - self._drag)
            entity.state.ang_vel = (
                entity.state.ang_vel
                + (self.torques_dict[entity] / entity.moment_of_inertia) * self._sub_dt
            )
            entity.state.rot = entity.state.rot + entity.state.ang_vel * self._sub_dt

    def _update_comm_state(self, agent):
        # set communication state (directly for now)
        if not agent.silent:
            agent.state.c = agent.action.c

    @override(TorchVectorizedObject)
    def to(self, device: torch.device):
        super().to(device)
        for e in self.entities:
            e.to(device)
