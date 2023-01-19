#  Copyright (c) 2022-2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

from __future__ import annotations

import math
import typing
from abc import ABC, abstractmethod
from typing import Callable, List, Tuple

import torch
from torch import Tensor

from vmas.simulator.joints import JointConstraint, Joint
from vmas.simulator.sensors import Sensor
from vmas.simulator.utils import (
    Color,
    X,
    Y,
    override,
    LINE_MIN_DIST,
    COLLISION_FORCE,
    JOINT_FORCE,
    Observable,
    DRAG,
    LINEAR_FRICTION,
    ANGULAR_FRICTION,
    TorchUtils,
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
        for attr in [self.pos, self.rot, self.vel, self.ang_vel]:
            if attr is not None:
                if env_index is None:
                    attr[:] = 0.0
                else:
                    attr[env_index] = 0.0

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

    @override(EntityState)
    def _reset(self, env_index: typing.Optional[int]):
        for attr in [self.c]:
            if attr is not None:
                if env_index is None:
                    attr[:] = 0.0
                else:
                    attr[env_index] = 0.0
        super()._reset(env_index)

    @override(EntityState)
    def _spawn(self, dim_c: int, dim_p: int):
        if dim_c > 0:
            self.c = torch.zeros(
                self.batch_dim, dim_c, device=self.device, dtype=torch.float32
            )
        super()._spawn(dim_c, dim_p)


# action of an agent
class Action(TorchVectorizedObject):
    def __init__(
        self,
        u_range: float,
        u_multiplier: float,
        u_noise: float,
        u_rot_range: float,
        u_rot_multiplier: float,
        u_rot_noise: float,
    ):
        super().__init__()
        # physical motor noise amount
        self._u_noise = u_noise
        # control range
        self._u_range = u_range
        # agent action is a force multiplied by this amount
        self._u_multiplier = u_multiplier

        # physical motor noise amount
        self._u_rot_noise = u_rot_noise
        # control range
        self._u_rot_range = u_rot_range
        # agent action is a force multiplied by this amount
        self._u_rot_multiplier = u_rot_multiplier

        # physical action
        self._u = None
        # rotation action
        self._u_rot = None
        # communication_action
        self._c = None

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
    def u_rot(self):
        return self._u_rot

    @u_rot.setter
    def u_rot(self, u_rot: Tensor):
        assert (
            self._batch_dim is not None and self._device is not None
        ), "First add an agent to the world before setting its action"
        assert (
            u_rot.shape[0] == self._batch_dim
        ), f"Action must match batch dim, got {u_rot.shape[0]}, expected {self._batch_dim}"

        self._u_rot = u_rot.to(self._device)

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
    def u_rot_range(self):
        return self._u_rot_range

    @property
    def u_rot_multiplier(self):
        return self._u_rot_multiplier

    @property
    def u_rot_noise(self):
        return self._u_rot_noise

    def _reset(self, env_index: typing.Optional[int]):
        for attr in [self.u, self.u_rot, self.c]:
            if attr is not None:
                if env_index is None:
                    attr[:] = 0.0
                else:
                    attr[env_index] = 0.0


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
        shape: Shape = Sphere(),
        v_range: float = None,
        max_speed: float = None,
        color=Color.GRAY,
        is_joint: bool = False,
        drag: float = None,
        linear_friction: float = None,
        angular_friction: float = None,
        collision_filter: Callable[[Entity], bool] = lambda _: True,
    ):
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
        if batch_index is None:
            if len(new.shape) > 1 and new.shape[0] == self.batch_dim:
                prop.fset(entity, new)
            else:
                prop.fset(entity, new.repeat(self.batch_dim, 1))
        else:
            new = new.to(self.device)
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
        shape: Shape = Sphere(),
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
            collision_filter,
        )


# properties of agent entities
class Agent(Entity):
    def __init__(
        self,
        name: str,
        shape: Shape = Sphere(),
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
        u_noise: float = None,
        u_range: float = 1.0,
        u_multiplier: float = 1.0,
        u_rot_noise: float = None,
        u_rot_range: float = 0.0,
        u_rot_multiplier: float = 1.0,
        action_script: Callable[[Agent, World], None] = None,
        sensors: List[Sensor] = None,
        c_noise: float = None,
        silent: bool = True,
        adversary: bool = False,
        drag: float = None,
        linear_friction: float = None,
        angular_friction: float = None,
        collision_filter: Callable[[Entity], bool] = lambda _: True,
        render_action: bool = False,
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
            collision_filter=collision_filter,
        )
        if obs_range == 0.0:
            assert sensors is None, f"Blind agent cannot have sensors, got {sensors}"

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

        # action
        self._action = Action(
            u_range=u_range,
            u_multiplier=u_multiplier,
            u_noise=u_noise,
            u_rot_range=u_rot_range,
            u_rot_multiplier=u_rot_multiplier,
            u_rot_noise=u_rot_noise,
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
            self._action.u.shape[1] == world.dim_p
        ), f"Scripted physical action of agent {self.name} has wrong shape"
        assert (
            (self._action.u / self.u_multiplier).abs() <= self.u_range
        ).all(), f"Scripted physical action of {self.name} is out of range"
        if self.u_rot_range != 0:
            assert (
                self._action.u_rot is not None
            ), f"Action script of {self.name} should set u_rot action"
            assert (
                self._action.u_rot.shape[1] == 1
            ), f"Scripted physical rotation action of agent {self.name} has wrong shape"
            assert (
                (self._action.u_rot / self._action.u_rot_multiplier).abs()
                <= self.u_rot_range
            ).all(), f"Scripted physical rotation action of {self.name} is out of range"

    @property
    def u_range(self):
        return self.action.u_range

    @property
    def u_rot_range(self):
        return self.action.u_rot_range

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
    def u_rot_multiplier(self):
        return self.action.u_rot_multiplier

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
        super()._reset(env_index)

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
        if self._render_action and self.action.u is not None:
            velocity = rendering.Line(
                self.state.pos[env_index],
                self.state.pos[env_index]
                + self.action.u[env_index] * 10 * self.shape.circumscribed_radius(),
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
        # Horizontal unit vector
        self._normal_vector = torch.tensor(
            [1.0, 0.0], dtype=torch.float32, device=self.device
        ).repeat(self._batch_dim, 1)

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

        closest_point = self._get_closest_point_line(
            line_pos, line_rot, line_length, test_point_pos, limit_to_line_length=False
        )

        d = test_point_pos - closest_point
        d_norm = torch.linalg.vector_norm(d, dim=1)
        ray_intersects = d_norm < sphere.shape.radius
        m = torch.sqrt(sphere.shape.radius**2 - d_norm**2)

        u = test_point_pos - ray_origin
        u1 = closest_point - ray_origin

        # Dot product of u and u1
        u_dot_ray = torch.einsum("bs,bs->b", u, ray_dir_world)
        sphere_is_in_front = u_dot_ray > 0.0
        dist = torch.linalg.vector_norm(u1, dim=1) - m
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
        t = TorchUtils.cross(q - p, s / rxs.unsqueeze(1))
        u = TorchUtils.cross(q - p, r / rxs.unsqueeze(1))
        d = torch.linalg.norm(u.unsqueeze(1) * s, dim=1)

        perpendicular = rxs == 0.0
        above_line = t > 0.5
        below_line = t < -0.5
        behind_line = u < 0.0
        d[perpendicular] = max_range
        d[above_line] = max_range
        d[below_line] = max_range
        d[behind_line] = max_range
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
                assert False, f"Shape {e.shape} currently not handled by cast_ray"
            dists.append(d)
        dist, _ = torch.min(torch.stack(dists, dim=-1), dim=-1)
        return dist

    def get_distance_from_point(
        self, entity: Entity, test_point_pos, env_index: int = None
    ):
        self._check_batch_index(env_index)

        if isinstance(entity.shape, Sphere):
            delta_pos = entity.state.pos - test_point_pos
            dist = torch.linalg.vector_norm(delta_pos, dim=1)
            return_value = dist - entity.shape.radius
        elif isinstance(entity.shape, Box):
            closest_point = self._get_closest_point_box(entity, test_point_pos)
            distance = torch.linalg.vector_norm(test_point_pos - closest_point, dim=1)
            return_value = distance - LINE_MIN_DIST
        elif isinstance(entity.shape, Line):
            closest_point = self._get_closest_point_line(
                entity.state.pos, entity.state.rot, entity.shape.length, test_point_pos
            )
            distance = torch.linalg.vector_norm(test_point_pos - closest_point, dim=1)
            return_value = distance - LINE_MIN_DIST
        else:
            assert False, "Distance not computable for given entity"
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
            point_a, point_b = self._get_closest_points_line_line(
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
            point_box, point_line = self._get_closest_line_box(
                box, line.state.pos, line.state.rot, line.shape.length
            )
            dist = torch.linalg.vector_norm(point_box - point_line, dim=1)
            return_value = dist - LINE_MIN_DIST
        elif isinstance(entity_a.shape, Box) and isinstance(entity_b.shape, Box):
            point_a, point_b = self._get_closest_box_box(entity_a, entity_b)
            dist = torch.linalg.vector_norm(point_a - point_b, dim=1)
            return_value = dist - LINE_MIN_DIST
        else:
            assert False, "Distance not computable for given entities"
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
            closest_point = self._get_closest_point_box(box, sphere.state.pos)

            distance_sphere_closest_point = torch.linalg.vector_norm(
                sphere.state.pos - closest_point, dim=1
            )
            distance_sphere_box = torch.linalg.vector_norm(
                sphere.state.pos - box.state.pos, dim=1
            )
            distance_closest_point_box = torch.linalg.vector_norm(
                box.state.pos - closest_point, dim=1
            )
            dist_min = sphere.shape.radius + LINE_MIN_DIST
            return_value = (distance_sphere_box < distance_closest_point_box) + (
                distance_sphere_closest_point < dist_min
            )
        else:
            assert False, "Overlap not computable for give entities"
        if env_index is not None:
            return_value = return_value[env_index]
        return return_value

    # update state of the world
    def step(self):
        # forces
        self.force = torch.zeros(
            self._batch_dim,
            len(self.entities),
            self._dim_p,
            device=self.device,
            dtype=torch.float32,
        )
        self.torque = torch.zeros(
            self._batch_dim,
            len(self.entities),
            1,
            device=self.device,
            dtype=torch.float32,
        )

        for substep in range(self._substeps):
            # gather forces applied to entities
            self.force[:] = 0
            self.torque[:] = 0

            for i, entity in enumerate(self.entities):
                # apply agent force controls
                self._apply_action_force(entity, i)
                # apply agent torque controls
                self._apply_action_torque(entity, i)
                # apply friction
                self._apply_friction_force(entity, i)
                # apply gravity
                self._apply_gravity(entity, i)
                # apply environment forces (constraints)
                self._apply_environment_force(entity, i)
            for i, entity in enumerate(self.entities):
                # integrate physical state
                self._integrate_state(entity, i, substep)

        # update non-differentiable comm state
        if self._dim_c > 0:
            for agent in self._agents:
                self._update_comm_state(agent)

    # gather agent action forces
    def _apply_action_force(self, entity: Entity, index: int):
        if isinstance(entity, Agent):
            # set applied forces
            if entity.movable:
                noise = (
                    torch.randn(
                        *entity.action.u.shape, device=self.device, dtype=torch.float32
                    )
                    * entity.u_noise
                    if entity.u_noise
                    else 0.0
                )
                entity.action.u += noise
                if entity.max_f is not None:
                    entity.action.u = TorchUtils.clamp_with_norm(
                        entity.action.u, entity.max_f
                    )
                if entity.f_range is not None:
                    entity.action.u = torch.clamp(
                        entity.action.u, -entity.f_range, entity.f_range
                    )
                self.force[:, index] += entity.action.u
            assert not self.force.isnan().any()

    def _apply_action_torque(self, entity: Entity, index: int):
        if isinstance(entity, Agent) and entity.u_rot_range != 0:
            # set applied forces
            if entity.rotatable:
                noise = (
                    torch.randn(
                        *entity.action.u_rot.shape,
                        device=self.device,
                        dtype=torch.float32,
                    )
                    * entity.action.u_rot_noise
                    if entity.action.u_rot_noise
                    else 0.0
                )
                entity.action.u_rot = entity.action.u_rot + noise
                if len(entity.action.u_rot.shape) == 1:
                    entity.action.u_rot.unsqueeze_(-1)
                if entity.max_t is not None:
                    entity.action.u_rot = TorchUtils.clamp_with_norm(
                        entity.action.u_rot, entity.max_t
                    )
                if entity.t_range is not None:
                    entity.action.u_rot = torch.clamp(
                        entity.action.u_rot, -entity.t_range, entity.t_range
                    )
                self.torque[:, index] += entity.action.u_rot
            assert not self.torque.isnan().any()

    def _apply_gravity(self, entity: Entity, index: int):
        if not (self._gravity == 0.0).all():
            if entity.movable:
                self.force[:, index] += entity.mass * self._gravity

    def _apply_friction_force(self, entity: Entity, index: int):
        def get_friction_force(vel, coeff, force, mass):
            speed = torch.linalg.vector_norm(vel, dim=1)
            static = speed == 0

            friction_force_constant = torch.full_like(
                force, coeff * mass, device=self.device
            )
            friction_force = -(vel / speed.unsqueeze(-1)) * torch.minimum(
                friction_force_constant, (vel.abs() / self._sub_dt) * mass
            )
            friction_force[static] = 0

            return friction_force

        if entity.linear_friction is not None:
            self.force[:, index] += get_friction_force(
                entity.state.vel,
                entity.linear_friction,
                self.force[:, index],
                entity.mass,
            )
        elif self._linear_friction > 0:
            self.force[:, index] += get_friction_force(
                entity.state.vel,
                self._linear_friction,
                self.force[:, index],
                entity.mass,
            )
        if entity.angular_friction is not None:
            self.torque[:, index] += get_friction_force(
                entity.state.ang_vel,
                entity.angular_friction,
                self.torque[:, index],
                entity.moment_of_inertia,
            )
        elif self._angular_friction > 0:
            self.torque[:, index] += get_friction_force(
                entity.state.ang_vel,
                self._angular_friction,
                self.torque[:, index],
                entity.moment_of_inertia,
            )

    # gather physical forces acting on entities
    def _apply_environment_force(self, entity_a: Entity, a: int):
        def apply_env_forces(f_a, t_a, f_b, t_b):
            if entity_a.movable:
                self.force[:, a] += f_a
            if entity_a.rotatable:
                self.torque[:, a] += t_a
            if entity_b.movable:
                self.force[:, b] += f_b
            if entity_b.rotatable:
                self.torque[:, b] += t_b

        for b, entity_b in enumerate(self.entities):
            if b <= a:
                continue
            # Joints
            if frozenset({entity_a.name, entity_b.name}) in self._joints:
                joint = self._joints[frozenset({entity_a.name, entity_b.name})]
                apply_env_forces(*self._get_joint_forces(entity_a, entity_b, joint))
                if joint.dist == 0:
                    continue
            # Collisions
            if self.collides(entity_a, entity_b):
                apply_env_forces(*self._get_collision_force(entity_a, entity_b))

    def collides(self, a: Entity, b: Entity) -> bool:
        if (not a.collides(b)) or (not b.collides(a)) or a is b:
            return False
        a_shape = a.shape
        b_shape = b.shape
        if (
            torch.linalg.vector_norm(a.state.pos - b.state.pos, dim=1)
            > a.shape.circumscribed_radius() + b.shape.circumscribed_radius()
        ).all():
            return False
        if not a.movable and not a.rotatable and not b.movable and not b.rotatable:
            return False
        if {a_shape.__class__, b_shape.__class__} in self._collidable_pairs:
            return True
        return False

    def _get_joint_forces(
        self, entity_a: Entity, entity_b: Entity, joint: JointConstraint
    ):
        pos_point_a = joint.pos_point(entity_a)
        pos_point_b = joint.pos_point(entity_b)
        force_a_attractive, force_b_attractive = self._get_constraint_forces(
            pos_point_a,
            pos_point_b,
            dist_min=joint.dist,
            attractive=True,
            force_multiplier=self._joint_force,
        )
        force_a_repulsive, force_b_repulsive = self._get_constraint_forces(
            pos_point_a,
            pos_point_b,
            dist_min=joint.dist,
            attractive=False,
            force_multiplier=self._joint_force,
        )
        force_a = force_a_attractive + force_a_repulsive
        force_b = force_b_attractive + force_b_repulsive
        r_a = pos_point_a - entity_a.state.pos
        r_b = pos_point_b - entity_b.state.pos
        if joint.rotate:
            torque_a = TorchUtils.compute_torque(force_a, r_a)
            torque_b = TorchUtils.compute_torque(force_b, r_b)
        else:
            torque_a = torque_b = 0
        return force_a, torque_a, force_b, torque_b

    # get collision forces for any contact between two entities
    # collisions among lines and boxes or these objects among themselves will be ignored
    def _get_collision_force(self, entity_a, entity_b):
        # Sphere and sphere
        if isinstance(entity_a.shape, Sphere) and isinstance(entity_b.shape, Sphere):
            force_a, force_b = self._get_constraint_forces(
                entity_a.state.pos,
                entity_b.state.pos,
                dist_min=entity_a.shape.radius + entity_b.shape.radius,
                force_multiplier=self._collision_force,
            )
            torque_a = 0
            torque_b = 0
        # Sphere and line
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
            closest_point = self._get_closest_point_line(
                line.state.pos, line.state.rot, line.shape.length, sphere.state.pos
            )
            force_sphere, force_line = self._get_constraint_forces(
                sphere.state.pos,
                closest_point,
                dist_min=sphere.shape.radius + LINE_MIN_DIST,
                force_multiplier=self._collision_force,
            )
            r = closest_point - line.state.pos
            torque_line = TorchUtils.compute_torque(force_line, r)

            force_a, torque_a, force_b, torque_b = (
                (force_sphere, 0, force_line, torque_line)
                if isinstance(entity_a.shape, Sphere)
                else (force_line, torque_line, force_sphere, 0)
            )
        # Sphere and box
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
            closest_point_box = self._get_closest_point_box(box, sphere.state.pos)
            if not box.shape.hollow:
                inner_point_box, d = self.get_inner_point_box(
                    sphere.state.pos, closest_point_box, box
                )
            else:
                inner_point_box = closest_point_box
                d = 0
            force_sphere, force_box = self._get_constraint_forces(
                sphere.state.pos,
                inner_point_box,
                dist_min=sphere.shape.radius + LINE_MIN_DIST + d,
                force_multiplier=self._collision_force,
            )
            r = closest_point_box - box.state.pos
            torque_box = TorchUtils.compute_torque(force_box, r)

            force_a, torque_a, force_b, torque_b = (
                (force_sphere, 0, force_box, torque_box)
                if isinstance(entity_a.shape, Sphere)
                else (force_box, torque_box, force_sphere, 0)
            )
        # Line and line
        elif isinstance(entity_a.shape, Line) and isinstance(entity_b.shape, Line):
            point_a, point_b = self._get_closest_points_line_line(
                entity_a.state.pos,
                entity_a.state.rot,
                entity_a.shape.length,
                entity_b.state.pos,
                entity_b.state.rot,
                entity_b.shape.length,
            )
            force_a, force_b = self._get_constraint_forces(
                point_a,
                point_b,
                dist_min=LINE_MIN_DIST,
                force_multiplier=self._collision_force,
            )
            r_a = point_a - entity_a.state.pos
            r_b = point_b - entity_b.state.pos

            torque_a = TorchUtils.compute_torque(force_a, r_a)
            torque_b = TorchUtils.compute_torque(force_b, r_b)
        # Line and box
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
            point_box, point_line = self._get_closest_line_box(
                box, line.state.pos, line.state.rot, line.shape.length
            )
            if not box.shape.hollow:
                inner_point_box, d = self.get_inner_point_box(
                    point_line, point_box, box
                )
            else:
                inner_point_box = point_box
                d = 0

            force_box, force_line = self._get_constraint_forces(
                inner_point_box,
                point_line,
                dist_min=LINE_MIN_DIST + d,
                force_multiplier=self._collision_force,
            )
            r_box = point_box - box.state.pos
            r_line = point_line - line.state.pos

            torque_box = TorchUtils.compute_torque(force_box, r_box)
            torque_line = TorchUtils.compute_torque(force_line, r_line)

            force_a, torque_a, force_b, torque_b = (
                (force_line, torque_line, force_box, torque_box)
                if isinstance(entity_a.shape, Line)
                else (force_box, torque_box, force_line, torque_line)
            )
        # Box and box
        elif isinstance(entity_a.shape, Box) and isinstance(entity_b.shape, Box):
            point_a, point_b = self._get_closest_box_box(entity_a, entity_b)
            if not entity_a.shape.hollow:
                inner_point_a, d_a = self.get_inner_point_box(
                    point_b, point_a, entity_a
                )
            else:
                inner_point_a = point_a
                d_a = 0
            if not entity_b.shape.hollow:
                inner_point_b, d_b = self.get_inner_point_box(
                    point_a, point_b, entity_b
                )
            else:
                inner_point_b = point_b
                d_b = 0
            force_a, force_b = self._get_constraint_forces(
                inner_point_a,
                inner_point_b,
                dist_min=d_a + d_b + LINE_MIN_DIST,
                force_multiplier=self._collision_force,
            )
            r_a = point_a - entity_a.state.pos
            r_b = point_b - entity_b.state.pos
            torque_a = TorchUtils.compute_torque(force_a, r_a)
            torque_b = TorchUtils.compute_torque(force_b, r_b)
        else:
            assert False

        return force_a, torque_a, force_b, torque_b

    def get_inner_point_box(self, outside_point, surface_point, box):
        v = surface_point - outside_point
        u = box.state.pos - surface_point
        v_norm = torch.linalg.vector_norm(v, dim=1).unsqueeze(-1)
        x_magnitude = torch.einsum("bs,bs->b", v, u).unsqueeze(-1) / v_norm
        x = (v / v_norm) * x_magnitude
        x[v_norm.repeat(1, 2) == 0] = surface_point[v_norm.repeat(1, 2) == 0]
        x_magnitude[v_norm == 0] = 0
        return surface_point + x, torch.abs(x_magnitude.squeeze(-1))

    def _get_closest_box_box(self, a: Entity, b: Entity):
        lines_a = self._get_all_lines_box(a)
        lines_b = self._get_all_lines_box(b)

        point_pairs = []

        for line_a in lines_a:
            point_line_a_b = self._get_closest_line_box(b, *line_a)
            point_pairs.append((point_line_a_b[1], point_line_a_b[0]))
        for line_b in lines_b:
            point_pairs.append(self._get_closest_line_box(a, *line_b))

        closest_point_1 = torch.full(
            (self.batch_dim, self.dim_p),
            float("inf"),
            device=self.device,
            dtype=torch.float32,
        )
        closest_point_2 = torch.full(
            (self.batch_dim, self.dim_p),
            float("inf"),
            device=self.device,
            dtype=torch.float32,
        )
        distance = torch.full(
            (self.batch_dim,), float("inf"), device=self.device, dtype=torch.float32
        )
        for p1, p2 in point_pairs:
            d = torch.linalg.vector_norm(p1 - p2, dim=1)
            is_closest = d < distance
            closest_point_1[is_closest] = p1[is_closest]
            closest_point_2[is_closest] = p2[is_closest]
            distance[is_closest] = d[is_closest]

        return closest_point_1, closest_point_2

    @staticmethod
    def get_line_extrema(line_pos, line_rot, line_length):
        x = (line_length / 2) * torch.cos(line_rot)
        y = (line_length / 2) * torch.sin(line_rot)
        xy = torch.cat([x, y], dim=1)

        point_a = line_pos + xy
        point_b = line_pos - xy

        return point_a, point_b

    def _get_closest_points_line_line(
        self, line_pos, line_rot, line_length, line2_pos, line2_rot, line2_length
    ):

        point_a1, point_a2 = self.get_line_extrema(line_pos, line_rot, line_length)
        point_b1, point_b2 = self.get_line_extrema(line2_pos, line2_rot, line2_length)

        point_i, d_i = self._get_intersection_point_line_line(
            point_a1, point_a2, point_b1, point_b2
        )

        point_a1_line_b = self._get_closest_point_line(
            line2_pos, line2_rot, line2_length, point_a1
        )
        point_a2_line_b = self._get_closest_point_line(
            line2_pos, line2_rot, line2_length, point_a2
        )
        point_b1_line_a = self._get_closest_point_line(
            line_pos, line_rot, line_length, point_b1
        )
        point_b2_line_a = self._get_closest_point_line(
            line_pos, line_rot, line_length, point_b2
        )
        point_pairs = (
            (point_a1, point_a1_line_b),
            (point_a2, point_a2_line_b),
            (point_b1_line_a, point_b1),
            (point_b2_line_a, point_b2),
        )

        closest_point_1 = torch.full(
            (self.batch_dim, self.dim_p),
            float("inf"),
            device=self.device,
            dtype=torch.float32,
        )
        closest_point_2 = torch.full(
            (self.batch_dim, self.dim_p),
            float("inf"),
            device=self.device,
            dtype=torch.float32,
        )
        min_distance = torch.full(
            (self.batch_dim,), float("inf"), device=self.device, dtype=torch.float32
        )
        for p1, p2 in point_pairs:
            d = torch.linalg.vector_norm(p1 - p2, dim=1)
            is_closest = d < min_distance
            closest_point_1[is_closest] = p1[is_closest]
            closest_point_2[is_closest] = p2[is_closest]
            min_distance[is_closest] = d[is_closest]

        closest_point_1[d_i == 0] = point_i[d_i == 0]
        closest_point_2[d_i == 0] = point_i[d_i == 0]

        return closest_point_1, closest_point_2

    def _get_intersection_point_line_line(self, point_a1, point_a2, point_b1, point_b2):
        """
        Taken from:
        https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
        """
        r = point_a2 - point_a1
        s = point_b2 - point_b1
        p = point_a1
        q = point_b1
        cross_q_minus_p_r = TorchUtils.cross(q - p, r)
        cross_q_minus_p_s = TorchUtils.cross(q - p, s)
        cross_r_s = TorchUtils.cross(r, s)
        u = cross_q_minus_p_r / cross_r_s
        t = cross_q_minus_p_s / cross_r_s
        t_in_range = (0 <= t) * (t <= 1)
        u_in_range = (0 <= u) * (u <= 1)

        cross_r_s_is_zero = cross_r_s == 0

        distance = torch.full(
            (self.batch_dim,), float("inf"), device=self.device, dtype=torch.float32
        )
        point = torch.full(
            (self.batch_dim, self.dim_p),
            float("inf"),
            device=self.device,
            dtype=torch.float32,
        )

        point[
            ~cross_r_s_is_zero.repeat(1, 2)
            * u_in_range.repeat(1, 2)
            * t_in_range.repeat(1, 2)
        ] = (p + t * r)[
            ~cross_r_s_is_zero.repeat(1, 2)
            * u_in_range.repeat(1, 2)
            * t_in_range.repeat(1, 2)
        ]
        distance[
            ~cross_r_s_is_zero.squeeze(-1)
            * u_in_range.squeeze(-1)
            * t_in_range.squeeze(-1)
        ] = 0

        return point, distance

    def _get_closest_point_box(self, box: Entity, test_point_pos):
        assert isinstance(box.shape, Box)

        closest_points = self._get_all_points_box(box, test_point_pos)
        closest_point = torch.full(
            (self.batch_dim, self.dim_p),
            float("inf"),
            device=self.device,
            dtype=torch.float32,
        )
        distance = torch.full(
            (self.batch_dim,), float("inf"), device=self.device, dtype=torch.float32
        )
        for p in closest_points:
            d = torch.linalg.vector_norm(test_point_pos - p, dim=1)
            is_closest = d < distance
            closest_point[is_closest] = p[is_closest]
            distance[is_closest] = d[is_closest]

        assert not closest_point.isinf().any()

        return closest_point

    def _get_all_lines_box(self, box: Entity):
        # Rotate normal vector by the angle of the box
        rotated_vector = TorchUtils.rotate_vector(self._normal_vector, box.state.rot)
        rotated_vector2 = TorchUtils.rotate_vector(
            self._normal_vector, box.state.rot + torch.pi / 2
        )

        # Middle points of the sides
        p1 = box.state.pos + rotated_vector * (box.shape.length / 2)
        p2 = box.state.pos - rotated_vector * (box.shape.length / 2)
        p3 = box.state.pos + rotated_vector2 * (box.shape.width / 2)
        p4 = box.state.pos - rotated_vector2 * (box.shape.width / 2)

        lines = []
        for i, p in enumerate([p1, p2, p3, p4]):
            lines.append(
                [
                    p,
                    box.state.rot + torch.pi / 2 if i <= 1 else box.state.rot,
                    box.shape.width if i <= 1 else box.shape.length,
                ]
            )
        return lines

    def _get_closest_line_box(self, box: Entity, line_pos, line_rot, line_length):
        lines = self._get_all_lines_box(box)

        closest_point_1 = torch.full(
            (self.batch_dim, self.dim_p),
            float("inf"),
            device=self.device,
            dtype=torch.float32,
        )
        closest_point_2 = torch.full(
            (self.batch_dim, self.dim_p),
            float("inf"),
            device=self.device,
            dtype=torch.float32,
        )
        distance = torch.full(
            (self.batch_dim,), float("inf"), device=self.device, dtype=torch.float32
        )

        for box_line in lines:
            p_box, p_line = self._get_closest_points_line_line(
                *box_line, line_pos, line_rot, line_length
            )
            d = torch.linalg.vector_norm(p_box - p_line, dim=1)
            is_closest = d < distance
            closest_point_1[is_closest] = p_box[is_closest]
            closest_point_2[is_closest] = p_line[is_closest]
            distance[is_closest] = d[is_closest]
        return closest_point_1, closest_point_2

    def _get_all_points_box(self, box: Entity, test_point_pos):
        assert isinstance(box.shape, Box)

        lines = self._get_all_lines_box(box)
        closest_points = []

        for line in lines:
            point = self._get_closest_point_line(
                *line,
                test_point_pos,
            )
            closest_points.append(point)

        return closest_points

    def _get_closest_point_line(
        self,
        line_pos,
        line_rot,
        line_length,
        test_point_pos,
        limit_to_line_length: bool = True,
    ):
        # Rotate it by the angle of the line
        rotated_vector = TorchUtils.rotate_vector(self._normal_vector, line_rot)
        # Get distance between line and sphere
        delta_pos = line_pos - test_point_pos
        # Dot product of distance and line vector
        dot_p = torch.einsum("bs,bs->b", delta_pos, rotated_vector).unsqueeze(-1)
        # Coordinates of the closes point
        sign = torch.sign(dot_p)
        distance_from_line_center = (
            torch.min(
                torch.abs(dot_p),
                torch.tensor(line_length / 2, dtype=torch.float32, device=self.device),
            )
            if limit_to_line_length
            else torch.abs(dot_p)
        )
        closest_point = line_pos - sign * distance_from_line_center * rotated_vector
        return closest_point

    def _get_constraint_forces(
        self,
        pos_a: Tensor,
        pos_b: Tensor,
        dist_min: float,
        force_multiplier: float,
        attractive: bool = False,
    ) -> Tensor:
        min_dist = 1e-6
        delta_pos = pos_a - pos_b
        dist = torch.linalg.vector_norm(delta_pos, dim=1)
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
            / dist.unsqueeze(-1)
            * penetration.unsqueeze(-1)
        )
        force[dist < min_dist] = 0
        if not attractive:
            force[dist > dist_min] = 0
        else:
            force[dist < dist_min] = 0
        assert not force.isnan().any()
        return force, -force

    # integrate physical state
    # uses semi-implicit euler with sub-stepping
    def _integrate_state(self, entity: Entity, index: int, substep: int):
        if entity.movable:
            # Compute translation
            if substep == 0:
                if entity.drag is not None:
                    entity.state.vel = entity.state.vel * (1 - entity.drag)
                else:
                    entity.state.vel = entity.state.vel * (1 - self._drag)
            accel = self.force[:, index] / entity.mass
            entity.state.vel += accel * self._sub_dt
            if entity.max_speed is not None:
                entity.state.vel = TorchUtils.clamp_with_norm(
                    entity.state.vel, entity.max_speed
                )
            if entity.v_range is not None:
                entity.state.vel = entity.state.vel.clamp(
                    -entity.v_range, entity.v_range
                )
            new_pos = entity.state.pos + entity.state.vel * self._sub_dt
            if self._x_semidim is not None:
                new_pos[:, X] = torch.clamp(
                    new_pos[:, X], -self._x_semidim, self._x_semidim
                )
            if self._y_semidim is not None:
                new_pos[:, Y] = torch.clamp(
                    new_pos[:, Y], -self._y_semidim, self._y_semidim
                )
            entity.state.pos = new_pos
        if entity.rotatable:
            # Compute rotation
            if substep == 0:
                if entity.drag is not None:
                    entity.state.ang_vel = entity.state.ang_vel * (1 - entity.drag)
                else:
                    entity.state.ang_vel = entity.state.ang_vel * (1 - self._drag)
            entity.state.ang_vel += (
                self.torque[:, index] / entity.moment_of_inertia
            ) * self._sub_dt
            entity.state.rot += entity.state.ang_vel * self._sub_dt

    def _update_comm_state(self, agent):
        # set communication state (directly for now)
        if not agent.silent:
            noise = (
                torch.randn(
                    *agent.action.c.shape, device=self.device, dtype=torch.float32
                )
                * agent.c_noise
                if agent.c_noise
                else 0.0
            )
            agent.state.c = agent.action.c + noise

    @override(TorchVectorizedObject)
    def to(self, device: torch.device):
        super().to(device)
        for e in self.entities:
            e.to(device)
