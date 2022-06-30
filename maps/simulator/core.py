#  Copyright (c) 2022. Matteo Bettini
#  All rights reserved.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, List, Union, Tuple

import torch
from torch import Tensor

from maps.simulator.utils import Color, SensorType, X, Y, override, LINE_MIN_DIST


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
        assert self._device is None, "You can set device only once"
        self._device = device

    def _check_batch_index(self, batch_index: int):
        if batch_index is not None:
            assert (
                0 <= batch_index < self.batch_dim
            ), f"Index must be between 0 and {self.batch_dim}, got {batch_index}"


class Shape(ABC):
    @abstractmethod
    def moment_of_inertia(self, mass: float):
        raise NotImplementedError


class Box(Shape):
    def __init__(self, length: float = 0.3, width: float = 0.1):
        super().__init__()
        assert length > 0, f"Length must be > 0, got {length}"
        assert width > 0, f"Width must be > 0, got {length}"
        self._length = length
        self._width = width

    @property
    def length(self):
        return self._length

    @property
    def width(self):
        return self._width

    def moment_of_inertia(self, mass: float):
        return (1 / 12) * mass * (self.length**2 + self.width**2)


class Sphere(Shape):
    def __init__(self, radius: float = 0.05):
        super().__init__()
        assert radius > 0, f"Radius must be > 0, got {radius}"
        self._radius = radius

    @property
    def radius(self):
        return self._radius

    def moment_of_inertia(self, mass: float):
        return (1 / 2) * mass * self.radius**2


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


# action of an agent
class Action(TorchVectorizedObject):
    def __init__(self):
        super().__init__()
        # physical action
        self._u = None
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


# properties and state of physical world entity
class Entity(TorchVectorizedObject, ABC):
    def __init__(
        self,
        name: str,
        movable: bool = False,
        rotatable: bool = False,
        collide: bool = True,
        density: float = 25.0,  # Unused for now
        mass: float = 1.0,
        shape: Shape = Sphere(),
        max_speed: float = None,
        color=Color.GRAY,
    ):
        super().__init__()
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
        # max speed and accel
        self._max_speed = max_speed
        # color
        self._color = color
        # shape
        self._shape = shape
        # state
        self._state = EntityState()
        # entity goal
        self._goal = None
        # Render the entity
        self._render = None

    @TorchVectorizedObject.batch_dim.setter
    def batch_dim(self, batch_dim: int):
        TorchVectorizedObject.batch_dim.fset(self, batch_dim)
        self._state.batch_dim = batch_dim

    @TorchVectorizedObject.device.setter
    def device(self, device: torch.device):
        TorchVectorizedObject.device.fset(self, device)
        self._state.device = device

    @property
    def render(self):
        if self._render is None:
            self.reset_render()
        return self._render

    def reset_render(self):
        self._render = torch.full((self.batch_dim,), True, device=self.device)

    @property
    def mass(self):
        return self._mass

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

    @goal.setter
    def goal(self, goal: Entity):
        self._goal = goal

    def _spawn(self, dim_p: int):
        self.state.pos = torch.zeros(
            self.batch_dim, dim_p, device=self.device, dtype=torch.float32
        )
        self.state.vel = torch.zeros(
            self.batch_dim, dim_p, device=self.device, dtype=torch.float32
        )
        self.state.rot = torch.zeros(
            self.batch_dim, 1, device=self.device, dtype=torch.float32
        )
        self.state.ang_vel = torch.zeros(
            self.batch_dim, 1, device=self.device, dtype=torch.float32
        )

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
        max_speed: float = None,
        color=Color.GRAY,
    ):
        super().__init__(
            name,
            movable,
            rotatable,
            collide,
            density,  # Unused for now
            mass,
            shape,
            max_speed,
            color,
        )


# properties of agent entities
class Agent(Entity):
    def __init__(
        self,
        name: str,
        shape: Shape = Sphere(),
        movable: bool = True,
        rotatable: bool = False,
        collide: bool = True,
        density: float = 25.0,  # Unused for now
        mass: float = 1.0,
        max_speed: float = None,
        color=Color.BLUE,
        obs_range: float = None,
        obs_noise: float = None,
        u_noise: float = None,
        u_range: float = 1.0,
        u_multiplier: float = 1.0,
        action_script: Callable[[Agent, World], None] = None,
        sensors: Union[SensorType, List[SensorType]] = None,
        c_noise: float = None,
        silent: bool = True,
        adversary: bool = False,
    ):
        super().__init__(
            name,
            movable,
            rotatable,
            collide,
            density,  # Unused for now
            mass,
            shape,
            max_speed,
            color,
        )
        if obs_range == 0.0:
            assert sensors is None, f"Blind agent cannot have sensors, got {sensors}"
        if shape is not None:
            assert isinstance(shape, Sphere), "Agents must be spheres"

        # cannot observe the world
        self._obs_range = obs_range
        # observation noise
        self._obs_noise = obs_noise
        # physical motor noise amount
        self._u_noise = u_noise
        # control range
        self._u_range = u_range
        # agent action is a force multiplied by this amount
        self._u_multiplier = u_multiplier
        # script behavior to execute
        self._action_script = action_script
        # agents sensors
        self._sensors = sensors
        # non differentiable communication noise
        self._c_noise = c_noise
        # cannot send communication signals
        self._silent = silent
        # is adversary
        self._adversary = adversary

        # action
        self._action = Action()
        # state
        self._state = AgentState()

    @Entity.batch_dim.setter
    def batch_dim(self, batch_dim: int):
        Entity.batch_dim.fset(self, batch_dim)
        self._action.batch_dim = batch_dim

    @Entity.device.setter
    def device(self, device: torch.device):
        Entity.device.fset(self, device)
        self._action.device = device

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

    @property
    def u_range(self):
        return self._u_range

    @property
    def action(self):
        return self._action

    @property
    def u_multiplier(self):
        return self._u_multiplier

    @property
    def silent(self):
        return self._silent

    @property
    def u_noise(self):
        return self._u_noise

    @property
    def c_noise(self):
        return self._c_noise

    @property
    def adversary(self):
        return self._adversary

    @override(Entity)
    def _spawn(self, dim_c, dim_p: int):
        super()._spawn(dim_p)
        if dim_c == 0:
            assert (
                self.silent
            ), f"Agent {self.name} must be silent when world has no communication"
        elif dim_c > 0 and not self.silent:
            self.state.c = torch.zeros(
                self.batch_dim, dim_c, device=self.device, dtype=torch.float32
            )


# Multi-agent world
class World(TorchVectorizedObject):
    def __init__(
        self,
        batch_dim: int,
        device: torch.device,
        dt: float = 0.1,
        damping: float = 0.25,
        x_semidim: float = None,
        y_semidim: float = None,
        dim_c: int = 0,
        contact_force: float = 1e2,
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
        # physical damping
        self._damping = damping
        # gravity
        self._gravity = torch.tensor(gravity, device=self.device, dtype=torch.float32)
        # contact response parameters
        self._contact_force = contact_force
        self._contact_margin = contact_margin
        # Pairs of collidable shapes
        self._collidable_pairs = [{Sphere, Sphere}, {Sphere, Box}, {Sphere, Line}]
        # Horizontal unit vector
        self._normal_vector = torch.tensor(
            [1.0, 0.0], dtype=torch.float32, device=self.device
        ).repeat(self._batch_dim, 1)

    def add_agent(self, agent: Agent):
        """Only way to add agents to the world"""
        agent.batch_dim = self._batch_dim
        agent.device = self._device
        agent._spawn(dim_p=self.dim_p, dim_c=self._dim_c)
        self._agents.append(agent)

    def add_landmark(self, landmark: Landmark):
        """Only way to add landmarks to the world"""
        landmark.batch_dim = self._batch_dim
        landmark.device = self._device
        landmark._spawn(self.dim_p)
        self._landmarks.append(landmark)

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
    def y_semidim(self):
        return self._y_semidim

    @property
    def dim_p(self):
        return self._dim_p

    @property
    def dim_c(self):
        return self._dim_c

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

    def seed(self, seed=None):
        if seed is None:
            seed = 0
        torch.manual_seed(seed)
        return [seed]

    def get_box_ray_intersection(
        self, box: Entity, pos: torch.Tensor, angles: torch.Tensor
    ):
        """
        Inspired from https://tavianator.com/2011/ray_box.html
        Checks if ray originating from pos at angle intersects with a box and if so
        at what point it intersects.
        """
        assert pos.ndim == 2 and angles.ndim == 1
        assert pos.shape[0] == angles.shape[0]
        assert isinstance(box.shape, Box)

        pos_origin = pos - box.state.pos
        pos_aabb = World._rotate_vector(pos_origin, -box.state.rot)
        ray_dir_world = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        ray_dir_aabb = World._rotate_vector(ray_dir_world, -box.state.rot)

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

        intersect_aabb = tmin * ray_dir_aabb + pos_aabb
        intersect_world = (
            World._rotate_vector(intersect_aabb, box.state.rot) + box.state.pos
        )

        collision = (tmax >= tmin) and (tmin > 0.0)
        return collision, intersect_world

    def get_sphere_ray_intersection(
        self, sphere: Entity, pos: torch.Tensor, angles: torch.Tensor
    ):
        """
        Inspired by https://www.bluebill.net/circle_ray_intersection.html
        Checks if ray originating from pos at angle intersects with a sphere and if so
        at what point it intersects.
        """
        assert pos.ndim == 2 and angles.ndim == 1
        assert pos.shape[0] == angles.shape[0]
        assert isinstance(sphere.shape, Sphere)

        u = sphere.state.pos - pos
        ray_dir_world = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        u_dot_ray = torch.bmm(u.unsqueeze(1), ray_dir_world.unsqueeze(2)).squeeze(1)
        u1 = u_dot_ray * ray_dir_world
        u2 = u - u1
        d = torch.linalg.norm(u2, dim=1)
        m = torch.sqrt(sphere.shape.radius**2 - d**2)
        first_intersection = pos + u1 - m * ray_dir_world
        ray_intersects = d <= sphere.shape.radius
        sphere_is_in_front = u_dot_ray.squeeze(1) > 0.0
        return ray_intersects and sphere_is_in_front, first_intersection

    def raycast(self, pos: torch.Tensor, angles: torch.Tensor):
        assert pos.ndim == 2 and angles.ndim == 1
        assert pos.shape[0] == angles.shape[0]

        dists = []
        for entity in self.landmarks:
            if isinstance(entity.shape, Box):
                collision, intersect = self.get_box_ray_intersection(
                    entity, pos, angles
                )
            elif isinstance(entity.shape, Sphere):
                collision, intersect = self.get_sphere_ray_intersection(
                    entity, pos, angles
                )
            else:
                assert False, f"Shape {entity.shape} currently not handled"
            d = torch.linalg.norm(pos - intersect, dim=1)
            d[~collision] = float("inf")
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
        else:
            assert False, "Distance not computable for given entities"
        return return_value

    def is_overlapping(self, entity_a: Entity, entity_b: Entity, env_index: int = None):
        a_shape = entity_a.shape
        b_shape = entity_b.shape
        self._check_batch_index(env_index)

        if isinstance(a_shape, Sphere) and isinstance(b_shape, Sphere):
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
        # Sphere and line
        elif (
            isinstance(entity_a.shape, Line)
            and isinstance(entity_b.shape, Sphere)
            or isinstance(entity_b.shape, Line)
            and isinstance(entity_a.shape, Sphere)
        ):
            return self.get_distance(entity_a, entity_b, env_index) < 0

        else:
            assert False, "Overlap not computable for give entities"
        if env_index is not None:
            return_value = return_value[env_index]
        return return_value

    # update state of the world
    def step(self):
        # set actions for scripted agents
        for agent in self.scripted_agents:
            agent.action_callback(self)
        # gather forces applied to entities
        force = torch.zeros(
            self._batch_dim,
            len(self.entities),
            self._dim_p,
            device=self.device,
            dtype=torch.float32,
        )
        torque = torch.zeros(
            self._batch_dim,
            len(self.entities),
            1,
            device=self.device,
            dtype=torch.float32,
        )

        # apply agent physical controls
        self._apply_action_force(force)
        # apply gravity
        self._apply_gravity(force)
        # apply environment forces
        self._apply_environment_force(force, torque)
        # integrate physical state
        self._integrate_state(force, torque)
        # update non-differentiable comm state
        if self._dim_c > 0:
            for agent in self._agents:
                self._update_comm_state(agent)

    def _apply_gravity(self, force):
        if not (self._gravity == 0.0).all():
            for i, entity in enumerate(self.entities):
                if entity.movable:
                    force[:, i] += entity.mass * self._gravity

    # gather agent action forces
    def _apply_action_force(self, force):
        # set applied forces
        for i, agent in enumerate(self._agents, start=len(self.landmarks)):
            if agent.movable:
                noise = (
                    torch.randn(
                        *agent.action.u.shape, device=self.device, dtype=torch.float32
                    )
                    * agent.u_noise
                    if agent.u_noise
                    else 0.0
                )
                force[:, i] = agent.action.u + noise
        assert not force.isnan().any()

    # gather physical forces acting on entities
    def _apply_environment_force(self, force, torque):
        # simple (but inefficient) collision response
        for a, entity_a in enumerate(self.entities):
            for b, entity_b in enumerate(self.entities):
                if b <= a or not self._collides(entity_a, entity_b):
                    continue
                (f_a, t_a), (f_b, t_b) = self._get_collision_force(entity_a, entity_b)
                if entity_a.movable:
                    force[:, a] += f_a
                if entity_a.rotatable:
                    torque[:, a] += t_a
                if entity_b.movable:
                    force[:, b] += f_b
                if entity_b.rotatable:
                    torque[:, b] += t_b

    def _collides(self, a: Entity, b: Entity) -> bool:
        if (not a.collide) or (not b.collide) or a is b:
            return False
        a_shape = a.shape
        b_shape = b.shape
        if {a_shape.__class__, b_shape.__class__} in self._collidable_pairs:
            return True
        return False

    # get collision forces for any contact between two entities
    # collisions among lines and boxes or these objects among themselves will be ignored
    def _get_collision_force(self, entity_a, entity_b):

        # Sphere and sphere
        if isinstance(entity_a.shape, Sphere) and isinstance(entity_b.shape, Sphere):
            force_a, force_b = self._get_collision_forces(
                entity_a.state.pos,
                entity_b.state.pos,
                dist_min=entity_a.shape.radius + entity_b.shape.radius,
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
            force_sphere, force_line = self._get_collision_forces(
                sphere.state.pos,
                closest_point,
                dist_min=sphere.shape.radius + LINE_MIN_DIST,
            )
            r = closest_point - line.state.pos
            torque_line = World._compute_torque(force_line, r)

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
            closest_point = self._get_closest_point_box(box, sphere.state.pos)
            force_sphere, force_box = self._get_collision_forces(
                sphere.state.pos,
                closest_point,
                dist_min=sphere.shape.radius + LINE_MIN_DIST,
            )
            r = closest_point - box.state.pos
            torque_box = World._compute_torque(force_box, r)

            force_a, torque_a, force_b, torque_b = (
                (force_sphere, 0, force_box, torque_box)
                if isinstance(entity_a.shape, Sphere)
                else (force_box, torque_box, force_sphere, 0)
            )
        else:
            assert False

        return (force_a, torque_a), (force_b, torque_b)

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

    def _get_all_points_box(self, box: Entity, test_point_pos):
        assert isinstance(box.shape, Box)

        # Rotate normal vector by the angle of the box
        rotated_vector = World._rotate_vector(self._normal_vector, box.state.rot)
        rotated_vector2 = World._rotate_vector(
            self._normal_vector, box.state.rot + torch.pi / 2
        )

        # Middle points of the sides
        p1 = box.state.pos + rotated_vector * (box.shape.length / 2)
        p2 = box.state.pos - rotated_vector * (box.shape.length / 2)
        p3 = box.state.pos + rotated_vector2 * (box.shape.width / 2)
        p4 = box.state.pos - rotated_vector2 * (box.shape.width / 2)

        closest_points = []

        for i, p in enumerate([p1, p2, p3, p4]):
            point = self._get_closest_point_line(
                p,
                box.state.rot + torch.pi / 2 if i <= 1 else box.state.rot,
                box.shape.width if i <= 1 else box.shape.length,
                test_point_pos,
            )
            closest_points.append(point)

        return closest_points

    def _get_closest_point_line(self, line_pos, line_rot, line_length, test_point_pos):
        # Rotate it by the angle of the line
        rotated_vector = World._rotate_vector(self._normal_vector, line_rot)
        # Get distance between line and sphere
        delta_pos = line_pos - test_point_pos
        # Dot product of distance and line vector
        dot_p = torch.einsum("bs,bs->b", delta_pos, rotated_vector).unsqueeze(-1)
        # Coordinates of the closes point
        sign = torch.sign(dot_p)
        closest_point = (
            line_pos
            - sign
            * torch.min(
                torch.abs(dot_p),
                torch.tensor(line_length / 2, dtype=torch.float32, device=self.device),
            )
            * rotated_vector
        )
        return closest_point

    def _get_collision_forces(
        self, pos_a: Tensor, pos_b: Tensor, dist_min: float
    ) -> Tensor:
        delta_pos = pos_a - pos_b
        dist = torch.linalg.vector_norm(delta_pos, dim=1)

        # softmax penetration
        k = self._contact_margin
        penetration = (
            torch.logaddexp(
                torch.tensor(0.0, dtype=torch.float32, device=self.device),
                -(dist - dist_min) / k,
            )
            * k
        )
        force = (
            self._contact_force
            * delta_pos
            / dist.unsqueeze(-1)
            * penetration.unsqueeze(-1)
        )
        force[dist == 0.0] = 0.0
        force[dist > dist_min] = 0
        assert not force.isnan().any()
        return +force, -force

    # integrate physical state
    def _integrate_state(self, force, torque):
        for i, entity in enumerate(self.entities):
            if entity.movable:
                # Compute translation
                entity.state.vel = entity.state.vel * (1 - self._damping)
                entity.state.vel += (force[:, i] / entity.mass) * self._dt
                if entity.max_speed is not None:
                    speed = torch.linalg.norm(entity.state.vel, dim=1)
                    new_vel = entity.state.vel / speed.unsqueeze(-1) * entity.max_speed
                    entity.state.vel[speed > entity.max_speed] = new_vel[
                        speed > entity.max_speed
                    ]
                new_pos = entity.state.pos + entity.state.vel * self._dt
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
                entity.state.ang_vel = entity.state.ang_vel * (1 - self._damping)
                entity.state.ang_vel += (
                    torque[:, i] / entity.moment_of_inertia
                ) * self._dt
                entity.state.rot += entity.state.ang_vel * self._dt

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

    @staticmethod
    def _rotate_vector(vector: Tensor, angle: Tensor):
        if len(angle.shape) > 1:
            angle = angle.squeeze(-1)
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        return torch.stack(
            [
                vector[:, X] * cos - vector[:, Y] * sin,
                vector[:, X] * sin + vector[:, Y] * cos,
            ],
            dim=-1,
        )

    @staticmethod
    def _compute_torque(f: Tensor, r: Tensor) -> Tensor:
        return (r[:, X] * f[:, Y] - r[:, Y] * f[:, X]).unsqueeze(1)
