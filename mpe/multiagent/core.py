from typing import Callable, Union, List

import torch
from torch import Tensor

from simulator.utils import Color
from simulator.utils import SensorType, X, Y


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


class Shape:
    pass


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


class Sphere(Shape):
    def __init__(self, radius: float = 0.05):
        super().__init__()
        assert radius > 0, f"Radius must be > 0, got {radius}"
        self._radius = radius

    @property
    def radius(self):
        return self._radius


class Line(Shape):
    def __init__(self, length: float = 0.5, width: float = 3):
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


class EntityState(TorchVectorizedObject):
    def __init__(self):
        super().__init__()
        # physical position
        self._pos = None
        # physical velocity
        self._vel = None
        # phyisical rotation -- from -pi to pi
        self._rot = None

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, pos: Tensor):
        assert (
            self._batch_dim is not None and self._device is not None
        ), f"First add an entity to the world before setting its state"
        assert (
            pos.shape[0] == self._batch_dim
        ), f"Internal state must match batch dim, got {pos.shape[0]}, expected {self._batch_dim}"
        if self._vel is not None:
            assert (
                pos.shape == self._vel.shape
            ), f"Position shape must match velocity shape, got {pos.shape} expected {self._vel.shape}"

        self._pos = pos.to(self._device)
        # Initialize rotation with position
        if self._rot is None:
            self.rot = torch.zeros(
                self._batch_dim,
            )

    @property
    def vel(self):
        return self._vel

    @vel.setter
    def vel(self, vel: Tensor):
        assert (
            self._batch_dim is not None and self._device is not None
        ), f"First add an entity to the world before setting its state"
        assert (
            vel.shape[0] == self._batch_dim
        ), f"Internal state must match batch dim, got {vel.shape[0]}, expected {self._batch_dim}"
        if self._pos is not None:
            assert (
                vel.shape == self._pos.shape
            ), f"Velocity shape must match position shape, got {vel.shape} expected {self._pos.shape}"

        self._vel = vel.to(self._device)

    @property
    def rot(self):
        return self._rot

    @rot.setter
    def rot(self, rot: Tensor):
        assert (
            self._batch_dim is not None and self._device is not None
        ), f"First add an entity to the world before setting its state"
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
        ), f"First add an entity to the world before setting its state"
        assert (
            c.shape[0] == self._batch_dim
        ), f"Internal state must match batch dim, got {c.shape[0]}, expected {self._batch_dim}"
        if self._pos is not None:
            assert (
                c.shape == self._pos.shape
            ), f"Communication shape must match position shape, got {c.shape} expected {self._pos.shape}"

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
        ), f"First add an agent to the world before setting its action"
        assert (
            u.shape[0] == self._batch_dim
        ), f"Action must match batch dim, got {u.shape[0]}, expected {self._batch_dim}"
        if self._c is not None:
            assert (
                u.shape == self._c.shape
            ), f"Physiscal action shape must match communication action shape, got {u.shape} expected {self._c.shape}"

        self._u = u.to(self._device)

    @property
    def c(self):
        return self._c

    @c.setter
    def c(self, c: Tensor):
        assert (
            self._batch_dim is not None and self._device is not None
        ), f"First add an agent to the world before setting its action"
        assert (
            c.shape[0] == self._batch_dim
        ), f"Action must match batch dim, got {c.shape[0]}, expected {self._batch_dim}"
        if self._u is not None:
            assert (
                c.shape == self._u.shape
            ), f"Physiscal action shape must match communication action shape, got {c.shape} expected {self._u.shape}"

        self._c = c.to(self._device)


# properties and state of physical world entity
class Entity(TorchVectorizedObject):
    def __init__(
        self,
        name: str,
        movable: bool = False,
        collide: bool = True,
        density: float = 25.0,  # Unused for now
        mass: float = 1.0,
        shape: Shape = Sphere(),
        max_speed: float = None,
        color: Color = Color.GRAY,
    ):
        super().__init__()
        # name
        self._name = name
        # entity can move / be pushed
        self._movable = movable
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

    @TorchVectorizedObject.batch_dim.setter
    def batch_dim(self, batch_dim: int):
        TorchVectorizedObject.batch_dim.fset(self, batch_dim)
        self._state.batch_dim = batch_dim

    @TorchVectorizedObject.device.setter
    def device(self, device: torch.device):
        TorchVectorizedObject.device.fset(self, device)
        self._state.device = device

    @property
    def mass(self):
        return self._mass

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
    def color(self):
        return self._color


# properties of landmark entities
class Landmark(Entity):
    def __init__(
        self,
        name: str,
        shape: Shape = Sphere(),
        movable: bool = False,
        collide: bool = True,
        density: float = 25.0,  # Unused for now
        mass: float = 1.0,
        max_speed: float = None,
        color: Color = Color.GRAY,
    ):
        super().__init__(
            name,
            movable,
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
        collide: bool = True,
        density: float = 25.0,  # Unused for now
        mass: float = 1.0,
        max_speed: float = None,
        color: Color = Color.BLUE,
        obs_range: float = None,
        obs_noise: float = None,
        u_noise: float = None,
        u_range: float = 1.0,
        u_multiplier: float = 5.0,
        action_script: Callable = None,
        sensors: Union[SensorType, List[SensorType]] = None,
        c_noise: float = None,
        silent=True,
    ):
        super().__init__(
            name,
            movable,
            collide,
            density,  # Unused for now
            mass,
            shape,
            max_speed,
            color,
        )
        if obs_range == 0.0:
            assert sensors is None, f"Blind agent cannot have sensors, got {sensors}"

        # cannot observe the world
        self._obs_range = obs_range
        # observation noise
        self._obs_noise = obs_noise
        # physical motor noise amount
        self._u_noise = u_noise
        # control range
        self._u_range = u_range
        # agent action is a force multplied by this amount
        self._u_multiplier = u_multiplier
        # script behavior to execute
        self._action_callback = action_script
        # agents sensors
        self._sensors = sensors
        # non diofferentiable communiation noise
        self._c_noise = c_noise
        # cannot send communication signals
        self._silent = silent
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
    def action_callback(self):
        return self._action_callback

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


# multiagent world
class World(TorchVectorizedObject):
    def __init__(
        self,
        batch_dim: int,
        device: str,
        dt: float = 0.1,
        damping: float = 0.25,
        x_semidim: float = None,
        y_semidim: float = None,
        dim_c: int = 0,
    ):
        assert batch_dim > 0, f"Batch dim must be greater than 0, got {batch_dim}"

        super().__init__(batch_dim, torch.device(device))
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
        # contact response parameters
        self._contact_force = 1e2
        self._contact_margin = 1e-3

        # Horizontal unit vector
        self._normal_vector = (
            torch.tensor([1.0, 0.0]).repeat(self._batch_dim, 1).to(self._device)
        )

    def add_agent(self, agent: Agent):
        """Only way to add agents to the world"""
        agent.batch_dim = self._batch_dim
        agent.device = self._device
        self._agents.append(agent)

    def add_landmark(self, landmark: Landmark):
        """Only way to add landmarks to the world"""
        landmark.batch_dim = self._batch_dim
        landmark.device = self._device
        self._landmarks.append(landmark)

    @property
    def agents(self):
        return self._agents

    @property
    def landmarks(self):
        return self._landmarks

    @property
    def dim_p(self):
        return self._dim_p

    @property
    def dim_c(self):
        return self._dim_p

    # return all entities in the world
    @property
    def entities(self):
        return self._agents + self._landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self._agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self._agents if agent.action_callback is not None]

    # update state of the world
    def step(self):
        # set actions for scripted agents
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        # gather forces applied to entities
        p_force = torch.zeros(self._batch_dim, len(self.entities), 2).to(self._device)
        # apply agent physical controls
        p_force = self._apply_action_force(p_force)
        # apply environment forces
        p_force = self._apply_environment_force(p_force)
        # integrate physical state
        self._integrate_state(p_force)
        # update non-differantiable comm state
        if self._dim_c > 0:
            for agent in self._agents:
                self._update_comm_state(agent)

    # gather agent action forces
    def _apply_action_force(self, p_force):
        # set applied forces
        for i, agent in enumerate(self._agents):
            if agent.movable:
                noise = (
                    torch.randn(*agent.action.u.shape).to(self._device) * agent.u_noise
                    if agent.u_noise
                    else 0.0
                )
                p_force[:, i] = agent.action.u + noise
        assert not p_force.isnan().any()
        return p_force

    # gather physical forces acting on entities
    def _apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        for a, entity_a in enumerate(self.entities):
            for b, entity_b in enumerate(self.entities):
                if b <= a:
                    continue
                f_a, f_b = self._get_collision_force(entity_a, entity_b)
                assert not f_a.isnan().any() or not f_b.isnan().any()
                p_force[:, a] += f_a
                p_force[:, b] += f_b
        return p_force

    # get collision forces for any contact between two entities
    def _get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None]  # not a collider
        if entity_a is entity_b:
            return [None, None]  # don't collide against itself

        force_a = torch.zeros(self._batch_dim, self._dim_p).to(self._device)
        force_b = torch.zeros(self._batch_dim, self._dim_p).to(self._device)

        # Sphere and sphere
        if isinstance(entity_a.shape, Sphere) and isinstance(entity_b.shape, Sphere):
            force_a, force_b = self._get_collision_forces(
                entity_a.state.pos,
                entity_b.state.pos,
                dist_min=entity_a.shape.radius + entity_b.shape.radius,
            )
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
            force_a, force_b = self._line_sphere_collision_forces(
                sphere.state.pos,
                sphere.shape.radius,
                line.state.pos,
                line.state.rot,
                line.shape.length,
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

            for i, p in enumerate([p1, p2, p3, p4]):
                f_a, f_b = self._line_sphere_collision_forces(
                    sphere.state.pos,
                    sphere.shape.radius,
                    p,
                    box.state.rot + torch.pi / 2 if i <= 1 else box.state.rot,
                    box.shape.width if i <= 1 else box.shape.length,
                )
                force_a += f_a
                force_b += f_b

        return (
            force_a if entity_a.movable else torch.tensor(0.0).to(self._device),
            force_b if entity_b.movable else torch.tensor(0.0).to(self._device),
        )

    def _line_sphere_collision_forces(
        self, sphere_pos, sphere_radius, line_pos, line_rot, line_length
    ):
        # Rotate it by the angle of the line
        rotated_vector = World._rotate_vector(self._normal_vector, line_rot)
        # Get distance between line and sphere
        delta_pos = line_pos - sphere_pos
        # Dot product of distance and line vector
        dot_p = torch.einsum("bs,bs->b", delta_pos, rotated_vector).unsqueeze(-1)
        # Coordinates of the closes poinht
        sign = torch.sign(dot_p)
        closest_point = (
            line_pos
            - sign
            * torch.min(
                torch.abs(dot_p),
                torch.tensor(line_length / 2).to(self._device),
            )
            * rotated_vector
        )

        return self._get_collision_forces(
            sphere_pos,
            closest_point,
            dist_min=sphere_radius,
        )

    def _get_collision_forces(self, pos_a, pos_b, dist_min):
        delta_pos = pos_a - pos_b
        dist = torch.sqrt(torch.sum(delta_pos**2, dim=-1))

        # softmax penetration
        k = self._contact_margin
        penetration = (
            torch.logaddexp(torch.tensor(0.0).to(self._device), -(dist - dist_min) / k)
            * k
        )
        force = (
            self._contact_force
            * delta_pos
            / dist.unsqueeze(-1)
            * penetration.unsqueeze(-1)
        )
        return +force, -force

    # integrate physical state
    def _integrate_state(self, p_force):
        for i, entity in enumerate(self.entities):
            if not entity.movable:
                continue
            entity.state.vel = entity.state.vel * (1 - self._damping)
            entity.state.vel += (p_force[:, i] / entity.mass) * self._dt
            if entity.max_speed is not None:
                speed = torch.sqrt(entity.state.vel[0] ** 2 + entity.state.vel[1] ** 2)
                if speed > entity.max_speed:
                    entity.state.vel = (
                        entity.state.vel
                        / torch.sqrt(
                            entity.state.vel[0] ** 2 + entity.state.vel[1] ** 2
                        )
                        * entity.max_speed
                    )
            entity.state.pos += entity.state.vel * self._dt

    def _update_comm_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = torch.zeros(self._batch_dim, self._dim_c).to(self._device)
        else:
            noise = (
                torch.randn(*agent.action.c.shape).to(self._device) * agent.c_noise
                if agent.c_noise
                else 0.0
            )
            agent.state.c = agent.action.c + noise

    @staticmethod
    def _rotate_vector(vector: Tensor, angle: Tensor):
        cos = torch.cos(angle)
        sin = torch.sin(angle)

        return torch.stack(
            [
                vector[:, X] * cos - vector[:, Y] * sin,
                vector[:, X] * sin + vector[:, Y] * cos,
            ],
            dim=-1,
        )
