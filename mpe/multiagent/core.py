from typing import Callable, Union, List

import torch
from torch import Tensor

# physical/external base state of all entites
from jan_env.vector_env import GRAY, BLUE
from simulator.utils import SensorType


class TorchVecotrizedObject(object):
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


class EntityState(TorchVecotrizedObject):
    def __init__(self):
        super().__init__()
        # physical position
        self._pos = None
        # physical velocity
        self._vel = None

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, pos: Tensor = None):
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

    @property
    def vel(self):
        return self._vel

    @vel.setter
    def vel(self, vel: Tensor = None):
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
    def c(self, c: Tensor = None):
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
class Action(TorchVecotrizedObject):
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
    def u(self, u: Tensor = None):
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
    def c(self, c: Tensor = None):
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
class Entity(TorchVecotrizedObject):
    def __init__(
        self,
        name: str,
        radius: float = 0.05,
        movable: bool = False,
        collide: bool = True,
        density: float = 25.0,  # Unused for now
        mass: float = 1.0,
        max_speed: float = None,
        color: tuple = GRAY,
    ):
        super().__init__()
        # name
        self._name = name
        # properties:
        self._radius = radius
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
        # state
        self._state = EntityState()

    @TorchVecotrizedObject.batch_dim.setter
    def batch_dim(self, batch_dim: int):
        TorchVecotrizedObject.batch_dim.fset(self, batch_dim)
        self._state.batch_dim = batch_dim

    @TorchVecotrizedObject.device.setter
    def device(self, device: torch.device):
        TorchVecotrizedObject.device.fset(self, device)
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
    def radius(self):
        return self._radius

    @property
    def max_speed(self):
        return self._max_speed

    @property
    def name(self):
        return self._name


# properties of landmark entities
class Landmark(Entity):
    def __init__(
        self,
        name: str,
        radius: float = 0.05,
        movable: bool = False,
        collide: bool = True,
        density: float = 25.0,  # Unused for now
        mass: float = 1.0,
        max_speed: float = None,
        color: tuple = GRAY,
    ):
        super().__init__(
            name,
            radius,
            movable,
            collide,
            density,  # Unused for now
            mass,
            max_speed,
            color,
        )


# properties of agent entities
class Agent(Entity):
    def __init__(
        self,
        name: str,
        radius: float = 0.05,
        movable: bool = True,
        collide: bool = True,
        density: float = 25.0,  # Unused for now
        mass: float = 1.0,
        max_speed: float = None,
        color: tuple = BLUE,
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
            radius,
            movable,
            collide,
            density,  # Unused for now
            mass,
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
class World(TorchVecotrizedObject):
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
                [f_a, f_b] = self._get_collision_force(entity_a, entity_b)
                assert not f_a.isnan().any() or not f_b.isnan().any()
                if f_a is not None:
                    p_force[:, a] += f_a
                if f_b is not None:
                    p_force[:, b] += f_b
        return p_force

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

    # get collision forces for any contact between two entities
    def _get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None]  # not a collider
        if entity_a is entity_b:
            return [None, None]  # don't collide against itself
        # compute actual distance between entities
        delta_pos = entity_a.state.pos - entity_b.state.pos

        dist = torch.sqrt(torch.sum(delta_pos**2, dim=-1))

        # minimum allowable distance
        dist_min = entity_a.radius + entity_b.radius
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
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]

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
