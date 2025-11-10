#  Copyright (c) ProrokLab.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
import typing
from abc import ABC, abstractmethod
from typing import List, Optional

import torch
from torch import Tensor

from vmas.simulator.core import Agent, World
from vmas.simulator.utils import (
    AGENT_INFO_TYPE,
    AGENT_OBS_TYPE,
    AGENT_REWARD_TYPE,
    INITIAL_VIEWER_SIZE,
    VIEWER_DEFAULT_ZOOM,
)

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom


class BaseScenario(ABC):
    """Base class for scenarios.

    This is the class that scenarios inherit from.

    The methods that are **compulsory to instantiate** are:

    - :class:`make_world`
    - :class:`reset_world_at`
    - :class:`observation`
    - :class:`reward`

    The methods that are **optional to instantiate** are:

    - :class:`info`
    - :class:`extra_render`
    - :class:`process_action`
    - :class:`pre_step`
    - :class:`post_step`

    """

    def __init__(self):
        """Do not override."""
        self._world = None
        self.viewer_size = INITIAL_VIEWER_SIZE
        """The size of the rendering viewer window. This can be changed in the :class:`~make_world` function. """
        self.viewer_zoom = VIEWER_DEFAULT_ZOOM
        """The zoom of the rendering camera (a lower value means more zoom). This can be changed in the :class:`~make_world` function. """
        self.render_origin = (0.0, 0.0)
        """The origin of the rendering camera when ``agent_index_to_focus`` is None in the ``render()`` arguments. This can be changed in the :class:`~make_world` function. """
        self.plot_grid = False
        """Whether to plot a grid in the scenario rendering background. This can be changed in the :class:`~make_world` function. """
        self.grid_spacing = 0.1
        """If :class:`~plot_grid`, the distance between lines in the background grid. This can be changed in the :class:`~make_world` function. """
        self.visualize_semidims = True
        """Whether to display boundaries in dimension-limited environment. This can be changed in the :class:`~make_world` function. """

    @property
    def world(self):
        """The :class:`~vmas.simulator.core.World` associated toi this scenario."""
        assert (
            self._world is not None
        ), "You first need to set `self._world` in the `make_world` method"
        return self._world

    def to(self, device: torch.device):
        """Casts the scenario to a different device.

        Args:
            device (Union[str, int, torch.device]): the device to cast to
        """
        for attr, value in self.__dict__.items():
            if isinstance(value, Tensor):
                self.__dict__[attr] = value.to(device)
        self.world.to(device)

    def env_make_world(self, batch_dim: int, device: torch.device, **kwargs) -> World:
        # Do not override
        self._world = self.make_world(batch_dim, device, **kwargs)
        return self._world

    def env_reset_world_at(self, env_index: typing.Optional[int]):
        # Do not override
        self.world.reset(env_index)
        self.reset_world_at(env_index)

    def env_process_action(self, agent: Agent):
        # Do not override
        if agent.action_script is not None:
            agent.action_callback(self.world)
        # Customizable action processor
        self.process_action(agent)
        agent.dynamics.check_and_process_action()

    @abstractmethod
    def make_world(self, batch_dim: int, device: torch.device, **kwargs) -> World:
        """
        This function needs to be implemented when creating a scenario.
        In this function the user should instantiate the world and insert agents and landmarks in it.

        Args:
            batch_dim (int): the number of vecotrized environments.
            device (Union[str, int, torch.device], optional): the device of the environmemnt.
            kwargs (dict, optional): named arguments passed from environment creation

        Returns:
            :class:`~vmas.simulator.core.World` : the :class:`~vmas.simulator.core.World`
            instance which is automatically set in :class:`~world`.

        Examples:
            >>> from vmas.simulator.core import Agent, World, Landmark, Sphere, Box
            >>> from vmas.simulator.scenario import BaseScenario
            >>> from vmas.simulator.utils import Color
            >>> class Scenario(BaseScenario):
            >>>     def make_world(self, batch_dim: int, device: torch.device, **kwargs):
            ...         # Pass any kwargs you desire when creating the environment
            ...         n_agents = kwargs.get("n_agents", 5)
            ...
            ...         # Create world
            ...         world = World(batch_dim, device, dt=0.1, drag=0.25, dim_c=0)
            ...         # Add agents
            ...         for i in range(n_agents):
            ...             agent = Agent(
            ...                 name=f"agent {i}",
            ...                 collide=True,
            ...                 mass=1.0,
            ...                 shape=Sphere(radius=0.04),
            ...                 max_speed=None,
            ...                 color=Color.BLUE,
            ...                 u_range=1.0,
            ...             )
            ...             world.add_agent(agent)
            ...         # Add landmarks
            ...         for i in range(5):
            ...             landmark = Landmark(
            ...                 name=f"landmark {i}",
            ...                 collide=True,
            ...                 movable=False,
            ...                 shape=Box(length=0.3,width=0.1),
            ...                 color=Color.RED,
            ...             )
            ...             world.add_landmark(landmark)
            ...         return world
        """
        raise NotImplementedError()

    @abstractmethod
    def reset_world_at(self, env_index: Optional[int] = None):
        """Resets the world at the specified env_index.

        When a ``None`` index is passed, the world should make a vectorized (batched) reset.
        The ``entity.set_x()`` methods already have this logic integrated and will perform
        batched operations when index is ``None``.

        When this function is called, all entities have already had their state reset to zeros according to the ``env_index``.
        In this function you shoud change the values of the reset states according to your task.
        For example, some functions you might want to use are:

        - ``entity.set_pos()``,
        - ``entity.set_vel()``,
        - ``entity.set_rot()``,
        - ``entity.set_ang_vel()``.

        Implementors can access the world at :class:`world`.

        To increase performance, torch tensors should be created with the device already set, like:
        ``torch.tensor(..., device=self.world.device)``

        Args:
            env_index (int, otpional): index of the environment to reset. If ``None`` a vectorized reset should be performed.

        Spawning at fixed positions

        Examples:
            >>> from vmas.simulator.scenario import BaseScenario
            >>> import torch
            >>> class Scenario(BaseScenario):
            >>>     def reset_world_at(self, env_index)
            ...        for i, agent in enumerate(self.world.agents):
            ...            agent.set_pos(
            ...                torch.tensor(
            ...                     [-0.2 + 0.1 * i, 1.0],
            ...                     dtype=torch.float32,
            ...                     device=self.world.device,
            ...                ),
            ...                 batch_index=env_index,
            ...            )
            ...        for i, landmark in enumerate(self.world.landmarks):
            ...            landmark.set_pos(
            ...                torch.tensor(
            ...                     [0.2 if i % 2 else -0.2, 0.6 - 0.3 * i],
            ...                     dtype=torch.float32,
            ...                     device=self.world.device,
            ...                ),
            ...                 batch_index=env_index,
            ...            )
            ...            landmark.set_rot(
            ...                torch.tensor(
            ...                     [torch.pi / 4 if i % 2 else -torch.pi / 4],
            ...                     dtype=torch.float32,
            ...                     device=self.world.device,
            ...                ),
            ...                 batch_index=env_index,
            ...            )

        Spawning at random positions

        Examples:
            >>> from vmas.simulator.scenario import BaseScenario
            >>> from vmas.simulator.utils import ScenarioUtils
            >>> class Scenario(BaseScenario):
            >>>     def reset_world_at(self, env_index)
            >>>         ScenarioUtils.spawn_entities_randomly(
            ...             self.world.agents + self.world.landmarks,
            ...             self.world,
            ...             env_index,
            ...             min_dist_between_entities=0.02,
            ...             x_bounds=(-1.0,1.0),
            ...             y_bounds=(-1.0,1.0),
            ...         )

        """
        raise NotImplementedError()

    @abstractmethod
    def observation(self, agent: Agent) -> AGENT_OBS_TYPE:
        """This function computes the observations for ``agent`` in a vectorized way.

        The returned tensor should contain the observations for ``agent`` in all envs and should have
        shape ``(self.world.batch_dim, n_agent_obs)``, or be a dict with leaves following that shape.

        Implementors can access the world at :class:`world`.

        To increase performance, torch tensors should be created with the device already set, like:
        ``torch.tensor(..., device=self.world.device)``

        Args:
            agent (Agent): the agent to compute the observations for

        Returns:
             Union[torch.Tensor, Dict[str, torch.Tensor]]: the observation

        Examples:
            >>> from vmas.simulator.scenario import BaseScenario
            >>> import torch
            >>> class Scenario(BaseScenario):
            >>>     def observation(self, agent):
            ...         # get positions of all landmarks in this agent's reference frame
            ...         landmark_rel_poses = []
            ...         for landmark in self.world.landmarks:
            ...             landmark_rel_poses.append(landmark.state.pos - agent.state.pos)
            ...         return torch.cat([agent.state.pos, agent.state.vel, *landmark_rel_poses], dim=-1)

        You can also return observations in a dictionary

        Examples:
            >>> from vmas.simulator.scenario import BaseScenario
            >>> from vmas.simulator.utils import Color
            >>> class Scenario(BaseScenario):
            >>>     def observation(self, agent):
            ...         return {"pos": agent.state.pos, "vel": agent.state.vel}

        """

        raise NotImplementedError()

    @abstractmethod
    def reward(self, agent: Agent) -> AGENT_REWARD_TYPE:
        """This function computes the reward for ``agent`` in a vectorized way.

        The returned tensor should contain the reward for ``agent`` in all envs and should have
        shape ``(self.world.batch_dim)`` and dtype ``torch.float``.

        Implementors can access the world at :class:`world`.

        To increase performance, torch tensors should be created with the device already set, like:
        ``torch.tensor(..., device=self.world.device)``

        Args:
            agent (Agent): the agent to compute the reward for

        Returns:
             torch.Tensor: reward tensor of shape ``(self.world.batch_dim)``

        Examples:
            >>> from vmas.simulator.scenario import BaseScenario
            >>> import torch
            >>> class Scenario(BaseScenario):
            >>>     def reward(self, agent):
            ...         # reward every agent proportionally to distance from first landmark
            ...         rew = -torch.linalg.vector_norm(agent.state.pos - self.world.landmarks[0].state.pos, dim=-1)
            ...         return rew
        """
        raise NotImplementedError()

    def done(self) -> Tensor:
        """This function computes the done flag for each env in a vectorized way.

        The returned tensor should contain the ``done`` for all envs and should have
        shape ``(n_envs)`` and dtype ``torch.bool``.

        Implementors can access the world at :class:`world`.

        To increase performance, torch tensors should be created with the device already set, like:
        ``torch.tensor(..., device=self.world.device)``

        By default, this function returns all ``False`` s.

        The scenario can still be done if ``max_steps`` has been set at envirtonment construction.

        Returns:
            torch.Tensor: done tensor of shape ``(self.world.batch_dim)``

        Examples:
            >>> from vmas.simulator.scenario import BaseScenario
            >>> import torch
            >>> class Scenario(BaseScenario):
            >>>     def done(self):
            ...         # retrun done when all agents have battery level lower than a threshold
            ...         return torch.stack([a.battery_level < threshold for a in self.world.agents], dim=-1).all(-1)
        """
        return torch.tensor([False], device=self.world.device).expand(
            self.world.batch_dim
        )

    def info(self, agent: Agent) -> AGENT_INFO_TYPE:
        """This function computes the info dict for ``agent`` in a vectorized way.

        The returned dict should have a key for each info of interest and the corresponding value should
        be a tensor of shape ``(n_envs, info_size)``

        By default this function returns an empty dictionary.

        Implementors can access the world at :class:`world`.

        To increase performance, torch tensors should be created with the device already set, like:
        ``torch.tensor(..., device=self.world.device)``

        Args:
            agent (Agent): the agent to compute the info for

        Returns:
             Union[torch.Tensor, Dict[str, torch.Tensor]]: the info
        """
        return {}

    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        """
        This function facilitates additional user/scenario-level rendering for a specific environment index.

        The returned list is a list of geometries. It is the user's responsibility to set attributes such as color,
        position and rotation.

        Args:
            env_index (int, optional): index of the environment to render. Defaults to ``0``.

        Returns: A list of geometries to render for the current time step.

        Examples:
            >>> from vmas.simulator.utils import Color
            >>> from vmas.simulator.scenario import BaseScenario
            >>> class Scenario(BaseScenario):
            >>>     def extra_render(self, env_index):
            >>>         from vmas.simulator import rendering
            >>>         color = Color.BLACK.value
            >>>         line = rendering.Line(
            ...            (self.world.agents[0].state.pos[env_index]),
            ...            (self.world.agents[1].state.pos[env_index]),
            ...            width=1,
            ...         )
            >>>         xform = rendering.Transform()
            >>>         line.add_attr(xform)
            >>>         line.set_color(*color)
            >>>         return [line]
        """
        return []

    def process_action(self, agent: Agent):
        """This function can be overridden to process the agent actions before the simulation step.

        It has access to the world through the :class:`world` attribute

        For example here you can manage additional actions before passing them to the dynamics.

        Args:
            agent (Agent): the agent process the action of

        Examples:
            >>> from vmas.simulator.scenario import BaseScenario
            >>> from vmas.simulator.utils import TorchUtils
            >>> class Scenario(BaseScenario):
            >>>     def process_action(self, agent):
            >>>         # Clamp square to circle
            >>>         agent.action.u = TorchUtils.clamp_with_norm(agent.action.u, agent.u_range)
            >>>         # Can use a PID controller to turn velocity actions into forces
            >>>         # (e.g., from vmas.simulator.controllers.velocity_controller)
            >>>         agent.controller.process_force()
            >>>         return
        """
        return

    def pre_step(self):
        """This function can be overridden to perform any computation that has to happen before the simulation step.
        Its intended use is for computation that has to happen only once before the simulation step has accured.

        For example, you can store temporal data before letting the world step.

        Examples:
            >>> from vmas.simulator.scenario import BaseScenario
            >>> class Scenario(BaseScenario):
            >>>     def pre_step(self):
            >>>         for agent in self.world.agents:
            >>>             agent.prev_state = agent.state
            >>>         return
        """
        return

    def post_step(self):
        """This function can be overridden to perform any computation that has to happen after the simulation step.
        Its intended use is for computation that has to happen only once after the simulation step has accured.

        For example, you can store temporal sensor data in this function.

        Examples:
            >>> from vmas.simulator.scenario import BaseScenario
            >>> class Scenario(BaseScenario):
            >>>     def post_step(self):
            >>>         for agent in self.world.agents:
            >>>             # Let the sensor take a measurement
            >>>             measurements = agent.sensors[0].measure()
            >>>             # Store sensor data in agent.sensor_history
            >>>             agent.sensor_history.append(measurements)
            >>>         return
        """
        return
