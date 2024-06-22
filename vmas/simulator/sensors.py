#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from typing import Callable, List, Union

import torch

import vmas.simulator.core
import vmas.simulator.utils

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom


class Sensor(ABC):
    def __init__(self, world: vmas.simulator.core.World):
        super().__init__()
        self._world = world
        self._agent: Union[vmas.simulator.core.Agent, None] = None

    @property
    def agent(self) -> Union[vmas.simulator.core.Agent, None]:
        return self._agent

    @agent.setter
    def agent(self, agent: vmas.simulator.core.Agent):
        self._agent = agent

    @abstractmethod
    def measure(self):
        raise NotImplementedError

    @abstractmethod
    def render(self, env_index: int = 0) -> "List[Geom]":
        raise NotImplementedError

    def to(self, device: torch.device):
        raise NotImplementedError


class Lidar(Sensor):
    def __init__(
        self,
        world: vmas.simulator.core.World,
        angle_start: float = 0.0,
        angle_end: float = 2 * torch.pi,
        n_rays: int = 8,
        max_range: float = 1.0,
        entity_filter: Callable[[vmas.simulator.core.Entity], bool] = lambda _: True,
        render_color: vmas.simulator.utils.Color = vmas.simulator.utils.Color.GRAY,
        render: bool = True,
    ):
        super().__init__(world)
        if (angle_start - angle_end) % (torch.pi * 2) < 1e-5:
            angles = torch.linspace(
                angle_start, angle_end, n_rays + 1, device=self._world.device
            )[:n_rays]
        else:
            angles = torch.linspace(
                angle_start, angle_end, n_rays, device=self._world.device
            )

        self._angles = angles.repeat(self._world.batch_dim, 1)
        self._max_range = max_range
        self._last_measurement = None
        self._render = render
        self._entity_filter = entity_filter
        self._render_color = render_color

    def to(self, device: torch.device):
        self._angles = self._angles.to(device)

    @property
    def entity_filter(self):
        return self._entity_filter

    @entity_filter.setter
    def entity_filter(
        self, entity_filter: Callable[[vmas.simulator.core.Entity], bool]
    ):
        self._entity_filter = entity_filter

    def measure(self):
        dists = []
        for angle in self._angles.unbind(1):
            dists.append(
                self._world.cast_ray(
                    self.agent,
                    angle + self.agent.state.rot.squeeze(-1),
                    max_range=self._max_range,
                    entity_filter=self.entity_filter,
                )
            )
        measurement = torch.stack(dists, dim=1)
        self._last_measurement = measurement
        return measurement

    def set_render(self, render: bool):
        self._render = render

    def render(self, env_index: int = 0) -> "List[Geom]":
        if not self._render:
            return []
        from vmas.simulator import rendering

        geoms: List[rendering.Geom] = []
        if self._last_measurement is not None:
            for angle, dist in zip(
                self._angles.unbind(1), self._last_measurement.unbind(1)
            ):
                angle = angle[env_index] + self.agent.state.rot.squeeze(-1)[env_index]
                ray = rendering.Line(
                    (0, 0),
                    (dist[env_index], 0),
                    width=0.05,
                )
                xform = rendering.Transform()
                xform.set_translation(*self.agent.state.pos[env_index])
                xform.set_rotation(angle)
                ray.add_attr(xform)

                ray_circ = rendering.make_circle(0.01)
                ray_circ.set_color(*self._render_color.value)
                xform = rendering.Transform()
                rot = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
                pos_circ = (
                    self.agent.state.pos[env_index] + rot * dist.unsqueeze(1)[env_index]
                )
                xform.set_translation(*pos_circ)
                ray_circ.add_attr(xform)

                geoms.append(ray)
                geoms.append(ray_circ)
        return geoms
