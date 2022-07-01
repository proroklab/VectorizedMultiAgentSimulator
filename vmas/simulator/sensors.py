#  Copyright (c) 2022.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Union

import torch

import vmas.simulator.core


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
    def render(self, env_index: int = 0):
        raise NotImplementedError


class Lidar(Sensor):
    def __init__(
        self,
        world: vmas.simulator.core.World,
        angle_start: float = 0.0,
        angle_end: float = 2 * torch.pi,
        n_rays: int = 8,
        max_range: float = 1.0,
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
        # repeat for n dims and make angles first dim so that we can iterate over them
        self._angles = angles.repeat(self._world.batch_dim, 1).swapaxes(1, 0)
        self._max_range = max_range
        self._last_measurement = None

    def measure(self):
        dists = []
        for angle in self._angles:
            dists.append(
                self._world.cast_ray(
                    self.agent.state.pos, angle, max_range=self._max_range
                )
            )
        measurement = torch.stack(dists, dim=1)
        self._last_measurement = measurement.swapaxes(1, 0)
        return measurement

    def render(self, env_index: int = 0):
        from vmas.simulator import rendering

        self.measure()

        geoms: List[rendering.Geom] = []
        if self._last_measurement is not None:
            for angle, dist in zip(self._angles, self._last_measurement):
                ray = rendering.Line(
                    (0, 0),
                    (dist[env_index], 0),
                    width=0.05,
                )
                xform = rendering.Transform()
                xform.set_translation(*self.agent.state.pos[env_index])
                xform.set_rotation(angle[env_index])
                ray.add_attr(xform)

                ray_circ = rendering.make_circle(0.01)
                xform = rendering.Transform()
                rot = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
                pos_circ = self.agent.state.pos + rot * dist
                xform.set_translation(*pos_circ[env_index])
                ray_circ.add_attr(xform)

                geoms.append(ray)
                geoms.append(ray_circ)
        return geoms
