#  Copyright (c) 2022-2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import math
from typing import Union

import torch
from torch import Tensor
import vmas.simulator.core
import vmas.simulator.utils


class DiffDriveDynamics:
    def __init__(
        self,
        agent: vmas.simulator.core.Agent,
        world: vmas.simulator.core.World,
        integration: str = "rk4",  # one of "euler", "rk4"
    ):
        assert integration == "rk4" or integration == "euler"
        assert (
            agent.action.u_rot_range != 0
        ), "Agent with diff drive dynamics needs non zero u_rot_range"

        self.agent = agent
        self.world = world
        self.dt = world.dt
        self.integration = integration

    def reset(self, index: Union[Tensor, int] = None):
        pass

    def euler(self, f, rot):
        return f(rot)

    def runge_kutta(self, f, rot):
        k1 = f(rot)
        k2 = f(rot + self.dt * k1[2] / 2)
        k3 = f(rot + self.dt * k2[2] / 2)
        k4 = f(rot + self.dt * k3[2])

        return (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def process_force(self):
        u_forward = self.agent.action.u[:, vmas.simulator.utils.X]
        u_rot = self.agent.action.u_rot.squeeze(-1)

        def f(rot):
            return torch.stack(
                [u_forward * torch.cos(rot), u_forward * torch.sin(rot), u_rot], dim=0
            )

        if self.integration == "euler":
            u = self.euler(f, self.agent.state.rot.squeeze(-1))
        else:
            u = self.runge_kutta(f, self.agent.state.rot.squeeze(-1))

        self.agent.action.u[:, vmas.simulator.utils.X] = u[vmas.simulator.utils.X]
        self.agent.action.u[:, vmas.simulator.utils.Y] = u[vmas.simulator.utils.Y]
