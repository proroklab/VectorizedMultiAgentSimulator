#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import math

import torch

import vmas.simulator.core
import vmas.simulator.utils
from vmas.simulator.dynamics.common import Dynamics


class DiffDrive(Dynamics):
    def __init__(
        self,
        world: vmas.simulator.core.World,
        integration: str = "rk4",  # one of "euler", "rk4"
    ):
        super().__init__()
        assert integration == "rk4" or integration == "euler"

        self.dt = world.dt
        self.integration = integration
        self.world = world

    def euler(self, f, rot):
        return f(rot)

    def runge_kutta(self, f, rot):
        k1 = f(rot)
        k2 = f(rot + self.dt * k1[2] / 2)
        k3 = f(rot + self.dt * k2[2] / 2)
        k4 = f(rot + self.dt * k3[2])

        return (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    @property
    def needed_action_size(self) -> int:
        return 2

    def process_action(self):
        u_forward = self.agent.action.u[:, 0]
        u_rot = self.agent.action.u[:, 1]

        def f(rot):
            return torch.stack(
                [u_forward * torch.cos(rot), u_forward * torch.sin(rot), u_rot], dim=0
            )

        if self.integration == "euler":
            u = self.euler(f, self.agent.state.rot.squeeze(-1))
        else:
            u = self.runge_kutta(f, self.agent.state.rot.squeeze(-1))

        self.agent.state.force[:, vmas.simulator.utils.X] = u[vmas.simulator.utils.X]
        self.agent.state.force[:, vmas.simulator.utils.Y] = u[vmas.simulator.utils.Y]
        self.agent.state.torque = u_rot.unsqueeze(-1)
