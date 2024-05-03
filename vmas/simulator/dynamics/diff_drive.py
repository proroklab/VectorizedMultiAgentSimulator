#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.


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

    def euler(self, f, state):
        return state + self.dt * f(state)

    def runge_kutta(self, f, state):
        k1 = f(state)
        k2 = f(state + self.dt * k1 / 2)
        k3 = f(state + self.dt * k2 / 2)
        k4 = f(state + self.dt * k3)

        return state + (self.dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    @property
    def needed_action_size(self) -> int:
        return 2

    def process_action(self):
        velocity = self.agent.action.u[:, 0]  # Forward velocity
        ang_velocity = self.agent.action.u[:, 1]  # Angular velocity

        # Current state of the agent
        state = torch.cat((self.agent.state.pos, self.agent.state.rot), dim=1)

        def f(state):
            theta = state[:, 2]
            return torch.stack(
                [
                    velocity * torch.cos(theta),
                    velocity * torch.sin(theta),
                    ang_velocity,
                ],
                dim=-1,
            )

        if self.integration == "euler":
            new_state = self.euler(f, state)
        else:
            new_state = self.runge_kutta(f, state)

        # Calculate the change in state
        delta_state = new_state - state

        # Calculate the accelerations required to achieve the change in state
        acceleration_x = delta_state[:, 0] / self.dt
        acceleration_y = delta_state[:, 1] / self.dt
        angular_acceleration = delta_state[:, 2] / self.dt

        # Calculate the forces required for the linear accelerations
        force_x = self.agent.mass * acceleration_x
        force_y = self.agent.mass * acceleration_y

        # Calculate the torque required for the angular acceleration
        torque = self.agent.moment_of_inertia * angular_acceleration

        # Update the physical force and torque required for the user inputs
        self.agent.state.force[:, vmas.simulator.utils.X] = force_x
        self.agent.state.force[:, vmas.simulator.utils.Y] = force_y
        self.agent.state.torque = torque.unsqueeze(-1)
