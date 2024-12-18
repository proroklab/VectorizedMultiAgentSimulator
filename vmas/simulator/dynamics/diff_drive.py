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

    def f(self, state, u_command, ang_vel_command):
        theta = state[:, 2]
        dx = u_command * torch.cos(theta)
        dy = u_command * torch.sin(theta)
        dtheta = ang_vel_command
        return torch.stack((dx, dy, dtheta), dim=-1)  # [batch_size,3]

    def euler(self, state, u_command, ang_vel_command):
        return self.dt * self.f(state, u_command, ang_vel_command)

    def runge_kutta(self, state, u_command, ang_vel_command):
        k1 = self.f(state, u_command, ang_vel_command)
        k2 = self.f(state + self.dt * k1 / 2, u_command, ang_vel_command)
        k3 = self.f(state + self.dt * k2 / 2, u_command, ang_vel_command)
        k4 = self.f(state + self.dt * k3, u_command, ang_vel_command)
        return (self.dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    @property
    def needed_action_size(self) -> int:
        return 2

    def process_action(self):
        u_command = self.agent.action.u[:, 0]  # Forward velocity
        ang_vel_command = self.agent.action.u[:, 1]  # Angular velocity

        # Current state of the agent
        state = torch.cat((self.agent.state.pos, self.agent.state.rot), dim=1)

        v_cur_x = self.agent.state.vel[:, 0]  # Current velocity in x-direction
        v_cur_y = self.agent.state.vel[:, 1]  # Current velocity in y-direction
        v_cur_angular = self.agent.state.ang_vel[:, 0]  # Current angular velocity

        # Select the integration method to calculate the change in state
        if self.integration == "euler":
            delta_state = self.euler(state, u_command, ang_vel_command)
        else:
            delta_state = self.runge_kutta(state, u_command, ang_vel_command)

        # Calculate the accelerations required to achieve the change in state
        acceleration_x = (delta_state[:, 0] - v_cur_x * self.dt) / self.dt**2
        acceleration_y = (delta_state[:, 1] - v_cur_y * self.dt) / self.dt**2
        acceleration_angular = (
            delta_state[:, 2] - v_cur_angular * self.dt
        ) / self.dt**2

        # Calculate the forces required for the linear accelerations
        force_x = self.agent.mass * acceleration_x
        force_y = self.agent.mass * acceleration_y

        # Calculate the torque required for the angular acceleration
        torque = self.agent.moment_of_inertia * acceleration_angular

        # Update the physical force and torque required for the user inputs
        self.agent.state.force[:, vmas.simulator.utils.X] = force_x
        self.agent.state.force[:, vmas.simulator.utils.Y] = force_y
        self.agent.state.torque = torque.unsqueeze(-1)
