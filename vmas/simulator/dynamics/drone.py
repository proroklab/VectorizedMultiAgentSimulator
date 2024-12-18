#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

from typing import Union

import torch
from torch import Tensor

import vmas.simulator.core
import vmas.simulator.utils
from vmas.simulator.dynamics.common import Dynamics
from vmas.simulator.utils import TorchUtils


class Drone(Dynamics):
    def __init__(
        self,
        world: vmas.simulator.core.World,
        I_xx: float = 8.1e-3,
        I_yy: float = 8.1e-3,
        I_zz: float = 14.2e-3,
        integration: str = "rk4",
    ):
        super().__init__()

        assert integration in (
            "rk4",
            "euler",
        )

        self.integration = integration
        self.I_xx = I_xx
        self.I_yy = I_yy
        self.I_zz = I_zz
        self.world = world
        self.g = 9.81
        self.dt = world.dt
        self.reset()

    def reset(self, index: Union[Tensor, int] = None):
        if index is None:
            # Drone state: phi(roll), theta (pitch), psi (yaw),
            #              p (roll_rate), q (pitch_rate), r (yaw_rate),
            #              x_dot (vel_x), y_dot (vel_y), z_dot (vel_z),
            #              x (pos_x), y (pos_y), z (pos_z)
            self.drone_state = torch.zeros(
                self.world.batch_dim,
                12,
                device=self.world.device,
            )
        else:
            self.drone_state = TorchUtils.where_from_index(index, 0.0, self.drone_state)

    def zero_grad(self):
        self.drone_state = self.drone_state.detach()

    def f(self, state, thrust_command, torque_command):
        phi = state[:, 0]
        theta = state[:, 1]
        psi = state[:, 2]
        p = state[:, 3]
        q = state[:, 4]
        r = state[:, 5]
        x_dot = state[:, 6]
        y_dot = state[:, 7]
        z_dot = state[:, 8]

        c_phi = torch.cos(phi)
        s_phi = torch.sin(phi)
        c_theta = torch.cos(theta)
        s_theta = torch.sin(theta)
        c_psi = torch.cos(psi)
        s_psi = torch.sin(psi)

        # Postion Dynamics
        x_ddot = (
            (c_phi * s_theta * c_psi + s_phi * s_psi) * thrust_command / self.agent.mass
        )
        y_ddot = (
            (c_phi * s_theta * s_psi - s_phi * c_psi) * thrust_command / self.agent.mass
        )
        z_ddot = (c_phi * c_theta) * thrust_command / self.agent.mass - self.g
        # Angular velocity dynamics
        p_dot = (torque_command[:, 0] - (self.I_yy - self.I_zz) * q * r) / self.I_xx
        q_dot = (torque_command[:, 1] - (self.I_zz - self.I_xx) * p * r) / self.I_yy
        r_dot = (torque_command[:, 2] - (self.I_xx - self.I_yy) * p * q) / self.I_zz

        return torch.stack(
            [
                p,
                q,
                r,
                p_dot,
                q_dot,
                r_dot,
                x_ddot,
                y_ddot,
                z_ddot,
                x_dot,
                y_dot,
                z_dot,
            ],
            dim=-1,
        )

    def needs_reset(self) -> Tensor:
        # Constraint roll and pitch within +-30 degrees
        return torch.any(self.drone_state[:, :2].abs() > 30 * (torch.pi / 180), dim=-1)

    def euler(self, state, thrust, torque):
        return self.dt * self.f(state, thrust, torque)

    def runge_kutta(self, state, thrust, torque):
        k1 = self.f(state, thrust, torque)
        k2 = self.f(state + self.dt * k1 / 2, thrust, torque)
        k3 = self.f(state + self.dt * k2 / 2, thrust, torque)
        k4 = self.f(state + self.dt * k3, thrust, torque)
        return (self.dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    @property
    def needed_action_size(self) -> int:
        return 4

    def process_action(self):
        u = self.agent.action.u
        thrust = u[:, 0]  # Thrust, sum of all propeller thrusts
        torque = u[:, 1:4]  # Torque in x, y, z direction

        thrust += self.agent.mass * self.g  # Ensure the drone is not falling

        self.drone_state[:, 9] = self.agent.state.pos[:, 0]  # x
        self.drone_state[:, 10] = self.agent.state.pos[:, 1]  # y
        self.drone_state[:, 2] = self.agent.state.rot[:, 0]  # psi (yaw)

        if self.integration == "euler":
            delta_state = self.euler(self.drone_state, thrust, torque)
        else:
            delta_state = self.runge_kutta(self.drone_state, thrust, torque)

        # Calculate the change in state
        self.drone_state = self.drone_state + delta_state

        v_cur_x = self.agent.state.vel[:, 0]  # Current velocity in x-direction
        v_cur_y = self.agent.state.vel[:, 1]  # Current velocity in y-direction
        v_cur_angular = self.agent.state.ang_vel[:, 0]  # Current angular velocity

        # Calculate the accelerations required to achieve the change in state
        acceleration_x = (delta_state[:, 6] - v_cur_x * self.dt) / self.dt**2
        acceleration_y = (delta_state[:, 7] - v_cur_y * self.dt) / self.dt**2
        acceleration_angular = (
            delta_state[:, 5] - v_cur_angular * self.dt
        ) / self.dt**2

        # Calculate the forces required for the linear accelerations
        force_x = self.agent.mass * acceleration_x
        force_y = self.agent.mass * acceleration_y

        # Calculate the torque required for the angular acceleration
        torque_yaw = self.agent.moment_of_inertia * acceleration_angular

        # Update the physical force and torque required for the user inputs
        self.agent.state.force[:, vmas.simulator.utils.X] = force_x
        self.agent.state.force[:, vmas.simulator.utils.Y] = force_y
        self.agent.state.torque = torque_yaw.unsqueeze(-1)
