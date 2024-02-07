from typing import Union

import torch
import vmas.simulator.core
import vmas.simulator.utils
from vmas.simulator.dynamics.common import Dynamics


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

        self.I_xx = I_xx
        self.I_yy = I_yy
        self.I_zz = I_zz
        self.world = world
        self.g = 9.81
        # Drone state: phi(roll), theta (pitch), psi (yaw),
        #              p (roll_rate), q (pitch_rate), r (yaw_rate),
        #              x_dot (vel_x), y_dot (vel_y), z_dot (vel_z),
        #              x (pos_x), y (pos_y), z (pos_z)
        self.drone_state = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], device=world.device
        ).unsqueeze(0).expand(world.batch_dim, 12)

    def euler(self, f, state):
        return state + self.dt * f(state)

    def runge_kutta(self, f, state):
        k1 = f(state)
        k2 = f(state + self.dt * k1 / 2)
        k3 = f(state + self.dt * k2 / 2)
        k4 = f(state + self.dt * k3)
        return state + (self.dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def needed_action_size(self) -> int:
        return 4

    def process_action(self):
        u = self.agent.action.u
        thrust = u[:, 0]  # Thrust, sum of all propeller thrusts
        torque = u[:, 1:4]  # Torque in x, y, z direction

        thrust += self.agent.mass * self.g  # Ensure the drone is not falling

        vmas_state = torch.cat(
            (self.agent.state.pos, self.agent.state.rot), dim=1
        )  # (x,y,rotation)
        self.drone_state[9] = vmas_state[:, 0]  # x
        self.drone_state[10] = vmas_state[:, 1]  # y
        self.drone_state[2] = vmas_state[:, 2]  # psi (yaw)

        def f(state):
            phi, theta, psi, p, q, r, x_dot, y_dot, z_dot, x, y, z = state

            c_phi = torch.cos(phi)
            s_phi = torch.sin(phi)
            c_theta = torch.cos(theta)
            s_theta = torch.sin(theta)
            c_psi = torch.cos(psi)
            s_psi = torch.sin(psi)

            # Postion Dynamics
            x_ddot = (c_phi * s_theta * c_psi + s_phi * s_psi) * thrust / self.agent.mass
            y_ddot = (c_phi * s_theta * s_psi - s_phi * c_psi) * thrust / self.agent.mass
            z_ddot = (c_phi * c_theta) * thrust / self.agent.mass - self.g
            # Angular velocity dynamics
            p_dot = (torque[:, 0] - (self.Iyy - self.Izz) * q * r) / self.Ixx
            q_dot = (torque[:, 1] - (self.Izz - self.Ixx) * p * r) / self.Iyy
            r_dot = (torque[:, 2] - (self.Ixx - self.Iyy) * p * q) / self.Izz

            return torch.stack(
                (
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
                ),
                dim=-1,
            )

        if self.integration == "euler":
            new_drone_state = self.euler(f, self.drone_state)
        else:
            new_drone_state = self.runge_kutta(f, self.drone_state)

        # Calculate the change in state
        delta_state = new_drone_state - self.drone_state

        # Calculate the accelerations required to achieve the change in state
        acceleration_x = delta_state[:, 9] / self.dt
        acceleration_y = delta_state[:, 10] / self.dt
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
