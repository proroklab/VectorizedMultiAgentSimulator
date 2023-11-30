#  Copyright (c) 2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

from typing import Union

import torch
from torch import Tensor

import vmas.simulator.core
import vmas.simulator.utils


class KinematicBicycleDynamics:
    # For the implementation of the kinematic bicycle model, see the equation (2) of the paper Polack, Philip, et al. "The kinematic bicycle model: A consistent model for planning feasible trajectories for autonomous vehicles?." 2017 IEEE intelligent vehicles symposium (IV). IEEE, 2017.
    def __init__(
        self,
        agent: vmas.simulator.core.Agent,
        world: vmas.simulator.core.World,
        width: float,
        l_f: float,
        l_r: float,
        max_steering_angle: float,
        integration: str = "euler",  # one of "euler", "rk4"
    ):
        assert integration in (
            "rk4",
            "euler",
        ), "Integration method must be 'euler' or 'rk4'."
        self.agent = agent
        self.world = world
        self.width = width
        self.l_f = l_f  # Distance between the front axle and the center of gravity
        self.l_r = l_r  # Distance between the rear axle and the center of gravity
        self.max_steering_angle = max_steering_angle
        self.dt = world.dt
        self.integration = integration

    def reset(self, index: Union[Tensor, int] = None):
        pass

    def euler(self, f, state):
        # Update the state using Euler's method
        # For Euler's method, see https://math.libretexts.org/Bookshelves/Calculus/Book%3A_Active_Calculus_(Boelkins_et_al.)/07%3A_Differential_Equations/7.03%3A_Euler's_Method (the full link may not be recognized properly, please copy and paste in your browser)
        return state + self.dt * f(state)

    def runge_kutta(self, f, state):
        # Update the state using fourth-order Runge-Kutta method
        # For Runge-Kutta method, see https://math.libretexts.org/Courses/Monroe_Community_College/MTH_225_Differential_Equations/3%3A_Numerical_Methods/3.3%3A_The_Runge-Kutta_Method
        k1 = f(state)
        k2 = f(state + self.dt * k1 / 2)
        k3 = f(state + self.dt * k2 / 2)
        k4 = f(state + self.dt * k3)
        return state + (self.dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def process_force(self):
        # Extracts the velocity and steering angle from the agent's actions and convert them to physical force and torque
        velocity = self.agent.action.u[:, vmas.simulator.utils.X]
        steering_angle = self.agent.action.u_rot.squeeze(-1)
        # Ensure steering angle is within bounds
        steering_angle = torch.clamp(
            steering_angle, -self.max_steering_angle, self.max_steering_angle
        )

        # Current state of the agent
        state = torch.cat((self.agent.state.pos, self.agent.state.rot), dim=1)

        def f(state):
            theta = state[:, 2]  # jaw angle
            beta = torch.atan(
                torch.tan(steering_angle) * self.l_r / (self.l_f + self.l_r)
            )  # slip angle
            dx = velocity * torch.cos(theta + beta)
            dy = velocity * torch.sin(theta + beta)
            dtheta = velocity / (self.l_f + self.l_r) * torch.sin(beta)
            return torch.stack(
                (dx, dy, dtheta), dim=1
            )  # Should return torch.Size([batch_size,3])

        # Select the integration method
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
        self.agent.action.u[:, vmas.simulator.utils.X] = force_x
        self.agent.action.u[:, vmas.simulator.utils.Y] = force_y
        self.agent.action.u_rot[:, 0] = torque
