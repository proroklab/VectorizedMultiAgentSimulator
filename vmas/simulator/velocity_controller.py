#  Copyright (c) 2022.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import torch
import vmas.simulator.core
import vmas.simulator.utils


class VelocityController:
    def __init__(self, agent: vmas.simulator.core.Agent, dt: float):
        self.agent = agent
        self.dt = dt
        self.positions = []
        self.velocities = []

    def process_force(self):
        desired_velocity = self.agent.action.u
        pos = self.agent.state.pos
        vel = self.agent.state.vel

        # Appending to the ques
        self.positions.append(pos)
        self.velocities.append(vel)

        # Controller
        force = ((desired_velocity - vel) / self.dt) * self.agent.mass

        # Clamping force to limits
        if self.agent.max_f is not None:
            force = vmas.simulator.utils.clamp_with_norm(force, self.agent.max_f)
        if self.agent.f_range is not None:
            force = torch.clamp(force, -self.agent.f_range, self.agent.f_range)

        if len(self.positions) > 5:
            self.positions.pop(0)
        if len(self.velocities) > 5:
            self.velocities.pop(0)

        self.agent.action.u = force
