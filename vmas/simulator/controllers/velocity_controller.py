#  Copyright (c) ProrokLab.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
import math
import warnings
from typing import Optional

import torch

import vmas.simulator.core
import vmas.simulator.utils
from vmas.simulator.utils import TorchUtils


class VelocityController:
    """
    Implements PID controller for velocity targets found in agent.action.u.
    Two forms of the PID controller are implemented: standard, and parallel. The controller takes 3 params, which
    are interpreted differently based on the form.
    > Standard form: ctrl_params=[gain, intg_ts, derv_ts]
                        intg_ts: rise time for integrator (err will be tolerated for this interval)
                        derv_ts: seek time for derivative (err is predicted over this interval)
                        These are specified in 1/dt scale (0.5 means 0.5/0.1==5sec)
    > Parallel form: ctrl_params=[kP, kI, kD]
                        kI and kD have no simple physical meaning, but are related to standard form params.
                        intg_ts = kP/kI and kD/kP = derv_ts
    """

    def __init__(
        self,
        agent: vmas.simulator.core.Agent,
        world: vmas.simulator.core.World,
        ctrl_params=(1, 0, 0),
        pid_form="standard",
    ):
        self.agent = agent
        self.world = world
        self.dt = world.dt
        # controller parameters: standard=[kP, intgTs ,dervTs], parallel=[kP, kI, kD]
        #    in parallel form, kI = kP/intgTs and kD = kP*dervTs
        self.ctrl_gain = ctrl_params[0]  # kP
        if pid_form == "standard":
            self.integralTs = ctrl_params[1]
            self.derivativeTs = ctrl_params[2]
        elif pid_form == "parallel":
            if ctrl_params[1] == 0:
                self.integralTs = 0.0
            else:
                self.integralTs = self.ctrl_gain / ctrl_params[1]
            self.derivativeTs = ctrl_params[2] / self.ctrl_gain
        else:
            raise Exception("PID form is either standard or parallel.")

        # in either form:
        if self.integralTs == 0:
            self.use_integrator = False
        else:
            self.use_integrator = True
            # set windup limit to 50% of agent's max force
            fmax = min(
                self.agent.max_f,
                self.agent.f_range,
                key=lambda x: x if x is not None else math.inf,
            )

            if fmax is not None:
                self.integrator_windup_cutoff = (
                    0.5 * fmax * self.integralTs / (self.dt * self.ctrl_gain)
                )
            else:
                self.integrator_windup_cutoff = None
                warnings.warn("Force limits not specified. Integrator can wind up!")

        self.reset()

    def reset(self, index: Optional[int] = None):
        if index is None:
            self.accum_errs = torch.zeros(
                (self.world.batch_dim, self.world.dim_p),
                device=self.world.device,
            )
            self.prev_err = torch.zeros(
                (self.world.batch_dim, self.world.dim_p),
                device=self.world.device,
            )
        else:
            self.accum_errs = TorchUtils.where_from_index(index, 0.0, self.accum_errs)
            self.prev_err = TorchUtils.where_from_index(index, 0.0, self.prev_err)

    def integralError(self, err):
        if not self.use_integrator:
            return 0
        # fixed-length history (not recommended):
        # if len( self.accum_errs ) > self.integrator_hist-1:
        #    self.accum_errs.pop(0);
        # self.accum_errs.append( err );
        # return (1.0/self.integralTs) * torch.stack( self.accum_errs, dim=1 ).sum(dim=1) * self.dt;

        self.accum_errs += self.dt * err
        if self.integrator_windup_cutoff is not None:
            self.accum_errs = self.accum_errs.clamp(
                -self.integrator_windup_cutoff, self.integrator_windup_cutoff
            )

        return (1.0 / self.integralTs) * self.accum_errs

    def rateError(self, err):
        e = self.derivativeTs * (err - self.prev_err) / self.dt
        self.prev_err = err
        return e

    def process_force(self):
        self.accum_errs = self.accum_errs.to(self.world.device)
        self.prev_err = self.prev_err.to(self.world.device)

        des_vel = self.agent.action.u
        cur_vel = self.agent.state.vel

        # apply control
        err = des_vel - cur_vel
        u = self.ctrl_gain * (err + self.integralError(err) + self.rateError(err))
        u *= self.agent.mass

        self.agent.action.u = u
