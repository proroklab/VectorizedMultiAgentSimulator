#  Copyright (c) 2022.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import torch
import numpy as np;
import vmas.simulator.core
import vmas.simulator.utils


class VelocityController:
    def __init__(self, agent: vmas.simulator.core.Agent, dt: float, ctrl_params=[1, 0, 0], pid_form="standard"):
        self.agent = agent
        self.dt = dt
        # controller parameters: standard=[kP, intTs ,dervTs], parallel=[kP, kI, kD]
        #    in parallel form, kI = kP/intTs and kD = kP*dervTs
        self.ctrl_gain = ctrl_params[0];    # kP
        if pid_form == "standard":
            self.integralTs = ctrl_params[1];
            self.derivativeTs = ctrl_params[2];
        elif pid_form == "parallel":
            if ctrl_params[1] == 0:
                self.integralTs = 0.0;
            else:
                self.integralTs = self.ctrl_gain / ctrl_params[1];
            self.derivativeTs = ctrl_params[2] / self.ctrl_gain;
        else:
            raise Exception( "PID form is either standard or parallel." );
        
        # in either form:
        if self.integralTs == 0:
            self.use_integrator = False;
        else:
            self.use_integrator = True;
        
        self.integrator_hist = 200;
        self.integrator_windup_cutoff = 2;
        # containers for integral & derivative control
        self.vel_errs = []
        self.prev_err = 0.0;

        # do other initialisation bits
        self.initialise();
    
    def initialise(self):
        # initialise containter for integrator only if I-gain is not strictly zero
        #if self.use_integrator:
        #    self.vel_errs = np.zeros(self.integrator_hist);
        self.vel_errs = [];

    def reset(self):
        self.vel_errs = np.zeros(self.integrator_hist);
    
    def integralError(self, err, _idx=[0]):
        if not self.use_integrator:
            return 0;
        # update integrator container as a fast ring buffer
        ### self.vel_errs[_idx[0]] = err;
        ### _idx[0] = (_idx[0] + 1) % self.integrator_hist;
        ### return (1.0/self.integralTs)*np.sum( self.vel_errs ) * self.dt;
        if len( self.vel_errs ) > self.integrator_hist-1:
            self.vel_errs.pop(0);
        self.vel_errs.append( err );
        return (1.0/self.integralTs) * torch.stack( self.vel_errs, dim=1 ).sum(dim=1) * self.dt;
    
    def rateError(self, err):
        e = self.derivativeTs * (err - self.prev_err)/self.dt;
        self.prev_err = err;
        return e;
            

    def process_force(self):
        des_vel = self.agent.action.u;
        cur_vel = self.agent.state.vel;

        # apply control
        err = des_vel - cur_vel;
        u = self.ctrl_gain * ( err + self.integralError(err) + self.rateError(err) );
        u = u * self.agent.mass;

        # Clamping force to limits
        if self.agent.max_f is not None:
            force = vmas.simulator.utils.clamp_with_norm(force, self.agent.max_f)
        if self.agent.f_range is not None:
            force = torch.clamp(force, -self.agent.f_range, self.agent.f_range)

        self.agent.action.u = u;
        before_after = des_vel[0].tolist() + u[0].tolist() + cur_vel[0].tolist();
        print( *before_after );
