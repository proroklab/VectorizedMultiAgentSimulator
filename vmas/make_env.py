"""
Code for creating a multiagent environment with one of the scenarios listed
in ./scenarios/.
Can be called by using, for example:
    env = make_env('waterfall')
After producing the env object, can be used similarly to an OpenAI gym
environment.
"""

#  Copyright (c) 2022.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

from vmas import scenarios
from vmas.simulator.environment import Environment
from vmas.simulator.environment import Wrapper


def make_env(
    scenario_name,
    num_envs: int = 32,
    device: str = "cpu",
    continuous_actions: bool = True,
    wrapper: Wrapper = None,  # One of: None, vmas.Wrapper.RLLIB, and vmas.Wrapper.GYM
    max_steps: int = None,
    **kwargs,
):
    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    env = Environment(
        scenario,
        num_envs=num_envs,
        device=device,
        continuous_actions=continuous_actions,
        max_steps=max_steps,
        **kwargs,
    )

    return wrapper.get_env(env) if wrapper is not None else env
