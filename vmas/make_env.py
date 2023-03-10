#  Copyright (c) 2022-2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

from typing import Optional, Union

from vmas import scenarios
from vmas.simulator.environment import Environment
from vmas.simulator.environment import Wrapper
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import DEVICE_TYPING


def make_env(
    scenario: Union[str, BaseScenario],
    num_envs: int = 32,
    device: DEVICE_TYPING = "cpu",
    continuous_actions: bool = True,
    wrapper: Optional[
        Wrapper
    ] = None,  # One of: None, vmas.Wrapper.RLLIB, and vmas.Wrapper.GYM
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
    **kwargs,
):
    # load scenario from script
    if isinstance(scenario, str):
        if not scenario.endswith(".py"):
            scenario += ".py"
        scenario = scenarios.load(scenario).Scenario()
    env = Environment(
        scenario,
        num_envs=num_envs,
        device=device,
        continuous_actions=continuous_actions,
        max_steps=max_steps,
        seed=seed,
        **kwargs,
    )

    return wrapper.get_env(env) if wrapper is not None else env
