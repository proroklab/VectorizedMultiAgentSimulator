#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

from typing import Optional, Union

from vmas import scenarios
from vmas.simulator.environment import Environment, Wrapper
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import DEVICE_TYPING


def make_env(
    scenario: Union[str, BaseScenario],
    num_envs: int,
    device: DEVICE_TYPING = "cpu",
    continuous_actions: bool = True,
    wrapper: Optional[
        Wrapper
    ] = None,  # One of: None, vmas.Wrapper.RLLIB, and vmas.Wrapper.GYM
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
    dict_spaces: bool = False,
    multidiscrete_actions: bool = False,
    clamp_actions: bool = False,
    grad_enabled: bool = False,
    **kwargs,
):
    """Create a vmas environment.

    Args:
        scenario (Union[str, BaseScenario]): Scenario to load.
            Can be the name of a file in `vmas.scenarios` folder or a :class:`~vmas.simulator.scenario.BaseScenario` class,
        num_envs (int): Number of vectorized simulation environments. VMAS performs vectorized simulations using PyTorch.
            This argument indicates the number of vectorized environments that should be simulated in a batch. It will also
            determine the batch size of the environment.
        device (Union[str, int, torch.device]): Device for simulation. All the tensors created by VMAS
            will be placed on this device. Default is ``"cpu""``,
        continuous_actions (bool): Whether to use continuous actions. Defaults to ``True``. If ``False``, actions
            will be discrete. The number of actions and their size will depend on the chosen scenario. Default is ``True``,
        wrapper (:class:`~vmas.simulator.environment.Wrapper`, optional): Wrapper class to use. For example can be Wrapper.RLLIB. Default is ``None``.
        max_steps (int, optional): Horizon of the task. Defaults to ``None`` (infinite horizon). Each VMAS scenario can
            be terminating or not. If ``max_steps`` is specified,
            the scenario is also terminated whenever this horizon is reached.
        seed: seed
        dict_spaces:  Weather to use dictionary i/o spaces with format {agent_name: tensor}
            for obs, rewards, and info instead of tuples.
        multidiscrete_actions (bool): Whether to use multidiscrete_actions action spaces when continuous_actions=False.
            Otherwise, (default) the action space will be Discrete, and it will be the cartesian product of the
            action spaces of an agent.
        clamp_actions: Weather to clamp input actions to the range instead of throwing
            an error when continuous_actions is True and actions are out of bounds
        grad_enabled: (bool): Whether the simulator will keep track of gradients in the output. Default is ``False``.
        **kwargs ()

    """

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
        dict_spaces=dict_spaces,
        multidiscrete_actions=multidiscrete_actions,
        clamp_actions=clamp_actions,
        grad_enabled=grad_enabled,
        **kwargs,
    )

    return wrapper.get_env(env) if wrapper is not None else env
