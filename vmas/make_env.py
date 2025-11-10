#  Copyright (c) ProrokLab.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

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
    wrapper: Optional[Union[Wrapper, str]] = None,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
    dict_spaces: bool = False,
    multidiscrete_actions: bool = False,
    clamp_actions: bool = False,
    grad_enabled: bool = False,
    terminated_truncated: bool = False,
    wrapper_kwargs: Optional[dict] = None,
    **kwargs,
):
    """Create a vmas environment.

    Args:
        scenario (Union[str, BaseScenario]): Scenario to load.
            Can be the name of a file in `vmas.scenarios` folder or a :class:`~vmas.simulator.scenario.BaseScenario` class,
        num_envs (int): Number of vectorized simulation environments. VMAS performs vectorized simulations using PyTorch.
            This argument indicates the number of vectorized environments that should be simulated in a batch. It will also
            determine the batch size of the environment.
        device (Union[str, int, torch.device], optional): Device for simulation. All the tensors created by VMAS
            will be placed on this device. Default is ``"cpu"``,
        continuous_actions (bool, optional): Whether to use continuous actions. If ``False``, actions
            will be discrete. The number of actions and their size will depend on the chosen scenario. Default is ``True``,
        wrapper (Union[Wrapper, str], optional): Wrapper class to use. For example, it can be
            ``"rllib"``, ``"gym"``, ``"gymnasium"``, ``"gymnasium_vec"``. Default is ``None``.
        max_steps (int, optional): Horizon of the task. Defaults to ``None`` (infinite horizon). Each VMAS scenario can
            be terminating or not. If ``max_steps`` is specified,
            the scenario is also terminated whenever this horizon is reached,
        seed (int, optional): Seed for the environment. Defaults to ``None``,
        dict_spaces (bool, optional):  Weather to use dictionaries spaces with format ``{"agent_name": tensor, ...}``
            for obs, rewards, and info instead of tuples. Defaults to ``False``: obs, rewards, info are tuples with length number of agents,
        multidiscrete_actions (bool, optional): Whether to use multidiscrete action spaces when ``continuous_actions=False``.
            Default is ``False``: the action space will be ``Discrete``, and it will be the cartesian product of the
            discrete action spaces available to an agent,
        clamp_actions (bool, optional): Weather to clamp input actions to their range instead of throwing
            an error when ``continuous_actions==True`` and actions are out of bounds,
        grad_enabled (bool, optional): If ``True`` the simulator will not call ``detach()`` on input actions and gradients can
            be taken from the simulator output. Default is ``False``.
        terminated_truncated (bool, optional): Weather to use terminated and truncated flags in the output of the step method (or single done).
            Default is ``False``.
        wrapper_kwargs (dict, optional): Keyword arguments to pass to the wrapper class. Default is ``{}``.
        **kwargs (dict, optional): Keyword arguments to pass to the :class:`~vmas.simulator.scenario.BaseScenario` class.

    Examples:
        >>> from vmas import make_env
        >>> env = make_env(
        ...     "waterfall",
        ...     num_envs=3,
        ...     num_agents=2,
        ... )
        >>> print(env.reset())


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
        terminated_truncated=terminated_truncated,
        **kwargs,
    )

    if wrapper is not None and isinstance(wrapper, str):
        wrapper = Wrapper[wrapper.upper()]

    if wrapper_kwargs is None:
        wrapper_kwargs = {}

    return wrapper.get_env(env, **wrapper_kwargs) if wrapper is not None else env
