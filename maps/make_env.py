"""
Code for creating a multiagent environment with one of the scenarios listed
in ./scenarios/.
Can be called by using, for example:
    env = make_env('waterfall')
After producing the env object, can be used similarly to an OpenAI gym
environment.
"""

from maps import scenarios
from maps.simulator.environment import VectorEnvWrapper, Environment


def make_env(
    scenario_name,
    num_envs: int = 32,
    device: str = "cpu",
    continuous_actions: bool = True,
    rllib_wrapped: bool = False,
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

    return VectorEnvWrapper(env) if rllib_wrapped else env
