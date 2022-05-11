"""
Code for creating a multiagent environment with one of the scenarios listed
in ./scenarios/.
Can be called by using, for example:
    env = make_env('simple_speaker_listener')
After producing the env object, can be used similarly to an OpenAI gym
environment.

A policy using this environment must output actions in the form of a list
for all agents. Each element of the list should be a numpy array,
of size (env.world.dim_p + env.world.dim_c, 1). Physical actions precede
communication actions in this array. See environment.py for more details.
"""
import time

import numpy as np
import torch

from mpe.multiagent import scenarios
from multiagent.environment import VectorEnvWrapper, Environment


def make_env(
    scenario_name,
    num_envs: int = 32,
    device: str = "cpu",
    continuous_actions: bool = True,
):
    """
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    """
    from multiagent.environment import Environment
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()

    return VectorEnvWrapper(
        Environment(
            scenario,
            num_envs=num_envs,
            device=device,
            continuous_actions=continuous_actions,
        )
    )


if __name__ == "__main__":
    num_envs = 400
    continuous_actions = False
    init_time = time.time()
    device = "cpu"
    wrapped = False
    n_steps = 800
    n_agents = 5

    scenario = scenarios.load("simple" + ".py").Scenario()
    env = Environment(
        scenario,
        num_envs=num_envs,
        device=device,
        continuous_actions=continuous_actions,
    )
    if wrapped:
        env = VectorEnvWrapper(env)

    for _ in range(n_steps):
        actions = []
        if wrapped:
            for i in range(num_envs):
                actions_per_env = []
                for j in range(n_agents):
                    actions_per_env.append(
                        np.array([0.0, -0.03] if continuous_actions else [3])
                    )
                actions.append(actions_per_env)
            obs, rews, dones, info = env.vector_step(actions)
            env.try_render_at(31)
        else:
            for i in range(n_agents):
                actions.append(
                    torch.tensor(
                        [0.0, -0.03] if continuous_actions else [3], device=device
                    ).repeat(num_envs, 1)
                )
            obs, rews, dones, info = env.step(actions)
            env.render(index=0)
    total_time = time.time() - init_time
    print(
        f"It took: {total_time}s for {n_steps} steps of {num_envs} parallel environments on device {device} for {'wrapped' if wrapped else 'unwrapped'} simulator"
    )
