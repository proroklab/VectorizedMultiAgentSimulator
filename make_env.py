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

from maps import scenarios
from maps.environment import VectorEnvWrapper, Environment


def make_env(
    scenario_name,
    num_envs: int = 32,
    device: str = "cpu",
    continuous_actions: bool = True,
    rllib_wrapped: bool = False,
    **kwargs,
):
    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    env = Environment(
        scenario,
        num_envs=num_envs,
        device=device,
        continuous_actions=continuous_actions,
        **kwargs,
    )

    return VectorEnvWrapper(env) if rllib_wrapped else env


if __name__ == "__main__":
    num_envs = 32
    continuous_actions = True
    device = "cpu"
    wrapped = True
    n_steps = 800
    n_agents = 1

    env = make_env(
        scenario_name="simple",
        num_envs=num_envs,
        device=device,
        continuous_actions=continuous_actions,
        rllib_wrapped=wrapped,
        n_agents=n_agents,
    )

    frame_list = np.empty((n_steps, n_agents), dtype=object)
    init_time = time.time()
    for s in range(n_steps):
        actions = []
        if wrapped:
            for i in range(num_envs):
                actions_per_env = []
                for j in range(n_agents):
                    actions_per_env.append(
                        np.array([0.0, -1.0] if continuous_actions else [3])
                    )
                actions.append(actions_per_env)
            obs, rews, dones, info = env.vector_step(actions)
            env.try_render_at()

        else:
            for i in range(n_agents):
                actions.append(
                    torch.tensor(
                        [0.0, -1.0] if continuous_actions else [3],
                        device=device,
                    ).repeat(num_envs, 1)
                )
            obs, rews, dones, info = env.step(actions)
            env.render()

    total_time = time.time() - init_time
    print(
        f"It took: {total_time}s for {n_steps} steps of {num_envs} parallel environments on device {device} for {'wrapped' if wrapped else 'unwrapped'} simulator"
    )
