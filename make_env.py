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
import os
import time

import numpy as np
import torch
from PIL import Image

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
    scenario_name = "maps_simple"
    num_envs = 32
    continuous_actions = True
    device = "cpu"
    wrapped = True
    n_steps = 150
    n_agents = 5

    simple_2d_action = [0.0, -0.18] if continuous_actions else [3] # Smaple action tell each agent to go down

    env = make_env(
        scenario_name=scenario_name,
        num_envs=num_envs,
        device=device,
        continuous_actions=continuous_actions,
        rllib_wrapped=wrapped,
        n_agents=n_agents,
    )

    frame_list = [] # For creating a gif
    init_time = time.time()
    for s in range(n_steps):
        actions = []
        if wrapped: # Rllib interface
            for i in range(num_envs):
                actions_per_env = []
                for j in range(n_agents):
                    actions_per_env.append(
                        np.array(simple_2d_action)
                    )
                actions.append(actions_per_env)
            obs, rews, dones, info = env.vector_step(actions)
            frame_list.append(Image.fromarray(env.try_render_at(mode="rgb_array", agent_index_focus=None))) # Can give the camera an agent index to focus on

        else: # Same as before, with faster MAPS interface
            for i in range(n_agents):
                actions.append(
                    torch.tensor(
                        simple_2d_action,
                        device=device,
                    ).repeat(num_envs, 1)
                )
            obs, rews, dones, info = env.step(actions)
            frame_list.append(Image.fromarray(env.render(mode="rgb_array", agent_index_focus=None))) # Can give the camera an agent index to focus on

    gif_name = scenario_name + ".gif"

    # Produce a gif
    frame_list[0].save(
        gif_name,
        save_all=True,
        append_images=frame_list[1:],
        duration=3,
        loop=0,
    )
    # Requires software to bi installed to convert the gif in faster format
    os.system(f"convert -delay 1x30 -loop 0 {gif_name} {gif_name}")


    total_time = time.time() - init_time
    print(
        f"It took: {total_time}s for {n_steps} steps of {num_envs} parallel environments on device {device} for {'wrapped' if wrapped else 'unwrapped'} simulator"
    )
