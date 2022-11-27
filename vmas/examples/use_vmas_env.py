#  Copyright (c) 2022.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import time

import numpy as np
import torch

from vmas import make_env, Wrapper
from vmas.simulator.utils import save_video


def use_vmas_env(render: bool = False, save_render: bool = False):
    assert not (save_render and not render), "To save the video you have to render it"

    scenario_name = "waterfall"

    # Scenario specific variables
    n_agents = 4

    num_envs = 32
    continuous_actions = False
    device = "cpu"  # or cuda or any other torch device
    wrapper = None
    n_steps = 100

    simple_2d_action = (
        [0, 0.5] if continuous_actions else [3]
    )  # Sample action tell each agent to go down

    env = make_env(
        scenario_name=scenario_name,
        num_envs=num_envs,
        device=device,
        continuous_actions=continuous_actions,
        wrapper=wrapper,
        # Environment specific variables
        n_agents=n_agents,
    )

    frame_list = []  # For creating a gif
    init_time = time.time()
    step = 0
    for s in range(n_steps):
        actions = []
        step += 1
        print(f"Step {step}")
        if wrapper is Wrapper.RLLIB:  # Rllib interface
            for i in range(num_envs):
                actions_per_env = []
                for j in range(n_agents):
                    actions_per_env.append(np.array(simple_2d_action))
                actions.append(actions_per_env)
            obs, rews, dones, info = env.vector_step(actions)
            if render:
                frame_list.append(
                    env.try_render_at(
                        mode="rgb_array",
                        agent_index_focus=None,
                        visualize_when_rgb=True,
                    )  # Can give the camera an agent index to focus on
                )
        elif wrapper is None:  # Same as before, with faster VMAS interface
            for i in range(n_agents):
                actions.append(
                    torch.tensor(
                        simple_2d_action,
                        device=device,
                    ).repeat(num_envs, 1)
                )
            obs, rews, dones, info = env.step(actions)
            if render:
                frame_list.append(
                    env.render(
                        mode="rgb_array",
                        agent_index_focus=None,
                        visualize_when_rgb=True,
                    )
                )  # Can give the camera an agent index to focus on

    if render and save_render:
        if wrapper is not None:
            env = env.env
        save_video(scenario_name, frame_list, fps=1 / env.scenario.world.dt)

    total_time = time.time() - init_time
    print(
        f"It took: {total_time}s for {n_steps} steps of {num_envs} parallel environments on device {device}"
        f" for {wrapper.name + ' wrapped ' if wrapper is not None else ''}simulator"
    )


if __name__ == "__main__":
    use_vmas_env(render=True, save_render=False)
