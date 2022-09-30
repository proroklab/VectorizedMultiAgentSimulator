#  Copyright (c) 2022.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import time

import numpy as np
import torch
from matplotlib import pyplot as plt

from vmas import make_env, Wrapper


def use_vmas_env(render: bool = False, save_render: bool = False):

    scenario_name = "debug"

    # Scenario specific variables
    n_agents = 3

    num_envs = 1
    continuous_actions = True
    device = "cpu"  # or cuda or any other torch device
    wrapper = Wrapper.RLLIB
    n_steps = 200

    simple_2d_action = (
        [1, 0] if continuous_actions else [3]
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
    v0 = []
    v1 = []
    v2 = []
    for s in range(n_steps):
        if s > n_steps / 2:
            simple_2d_action = [0, 0] if continuous_actions else [3]
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
            v0.append(obs[0][0][2])
            v1.append(obs[0][1][2])
            v2.append(obs[0][2][2])
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

    fig, ax = plt.subplots()

    ax.plot(v0, label="v0")
    ax.plot(v1, label="v1")
    ax.plot(v2, label="v2")

    v0_shifted = v0[1:] + [v0[-1]]
    v0_shifted = np.array(v0_shifted)
    v0 = np.array(v0)
    ax.plot((v0_shifted - v0) / env.env.scenario.world.dt, label="a0")

    v1_shifted = v1[1:] + [v1[-1]]
    v1_shifted = np.array(v1_shifted)
    v1 = np.array(v1)
    ax.plot((v1_shifted - v1) / env.env.scenario.world.dt, label="a1")

    v2_shifted = v2[1:] + [v2[-1]]
    v2_shifted = np.array(v2_shifted)
    v2 = np.array(v2)
    ax.plot((v2_shifted - v2) / env.env.scenario.world.dt, label="a2")

    plt.legend()
    plt.show()

    if render and save_render:
        import cv2

        video_name = scenario_name + ".mp4"

        # Produce a video
        video = cv2.VideoWriter(
            video_name,
            cv2.VideoWriter_fourcc(*"mp4v"),
            30,  # FPS
            (frame_list[0].shape[1], frame_list[0].shape[0]),
        )
        for img in frame_list:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            video.write(img)
        video.release()

    total_time = time.time() - init_time
    print(
        f"It took: {total_time}s for {n_steps} steps of {num_envs} parallel environments on device {device}"
        f" for {wrapper.name}{' wrapped' if wrapper is not None else ''} simulator"
    )


if __name__ == "__main__":
    use_vmas_env(render=True, save_render=False)
