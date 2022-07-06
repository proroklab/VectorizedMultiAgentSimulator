#  Copyright (c) 2022.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import os
import time

import torch
from PIL import Image
from vmas import make_env
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy


class RandomPolicy(BaseHeuristicPolicy):
    def compute_action(self, observation: torch.Tensor, u_range: float) -> torch.Tensor:
        n_envs = observation.shape[0]
        return torch.clamp(torch.randn(n_envs, 2), -u_range, u_range)


def run_heuristic(
    scenario_name: str = "transport",
    heuristic: BaseHeuristicPolicy = RandomPolicy,
    n_steps: int = 200,
    n_envs: int = 32,
    env_kwargs: dict = {},
    render: bool = False,
    save: bool = False,
    device: str = "cpu",
):

    # Scenario specific variables
    policy = heuristic(continuous_action=True)

    env = make_env(
        scenario_name=scenario_name,
        num_envs=n_envs,
        device=device,
        continuous_actions=True,
        wrapper=None,
        # Environment specific variables
        **env_kwargs,
    )

    frame_list = []  # For creating a gif
    init_time = time.time()
    step = 0
    obs = env.reset()
    total_reward = 0
    for s in range(n_steps):
        step += 1
        actions = [None] * len(obs)
        for i in range(len(obs)):
            actions[i] = policy.compute_action(obs[i], u_range=env.agents[i].u_range)
        obs, rews, dones, info = env.step(actions)
        rewards = torch.stack(rews, dim=1)
        global_reward = rewards.mean(dim=1)
        mean_global_reward = global_reward.mean(dim=0)
        total_reward += mean_global_reward
        if render:
            if save:
                frame_list.append(
                    Image.fromarray(
                        env.render(
                            mode="rgb_array",
                            agent_index_focus=None,
                            visualize_when_rgb=True,
                        )
                    )
                )
            else:
                env.render(mode="human")

    total_time = time.time() - init_time
    if render and save:
        gif_name = scenario_name + ".gif"
        # Produce a gif
        frame_list[0].save(
            gif_name,
            save_all=True,
            append_images=frame_list[1:],
            duration=3,
            loop=0,
        )
        # Requires image magik to be installed to convert the gif in faster format
        os.system(f"convert -delay 1x30 -loop 0 {gif_name} {scenario_name}_fast.gif")
    print(
        f"It took: {total_time}s for {n_steps} steps of {n_envs} parallel environments on device {device}\n"
        f"The average total reward was {total_reward}"
    )


if __name__ == "__main__":
    from vmas.scenarios.transport import HeuristicPolicy as TransportHeuristic

    run_heuristic(
        scenario_name="transport",
        heuristic=TransportHeuristic,
        n_envs=300,
        n_steps=200,
        render=True,
        save=False,
    )
