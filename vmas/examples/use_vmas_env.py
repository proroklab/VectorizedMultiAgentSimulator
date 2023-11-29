#  Copyright (c) 2022-2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import pstats
import random
import time

import torch

from vmas import make_env
from vmas.simulator.utils import save_video, VecCollisions


def use_vmas_env(render: bool = False, save_render: bool = False):
    assert not (save_render and not render), "To save the video you have to render it"

    scenario_name = "waterfall"

    # Scenario specific variables
    n_agents = 4

    num_envs = 32  # Number of vectorized environments
    continuous_actions = False
    device = "cpu"  # or cuda or any other torch device
    n_steps = 100  # Number of steps before returning done
    dict_spaces = True  # Weather to return obs, rewards, and infos as dictionaries with agent names
    # (by default they are lists of len # of agents)

    simple_2d_action = (
        [0, 0.5] if continuous_actions else [3]
    )  # Simple action for an agent with 2d actions
    simple_3d_action = (
        [0, 0.5, 0.1] if continuous_actions else [3, 1]
    )  # Simple action for an agent with 3d actions (2d forces and torque)

    env = make_env(
        scenario=scenario_name,
        num_envs=num_envs,
        device=device,
        continuous_actions=continuous_actions,
        dict_spaces=dict_spaces,
        wrapper=None,
        seed=None,
        # Environment specific variables
        n_agents=n_agents,
    )

    frame_list = []  # For creating a gif
    init_time = time.time()
    step = 0

    for s in range(n_steps):
        step += 1
        # print(f"Step {step}")

        # VMAS actions can be either a list of tensors (one per agent)
        # or a dict of tensors (one entry per agent with its name as key)
        # Both action inputs can be used independently of what type of space its chosen
        dict_actions = random.choice([True, False])

        actions = {} if dict_actions else []
        for i, agent in enumerate(env.agents):
            action = torch.tensor(
                simple_2d_action if agent.u_rot_range == 0 else simple_3d_action,
                device=device,
            ).repeat(num_envs, 1)
            # action = torch.zeros(
            #     (num_envs, 2),
            #     device=device,
            #     dtype=torch.float32,
            # ).uniform_(
            #     -agent.action.u_range,
            #     agent.action.u_range,
            # )
            if dict_actions:
                actions.update({agent.name: action})
            else:
                actions.append(action)

        obs, rews, dones, info = env.step(actions)

        if render:
            frame = env.render(
                mode="rgb_array" if save_render else "human",
                agent_index_focus=None,  # Can give the camera an agent index to focus on
                visualize_when_rgb=True,
            )
            if save_render:
                frame_list.append(frame)

    if render and save_render:
        save_video(scenario_name, frame_list, fps=1 / env.scenario.world.dt)

    total_time = time.time() - init_time
    print(
        f"It took: {total_time}s for {n_steps} steps of {num_envs} parallel environments on device {device} "
        f"for {scenario_name} scenario."
    )
    return env


def check_env_same_state(env1, env2):
    for entity_a, entity_b in zip(env1.world.entities, env2.world.entities):
        assert torch.allclose(entity_a.state.pos, entity_b.state.pos, atol=1e-3)
        assert torch.allclose(entity_a.state.rot, entity_b.state.rot, atol=1e-3)


if __name__ == "__main__":
    import cProfile

    profiler = cProfile.Profile()

    print("No vec collisions")
    profiler.enable()
    torch.manual_seed(0)
    VecCollisions.VECTORIZED_COLLISIONS = False
    env1 = use_vmas_env(render=False, save_render=False)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("tottime")
    # stats.print_stats()

    profiler = cProfile.Profile()
    print("Vec collisions")
    torch.manual_seed(0)
    profiler.enable()
    VecCollisions.VECTORIZED_COLLISIONS = True
    env2 = use_vmas_env(render=False, save_render=False)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("tottime")
    # stats.print_stats()

    check_env_same_state(env1, env2)
