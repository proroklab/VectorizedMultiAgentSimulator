#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import random
import time

import torch

from vmas import make_env
from vmas.simulator.core import Agent
from vmas.simulator.utils import save_video


def _get_random_action(agent: Agent, continuous: bool, env):
    if continuous:
        actions = []
        for action_index in range(agent.action_size):
            actions.append(
                torch.zeros(
                    agent.batch_dim,
                    device=agent.device,
                    dtype=torch.float32,
                ).uniform_(
                    -agent.action.u_range_tensor[action_index],
                    agent.action.u_range_tensor[action_index],
                )
            )
        if env.world.dim_c != 0 and not agent.silent:
            # If the agent needs to communicate
            for _ in range(env.world.dim_c):
                actions.append(
                    torch.zeros(
                        agent.batch_dim,
                        device=agent.device,
                        dtype=torch.float32,
                    ).uniform_(
                        0,
                        1,
                    )
                )
        action = torch.stack(actions, dim=-1)
    else:
        action = torch.randint(
            low=0,
            high=env.get_agent_action_space(agent).n,
            size=(agent.batch_dim,),
            device=agent.device,
        )
    return action


def _get_deterministic_action(agent: Agent, continuous: bool, env):
    if continuous:
        action = -agent.action.u_range_tensor.expand(env.batch_dim, agent.action_size)
    else:
        action = (
            torch.tensor([1], device=env.device, dtype=torch.long)
            .unsqueeze(-1)
            .expand(env.batch_dim, 1)
        )
    return action


def use_vmas_env(
    render: bool = False,
    save_render: bool = False,
    num_envs: int = 32,
    n_steps: int = 100,
    random_action: bool = False,
    device: str = "cpu",
    scenario_name: str = "waterfall",
    n_agents: int = 4,
    continuous_actions: bool = True,
):
    """Example function to use a vmas environment

    Args:
        continuous_actions (bool): Whether the agents have continuous or discrete actions
        n_agents (int): Number of agents
        scenario_name (str): Name of scenario
        device (str): Torch device to use
        render (bool): Whether to render the scenario
        save_render (bool):  Whether to save render of the scenario
        num_envs (int): Number of vectorized environments
        n_steps (int): Number of steps before returning done
        random_action (bool): Use random actions or have all agents perform the down action

    Returns:

    """
    assert not (save_render and not render), "To save the video you have to render it"

    dict_spaces = True  # Weather to return obs, rewards, and infos as dictionaries with agent names
    # (by default they are lists of len # of agents)

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
        print(f"Step {step}")

        # VMAS actions can be either a list of tensors (one per agent)
        # or a dict of tensors (one entry per agent with its name as key)
        # Both action inputs can be used independently of what type of space its chosen
        dict_actions = random.choice([True, False])

        actions = {} if dict_actions else []
        for i, agent in enumerate(env.agents):
            if not random_action:
                action = _get_deterministic_action(agent, continuous_actions, env)
            else:
                action = _get_random_action(agent, continuous_actions, env)
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

    total_time = time.time() - init_time
    print(
        f"It took: {total_time}s for {n_steps} steps of {num_envs} parallel environments on device {device} "
        f"for {scenario_name} scenario."
    )

    if render and save_render:
        save_video(scenario_name, frame_list, fps=1 / env.scenario.world.dt)


if __name__ == "__main__":
    use_vmas_env(
        scenario_name="waterfall",
        render=True,
        save_render=False,
        random_action=False,
        continuous_actions=False,
    )
