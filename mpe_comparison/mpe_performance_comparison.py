#  Copyright (c) 2022. Matteo Bettini
#  All rights reserved.

import argparse
import os
import platform
import re
import subprocess
import time

import torch
from matplotlib import pyplot as plt
from matplotlib.pyplot import xticks

import maps


def mpe_make_env(scenario_name):
    from mpe.multiagent.environment import MultiAgentEnv
    import mpe.multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(
        world, scenario.reset_world, scenario.reward, scenario.observation
    )
    return env


def run_mpe_simple_spread(n_envs: int, n_steps: int):
    n_envs = int(n_envs)
    n_steps = int(n_steps)
    n_agents = 3
    envs = [mpe_make_env("simple_spread") for _ in range(n_envs)]
    simple_shared_action = [0, 1, 0, 0, 0]

    [env.reset() for env in envs]
    init_time = time.time()

    for step in range(n_steps):
        for env_idx in range(n_envs):
            actions = []
            for i in range(n_agents):
                actions.append(simple_shared_action)
            envs[env_idx].step(actions)

    total_time = time.time() - init_time
    return total_time


def run_maps_simple_spread(n_envs: int, n_steps: int, device: str):
    n_envs = int(n_envs)
    n_steps = int(n_steps)
    n_agents = 3
    env = maps.make_env(
        "simple_spread",
        device=device,
        num_envs=n_envs,
        continuous_actions=False,
        # Scenario specific config
        n_agents=n_agents,
    )
    simple_shared_action = [2]

    env.reset()
    init_time = time.time()

    for step in range(n_steps):
        actions = []
        for i in range(n_agents):
            actions.append(
                torch.tensor(
                    simple_shared_action,
                    device=device,
                ).repeat(n_envs, 1)
            )
        env.step(actions)

    total_time = time.time() - init_time
    return total_time


def get_device_name(torch_device: str):
    if torch_device == "cpu":
        if platform.system() == "Darwin":
            return "Apple M1 Pro"
        else:
            if platform.system() == "Windows":
                return platform.processor()
            elif platform.system() == "Linux":
                command = "cat /proc/cpuinfo"
                all_info = subprocess.check_output(command, shell=True).decode().strip()
                for line in all_info.split("\n"):
                    if "model name" in line:
                        return re.sub(".*model name.*:", "", line, 1)
            else:
                assert False
    elif torch_device == "cuda":
        return torch.cuda.get_device_name()
    else:
        assert False


def run_comparison(maps_device: str, n_steps: int = 100):
    device_name = get_device_name(maps_device)

    mpe_times = []
    maps_times = []

    list_n_envs = [1, 10, 100, 1e3, 1e4, 1e5]

    for n_envs in list_n_envs:
        mpe_times.append(run_mpe_simple_spread(n_envs=n_envs, n_steps=n_steps))
        maps_times.append(
            run_maps_simple_spread(n_envs=n_envs, n_steps=n_steps, device=maps_device)
        )

    fig, ax = plt.subplots()
    ax.plot(
        list_n_envs,
        mpe_times,
        label="MPE",
    )
    ax.plot(
        list_n_envs,
        maps_times,
        label="MAPS",
    )
    plt.xlabel("Number of parallel environments", fontsize=14)
    plt.ylabel("Seconds", fontsize=14)
    xticks(list_n_envs)
    ax.legend(loc="upper left")

    fig.suptitle("MAPS vs MPE", fontsize=16)
    ax.set_title(
        f"Execution time of 'simple_spread' for {n_steps} steps on {device_name}",
        fontsize=10,
    )

    figure_name = f"MAPS_vs_MPE_{n_steps}_steps_{device_name}"
    save_folder = os.path.dirname(os.path.realpath(__file__))
    plt.savefig(f"{save_folder}/maps_vs_mpe_graphs/{figure_name}.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a time comparison between MAPS and MPE"
    )

    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Use cuda device for MAPS",
    )

    args = parser.parse_args()

    run_comparison(maps_device="cuda" if args.cuda else "cpu")
