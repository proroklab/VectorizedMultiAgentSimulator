#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import argparse
import os
import pickle
import platform
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import tikzplotlib
import torch

import vmas
from matplotlib import pyplot as plt


def mpe_make_env(scenario_name):
    import mpe.multiagent.scenarios as scenarios
    from mpe.multiagent.environment import MultiAgentEnv

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

    for _ in range(n_steps):
        for env_idx in range(n_envs):
            actions = []
            for _ in range(n_agents):
                actions.append(simple_shared_action)
            envs[env_idx].step(actions)

    total_time = time.time() - init_time
    return total_time


def run_vmas_simple_spread(n_envs: int, n_steps: int, device: str):
    n_envs = int(n_envs)
    n_steps = int(n_steps)
    n_agents = 3
    env = vmas.make_env(
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

    for _ in range(n_steps):
        actions = []
        for _ in range(n_agents):
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
                raise AssertionError()
    elif torch_device == "cuda":
        return torch.cuda.get_device_name()
    else:
        raise AssertionError()


def store_pickled_evaluation(name: str, evaluation: list):
    save_folder = (
        f"{os.path.dirname(os.path.realpath(__file__))}/vmas_vs_mpe_graphs/pickled"
    )
    file = f"{save_folder}/{name}.pkl"

    pickle.dump(evaluation, open(file, "wb"))


def load_pickled_evaluation(
    name: str,
):
    save_folder = (
        f"{os.path.dirname(os.path.realpath(__file__))}/vmas_vs_mpe_graphs/pickled"
    )
    file = Path(f"{save_folder}/{name}.pkl")

    if file.is_file():
        return pickle.load(open(file, "rb"))
    return None


def run_comparison(vmas_device: str, n_steps: int = 100):
    device_name = get_device_name(vmas_device)

    mpe_times = []
    vmas_times = []

    low = 1
    high = 30000
    num = 100

    list_n_envs = np.linspace(low, high, num)

    figure_name = f"VMAS_vs_MPE_{n_steps}_steps_{device_name.lower().replace(' ','_')}"
    figure_name_pkl = figure_name + f"_range_{low}_{high}_num_{num}"

    evaluation = load_pickled_evaluation(figure_name_pkl)
    if evaluation is None:
        for n_envs in list_n_envs:
            mpe_times.append(run_mpe_simple_spread(n_envs=n_envs, n_steps=n_steps))
            vmas_times.append(
                run_vmas_simple_spread(
                    n_envs=n_envs, n_steps=n_steps, device=vmas_device
                )
            )
        store_pickled_evaluation(
            name=figure_name_pkl, evaluation=[mpe_times, vmas_times]
        )
    else:
        mpe_times = evaluation[0]
        vmas_times = evaluation[1]

    fig, ax = plt.subplots()
    ax.plot(
        list_n_envs,
        mpe_times,
        label="MPE",
    )
    ax.plot(
        list_n_envs,
        vmas_times,
        label="VMAS",
    )
    plt.xlabel("Number of parallel environments", fontsize=14)
    plt.ylabel("Seconds", fontsize=14)
    ax.legend(loc="upper left")

    fig.suptitle("VMAS vs MPE", fontsize=16)
    ax.set_title(
        f"Execution time of 'simple_spread' for {n_steps} steps on {device_name}",
        fontsize=8,
    )

    save_folder = os.path.dirname(os.path.realpath(__file__))
    tikzplotlib.clean_figure()
    tikzplotlib.save(
        f"{save_folder}/vmas_vs_mpe_graphs/{figure_name}.tex",
    )
    plt.savefig(f"{save_folder}/vmas_vs_mpe_graphs/{figure_name}.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a time comparison between VMAS and MPE"
    )

    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Use cuda device for VMAS",
    )

    args = parser.parse_args()

    run_comparison(vmas_device="cuda" if args.cuda else "cpu")
