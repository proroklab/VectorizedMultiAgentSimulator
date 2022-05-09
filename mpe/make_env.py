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
import numpy as np
import torch
from matplotlib import pyplot as plt


def make_env(scenario_name, benchmark=False):
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
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(
            world,
            scenario.reset_world,
            scenario.reward,
            scenario.observation,
            scenario.benchmark_data,
        )
    else:
        env = MultiAgentEnv(
            world, scenario.reset_world, scenario.reward, scenario.observation
        )
    return env


if __name__ == "__main__":
    env = make_env("simple")

    n_steps = 800
    fig, ax = plt.subplots()
    x = np.arange(n_steps)
    agent_actions = []
    agent_action_forces = []
    agent_coll_forces = []
    landm_coll_forces = []
    landm_tot_forces = []
    agent_tot_forces = []
    distance = []
    landm_vels = []

    obs = env.reset()[0]
    done = False
    for _ in range(n_steps):
        # print(f"Observation: {obs}")
        action = torch.tensor([0.0, 0.02]).repeat(32, 1)
        agent_actions.append(action[1])
        obss, rews, dones, info = env.step([action])
        #
        # agent_action_forces.append(env.world.entities[0].f_act[1])
        # agent_coll_forces.append(env.world.entities[0].f_coll[1])
        # landm_coll_forces.append(env.world.entities[1].f_coll[1])
        # landm_tot_forces.append(env.world.entities[1].f_tot[1])
        # agent_tot_forces.append(env.world.entities[0].f_tot[1])
        # distance.append(
        #     env.world.entities[0].state.p_pos[1] - env.world.entities[1].state.p_pos[1]
        # )
        # landm_vels.append(env.world.entities[1].state.p_vel[1])

        obs = obss[0]
        rew = rews[0]
        done = dones[0]

        env.render()
    # ax.plot(x, agent_actions, label="Agent action")
    ax.plot(x, agent_action_forces, label="Agent action f")
    ax.plot(x, agent_coll_forces, label="Agent coll f")
    ax.plot(x, landm_coll_forces, label="Land call f")
    # ax.plot(x, landm_tot_forces, label="Land tot f")
    ax.plot(x, agent_tot_forces, label="Agent tot f")
    ax.plot(x, distance, label="Distance")
    ax.plot(x, landm_vels, label="Landm vel")
    plt.legend()
    plt.show()
