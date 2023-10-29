'''
This is a proof of concept expert.
The goal of this is to act as a template on how to create the real expert once the policy architecture is created
This is not meant to be the final code that we are using
'''

import argparse
import torch
from torch.nn import ModuleDict
import time
import os
import xml.etree.ElementTree as ET
from PIL import Image
from vmas import make_env
import numpy as np

stepsPerTimeInterval = 30 #This still assumes constant velocity... Needs to be large enough so agents actually reach their goal before moving on

def load_map(map_file, device):
    #Load in map
    f = os.path.dirname(__file__)
    map = ET.parse(f'{f}/{map_file}')
    map_root = map.getroot()

    # Extract grid dimensions
    width = int(map_root.find('.//width').text)
    height = int(map_root.find('.//height').text)

    # Initialize a numpy array to store the grid
    grid = torch.zeros((height, width), dtype=torch.int, device=device)

    # Find all row elements and populate the numpy array
    row_elements = map_root.findall('.//row')
    for i, row_element in enumerate(row_elements):
        row_data = row_element.text.split()
        for j, value in enumerate(row_data):
            grid[height - 1 - i, j] = int(value)
    return grid, width, height

def load_episodes(log_dir, device, width, height):
    f = os.path.dirname(__file__)
    episodes = {}
    for file in os.listdir(f'{f}/{log_dir}/'):
        if file[-3:] != 'xml':
            continue

        #episode name must be a number!
        #Assumes the files are named <what ever you want>_<id>_log.xml
        episode_name = int(file[:-4].split('_')[-2])
        
        agent_paths = {}
        log = ET.parse(f'{f}/{log_dir}/{file}')
        log_root = log.getroot()
        agents = log_root.findall(".//agent[@number]")
        for agent in agents:
            number = int(agent.get('number'))
            waypoints = []
            path = agent.find('.//path')
            sections = path.findall('.//section')
            first = True
            for waypoint in sections:
                if first:
                    y = height - 1 - int(waypoint.get('start_i')) #i is row, j is column...
                    x = int(waypoint.get('start_j'))
                    duration = 0.0
                    waypoints.append(torch.tensor([duration, x, y], device=device))
                    first=False
                y = height - 1 - int(waypoint.get("goal_i"))
                x = int(waypoint.get('goal_j'))
                duration = float(waypoint.get('duration')) + waypoints[-1][0]
                waypoints.append(torch.tensor([duration, x, y], device=device))
            agent_paths[number] = waypoints
        episodes[episode_name] = agent_paths.copy()
    return episodes

def select_action(position, waypoint):   
    #Proof of concept action selector based on agent's position and desired waypoint
    #This can be modified however needed to select better actions

    ''' 
    #For discrete actions
    x_mag = waypoint[0] - position[0]
    y_mag = waypoint[1] - position[1]
    if abs(x_mag) > abs(y_mag):
        if x_mag < 0:
            return [1]
        else:
            return [2]
    else:
        if y_mag < 0:
            return [3]
        else:
            return [4]
    '''

    #For continuous actions
    action = []
    if (waypoint[0] - position[0]) > .1:
        action.append(1)
    elif (waypoint[0] - position[0]) < -.1:
        action.append(-1)
    else:
        action.append(5 * (waypoint[0] - position[0]))


    if (waypoint[1] - position[1]) > .1:
        action.append(1)
    elif (waypoint[1] - position[1]) < -.1:
        action.append(-1)
    else:
        action.append(5 * (waypoint[1] - position[1]))

    return action

def get_agent_action(episodes, obs, steps):
    scenario_time = steps / stepsPerTimeInterval
    actions = []
    for i, ob in enumerate(obs.values()):
        env_actions = []
        for env_ob in ob:
            episode = episodes[int(env_ob[0])]
            x, y = env_ob[1:3]
            appended = False
            for waypoint in episode[i]:
                if waypoint[0] > scenario_time:
                    env_actions.append(select_action([x,y],waypoint[1:]))
                    appended = True
                    break
            if not appended:
                env_actions.append(select_action([x,y],episode[i][-1][1:]))
        actions.append(env_actions)
        
    return actions


def main(args):
    device = args.device  # or cuda or any other torch device

    #Load in the config to get agent sizes and the time limit
    f = os.path.dirname(__file__)
    config = ET.parse(f'{f}/{args.config}')
    config_root = config.getroot()
    agent_size = float(config_root.find('.//agent_size').text)

    #load map. Currently assumes every task takes place on the same map
    grid_map, width, height = load_map(args.map, device)
    
    #Load in all of the pre generated episodes
    episodes = load_episodes(args.log_dir, device, width, height)

    scenario_name = 'navigation2' #Scenario name

    # Scenario specific variables
    n_agents = len(episodes[next(iter(episodes))].keys()) #Assumes all episodes has same number of agents

    num_envs = args.num_envs  # Number of vectorized environments
    continuous_actions = True
    n_steps = args.max_steps # Number of steps before returning done
    dict_spaces = True  # Weather to return obs, rewards, and infos as dictionaries with agent names (by default they are lists of len # of agents)

    start_action = (
        [0, 0] if continuous_actions else [0]
    )  # Simple action to start the program 

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
        episodes = episodes,
        map=grid_map,
        agent_radius = agent_size/2
    )

    frame_list = []  # For creating a gif
    init_time = time.time()
    step = 0
    actions = {} 
    obs = []
    for s in range(n_steps):
        step += 1
        print(f"Step {step}")
        
        if len(obs) == 0:
            agent_actions = [[start_action] * num_envs]* len(env.agents)
        else:
            agent_actions = get_agent_action(episodes, obs, step)

        for i, agent in enumerate(env.agents):
            action = torch.tensor(
                agent_actions[i],
                device=device,
            )
            actions.update({agent.name: action})

        obs, rews, dones, info = env.step(actions)

        frame_list.append(
            Image.fromarray(env.render(mode="rgb_array", agent_index_focus=None))
        )

        if dones.all():
            break

    total_time = time.time() - init_time
    print(
        f"It took: {total_time}s for {n_steps} steps of {num_envs} parallel environments on device {device} "
        f"for {scenario_name} scenario."
    )

    gif_name = scenario_name + ".gif"
    # Produce a gif
    frame_list[0].save(
        gif_name,
        save_all=True,
        append_images=frame_list[1:],
        duration=3,
        loop=0,
    )


def create_parser():
    parser = argparse.ArgumentParser(description='Use the expert to solve the navigation2 scenario')
    parser.add_argument('--map', '-m', default='grid_map.xml', type=str, help='The map to generate tasks for. Only supports grid type maps')
    parser.add_argument('--config', '-c', default='config.xml', help='The config file for the task')
    parser.add_argument('--log_dir', '-l', default='logs', help='local path to logs directory')
    parser.add_argument('--device', '-d', default = 'cpu', help='device to run on')
    parser.add_argument('--num_envs', type=int, default = 10, help='number of environments to run at once')
    parser.add_argument('--max_steps', type=int, default = 500, help='Max number of steps to run for')
    return parser

if __name__ == "__main__":
    args = create_parser().parse_args()
    main(args)