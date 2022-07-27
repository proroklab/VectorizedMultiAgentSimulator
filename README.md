# VectorizedMultiAgentSimulator (VMAS)

<p align="center">
<img src="https://github.com/matteobettini/vmas-media/blob/main/media/VMAS_scenarios.gif?raw=true" alt="drawing"/>  
</p>

### Welcome to **VMAS**!

This repository contains the code for the Vectorized Multi-Agent Simulator (VMAS).

VMAS is a vectorized framework designed for efficient MARL benchmarking.
It is comprised of a vectorized 2D physics engine written in PyTorch and a set of challenging multi-robot scenarios.
Scenario creation is made simple and modular to incentivize contributions.
VMAS simulates agents and landmarks of different shapes and supports rotations, elastic collisions and custom gravity.
Holonomic motion models are used for the agents to simplify simulation. Custom sensors such as LIDARs are available and the simulator supports inter-agent communication.
Vectorization in [PyTorch](https://pytorch.org/) allows VMAPS to perform simulations in a batch, seamlessly scaling to tens of thousands of parallel environments on accelerated hardware.
VMAS has an interface compatible with [OpenAI Gym](https://github.com/openai/gym) and with the [RLlib](https://docs.ray.io/en/latest/rllib/index.html) library, enabling out-of-the-box integration with a wide range of RL algorithms. 
The implementation is inspired by [OpenAI's MPE](https://github.com/openai/multiagent-particle-envs). 
Alongside VMAS's scenarios, we port and vectorize all the scenarios in MPE.

### [Paper](https://arxiv.org/abs/2207.03530)
The arXiv paper can be found [here](https://arxiv.org/abs/2207.03530).

If you use VMAS in your research, **cite** it using:
```
@article{bettini2022vmas
  title = {VMAS: A Vectorized Multi-Agent Simulator for Collective Robot Learning},
  author = {Bettini, Matteo and Kortvelesy, Ryan and Blumenkamp, Jan and Prorok, Amanda},
  year = {2022},
  journal={arXiv preprint arXiv:2207.03530},
  url = {https://arxiv.org/abs/2207.03530}
}
```

### [Video](https://youtu.be/aaDRYfiesAY)
Watch the presentation video of VMAS, showing its structure, scenarios, and experiments.

<p align="center">

[![VMAS Video](https://img.youtube.com/vi/aaDRYfiesAY/0.jpg)](https://www.youtube.com/watch?v=aaDRYfiesAY)
</p>

## How to use
### Notebooks
-  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/proroklab/VectorizedMultiAgentSimulator/blob/main/notebooks/VMAS_Use_vmas_environment.ipynb) &ensp; **Using a VMAS environment**.
 Here is a simple notebook that you can run to create, step and render any scenario in VMAS. It reproduces the `use_vmas_env.py` script in the `examples` folder.
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/proroklab/VectorizedMultiAgentSimulator/blob/main/notebooks/VMAS_RLlib.ipynb)  &ensp;  **Using VMAS in RLlib**.  In this notebook, we show how to use any VMAS scenario in RLlib. It reproduces the `rllib.py` script in the `examples` folder.



### Install

To install the simulator, simply install the requirements using:
```
pip install -r requirements.txt
```
rllib dependencies are outdated, so then run:
```
pip install gym==0.22
```
and then install the package with:
```
pip install -e .
```

### Run 

To use the simulator, simply create an environment by passing the name of the scenario
you want (from the `scenarios` folder) to the `make_env` function.
The function arguments are explained in the documentation. The function returns an environment
object with the OpenAI gym interface:

Here is an example:
```
 env = vmas.make_env(
        scenario_name="waterfall",
        num_envs=32,
        device="cpu", # Or "cuda" for GPU
        continuous_actions=True,
        wrapper=None,  # One of: None, vmas.Wrapper.RLLIB, and vmas.Wrapper.GYM
        **kwargs # Additional arguments you want to pass to the scenario initialization
    )
```
A further example that you can run is contained in `use_vmas_env.py` in the `examples` directory.

#### RLlib

To see how to use VMAS in RLlib, check out the script in `examples/rllib.py`.

## Simulator features

- **Vectorized**: VMAS vectorization can step any number of environments in parallel. This significantly reduces the time needed to collect rollouts for training in MARL.
- **Simple**: Complex vectorized physics engines exist (e.g., Brax~\cite{brax2021github}), but they do not scale efficiently when dealing with multiple agents. This defeats the computational speed goal set by vectorization. VMAS uses a simple custom 2D dynamics engine written in PyTorch to provide fast simulation. 
- **General**: The core of VMAS is structured so that it can be used to implement general high-level multi-robot problems in 2D. It can support adversarial as well as cooperative scenarios. Holonomic point-robot simulation has been chosen to focus on general high-level problems, without learning low-level custom robot controls through MARL.
- **Extensible**: VMAS is not just a simulator with a set of environments. It is a framework that can be used to create new multi-agent scenarios in a format that is usable by the whole MARL community. For this purpose, we have modularized the process of creating a task and introduced interactive rendering to debug it. You can define your own scenario in minutes. Have a look at the dedicated section in this document.
- **Compatible**: VMAS has wrappers for [RLlib](https://docs.ray.io/en/latest/rllib/index.html) and [OpenAI Gym](https://github.com/openai/gym). RLlib has a large number of already implemented RL algorithms.
Keep in mind that this interface is less efficient than the unwrapped version. For an example of wrapping, see the main of `make_env`.
- **Entity shapes**: Our entities (agent and landmarks) can have different customizable shapes (spheres, boxes, lines).
All these shapes are supported for elastic collisions.
- **Faster than physics engines**: Our simulator is extremely lightweight, using only tensor operations. It is perfect for 
running MARL training at scale with multi-agent collisions and interactions.
- **Customizable**: When creating a new scenario of your own, the world, agent and landmarks are highly
customizable. Examples are: drag, friction, gravity, simulation timestep, non-differentiable communication, agent sensors (e.g. LIDAR), and masses.
- **Non-differentiable communication**: Scenarios can require agents to perform discrete or continuous communication actions.
- **Gravity**: VMAS supports customizable gravity.
- **Sensors**: Our simulator implements ray casting, which can be used to simulate a wide range of distance-based sensors that can be added to agents. We currently support LIDARs. To see available sensors, have a look at the `sensors` script.
- **Joints**: Our simulator supports joints. Joints are constraints that keep entities at a specified distance. The user can specify the anchor points on the two objects, the distance (including 0), the thickness of the joint, if the joint is allowed to rotate at either anchor point, and if he wants the joint to be collidable. Have a look at the waterfall scenario to see how you can use joints.

## Creating a new scenario

To create a new scenario, just extend the `BaseScenario` class in `scenario`.

You will need to implement at least `make_world`, `reset_world_at`, `observation`, and `reward`. Optionally, you can also implement `done`, `info`, and `extra_render`.

To know how, just read the documentation of `BaseScenario` and look at the implemented scenarios. 

## Play a scenario

You can play with a scenario interactively!

Just use the `render_interactively` script. Relevant values will be plotted to screen.
Move the agent with the arrow keys and switch agents with TAB. You can reset the environment by pressing R.
If you have more than 1 agent, you can control another one with W,A,S,D and switch the second agent using LSHIFT.

On the screen you will see some data from the agent controlled with arrow keys. This data includes: name, current obs, 
current reward, total reward so far and environment done flag.

Here is an overview of what it looks like:

<p align="center">
<img src="https://github.com/matteobettini/vmas-media/blob/main/media/interactive.png?raw=true"  alt="drawing" width="500"/>
</p>

## Rendering

To render the environment, just call the `render` or the `try_render_at` functions (depending on environment wrapping).

Example:
```
env.render(
    mode="rgb_array", # "rgb_array" returns image, "human" renders in display
    agent_index_focus=4, # If None keep all agents in camera, else focus camera on specific agent
    index=0, # Index of batched environment to render
    visualize_when_rgb: bool = False, # Also run human visualization when mode=="rgb_array"
)
```

|                                                                    Gif                                                                    |                             Agent focus                             |
|:-----------------------------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------:|
|        <img src="https://github.com/matteobettini/vmas-media/blob/main/media/vmas_simple.gif?raw=true" alt="drawing" width="260"/>        | With ` agent_index_focus=None` the camera keeps focus on all agents |
| <img src="https://github.com/matteobettini/vmas-media/blob/main/media/vmas_simple_focus_agent_0.gif?raw=true" alt="drawing" width="260"/> |       With ` agent_index_focus=0` the camera follows agent 0        |
| <img src="https://github.com/matteobettini/vmas-media/blob/main/media/vmas_simple_focus_agent_4.gif?raw=true" alt="drawing" width="260"/> |       With ` agent_index_focus=4` the camera follows agent 4        |

### Rendering on server machines
To render in machines without a display use `mode="rgb_array"`. Make sure you have OpenGL and Pyglet installed.
To enable rendering on headless machines you should install EGL.
If you do not have EGL, you need to create a fake screen. You can do this by running these commands before the script: 
```
export DISPLAY=':99.0'
Xvfb :99 -screen 0 1400x900x24 > /dev/null 2>&1 &
```
or in this way:
```
xvfb-run -s \"-screen 0 1400x900x24\" python <your_script.py>
```
To create a fake screen you need to have `Xvfb` installed.

## List of environments
### VMAS
|                                                                                                                                                     |                                                                                                                                                   |                                                                                                                                                                           |
|-----------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **<p align="center">dropout</p>** <br/> <img src="https://github.com/matteobettini/vmas-media/blob/main/media/scenarios/dropout.gif?raw=true"/>     | **<p align="center">football</p>** <br/> <img src="https://github.com/matteobettini/vmas-media/blob/main/media/scenarios/football.gif?raw=true"/> | **<p align="center">transport</p>** <br/> <img src="https://github.com/matteobettini/vmas-media/blob/main/media/scenarios/transport.gif?raw=true"/>                       |
| **<p align="center">wheel</p>** <br/> <img src="https://github.com/matteobettini/vmas-media/blob/main/media/scenarios/wheel.gif?raw=true"/>         | **<p align="center">balance</p>** <br/> <img src="https://github.com/matteobettini/vmas-media/blob/main/media/scenarios/balance.gif?raw=true"/>   | **<p align="center">reverse <br/> transport</p>** <br/> <img src="https://github.com/matteobettini/vmas-media/blob/main/media/scenarios/reverse_transport.gif?raw=true"/> |
| **<p align="center">give_way</p>** <br/> <img src="https://github.com/matteobettini/vmas-media/blob/main/media/scenarios/give_way.gif?raw=true"/>   | **<p align="center">passage</p>** <br/> <img src="https://github.com/matteobettini/vmas-media/blob/main/media/scenarios/passage.gif?raw=true"/>   | **<p align="center">dispersion</p>** <br/> <img src="https://github.com/matteobettini/vmas-media/blob/main/media/scenarios/dispersion.gif?raw=true"/>                     |
| **<p align="center">waterfall</p>** <br/> <img src="https://github.com/matteobettini/vmas-media/blob/main/media/scenarios/waterfall.gif?raw=true"/> | **<p align="center">flocking</p>** <br/> <img src="https://github.com/matteobettini/vmas-media/blob/main/media/scenarios/flocking.gif?raw=true"/> | **<p align="center">discovery</p>** <br/> <img src="https://github.com/matteobettini/vmas-media/blob/main/media/scenarios/discovery.gif?raw=true"/>                       |


| Env name               | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | GIF                                                                                                                                         |
|------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| `waterfall.py`         | Debug environment. `n_agents` agents are spawned in the top of the environment. Each agent is rewarded based on how close it is to the center of the black line at the bottom. Agents have to reach the line and in doing so they might collide with each other and with boxes in the environment.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | <img src="https://github.com/matteobettini/vmas-media/blob/main/media/scenarios/waterfall.gif?raw=true" alt="drawing" width="300"/>         |
| `dropout.py`           | In this scenario, `n_agents` and a goal are spawned at random positions between -1 and 1. Agents cannot collide with each other and with the goal. The reward is shared among all agents. The team receives a reward of 1 when at least one agent reaches the goal. A penalty is given to the team proportional to the sum of the magnitude of actions of every agent. This penalises agents for moving. The impact of the energy reward can be tuned by setting `energy_coeff`. The default coefficient is 0.02 makes it so that for one agent it is always worth reaching the goal. The optimal policy consists in agents sending only the closest agent to the goal and thus saving as much energy as possible. Every agent observes its position, velocity, relative position to the goal and a flag that is set when someone reaches the goal. The environment terminates when when someone reaches the goal. To solve this environment, communication is needed.                                                                                                                                                                                                                                                | <img src="https://github.com/matteobettini/vmas-media/blob/main/media/scenarios/dropout.gif?raw=true" alt="drawing" width="300"/>           |
| `dispersion.py`        | In this scenario, `n_agents` agents and goals are spawned. All agents spawn in [0,0] and goals spawn at random positions between -1 and 1.   Agents cannot collide with each other and with the goals. Agents are tasked with reaching the goals. When a goal is reached, the team gets a reward of 1 if `share_reward` is true, otherwise the agents which reach that goal in the same step split the reward of 1. If `penalise_by_time` is true, every agent gets an additional reward of -0.01 at each step. The optimal policy is for agents to disperse and each tackle a different goal. This requires high coordination and diversity. Every agent observes its position and velocity. For every goal it also observes the relative position and a flag indicating if the goal has been already reached by someone or not. The environment terminates when all the goals are reached.                                                                                                                                                                                                                                                                                                                          | <img src="https://github.com/matteobettini/vmas-media/blob/main/media/scenarios/dispersion.gif?raw=true" alt="drawing" width="300"/>        |
| `transport.py`         | In this scenario, `n_agents`, `n_packages` (default 1) and a goal are spawned at random positions between -1 and 1. Packages are boxes with `package_mass` mass (default 50 times agent mass) and `package_width` and `package_length` as sizes.  The goal is for agents to push all packages to the goal. When all packages overlap with the goal, the scenario ends. Each agent receives the same reward which is proportional to the sum of the distance variations between the packages and the goal. In other words, pushing a package towards the goal will give a positive reward, while pushing it away, a negative one. Once a package overlaps with the goal, it becomes green and its contribution to the reward becomes 0. Each agent observes its position, velocity, relative position to packages, package velocities, relative positions between packages and the goal and a flag for each package indicating if it is on the goal. By default packages are very heavy and one agent is barely able to push them. Agents need to collaborate and push packages together to be able to move them faster.                                                                                               | <img src="https://github.com/matteobettini/vmas-media/blob/main/media/scenarios/transport.gif?raw=true" alt="drawing" width="300"/>         |
| `reverse_transport.py` | This is exactly the same of transport except with `n_agents` spawned inside a single package. All the rest is the same.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | <img src="https://github.com/matteobettini/vmas-media/blob/main/media/scenarios/reverse_transport.gif?raw=true" alt="drawing" width="300"/> |
| `give_way.py`          | In this scenario, two agents and two goals are spawned in a narrow corridor. The agents need to reach the goal with their color. The agents are standing in front of each other's goal and thus need to swap places. In the middle of the corridor there is an asymmetric opening which fits one agent only. Therefore the optimal policy is for one agent to give way to the other. This requires heterogeneous behaviour. If `shared_reward` is true, each agent gets a reward of one when someone reaches a goal, otherwise each agent gets a reward of 1 for reaching its goal. If `dense_reward` is True, the agents get dense rewards instead of sparse ones. Each agent observes its position, velocity and the relative position to its goal. The scenario terminates when both agents reach their goals.                                                                                                                                                                                                                                                                                                                                                                                                     | <img src="https://github.com/matteobettini/vmas-media/blob/main/media/scenarios/give_way.gif?raw=true" alt="drawing" width="300"/>          |
| `wheel.py`             | In this scenario, `n_agents` are spawned at random positions between -1 and 1. One line with `line_length` and `line_mass` is spawned in the middle. The line is constrained in the origin and can rotate. The goal of the agents is to make the absolute angular velocity of the line match `desired_velocity`. Therefore, it is not sufficient for the agents to all push in the extrema of the line, but they need to organize to achieve, and not exceed, the desired velocity. Each agent observes its position, velocity, the current angle of the line module pi, the absolute difference between the current angular velocity of the line and the desired one, and the relative position to the two line extrema. The reward is shared and it is the absolute difference between the current angular velocity of the line and the desired one.                                                                                                                                                                                                                                                                                                                                                                | <img src="https://github.com/matteobettini/vmas-media/blob/main/media/scenarios/wheel.gif?raw=true" alt="drawing" width="300"/>             |  
| `balance.py`           | In this scenario, `n_agents` are spawned uniformly spaced out under a line upon which lies a spherical package of mass `package_mass`. The team and the line are spawned at a random X position at the bottom of the environment. The environment has vertical gravity. If `random_package_pos_on_line` is True (default), the relative X position of the package on the line is random. In the top half of the environment a goal is spawned. The agents have to carry the package to the goal.  Each agent receives the same reward which is proportional to the distance variation between the package and the goal. In other words, getting the package closer to the goal will give a positive reward, while moving it away, a negative one. The team receives a negative reward of -10 for making the package or the line fall to the floor. The observations for each agent are: its position, velocity, relative position to the package, relative position to the line, relative position between package and goal, package velocity, line velocity, line angular velocity, and line rotation mod pi. The environment is done either when the package or the line fall or when the package touches the goal. | <img src="https://github.com/matteobettini/vmas-media/blob/main/media/scenarios/balance.gif?raw=true" alt="drawing" width="300"/>           |  
| `passage.py`           | In this scenario, a team of 5 robots is spawned in formation at a random location in the bottom part of the environment. A simular formation of goals is spawned at random in the top part. Each robot has to reach its corresponding goal. In the middle of the environment there is a wall with `n_passages`. Each passage is large enough to fit one robot at a time. Each agent receives a reward which is proportional to the distance variation between itself and the goal. In other words, getting closer to the goal will give a positive reward, while moving it away, a negative one. This reward will be shared in case `shared_reward` is true. If collisions among robots occur, each robot involved will get a reward of -10. Each agent observes: its position, velocity, relative position to the goal and relative position to the center of each passage. The environment terminates when all the robots reach their goal.                                                                                                                                                                                                                                                                         | <img src="https://github.com/matteobettini/vmas-media/blob/main/media/scenarios/passage.gif?raw=true" alt="drawing" width="300"/>           | 
| `football.py`          | In this scenario, a team of `n_blue_agents` play football against a team of `n_red_agents`. The boolean parameters `ai_blue_agents` and `ai_red_agents` specify whether each team is controlled by action inputs or a programmed AI. Consequently, football can be treated as either a cooperative or competitive task. The reward in this scenario can be tuned with `dense_reward_ratio`, where a value of 0 denotes a fully sparse reward (1 for a goal scored, -1 for a goal conceded), and 1 denotes a fully dense reward (based on the the difference of the "attacking value" of each team, which considers the distance from the ball to the goal and the presence of open dribbling/shooting lanes to the goal). Every agent observes its position, velocity, relative position to the ball, and relative velocity to the ball. The episode terminates when one team scores a goal.                                                                                                                                                                                                                                                                                                                          | <img src="https://github.com/matteobettini/vmas-media/blob/main/media/scenarios/football.gif?raw=true" alt="drawing" width="300"/>          | 
| `discovery.py`         | In this scenario, a team of `n_agents` has to coordinate to cover `n_targets` targets as quickly as possible while avoiding collisions. A target is considered covered if `agents_per_target` agents have approached a target at a distance of at least `covering_range`. After a target is covered, the `agents_per_target` each receive a reward and the target is respawned to a new random position. Agents receive a penalty if they collide with each other. Every agent observes its position, velocity, LIDAR range measurements to other agents and targets (independently). The episode terminates after a fixed number of time steps.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | <img src="https://github.com/matteobettini/vmas-media/blob/main/media/scenarios/discovery.gif?raw=true" alt="drawing" width="300"/>         | 
| `flocking.py`          | In this scenario, a team of `n_agents` has to flock around a target while staying together and maximising their velocity without colliding with each other and a number of `n_obstacles` obstacles. Agents are penalized for colliding with each other and with obstacles, and are rewarded for maximising velocity and minimising the span of the flock (cohesion). Every agent observes its position, velocity, and LIDAR range measurements to other agents. The episode terminates after a fixed number of time steps.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | <img src="https://github.com/matteobettini/vmas-media/blob/main/media/scenarios/flocking.gif?raw=true" alt="drawing" width="300"/>          |



### [MPE](https://github.com/openai/multiagent-particle-envs)

| Env name in code (name in paper)                         | Communication? | Competitive? | Notes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
|----------------------------------------------------------|----------------|--------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `simple.py`                                              | N              | N            | Single agent sees landmark position, rewarded based on how close it gets to landmark. Not a multi-agent environment -- used for debugging policies.                                                                                                                                                                                                                                                                                                                                                                                                               |
| `simple_adversary.py` (Physical deception)               | N              | Y            | 1 adversary (red), N good agents (green), N landmarks (usually N=2). All agents observe position of landmarks and other agents. One landmark is the ‘target landmark’ (colored green). Good agents rewarded based on how close one of them is to the target landmark, but negatively rewarded if the adversary is close to target landmark. Adversary is rewarded based on how close it is to the target, but it doesn’t know which landmark is the target landmark. So good agents have to learn to ‘split up’ and cover all landmarks to deceive the adversary. |
| `simple_crypto.py` (Covert communication)                | Y              | Y            | Two good agents (alice and bob), one adversary (eve). Alice must sent a private message to bob over a public channel. Alice and bob are rewarded based on how well bob reconstructs the message, but negatively rewarded if eve can reconstruct the message. Alice and bob have a private key (randomly generated at beginning of each episode), which they must learn to use to encrypt the message.                                                                                                                                                             |
| `simple_push.py` (Keep-away)                             | N              | Y            | 1 agent, 1 adversary, 1 landmark. Agent is rewarded based on distance to landmark. Adversary is rewarded if it is close to the landmark, and if the agent is far from the landmark. So the adversary learns to push agent away from the landmark.                                                                                                                                                                                                                                                                                                                 |
| `simple_reference.py`                                    | Y              | N            | 2 agents, 3 landmarks of different colors. Each agent wants to get to their target landmark, which is known only by other agent. Reward is collective. So agents have to learn to communicate the goal of the other agent, and navigate to their landmark. This is the same as the simple_speaker_listener scenario where both agents are simultaneous speakers and listeners.                                                                                                                                                                                    |
| `simple_speaker_listener.py` (Cooperative communication) | Y              | N            | Same as simple_reference, except one agent is the ‘speaker’ (gray) that does not move (observes goal of other agent), and other agent is the listener (cannot speak, but must navigate to correct landmark).                                                                                                                                                                                                                                                                                                                                                      |
| `simple_spread.py` (Cooperative navigation)              | N              | N            | N agents, N landmarks. Agents are rewarded based on how far any agent is from each landmark. Agents are penalized if they collide with other agents. So, agents have to learn to cover all the landmarks while avoiding collisions.                                                                                                                                                                                                                                                                                                                               |
| `simple_tag.py` (Predator-prey)                          | N              | Y            | Predator-prey environment. Good agents (green) are faster and want to avoid being hit by adversaries (red). Adversaries are slower and want to hit good agents. Obstacles (large black circles) block the way.                                                                                                                                                                                                                                                                                                                                                    |
| `simple_world_comm.py`                                   | Y              | Y            | Environment seen in the video accompanying the paper. Same as simple_tag, except (1) there is food (small blue balls) that the good agents are rewarded for being near, (2) we now have ‘forests’ that hide agents inside from being seen from outside; (3) there is a ‘leader adversary” that can see the agents at all times, and can communicate with the other adversaries to help coordinate the chase.                                                                                                                                                      |

## TODOS

- [ ] Custom actions for scenario
- [X] Link video of experiments
- [X] Add LIDAR section
- [X] Implement LIDAR
- [ ] Implement 1D camera sensor
- [ ] Make football heuristic efficient
- [X] Rewrite all MPE scenarios
  - [X] simple
  - [x] simple_adversary
  - [X] simple_crypto
  - [X] simple_push
  - [X] simple_reference
  - [X] simple_speaker_listener
  - [X] simple_spread
  - [X] simple_tag
  - [X] simple_world_comm
