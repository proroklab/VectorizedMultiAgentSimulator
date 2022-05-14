# MultiAgentParticleSimulator (MAPS)
 
Welcome to **MAPS**!

This repository contains the code for the Multi Agent Particle Simulator (MAPS).

<img src="media/maps_simple_2.gif" align="right" alt="drawing" width="300"/>  

MAPS is a vectorized simulator designed for simulating agents and entities in a 2D particle world.
Its primary use is Multi Agent Reinforcement Learning (MARL) as it provides OpenAI gym interfaces
for all scenarios. It simulates basic body dynamics and elastic collisions. The implementation 
is written in PyTorch to provide environment vectorization (multiple environments can be stepped parallely 
in a batch). The implementation is based on [OpenAI's MPE](https://github.com/openai/multiagent-particle-envs). 
We support all the scenarios in MPE. Additionally, we created new MARL robotics benchmarking scenarios.
With respect to MPE, we also introduced environment vectorization and new simulator features like
additional shapes for entities (boxes, lines) and related collision rules.

## How to use

To use the simulator, simply create an environment by passing the name of the scenario
you want (from the `scenarios` folder) to the `make_env` function.
The function arguments are explained in the documentation. The function returns an environment
object with the OpenAI gym interface:

Here is an example:
```
 env = make_env(
        scenario_name="simple",
        num_envs=32,
        device="cuda",
        continuous_actions=True,
        rllib_wrapped=False,
        **kwargs # Additional arguments you want to pass to the scenario initialization
    )
```
A further example that you can run is available in the `make_env` main function.

## Simulator features

- **Vectorization**: the simulator is vectorized and uses torch tensor operations to step environments parallely
- **Rllib compatible**: A wrapper is already available in `environment` to use the scenarios in [rllib](https://docs.ray.io/en/latest/rllib/index.html) as a rllib.VectorEnv object.
Keep in mind that this interface is less efficient than the unwrapped version. For an example of wrapping, see the main of `make_env`.
- **Entity shapes**: Our entities (agent and landmarks) can have different customizable shapes (spheres, boxes, lines).
These shapes are supported for elastic collisions. For details on which collidable pairs are supported look at the `core`.
- **Faster than physics engines**: Our simulator is extremely lightweight, using only tensor operations. It is perfect for 
running MARL training at scale with multi-agent collisions and interactions.
- **Customizable**: When creating a new scenario of your own, the world, agent and landmarks are highly
customizable. Examples are: _world damping, simulation timestep, non-differentiable communication, agent sensors, masses and densities_.
- **Easy to extend**: You can define your own scenario in minutes. Have a look at the dedicated section in this document.
- **Non-differentiable communication**: Scenarios can require agents to perform discrete or continuous communication actions.

## Creating a new scenario

To create a new scenario, just extend the `BaseScenario` class in `scenario`.

You will need to implement at least `make_world`, `reset_world_at`, `observation`, and `reward`. Optionally, you can also implement `done` and `info`.

To know how, just read the documentation of `BaseScenario` and look at the implemented scenarios. 

## Rendering

To render the environment, just call the `render` or the `try_render_at` functions (depending on environment wrapping).

Example:
```
env.render(
    mode="rgb_array", # Rgb array returns image, "human" renders in display
    agent_index_focus=4, # If None keep all agents in camera, else focus camera on specific agent
    index=0 # Index of batched environment to render
)
```

|                                    Gif                                     |                             Agent focus                             |
|:--------------------------------------------------------------------------:|:-------------------------------------------------------------------:|
|        <img src="media/maps_simple.gif" alt="drawing" width="260"/>        | With ` agent_index_focus=None` the camera keeps focus on all agents |
| <img src="media/maps_simple_focus_agent_0.gif" alt="drawing" width="260"/> |       With ` agent_index_focus=0` the camera follows agent 0        |
| <img src="media/maps_simple_focus_agent_4.gif" alt="drawing" width="260"/> |       With ` agent_index_focus=4` the camera follows agent 4        |

### Rendering on server machines
To render in machines without a display use `mode=rgb_array` and the following
```
export DISPLAY=':99.0'
Xvfb :99 -screen 0 1400x900x24 > /dev/null 2>&1 &
```

## List of environments
### MAPS
### [MPE](https://github.com/openai/multiagent-particle-envs)
> Note: not all the MPE scenarios are available. For a list of the already implemented ones see TODO section

| Env name in code (name in paper)                         | Communication? | Competitive? | Notes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
|----------------------------------------------------------|----------------|--------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `simple.py`                                              | N              | N            | Single agent sees landmark position, rewarded based on how close it gets to landmark. Not a multiagent environment -- used for debugging policies.                                                                                                                                                                                                                                                                                                                                                                                                                |
| `simple_adversary.py` (Physical deception)               | N              | Y            | 1 adversary (red), N good agents (green), N landmarks (usually N=2). All agents observe position of landmarks and other agents. One landmark is the ‘target landmark’ (colored green). Good agents rewarded based on how close one of them is to the target landmark, but negatively rewarded if the adversary is close to target landmark. Adversary is rewarded based on how close it is to the target, but it doesn’t know which landmark is the target landmark. So good agents have to learn to ‘split up’ and cover all landmarks to deceive the adversary. |
| `simple_crypto.py` (Covert communication)                | Y              | Y            | Two good agents (alice and bob), one adversary (eve). Alice must sent a private message to bob over a public channel. Alice and bob are rewarded based on how well bob reconstructs the message, but negatively rewarded if eve can reconstruct the message. Alice and bob have a private key (randomly generated at beginning of each episode), which they must learn to use to encrypt the message.                                                                                                                                                             |
| `simple_push.py` (Keep-away)                             | N              | Y            | 1 agent, 1 adversary, 1 landmark. Agent is rewarded based on distance to landmark. Adversary is rewarded if it is close to the landmark, and if the agent is far from the landmark. So the adversary learns to push agent away from the landmark.                                                                                                                                                                                                                                                                                                                 |
| `simple_reference.py`                                    | Y              | N            | 2 agents, 3 landmarks of different colors. Each agent wants to get to their target landmark, which is known only by other agent. Reward is collective. So agents have to learn to communicate the goal of the other agent, and navigate to their landmark. This is the same as the simple_speaker_listener scenario where both agents are simultaneous speakers and listeners.                                                                                                                                                                                    |
| `simple_speaker_listener.py` (Cooperative communication) | Y              | N            | Same as simple_reference, except one agent is the ‘speaker’ (gray) that does not move (observes goal of other agent), and other agent is the listener (cannot speak, but must navigate to correct landmark).                                                                                                                                                                                                                                                                                                                                                      |
| `simple_spread.py` (Cooperative navigation)              | N              | N            | N agents, N landmarks. Agents are rewarded based on how far any agent is from each landmark. Agents are penalized if they collide with other agents. So, agents have to learn to cover all the landmarks while avoiding collisions.                                                                                                                                                                                                                                                                                                                               |
| `simple_tag.py` (Predator-prey)                          | N              | Y            | Predator-prey environment. Good agents (green) are faster and want to avoid being hit by adversaries (red). Adversaries are slower and want to hit good agents. Obstacles (large black circles) block the way.                                                                                                                                                                                                                                                                                                                                                    |
| `simple_world_comm.py`                                   | Y              | Y            | Environment seen in the video accompanying the paper. Same as simple_tag, except (1) there is food (small blue balls) that the good agents are rewarded for being near, (2) we now have ‘forests’ that hide agents inside from being seen from outside; (3) there is a ‘leader adversary” that can see the agents at all times, and can communicate with the other adversaries to help coordinate the chase.                                                                                                                                                      |

## TODOS

- [ ] Implement resampling of random position until valid
- [ ] Implement new custom MAPS scenarios
- [ ] Rewrite all MPE scenarios
  - [X] simple
  - [x] simple_adversary
  - [ ] simple_crypto
  - [ ] simple_push
  - [ ] simple_reference
  - [ ] simple_speaker_listener
  - [ ] simple_spread
  - [ ] simple_tag
  - [ ] simple_world_comm
