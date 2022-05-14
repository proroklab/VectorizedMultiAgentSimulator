# MultiAgentParticleSimulator (MAPS)
 
Welcome to MAPS!

This repository cintains the code for the Multi Agent Prticle Simulator (MAPS)

 <img src="media/maps_simple_2.gif" align="right" alt="drawing" width="200"/>  

MAPS is a vecotrized simulator designed for simulating agents and entities in a 2D particle world.
Its primary use is Multi Agent Reinforcement Learning (MARL) as it provides OpenAI gym interfaces
for all scenarios. It simulates basic body dynamics and elastic collisions. The implementation 
is written in PyTorch to provide environment vectorization (multiple environments can be stepped parallely 
in a batch). The implmentation is based on [OpenAI's MPE](https://github.com/openai/multiagent-particle-envs). 
We support all the scenarios in MPE. Additionally, we created new MARL robotics benchmarking scenarios.
With respect to MPE, we also introduced environment vectorization and new simulator feactures like
additional shaoes for entities (boxes, lines) and related collision rules.

## How to use

To use the simulator, simply create an environment by passing the name of the scenario
you want (from the `scenarios` folder) to the `make_env` function in the omonimous file.
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
        **kwargs # Additional arguments you want to pass to the scenarion initialization
    )
```
A further example that you can run is available in the `make_env` main function.

## Simulator features

- **Vectorization**: the simulator is vectorized and uses torch tensor operations to step environments parallely
- **Rllib compatible**: A wrapper is already available in `environment` to use the scenarios in [rllib](https://docs.ray.io/en/latest/rllib/index.html) as an rllib.VectorEnv object.
- **Entity shapes**: Our entities (agent and landmarks) can have different customizable shapes (spheres, boxes, lines).
These shapes are supported for elastic collisions. For details on which collidable pairs are supported look at the `core`.
- **Faster than physics engines**: Our simulator is extremely light weight, using only tensor operations. It is perfect for 
running MARL training at scale with multi-agent collisions and interactions.
- **Customizable**: When creating a new scenario of your own, the world, agent and landmarks are highly
customizable. Examples are: _world damping, simulation timestep, non-differentiable communication, agent sensors, masses and densities_.
- **Easy to extend**: You can define your own scenario in minutes. Have a look at the dedicated section in this document.

## Creating a new scenario

To create a new scenario, just extend the `BaseScenario` class in `scenario`.

You will need to implement at least `make_world`, `reset_world_at`, `observation`, and `reward`. Optionally, you can also implement `done` and `info`.

To know how, just read the documentation of `BaseScenario` and look at the implmented scenarios. 

## Rnedering

To render the environment, just call the `render` or the `try_render_at` functions (depending on environment wrapping).

Example:
```
env.try_render_at(
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

## TODOS

- [ ] Implement resampling of random position until valid
- [ ] Rewrite all MPE scenarios
- [ ] Implement new custom MPE environments