Running
=======

To use the simulator, simply create an environment by passing the name of the scenario
you want (from the ``scenarios`` folder) to the :class:`vmas.make_env` function.
The function arguments are explained in the documentation. The function returns an environment
object which you can step and reset.

.. code-block:: python

    import vmas

    # Create the environment
    env = vmas.make_env(
        scenario="waterfall", # can be scenario name or BaseScenario class
        num_envs=32,
        device="cpu", # Or "cuda" for GPU
        continuous_actions=True,
        max_steps=None, # Defines the horizon. None is infinite horizon.
        seed=None, # Seed of the environment
        n_agents=3  # Additional arguments you want to pass to the scenario
    )
    # Reset it
    obs = env.reset()

    # Step it with deterministic actions (all agents take their maximum range action)
    for _ in range(10):
        obs, rews, dones, info = env.step(env.get_random_actions())

Here is a python example on how you can execute vmas environments.

.. python_example_button::
   https://github.com/proroklab/VectorizedMultiAgentSimulator/blob/main/vmas/examples/use_vmas_env.py

The `Concepts` documentation contains a series of sections that
can help you get familiar with further VMAS functionalities.
