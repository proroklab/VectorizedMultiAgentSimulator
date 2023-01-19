#  Copyright (c) 2022-2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import random
from ctypes import byref
from typing import List, Tuple, Callable, Optional

import numpy as np
import torch
from gym import spaces
from torch import Tensor

import vmas.simulator.utils
from vmas.simulator.core import Agent, TorchVectorizedObject
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import VIEWER_MIN_ZOOM
from vmas.simulator.utils import X, Y, ALPHABET, DEVICE_TYPING, override


# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class Environment(TorchVectorizedObject):
    metadata = {
        "render.modes": ["human", "rgb_array"],
        "runtime.vectorized": True,
    }

    def __init__(
        self,
        scenario: BaseScenario,
        num_envs: int = 32,
        device: DEVICE_TYPING = "cpu",
        max_steps: Optional[int] = None,
        continuous_actions: bool = True,
        seed: Optional[int] = None,
        **kwargs,
    ):

        self.scenario = scenario
        self.num_envs = num_envs
        TorchVectorizedObject.__init__(self, num_envs, torch.device(device))
        self.world = self.scenario.env_make_world(self.num_envs, self.device, **kwargs)

        self.agents = self.world.policy_agents
        self.n_agents = len(self.agents)
        self.max_steps = max_steps
        self.continuous_actions = continuous_actions

        self.reset(seed=seed)

        # configure spaces
        self.action_space = spaces.Tuple(
            [self.get_action_space(agent) for agent in self.agents]
        )
        self.observation_space = spaces.Tuple(
            [
                spaces.Box(
                    low=-np.float32("inf"),
                    high=np.float32("inf"),
                    shape=(len(self.scenario.observation(agent)[0]),),
                    dtype=np.float32,
                )
                for agent in self.agents
            ]
        )

        # rendering
        self.render_geoms_xform = None
        self.render_geoms = None
        self.viewer = None
        self.headless = None
        self.visible_display = None

    def reset(self, seed: Optional[int] = None, return_info: bool = False):
        """
        Resets the environment in a vectorized way
        Returns observations for all envs and agents
        """
        if seed is not None:
            self.seed(seed)
        # reset world
        self.scenario.env_reset_world_at(env_index=None)
        self.steps = torch.zeros(self.num_envs, device=self.device)
        # record observations for each agent
        obs = []
        info = []
        for agent in self.agents:
            obs.append(self.scenario.observation(agent))
            if return_info:
                info.append(self.scenario.info(agent))
        return (obs, info) if return_info else obs

    def reset_at(self, index: int, return_info: bool = False):
        """
        Resets the environment at index
        Returns observations for all agents in that environment
        """
        self._check_batch_index(index)
        self.scenario.env_reset_world_at(index)
        self.steps[index] = 0
        obs = []
        info = []
        for agent in self.agents:
            obs.append(self.scenario.observation(agent)[index].unsqueeze(0))
            if return_info:
                info.append(
                    {
                        key: val[index].unsqueeze(0)
                        for key, val in self.scenario.info(agent).items()
                    }
                )
        return (obs, info) if return_info else obs

    def seed(self, seed=None):
        if seed is None:
            seed = 0
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        return [seed]

    def step(self, actions: List):
        """Performs a vectorized step on all sub environments using `actions`.
        Args:
            actions: Is a list on len 'self.n_agents' of which each element is a torch.Tensor of shape
            '(self.num_envs, action_size_of_agent)'.
        Returns:
            obs: List on len 'self.n_agents' of which each element is a torch.Tensor
                 of shape '(self.num_envs, obs_size_of_agent)'
            rewards: List on len 'self.n_agents' of which each element is a torch.Tensor of shape '(self.num_envs)'
            dones: Tensor of len 'self.num_envs' of which each element is a bool
            infos : List on len 'self.n_agents' of which each element is a dictionary for which each key is a metric
                    and the value is a tensor of shape '(self.num_envs, metric_size_per_agent)'
        """
        assert (
            len(actions) == self.n_agents
        ), f"Expecting actions for {self.n_agents}, got {len(actions)} actions"
        for i in range(len(actions)):
            if not isinstance(actions[i], Tensor):
                actions[i] = torch.tensor(
                    actions[i], dtype=torch.float32, device=self.device
                )
            if len(actions[i].shape) == 1:
                actions[i].unsqueeze_(-1)
            assert (
                actions[i].shape[0] == self.num_envs
            ), f"Actions used in input of env must be of len {self.num_envs}, got {actions[i].shape[0]}"
            assert actions[i].shape[1] == self.get_agent_action_size(self.agents[i]), (
                f"Action for agent {self.agents[i].name} has shape {actions[i].shape[1]},"
                f" but should have shape {self.get_agent_action_size(self.agents[i])}"
            )

        # set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(actions[i], agent)
        # Scenarios can define a custom action processor. This step takes care also of scripted agents automatically
        for agent in self.world.agents:
            self.scenario.env_process_action(agent)

        # advance world state
        self.world.step()

        obs = []
        rewards = []
        infos = []
        for agent in self.agents:
            rewards.append(self.scenario.reward(agent).clone())
            obs.append(self.scenario.observation(agent).clone())
            # A dictionary per agent
            infos.append(self.scenario.info(agent))

        dones = self.scenario.done().clone()

        self.steps += 1
        if self.max_steps is not None:
            dones += self.steps >= self.max_steps
        # print("\nStep results in unwrapped environment")
        # print(
        #     f"Actions len (n_agents): {len(actions)}, "
        #     f"actions[0] shape (num_envs, agent 0 action shape): {actions[0].shape}, "
        #     f"actions[0][0] (action agent 0 env 0): {actions[0][0]}"
        # )
        # print(
        #     f"Obs len (n_agents): {len(obs)}, "
        #     f"obs[0] shape (num_envs, agent 0 obs shape): {obs[0].shape}, obs[0][0] (obs agent 0 env 0): {obs[0][0]}"
        # )
        # print(
        #     f"Rewards len (n_agents): {len(rewards)}, rewards[0] shape (num_envs, 1): {rewards[0].shape}, "
        #     f"rewards[0][0] (agent 0 env 0): {rewards[0][0]}"
        # )
        # print(f"Dones len (n_envs): {len(dones)}, dones[0] (done env 0): {dones[0]}")
        # print(f"Info len (n_agents): {len(infos)}, info[0] (infos agent 0): {infos[0]}")
        return obs, rewards, dones, infos

    def get_agent_action_size(self, agent: Agent):
        return (
            self.world.dim_p
            + (1 if agent.action.u_rot_range != 0 else 0)
            + (self.world.dim_c if not agent.silent else 0)
            if self.continuous_actions
            else 1
            + (1 if agent.action.u_rot_range != 0 else 0)
            + (1 if not agent.silent else 0)
        )

    def get_action_space(self, agent: Agent):
        if self.continuous_actions:
            return spaces.Box(
                low=np.array(
                    [-agent.u_range] * self.world.dim_p
                    + [-agent.u_rot_range] * (1 if agent.u_rot_range != 0 else 0)
                    + [0] * (self.world.dim_c if not agent.silent else 0),
                    dtype=np.float32,
                ),
                high=np.array(
                    [agent.u_range] * self.world.dim_p
                    + [agent.u_rot_range] * (1 if agent.u_rot_range != 0 else 0)
                    + [1] * (self.world.dim_c if not agent.silent else 0),
                    dtype=np.float32,
                ),
                shape=(self.get_agent_action_size(agent),),
                dtype=np.float32,
            )
        else:
            if (self.world.dim_c == 0 or agent.silent) and agent.u_rot_range == 0.0:
                return spaces.Discrete(self.world.dim_p * 2 + 1)
            else:
                actions = (
                    [self.world.dim_p * 2 + 1]
                    + ([3] if agent.u_rot_range != 0 else [])
                    + (
                        [self.world.dim_c]
                        if not agent.silent and self.world.dim_c != 0
                        else []
                    )
                )
                return spaces.MultiDiscrete(actions)

    def _check_discrete_action(self, action: Tensor, low: int, high: int, type: str):
        assert torch.all(
            torch.any(
                torch.arange(low, high, device=self.device).repeat(self.num_envs)
                == action,
                dim=-1,
            )
        ), f"Discrete {type} actions are out of bounds, allowed int range [{low},{high})"

    # set env action for a particular agent
    def _set_action(self, action, agent):
        action = action.clone().to(self.device)
        agent.action.u = torch.zeros(
            self.batch_dim, self.world.dim_p, device=self.device, dtype=torch.float32
        )

        assert action.shape[1] == self.get_agent_action_size(agent), (
            f"Agent {agent.name} has wrong action size, got {action.shape[1]}, "
            f"expected {self.get_agent_action_size(agent)}"
        )
        action_index = 0

        if self.continuous_actions:
            physical_action = action[:, action_index : action_index + self.world.dim_p]
            action_index += self.world.dim_p
            assert not torch.any(
                torch.abs(physical_action) > agent.u_range
            ), f"Physical actions of agent {agent.name} are out of its range {agent.u_range}"

            agent.action.u = physical_action.to(torch.float32)
        else:
            physical_action = action[:, action_index].unsqueeze(-1)
            action_index += 1
            self._check_discrete_action(
                physical_action,
                low=0,
                high=self.world.dim_p * 2 + 1,
                type="physical",
            )

            arr1 = physical_action == 1
            arr2 = physical_action == 2
            arr3 = physical_action == 3
            arr4 = physical_action == 4

            disc_action_value = agent.u_range

            agent.action.u[:, X] -= disc_action_value * arr1.squeeze(-1)
            agent.action.u[:, X] += disc_action_value * arr2.squeeze(-1)
            agent.action.u[:, Y] -= disc_action_value * arr3.squeeze(-1)
            agent.action.u[:, Y] += disc_action_value * arr4.squeeze(-1)

        agent.action.u *= agent.u_multiplier
        if agent.u_rot_range != 0:
            agent.action.u_rot = torch.zeros(
                self.batch_dim, 1, device=self.device, dtype=torch.float32
            )
            if self.continuous_actions:
                physical_action = action[:, action_index]
                action_index += 1
                assert not torch.any(
                    torch.abs(physical_action) > agent.u_rot_range
                ), f"Physical rotation actions of agent {agent.name} are out of its range {agent.u_rot_range}"

                agent.action.u_rot = physical_action.to(torch.float32)
            else:
                physical_action = action[:, action_index].unsqueeze(-1)
                action_index += 1
                self._check_discrete_action(
                    physical_action,
                    low=0,
                    high=3,
                    type="rotation",
                )

                arr1 = physical_action == 1
                arr2 = physical_action == 2

                disc_action_value = agent.u_rot_range

                agent.action.u_rot[:] -= disc_action_value * arr1.squeeze(-1)
                agent.action.u_rot[:] += disc_action_value * arr2.squeeze(-1)

            agent.action.u_rot *= agent.u_rot_multiplier
        if self.world.dim_c > 0 and not agent.silent:
            if not self.continuous_actions:
                comm_action = action[:, action_index:]
                self._check_discrete_action(
                    comm_action, 0, self.world.dim_c, "communication"
                )
                comm_action = comm_action.long()
                agent.action.c = torch.zeros(
                    self.num_envs,
                    self.world.dim_c,
                    device=self.device,
                    dtype=torch.float32,
                )
                # Discrete to one-hot
                agent.action.c.scatter_(1, comm_action, 1)
            else:
                comm_action = action[:, action_index:]
                assert not torch.any(comm_action > 1) and not torch.any(
                    comm_action < 0
                ), "Comm actions are out of range [0,1]"
                agent.action.c = comm_action

    def render(
        self,
        mode="human",
        env_index=0,
        agent_index_focus: int = None,
        visualize_when_rgb: bool = False,
        plot_position_function: Callable[[Tuple[float, float]], float] = None,
        plot_position_function_precision: float = 0.05,
        plot_position_function_range: float = 1,
    ):
        """
        Render function for environment using pyglet

        On servers use mode="rgb_array" and set
        ```
        export DISPLAY=':99.0'
        Xvfb :99 -screen 0 1400x900x24 > /dev/null 2>&1 &
        ```

        :param mode: One of human or rgb_array
        :param env_index: Index of the environment to render
        :param agent_index_focus: If specified the camera will stay on the agent with this index.
                                  If None, the camera will stay in the center and zoom out to contain all agents
        :param visualize_when_rgb: Also run human visualization when mode=="rgb_array"
        :param plot_position_function: A function to plot under the rendering. This function takes
         the position (x,y) as input and outputs a transparency alpha value
        :param plot_position_function_precision: The precision to use for plotting the function
        :param plot_position_function_range: The position range to plot the function in
        :return: Rgb array or None, depending on the mode
        """
        self._check_batch_index(env_index)
        assert (
            mode in self.metadata["render.modes"]
        ), f"Invalid mode {mode} received, allowed modes: {self.metadata['render.modes']}"
        if agent_index_focus is not None:
            assert 0 <= agent_index_focus < self.n_agents, (
                f"Agent focus in rendering should be a valid agent index"
                f" between 0 and {self.n_agents}, got {agent_index_focus}"
            )
        shared_viewer = agent_index_focus is None
        aspect_ratio = self.scenario.viewer_size[X] / self.scenario.viewer_size[Y]

        headless = mode == "rgb_array" and not visualize_when_rgb
        # First time rendering
        if self.visible_display is None:
            self.visible_display = not headless
            self.headless = headless
        # All other times headless should be the same
        else:
            assert self.visible_display is not headless

        # First time rendering
        if self.viewer is None:
            try:
                import pyglet
            except ImportError:
                raise ImportError(
                    "Cannot import pyg;et: you can install pyglet directly via 'pip install pyglet'."
                )

            try:
                # Try to use EGL
                pyglet.lib.load_library("EGL")

                # Only if we have GPUs
                from pyglet.libs.egl import egl
                from pyglet.libs.egl import eglext

                num_devices = egl.EGLint()
                eglext.eglQueryDevicesEXT(0, None, byref(num_devices))
                assert num_devices.value > 0

            except (ImportError, AssertionError):
                self.headless = False
            pyglet.options["headless"] = self.headless

            self._init_rendering()

        # Render comm messages
        if self.world.dim_c > 0:
            idx = 0
            for agent in self.world.agents:
                if agent.silent:
                    continue
                assert (
                    agent.state.c is not None
                ), "Agent has no comm state but it should"
                if self.continuous_actions:
                    word = (
                        "["
                        + ",".join([f"{comm:.2f}" for comm in agent.state.c[env_index]])
                        + "]"
                    )
                else:
                    word = ALPHABET[torch.argmax(agent.state.c[env_index]).item()]

                message = agent.name + " sends " + word + "   "
                self.viewer.text_lines[idx].set_text(message)
                idx += 1

        zoom = max(VIEWER_MIN_ZOOM, self.scenario.viewer_zoom)

        if aspect_ratio < 1:
            cam_range = torch.tensor([zoom, zoom / aspect_ratio], device=self.device)
        else:
            cam_range = torch.tensor([zoom * aspect_ratio, zoom], device=self.device)

        if shared_viewer:
            # zoom out to fit everyone
            all_poses = torch.stack(
                [agent.state.pos[env_index] for agent in self.world.agents], dim=0
            )
            max_agent_radius = np.max(
                [agent.shape.circumscribed_radius() for agent in self.world.agents]
            )
            viewer_size_fit = (
                torch.stack(
                    [
                        torch.max(torch.abs(all_poses[:, X])),
                        torch.max(torch.abs(all_poses[:, Y])),
                    ]
                )
                + 2 * max_agent_radius
            )

            viewer_size = torch.maximum(
                viewer_size_fit / cam_range, torch.tensor(zoom, device=self.device)
            )
            cam_range *= torch.max(viewer_size)
            self.viewer.set_bounds(
                -cam_range[X],
                cam_range[X],
                -cam_range[Y],
                cam_range[Y],
            )
        else:
            # update bounds to center around agent
            pos = self.agents[agent_index_focus].state.pos[env_index]
            self.viewer.set_bounds(
                pos[X] - cam_range[X],
                pos[X] + cam_range[X],
                pos[Y] - cam_range[Y],
                pos[Y] + cam_range[Y],
            )
        from vmas.simulator.rendering import Grid

        if plot_position_function is not None:
            self.plot_function(
                plot_position_function,
                precision=plot_position_function_precision,
                plot_range=plot_position_function_range,
            )

        if self.scenario.plot_grid:
            grid = Grid(spacing=self.scenario.grid_spacing)
            grid.set_color(*vmas.simulator.utils.Color.BLACK.value, alpha=0.3)
            self.viewer.add_onetime(grid)

        for geom in self.scenario.extra_render(env_index):
            self.viewer.add_onetime(geom)

        for entity in self.world.entities:
            [
                self.viewer.add_onetime(geom)
                for geom in entity.render(env_index=env_index)
            ]

        # render to display or array
        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def plot_function(
        self,
        f: Callable[[Tuple[float, float]], float],
        precision: float,
        plot_range: float,
    ):
        from vmas.simulator.rendering import Transform, make_polygon

        l, r, t, b = (
            0,
            precision,
            precision,
            0,
        )
        poly_points = [(l, b), (l, t), (r, t), (r, b)]

        points = np.arange(-plot_range, plot_range, precision)
        for x in points:
            for y in points:
                alpha = f(x, y)
                box = make_polygon(poly_points, draw_border=False)
                transform = Transform()
                transform.set_translation(x, y)
                box.add_attr(transform)
                box.set_color(*vmas.simulator.utils.Color.BLACK.value, alpha=alpha)
                self.viewer.add_onetime(box)

    def _init_rendering(self):
        from vmas.simulator import rendering

        self.viewer = rendering.Viewer(
            *self.scenario.viewer_size, visible=self.visible_display
        )

        idx = 0
        if self.world.dim_c > 0:
            self.viewer.text_lines = []
            for agent in self.world.agents:
                if not agent.silent:
                    text_line = rendering.TextLine(self.viewer.window, idx)
                    self.viewer.text_lines.append(text_line)
                    idx += 1

    @override(TorchVectorizedObject)
    def to(self, device: DEVICE_TYPING):
        device = torch.device(device)
        self.scenario.to(device)
        super().to(device)
