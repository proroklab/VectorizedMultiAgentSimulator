from typing import List, Tuple, Optional

import gym
import numpy as np
import torch
from gym import spaces
from ray import rllib
from ray.rllib.utils.typing import EnvActionType, EnvObsType, EnvInfoDict
from torch import Tensor

from maps import core
from maps.core import TorchVectorizedObject, Line, Box, Agent
from maps.scenario import BaseScenario
from maps.utils import X, Y


# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class Environment(gym.vector.VectorEnv, TorchVectorizedObject):

    metadata = {
        "render.modes": ["human", "rgb_array"],
        "runtime.vectorized": True,
    }

    def __init__(
        self,
        scenario: BaseScenario,
        num_envs: int = 32,
        device: str = "cpu",
        max_steps=None,
        continuous_actions=True,
        **kwargs,
    ):
        self.current_rendering_index = None
        self.scenario = scenario
        self.num_envs = num_envs
        TorchVectorizedObject.__init__(self, num_envs, torch.device(device))
        self.world = self.scenario.env_make_world(self.num_envs, self.device, **kwargs)

        self.agents = self.world.policy_agents
        self.n_agents = len(self.agents)
        self.max_steps = max_steps
        self.continuous_actions = continuous_actions

        self.steps = 0

        # configure spaces
        self.action_space = gym.spaces.Tuple(
            (
                (
                    spaces.Box(
                        low=np.array(
                            [-agent.u_range] * self.world.dim_p
                            + [0.0] * (self.world.dim_c if not agent.silent else 0)
                        ),
                        high=np.array(
                            [agent.u_range] * self.world.dim_p
                            + [1.0] * (self.world.dim_c if not agent.silent else 0)
                        ),
                        shape=(self.get_agent_action_size(agent),),
                        dtype=float,
                    )
                    if self.continuous_actions
                    else (
                        spaces.Discrete(self.world.dim_p * 2 + 1)
                        if self.world.dim_c == 0 or agent.silent
                        else spaces.MultiDiscrete(
                            [self.world.dim_p * 2 + 1, self.world.dim_c]
                        )
                    )
                )
                for agent in self.agents
            )
        )
        self.observation_space = gym.spaces.Tuple(
            (
                (
                    spaces.Box(
                        low=-float("inf"),
                        high=float("inf"),
                        shape=(len(self.scenario.observation(agent)[0]),),
                        dtype=float,
                    )
                )
                for agent in self.agents
            )
        )

        # rendering
        self.render_geoms_xform = None
        self.render_geoms = None
        self.viewer = None

    def reset(self, seed: int = None):
        """
        Resets the environment in a vectorized way
        Returns observations for all envs and agents
        """
        # Seed will be none at first call of this fun from constructor
        if seed is not None:
            self.seed(seed)
        # reset world
        self.scenario.reset_world()
        # record observations for each agent
        obs = []
        for agent in self.agents:
            obs.append(self.scenario.observation(agent))
        return obs

    def reset_at(self, index: int = None):
        """
        Resets the environment at index
        Returns observations for all agents in that environment
        """
        self._check_batch_index(index)
        self.scenario.reset_world_at(index)
        obs = []
        for agent in self.agents:
            obs.append(self.scenario.observation(agent)[index])
        return obs

    def seed(self, seed=None):
        if seed is None:
            seed = 0
        torch.manual_seed(seed)
        self.scenario.seed()
        return [seed]

    def step(self, actions: List):
        """Performs a vectorized step on all sub environments using `actions`.
        Args:
            actions: Is a list on len 'self.n_agents' of which each element is a torch.Tensor of shape '(self.num_envs, action_size_of_agent)'.
        Returns:
            obs: List on len 'self.n_agents' of which each element is a torch.Tensor of shape '(self.num_envs, obs_size_of_agent)'
            rewards: List on len 'self.n_agents' of which each element is a torch.Tensor of shape '(self.num_envs)'
            dones: Tensor of len 'self.num_envs' of which each element is a bool
            infos : List on len 'self.n_agents' of which each element is a dictionary for which each key is a metric and the value is a tensor of shape '(self.num_envs, metric_size_per_agent)'
        """
        assert (
            len(actions) == self.n_agents
        ), f"Expecting actions for {self.n_agents}, got {len(actions)} actions"
        for i in range(len(actions)):
            assert (
                actions[i].shape[0] == self.num_envs
            ), f"Actions used in input of env must be of len {self.num_envs}, got {actions[i].shape[0]}"
            assert actions[i].shape[1] == self.get_agent_action_size(self.agents[i]), (
                f"Action for agent {self.agents[i].name} has shape {actions[i].shape[1]},"
                f" but should have shape {self.get_agent_action_size(self.agents[i])}"
            )
            if not isinstance(actions[i], Tensor):
                actions[i] = torch.tensor(
                    actions[i], dtype=torch.float64, device=self.device
                )

        # set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(actions[i], agent)
        # advance world state
        self.world.step()
        # record observation for each agent
        obs = []
        rews = []
        infos = []
        for agent in self.agents:
            obs.append(self.scenario.observation(agent))
            rews.append(self.scenario.reward(agent))
            # A dictionary per agent
            infos.append(self.scenario.info(agent))

        dones = self.scenario.done()

        self.steps += 1
        if self.max_steps is not None and self.steps > self.max_steps:
            dones = torch.tensor([True], device=self.device).repeat(
                self.world.batch_dim
            )

        # print("\nStep results in unwrapped environment")
        # print(
        #     f"Actions len (n_agents): {len(actions)}, actions[0] shape (num_envs, agent 0 action shape): {actions[0].shape}, actions[0][0] (action agent 0 env 0): {actions[0][0]}"
        # )
        # print(
        #     f"Obs len (n_agents): {len(obs)}, obs[0] shape (num_envs, agent 0 obs shape): {obs[0].shape}, obs[0][0] (obs agent 0 env 0): {obs[0][0]}"
        # )
        # print(
        #     f"Rews len (n_agents): {len(rews)}, rews[0] shape (num_envs, 1): {rews[0].shape}, rews[0][0] (agent 0 env 0): {rews[0][0]}"
        # )
        # print(f"Dones len (n_envs): {len(dones)}, dones[0] (done env 0): {dones[0]}")
        # print(f"Info len (n_agents): {len(infos)}, info[0] (infos agent 0): {infos[0]}")

        return obs, rews, dones, infos

    def get_agent_action_size(self, agent: Agent):
        return (
            self.world.dim_p + (self.world.dim_c if not agent.silent else 0)
            if self.continuous_actions
            else 1 + (1 if not agent.silent else 0)
        )

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
            self.batch_dim, self.world.dim_p, device=self.device, dtype=torch.float64
        )

        assert action.shape[1] == self.get_agent_action_size(agent), (
            f"Agent {agent.name} has wrong action size, got {action.shape[1]}, "
            f"expected {self.get_agent_action_size(agent)}"
        )

        if agent.movable:
            if self.continuous_actions:
                physical_action = action[:, : self.world.dim_p]
                assert not torch.any(
                    torch.abs(physical_action) > agent.u_range
                ), f"Physical actions of agent {agent.name} are out of its range {agent.u_range}"

                agent.action.u = physical_action
            else:
                physical_action = action[:, :1]

                self._check_discrete_action(
                    physical_action, 0, self.world.dim_p * 2 + 1, "physical"
                )

                arr1 = physical_action == 1
                arr2 = physical_action == 2
                arr3 = physical_action == 3
                arr4 = physical_action == 4

                disc_action_value = agent.u_range

                agent.action.u[:, X] += -disc_action_value * arr1.squeeze(-1)
                agent.action.u[:, X] += disc_action_value * arr2.squeeze(-1)
                agent.action.u[:, Y] += -disc_action_value * arr3.squeeze(-1)
                agent.action.u[:, Y] += disc_action_value * arr4.squeeze(-1)

            agent.action.u *= agent.u_multiplier
        if self.world.dim_c > 0 and not agent.silent:
            if not self.continuous_actions:
                comm_action = action[:, 1:]
                self._check_discrete_action(
                    comm_action, 0, self.world.dim_c, "communication"
                )
                comm_action = comm_action.long()
                agent.action.c = torch.zeros(
                    self.num_envs,
                    self.world.dim_c,
                    device=self.device,
                    dtype=torch.float64,
                )
                agent.action.c[:, comm_action] = 1.0
            else:
                comm_action = action[:, self.world.dim_p :]
                assert not torch.any(comm_action > 1) and not torch.any(
                    0 > comm_action
                ), f"Comm actions are out of range [0,1]"
                agent.action.c = comm_action

    def render(self, mode="human", index=0, agent_index_focus: int = None):
        """
        Render function for environment using pyglet

        On servers use mode="rgb_array" and set
        ```
        export DISPLAY=':99.0'
        Xvfb :99 -screen 0 1400x900x24 > /dev/null 2>&1 &
        ```

        :param mode: One of human or rgb_array
        :param index: Index of the environment to render
        :param agent_index_focus: If specified the camera will stay on the agent with this index.
                                  If None, the camera will stay in the center and zoom out to contain all agents
        :return: Rgb array or None, depending on the mode
        """
        self._check_batch_index(index)
        if agent_index_focus is not None:
            assert -1 <= agent_index_focus <= self.n_agents, (
                f"Agent focus in rendering should be a valid agent index"
                f" between 0 and {self.n_agents}, got {agent_index_focus}"
            )
        shared_viewer = agent_index_focus is None

        if self.viewer is None:
            from maps import rendering

            self.viewer = rendering.Viewer(700, 700)

        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            from maps import rendering

            self.render_geoms = []
            self.render_geoms_xform = []
            for entity in self.world.entities:
                if isinstance(entity.shape, core.Sphere):
                    geom = rendering.make_circle(entity.shape.radius)
                elif isinstance(entity.shape, Line):
                    geom = rendering.Line(
                        (-entity.shape.length / 2, 0),
                        (entity.shape.length / 2, 0),
                        width=entity.shape.width,
                    )
                elif isinstance(entity.shape, Box):
                    l, r, t, b = (
                        -entity.shape.length / 2,
                        entity.shape.length / 2,
                        entity.shape.width / 2,
                        -entity.shape.width / 2,
                    )
                    geom = rendering.make_polygon([(l, b), (l, t), (r, t), (r, b)])
                else:
                    assert (
                        False
                    ), f"Entity shape not supported in rendering for {entity.name}"
                xform = rendering.Transform()
                color = entity.color
                if isinstance(color, torch.Tensor) and len(color.shape) > 1:
                    color = color[index]
                if isinstance(entity, Agent):
                    geom.set_color(*color, alpha=0.5)
                else:
                    geom.set_color(*color)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # add geoms to viewer
            self.viewer.geoms = []
            for geom in self.render_geoms:
                self.viewer.add_geom(geom)
            self.viewer.text_lines = []
            idx = 0
            if self.world.dim_c > 0:
                for agent in self.world.agents:
                    if not agent.silent:
                        tline = rendering.TextLine(self.viewer.window, idx)
                        self.viewer.text_lines.append(tline)
                        idx += 1

        if self.world.dim_c > 0:
            alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
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
                        + ",".join([f"{comm:.2f}" for comm in agent.state.c[index]])
                        + "]"
                    )
                else:
                    word = alphabet[torch.argmax(agent.state.c[index]).item()]

                message = agent.name + " sends " + word + "   "
                self.viewer.text_lines[idx].set_text(message)
                idx += 1

        if shared_viewer:
            # zoom out to fit eveyone
            all_poses = torch.cat(
                [entity.state.pos[index] for entity in self.world.entities], dim=0
            )
            cam_range = torch.max(torch.abs(all_poses)) + 1
            self.viewer.set_max_size(cam_range)
        else:
            # update bounds to center around agent
            cam_range = 1
            pos = self.agents[agent_index_focus].state.pos[index]
            self.viewer.set_bounds(
                pos[0] - cam_range,
                pos[0] + cam_range,
                pos[1] - cam_range,
                pos[1] + cam_range,
            )
        # update geometry positions
        for e, entity in enumerate(self.world.entities):
            self.render_geoms_xform[e].set_translation(*entity.state.pos[index])
            self.render_geoms_xform[e].set_rotation(entity.state.rot[index])
        # render to display or array
        return self.viewer.render(return_rgb_array=mode == "rgb_array")


class VectorEnvWrapper(rllib.VectorEnv):
    def __init__(
        self,
        env: Environment,
    ):
        self._env = env
        super().__init__(
            observation_space=self._env.observation_space,
            action_space=self._env.action_space,
            num_envs=self._env.num_envs,
        )

    @property
    def env(self):
        return self._env

    def vector_reset(self) -> List[EnvObsType]:
        obs = self._env.reset()
        return self._tensor_to_list(obs)

    def reset_at(self, index: Optional[int] = None) -> EnvObsType:
        obs = self._env.reset_at(index)
        return self._tensor_to_list(obs)

    def vector_step(
        self, actions: List[EnvActionType]
    ) -> Tuple[List[EnvObsType], List[float], List[bool], List[EnvInfoDict]]:
        saved_actions = actions
        actions = self._list_to_tensor(actions)
        obs, rews, dones, infos = self._env.step(actions)

        dones = dones.tolist()

        total_rews = []
        total_infos = []
        obs_list = []
        for j in range(self.num_envs):
            obs_list.append([])
            env_infos = {"rewards": {}}
            total_env_rew = 0.0
            for i, agent in enumerate(self._env.agents):
                obs_list[j].append(obs[i][j].cpu().detach().numpy())
                total_env_rew += rews[i][j].item()
                env_infos["rewards"].update({i: rews[i][j].item()})
                env_infos.update(
                    {agent.name: {k: v[j].tolist() for k, v in infos[i].items()}}
                )
            total_infos.append(env_infos)
            total_rews.append(total_env_rew)

        # print("\nStep results in wrapped environment")
        # print(
        #     f"Actions len (num_envs): {len(saved_actions)}, len actions[0] (n_agents): {len(saved_actions[0])}, actions[0][0] (action agent 0 env 0): {saved_actions[0][0]}"
        # )
        # print(
        #     f"Obs len (num_envs): {len(obs_list)}, len obs[0] (n_agents): {len(obs_list[0])}, obs[0][0] (obs agent 0 env 0): {obs_list[0][0]}"
        # )
        # print(
        #     f"Total rews len (num_envs): {len(total_rews)}, total_rews[0] (total rew env 0): {total_rews[0]}"
        # )
        # print(f"Dones len (num_envs): {len(dones)}, dones[0] (done env 0): {dones[0]}")
        # print(
        #     f"Total infos len (num_envs): {len(total_infos)}, total_infos[0] (infos env 0): {total_infos[0]}"
        # )

        return obs_list, total_rews, dones, total_infos

    def seed(self, seed=None):
        return self._env.seed(seed)

    def try_render_at(
        self,
        index: Optional[int] = 0,
        mode="human",
        agent_index_focus: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """
        Render function for environment using pyglet

        On servers use mode="rgb_array" and set
        ```
        export DISPLAY=':99.0'
        Xvfb :99 -screen 0 1400x900x24 > /dev/null 2>&1 &
        ```

        :param mode: One of human or rgb_array
        :param index: Index of the environment to render
        :param agent_index_focus: If specified the camera will stay on the agent with this index.
                                  If None, the camera will stay in the center and zoom out to contain all agents
        :return: Rgb array or None, depending on the mode
        """
        return self._env.render(
            mode=mode, index=index, agent_index_focus=agent_index_focus
        )

    def _list_to_tensor(self, list_in: List) -> List:
        if len(list_in) == self.num_envs:
            actions = []
            for agent in self._env.agents:
                actions.append(
                    torch.zeros(
                        self.num_envs,
                        self._env.get_agent_action_size(agent),
                        device=self._env.device,
                        dtype=torch.float64,
                    )
                )
            for j in range(self.num_envs):
                assert (
                    len(list_in[j]) == self._env.n_agents
                ), f"Expecting actions for {self._env.n_agents} agents, got {len(list_in[j])} actions"
                for i in range(self._env.n_agents):
                    act = torch.tensor(
                        list_in[j][i], dtype=torch.float64, device=self._env.device
                    )
                    assert act.shape[0] == self._env.get_agent_action_size(
                        self._env.agents[i]
                    ), (
                        f"Action of agent {i} in env {j} hase wrong shape: "
                        f"expected {self._env.get_agent_action_size(self._env.agents[i])}, got {act.shape[0]}"
                    )
                    actions[i][j] = act
            return actions
        else:
            assert False, f"Input action is not in correct format"

    def _tensor_to_list(self, list_in: List) -> List:
        assert (
            len(list_in) == self._env.n_agents
        ), f"Tensor used in output of env must be of len {self._env.n_agents}, got {len(list_in)}"
        list_out = []
        for j in range(self.num_envs):
            list_out.append([])
            for i in range(self._env.n_agents):
                list_out[j].append(list_in[i][j].cpu().detach().numpy())
        return list_out
