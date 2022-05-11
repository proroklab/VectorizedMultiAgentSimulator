from typing import List, Tuple, Optional

import gym
import numpy as np
import torch
from gym import spaces
from ray import rllib
from ray.rllib.utils.typing import EnvActionType, EnvObsType, EnvInfoDict
from torch import Tensor

from mpe.multiagent import core
from mpe.multiagent.core import Line, Box, TorchVectorizedObject
from mpe.multiagent.scenario import BaseScenario
from simulator.utils import Y, X


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
        shared_viewer=True,
        continuous_actions=True,
    ):
        self.current_rendering_index = None
        self.scenario = scenario
        self.num_envs = num_envs
        TorchVectorizedObject.__init__(self, num_envs, torch.device(device))
        self.world = self.scenario.env_make_world(self.num_envs, self.device)

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
                        shape=(
                            self.world.dim_p
                            + (self.world.dim_c if not agent.silent else 0),
                        ),
                        dtype=float,
                    )
                    if self.continuous_actions
                    else spaces.Discrete(
                        self.world.dim_p * 2
                        + 1
                        + (self.world.dim_c if not agent.silent else 0)
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
        self.shared_viewer = shared_viewer
        self.render_geoms_xform = None
        self.render_geoms = None
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n_agents

    def reset(self, seed: int = None):
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
            actions: Should be a list on len 'self.n_agents' of which each element has shape '(self.num_envs, action_size_of_agent)'.

            For compatibility purposes, we also allow actions to be a list of len 'self.num_envs' of which each element is
            a list of len 'self.n_agents' of which each element has shape '(action_size_of_agent,)'.
            Conversion of this second type is handled automatically, however it can be expensive, thus the first method
            is suggested.
        Returns:
            obs (List[any]): New observations for each sub-env.
            rewards (List[any]): Reward values for each sub-env.
            dones (List[any]): Done values for each sub-env.
            infos (List[any]): Info values for each sub-env.
        """
        for i in range(len(actions)):
            assert (
                actions[i].shape[0] == self.num_envs
            ), f"Actions used in input of env must be of len {self.num_envs}, got {actions[i].shape[0]}"
            if isinstance(self.action_space[i], spaces.Box):
                assert actions[i].shape[1] == self.action_space[i].shape[0], (
                    f"Action for agent {self.agents[i].name} has shape, {actions[i].shape[1]},"
                    f" but should have shape {self.action_space[i].shape[0]}"
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

        print("\nStep results in unwrapped environment")
        print(
            f"Actions len (n_agents): {len(actions)}, actions[0] shape (num_envs, agent 0 action shape): {actions[0].shape}, actions[0][0] (action agent 0 env 0): {actions[0][0]}"
        )
        print(
            f"Obs len (n_agents): {len(obs)}, obs[0] shape (num_envs, agent 0 obs shape): {obs[0].shape}, obs[0][0] (obs agent 0 env 0): {obs[0][0]}"
        )
        print(
            f"Rews len (n_agents): {len(rews)}, rews[0] shape (num_envs, 1): {rews[0].shape}, rews[0][0] (agent 0 env 0): {rews[0][0]}"
        )
        print(f"Dones len (n_envs): {len(dones)}, dones[0] (done env 0): {dones[0]}")
        print(f"Info len (n_agents): {len(infos)}, info[0] (infos agent 0): {infos[0]}")

        return obs, rews, dones, infos

    # set env action for a particular agent
    def _set_action(self, action, agent):
        action = action.clone().to(self.device)
        agent.action.u = torch.zeros(
            self.batch_dim, self.world.dim_p, device=self.device, dtype=torch.float64
        )

        if agent.movable:
            if self.continuous_actions:
                physical_action = action[:, : self.world.dim_p]
                agent.action.u = physical_action
            else:
                physical_action = action[:, :1]
                arr1 = physical_action == 1
                arr2 = physical_action == 2
                arr3 = physical_action == 3
                arr4 = physical_action == 4

                cont_action = 0.03

                agent.action.u[:, X] += -cont_action * arr1.squeeze(-1)
                agent.action.u[:, X] += cont_action * arr2.squeeze(-1)
                agent.action.u[:, Y] += -cont_action * arr3.squeeze(-1)
                agent.action.u[:, Y] += cont_action * arr4.squeeze(-1)

            agent.action.u *= agent.u_multiplier
        if self.world.dim_c > 0 and not agent.silent:
            comm_action = action[:, -self.world.dim_c :]
            agent.action.c = comm_action

    # render environment
    def render(self, mode="human", index=0):
        self._check_batch_index(index)

        # if mode == "human":
        #     alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        #     message = ""
        #     for agent in self.world.agents:
        #         comm = []
        #         for other in self.world.agents:
        #             if other is agent:
        #                 continue
        #             if torch.all(other.state.c == 0):
        #                 word = "_"
        #             else:
        #                 word = alphabet[torch.argmax(other.state.c[index]).item()]
        #             message += other.name + " to " + agent.name + ": " + word + "   "
        # if len(message) > 0:
        # print(message)

        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                # from gym.envs.classic_control import rendering
                from mpe.multiagent import rendering

                self.viewers[i] = rendering.Viewer(700, 700)

        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            from mpe.multiagent import rendering

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
                if "agent" in entity.name:
                    geom.set_color(*entity.color.value, alpha=0.5)
                else:
                    geom.set_color(*entity.color.value)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # add geoms to viewer
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)

        results = []
        for i in range(len(self.viewers)):
            # update bounds to center around agent
            cam_range = 1
            if self.shared_viewer:
                pos = torch.zeros(
                    self.world.dim_p, device=self.device, dtype=torch.float64
                )
            else:
                pos = self.agents[i].state.pos[index]
            self.viewers[i].set_bounds(
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
            results.append(self.viewers[i].render(return_rgb_array=mode == "rgb_array"))

        return results


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
        self, index: Optional[int] = 0, mode="human"
    ) -> Optional[np.ndarray]:
        return np.array(self._env.render(mode=mode, index=index))

    def _list_to_tensor(self, list_in: List) -> List:
        if len(list_in) == self._env.n_agents:
            for i in range(len(list_in)):
                assert (
                    list_in[i].shape[0] == self.num_envs
                ), f"Actions used in input of env must be of len {self.num_envs}, got {list_in[i].shape[0]}"
                if isinstance(self._env.action_space[i], spaces.Box):
                    assert list_in[i].shape[1] == self._env.action_space[i].shape[0], (
                        f"Action for agent {self._env.agents[i].name} has shape, {list_in[i].shape[1]},"
                        f" but should have shape {self._env.action_space[i].shape[0]}"
                    )
                if not isinstance(list_in[i], Tensor):
                    list_in[i] = torch.tensor(
                        list_in[i], dtype=torch.float64, device=self._env.device
                    )
            return list_in
        elif len(list_in) == self.num_envs:
            actions = []
            for i in range(self._env.n_agents):
                actions.append(
                    torch.zeros(
                        self.num_envs,
                        self._env.world.dim_p if self._env.continuous_actions else 1,
                        device=self._env.device,
                        dtype=torch.float64,
                    )
                )
            for j in range(self.num_envs):
                for i in range(self._env.n_agents):
                    actions[i][j] = torch.tensor(
                        list_in[j][i], dtype=torch.float64, device=self._env.device
                    )
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
