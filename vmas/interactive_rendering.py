#  Copyright (c) 2022-2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
"""
Use this script to interactively play with scenarios

You can change agent by pressing TAB
You can reset the environment by pressing R
You can move agents with the arrow keys
If you have more than 1 agent, you can control another one with W,A,S,D
and switch the agent with these controls using LSHIFT
"""
from operator import add
from typing import Union, Dict

import numpy as np
from torch import Tensor

from vmas.make_env import make_env
from vmas.simulator.environment import Wrapper
from vmas.simulator.environment.gym import GymWrapper
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import save_video

N_TEXT_LINES_INTERACTIVE = 6


class InteractiveEnv:
    """
    Use this script to interactively play with scenarios

    You can change agent by pressing TAB
    You can reset the environment by pressing R
    You can move agents with the arrow keys and if the agent has a rotational action you can control it with M, N
    If you have more than 1 agent, you can control another one with W,A,S,D and Q,E for eventual rotational actions
    and switch the agent with these controls using LSHIFT
    """

    def __init__(
        self,
        env: GymWrapper,
        control_two_agents: bool = False,
        display_info: bool = True,
        save_render: bool = False,
        render_name: str = "interactive",
    ):
        self.env = env
        self.control_two_agents = control_two_agents
        # hard-coded keyboard events
        self.current_agent_index = 0
        self.current_agent_index2 = 1
        self.n_agents = self.env.unwrapped().n_agents
        self.agents = self.env.unwrapped().agents
        self.continuous = self.env.unwrapped().continuous_actions
        self.reset = False
        self.keys = np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )  # up, down, left, right, rot+, rot-
        self.keys2 = np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )  # up, down, left, right, rot+, rot-
        self.u = [0] * (3 if self.continuous else 2)
        self.u2 = [0] * (3 if self.continuous else 2)
        self.frame_list = []
        self.display_info = display_info
        self.save_render = save_render
        self.render_name = render_name

        if self.control_two_agents:
            assert (
                self.n_agents >= 2
            ), "Control_two_agents is true but not enough agents in scenario"

        self.text_lines = []
        self.font_size = 15
        self.env.render()
        self.text_idx = len(self.env.unwrapped().text_lines)
        self._init_text()
        self.env.unwrapped().viewer.window.on_key_press = self._key_press
        self.env.unwrapped().viewer.window.on_key_release = self._key_release

        self._cycle()

    def _increment_selected_agent_index(self, index: int):
        index += 1
        if index == self.n_agents:
            index = 0
        return index

    def _cycle(self):
        total_rew = [0] * self.n_agents
        while True:
            if self.reset:
                if self.save_render:
                    save_video(
                        self.render_name,
                        self.frame_list,
                        fps=1 / self.env.env.world.dt,
                    )
                self.env.reset()
                self.reset = False
                total_rew = [0] * self.n_agents

            action_list = [
                [0.0] * self.env.unwrapped().get_agent_action_size(agent)
                for agent in self.agents
            ]
            action_list[self.current_agent_index] = self.u[
                : self.env.unwrapped().get_agent_action_size(
                    self.agents[self.current_agent_index]
                )
            ]

            if self.n_agents > 1 and self.control_two_agents:
                action_list[self.current_agent_index2] = self.u2[
                    : self.env.unwrapped().get_agent_action_size(
                        self.agents[self.current_agent_index2]
                    )
                ]
            obs, rew, done, info = self.env.step(action_list)

            if self.display_info:
                # TODO: Determine number of lines of obs_str and render accordingly
                obs_str = str(InteractiveEnv.format_obs(obs[self.current_agent_index]))
                message = f"\t\t{obs_str[len(obs_str) // 2:]}"
                self._write_values(0, message)
                message = f"Obs: {obs_str[:len(obs_str) // 2]}"
                self._write_values(1, message)

                message = f"Rew: {round(rew[self.current_agent_index],3)}"
                self._write_values(2, message)

                total_rew = list(map(add, total_rew, rew))
                message = f"Total rew: {round(total_rew[self.current_agent_index], 3)}"
                self._write_values(3, message)

                message = f"Done: {done}"
                self._write_values(4, message)

                message = f"Selected: {self.env.unwrapped().agents[self.current_agent_index].name}"
                self._write_values(5, message)

            frame = self.env.render(
                mode="rgb_array" if self.save_render else "human",
                visualize_when_rgb=True,
            )
            if self.save_render:
                self.frame_list.append(frame)

            if done:
                self.reset = True

    def _init_text(self):
        from vmas.simulator import rendering

        for i in range(N_TEXT_LINES_INTERACTIVE):
            text_line = rendering.TextLine(
                y=(self.text_idx + i) * 40, font_size=self.font_size
            )
            self.env.unwrapped().viewer.add_geom(text_line)
            self.text_lines.append(text_line)

    def _write_values(self, index: int, message: str):
        self.text_lines[index].set_text(message)

    # keyboard event callbacks
    def _key_press(self, k, mod):
        from pyglet.window import key

        agent_range = self.agents[self.current_agent_index].u_range
        agent_rot_range = self.agents[self.current_agent_index].u_rot_range

        if k == key.LEFT:
            self.keys[0] = agent_range
        elif k == key.RIGHT:
            self.keys[1] = agent_range
        elif k == key.DOWN:
            self.keys[2] = agent_range
        elif k == key.UP:
            self.keys[3] = agent_range
        elif k == key.M:
            self.keys[4] = agent_rot_range
        elif k == key.N:
            self.keys[5] = agent_rot_range
        elif k == key.TAB:
            self.current_agent_index = self._increment_selected_agent_index(
                self.current_agent_index
            )
            if self.control_two_agents:
                while self.current_agent_index == self.current_agent_index2:
                    self.current_agent_index = self._increment_selected_agent_index(
                        self.current_agent_index
                    )

        if self.control_two_agents:
            agent2_range = self.agents[self.current_agent_index2].u_range
            agent2_rot_range = self.agents[self.current_agent_index2].u_rot_range

            if k == key.A:
                self.keys2[0] = agent2_range
            elif k == key.D:
                self.keys2[1] = agent2_range
            elif k == key.S:
                self.keys2[2] = agent2_range
            elif k == key.W:
                self.keys2[3] = agent2_range
            elif k == key.E:
                self.keys2[4] = agent2_rot_range
            elif k == key.Q:
                self.keys2[5] = agent2_rot_range

            elif k == key.LSHIFT:
                self.current_agent_index2 = self._increment_selected_agent_index(
                    self.current_agent_index2
                )
                while self.current_agent_index == self.current_agent_index2:
                    self.current_agent_index2 = self._increment_selected_agent_index(
                        self.current_agent_index2
                    )

        if k == key.R:
            self.reset = True

        self.set_u()

    def _key_release(self, k, mod):
        from pyglet.window import key

        if k == key.LEFT:
            self.keys[0] = 0
        elif k == key.RIGHT:
            self.keys[1] = 0
        elif k == key.DOWN:
            self.keys[2] = 0
        elif k == key.UP:
            self.keys[3] = 0
        elif k == key.M:
            self.keys[4] = 0
        elif k == key.N:
            self.keys[5] = 0

        if self.control_two_agents:
            if k == key.A:
                self.keys2[0] = 0
            elif k == key.D:
                self.keys2[1] = 0
            elif k == key.S:
                self.keys2[2] = 0
            elif k == key.W:
                self.keys2[3] = 0
            elif k == key.E:
                self.keys2[4] = 0
            elif k == key.Q:
                self.keys2[5] = 0

        self.set_u()

    def set_u(self):
        if self.continuous:
            self.u = [
                self.keys[1] - self.keys[0],
                self.keys[3] - self.keys[2],
                self.keys[4] - self.keys[5],
            ]
            self.u2 = [
                self.keys2[1] - self.keys2[0],
                self.keys2[3] - self.keys2[2],
                self.keys2[4] - self.keys2[5],
            ]
        else:
            if np.sum(self.keys[:4]) >= 1:
                self.u[0] = np.argmax(self.keys[:4]) + 1
            else:
                self.u[0] = 0
            if np.sum(self.keys[4:]) >= 1:
                self.u[1] = np.argmax(self.keys[4:]) + 1
            else:
                self.u[1] = 0

            if np.sum(self.keys2[:4]) >= 1:
                self.u2[0] = np.argmax(self.keys2[:4]) + 1
            else:
                self.u2[0] = 0
            if np.sum(self.keys2[4:]) >= 1:
                self.u2[1] = np.argmax(self.keys2[4:]) + 1
            else:
                self.u2[1] = 0

    @staticmethod
    def format_obs(obs):
        if isinstance(obs, Tensor):
            return list(np.around(obs.cpu().tolist(), decimals=2))
        elif isinstance(obs, Dict):
            return {key: InteractiveEnv.format_obs(value) for key, value in obs.items()}
        else:
            raise NotImplementedError(f"Invalid type of observation {obs}")


def render_interactively(
    scenario: Union[str, BaseScenario],
    control_two_agents: bool = False,
    display_info: bool = True,
    save_render: bool = False,
    **kwargs,
):
    """
    Use this script to interactively play with scenarios

    You can change agent by pressing TAB
    You can reset the environment by pressing R
    You can move agents with the arrow keys and if the agent has a rotational action you can control it with M, N
    If you have more than 1 agent, you can control another one with W,A,S,D and Q,E for eventual rotational actions
    and switch the agent with these controls using LSHIFT
    """

    InteractiveEnv(
        make_env(
            scenario=scenario,
            num_envs=1,
            device="cpu",
            continuous_actions=True,
            wrapper=Wrapper.GYM,
            seed=0,
            # Environment specific variables
            **kwargs,
        ),
        control_two_agents=control_two_agents,
        display_info=display_info,
        save_render=save_render,
        render_name=f"{scenario}",
    )


if __name__ == "__main__":
    # Use this script to interactively play with scenarios
    #
    # You can change agent by pressing TAB
    # You can reset the environment by pressing R
    # You can move agents with the arrow keys and if the agent has a rotational action you can control it with M, N
    # If you have more than 1 agent, you can control another one with W,A,S,D and Q,E for eventual rotational actions
    # and switch the agent with these controls using LSHIFT

    scenario_name = "waterfall"

    # Scenario specific variables

    render_interactively(
        scenario_name, control_two_agents=True, save_render=False, display_info=True
    )
