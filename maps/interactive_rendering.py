#  Copyright (c) 2022. Matteo Bettini
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

import numpy as np

from maps.make_env import make_env
from maps.simulator.environment import GymWrapper

N_TEXT_LINES_INTERACTIVE = 6


class InteractiveEnv:
    """
    Use this script to interactively play with scenarios

    You can change agent by pressing TAB
    You can reset the environment by pressing R
    You can move agents with the arrow keys
    If you have more than 1 agent, you can control another one with W,A,S,D
    and switch the agent with these controls using LSHIFT
    """

    def __init__(self, env: GymWrapper):

        self.env = env
        # hard-coded keyboard events
        self.current_agent_index = 0
        self.current_agent_index2 = 1
        self.n_agents = self.env.unwrapped().n_agents
        self.continuous = self.env.unwrapped().continuous_actions
        self.reset = False
        self.keys = np.array([0, 0, 0, 0])  # up, down, left, right
        self.keys2 = np.array([0, 0, 0, 0])  # up, down, left, right
        self.u = 0 if not self.continuous else (0.0, 0.0)
        self.u2 = 0 if not self.continuous else (0.0, 0.0)

        env.render()
        self._init_text()
        env.unwrapped().viewer.window.on_key_press = self._key_press
        env.unwrapped().viewer.window.on_key_release = self._key_release

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
                self.env.reset()
                self.reset = False
                total_rew = [0] * self.n_agents

            if self.continuous:
                action_list = [(0.0, 0.0)] * self.n_agents
            else:
                action_list = [0] * self.n_agents
            action_list[self.current_agent_index] = self.u
            if self.n_agents > 1:
                action_list[self.current_agent_index2] = self.u2
            obs, rew, done, info = self.env.step(action_list)

            obs[self.current_agent_index] = np.around(
                obs[self.current_agent_index].cpu().tolist(), decimals=2
            )
            len_obs = len(obs[self.current_agent_index])
            message = f"\t\t{obs[self.current_agent_index][len_obs//2:]}"
            self._write_values(self.text_idx, message)
            message = f"Obs: {obs[self.current_agent_index][:len_obs//2]}"
            self._write_values(self.text_idx + 1, message)

            message = f"Rew: {round(rew[self.current_agent_index],3)}"
            self._write_values(self.text_idx + 2, message)

            total_rew = list(map(add, total_rew, rew))
            message = f"Total rew: {round(total_rew[self.current_agent_index], 3)}"
            self._write_values(self.text_idx + 3, message)

            message = f"Done: {done}"
            self._write_values(self.text_idx + 4, message)

            message = f"Selected: {self.env.unwrapped().agents[self.current_agent_index].name}"
            self._write_values(self.text_idx + 5, message)

            self.env.render()

            if done:
                self.reset = True

    def _init_text(self):
        from maps.simulator import rendering

        try:
            self.text_idx = len(self.env.unwrapped().viewer.text_lines)
        except AttributeError:
            self.text_idx = 0

        for i in range(N_TEXT_LINES_INTERACTIVE):
            text_line = rendering.TextLine(
                self.env.unwrapped().viewer.window, self.text_idx + i
            )
            self.env.unwrapped().viewer.text_lines.append(text_line)

    def _write_values(self, index: int, message: str, font_size: int = 15):
        self.env.unwrapped().viewer.text_lines[index].set_text(
            message, font_size=font_size
        )

    # keyboard event callbacks
    def _key_press(self, k, mod):
        from pyglet.window import key

        u = self.u
        u2 = self.u2
        if k == key.LEFT:
            self.keys[0] = 1
            u = 1
        elif k == key.RIGHT:
            self.keys[1] = 1
            u = 2
        elif k == key.DOWN:
            self.keys[2] = 1
            u = 3
        elif k == key.UP:
            self.keys[3] = 1
            u = 4
        elif k == key.TAB:
            self.current_agent_index = self._increment_selected_agent_index(
                self.current_agent_index
            )
            while self.current_agent_index == self.current_agent_index2:
                self.current_agent_index = self._increment_selected_agent_index(
                    self.current_agent_index
                )

        elif k == key.A:
            self.keys2[0] = 1
            u2 = 1
        elif k == key.D:
            self.keys2[1] = 1
            u2 = 2
        elif k == key.S:
            self.keys2[2] = 1
            u2 = 3
        elif k == key.W:
            self.keys2[3] = 1
            u2 = 4
        elif k == key.LSHIFT:
            self.current_agent_index2 = self._increment_selected_agent_index(
                self.current_agent_index2
            )
            while self.current_agent_index == self.current_agent_index2:
                self.current_agent_index2 = self._increment_selected_agent_index(
                    self.current_agent_index2
                )

        elif k == key.R:
            self.reset = True

        if self.continuous:
            self.u = (self.keys[1] - self.keys[0], self.keys[3] - self.keys[2])
            self.u2 = (self.keys2[1] - self.keys2[0], self.keys2[3] - self.keys2[2])
        else:
            self.u = u
            self.u2 = u2

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

        elif k == key.A:
            self.keys2[0] = 0
        elif k == key.D:
            self.keys2[1] = 0
        elif k == key.S:
            self.keys2[2] = 0
        elif k == key.W:
            self.keys2[3] = 0

        elif k == key.R:
            self.reset = False

        if self.continuous:
            self.u = (self.keys[1] - self.keys[0], self.keys[3] - self.keys[2])
            self.u2 = (self.keys2[1] - self.keys2[0], self.keys2[3] - self.keys2[2])
        else:
            if np.sum(self.keys) == 1:
                self.u = np.argmax(self.keys) + 1
            else:
                self.u = 0
            if np.sum(self.keys2) == 1:
                self.u2 = np.argmax(self.keys2) + 1
            else:
                self.u2 = 0


def render_interactively(scenario_name: str, **kwargs):
    """
    Use this script to interactively play with scenarios

    You can change agent by pressing TAB
    You can reset the environment by pressing R
    You can move agents with the arrow keys
    If you have more than 1 agent, you can control another one with W,A,S,D
    and switch the agent with these controls using LSHIFT
    """
    InteractiveEnv(
        GymWrapper(
            make_env(
                scenario_name=scenario_name,
                num_envs=1,
                device="cpu",
                continuous_actions=True,
                rllib_wrapped=False,
                # Environment specific variables
                **kwargs,
            )
        )
    )


if __name__ == "__main__":
    # Use this script to interactively play with scenarios
    #
    # You can change agent by pressing TAB
    # You can reset the environment by pressing R
    # You can move agents with the arrow keys
    # If you have more than 1 agent, you can control another one with W,A,S,D
    # and switch the agent with these controls using LSHIFT

    scenario_name = "waterfall"

    # Scenario specific variables

    render_interactively(scenario_name)
