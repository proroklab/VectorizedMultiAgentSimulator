#  Copyright (c) 2022. Matteo Bettini
#  All rights reserved.
"""
Use this script to interactively play with scenarios

You can change agent by pressing TAB
You can move agents with the arrow keys
"""
import numpy as np

from maps import make_env
from maps.simulator.environment import GymWrapper

N_TEXT_LINES_INTERACTIVE = 5


class InteractiveEnv:
    """
    Use this script to interactively play with scenarios

    You can change agent by pressing TAB
    You can move agents with the arrow keys
    """

    def __init__(self, env: GymWrapper):

        self.env = env
        assert env.unwrapped().continuous_actions is False
        # hard-coded keyboard events
        self.u = 0
        self.current_agent_index = 0
        self.n_agents = self.env.unwrapped().n_agents

        env.render()
        self._init_text()
        env.unwrapped().viewer.window.on_key_press = self._key_press
        env.unwrapped().viewer.window.on_key_release = self._key_release

        self._cycle()

    def _increment_selected_agent_index(self):
        self.current_agent_index += 1
        if self.current_agent_index == self.n_agents:
            self.current_agent_index = 0

    def _cycle(self):
        while True:
            active_agent_index = self.current_agent_index

            action_list = [0] * self.n_agents
            action_list[active_agent_index] = self.u
            obs, rew, done, info = self.env.step(action_list)

            obs[active_agent_index] = np.around(
                obs[active_agent_index].cpu().tolist(), decimals=2
            )
            len_obs = len(obs[active_agent_index])
            message = f"\t\t{obs[active_agent_index][len_obs//2:]}"
            self._write_values(self.text_idx, message)
            message = f"Obs: {obs[active_agent_index][:len_obs//2]}"
            self._write_values(self.text_idx + 1, message)

            message = f"Rew: {round(rew[active_agent_index],3)}"
            self._write_values(self.text_idx + 2, message)

            message = f"Done: {done}"
            self._write_values(self.text_idx + 3, message)

            message = (
                f"Selected: {self.env.unwrapped().agents[active_agent_index].name}"
            )
            self._write_values(self.text_idx + 4, message)

            self.env.render()

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

        if k == key.LEFT:
            self.u = 1
        elif k == key.RIGHT:
            self.u = 2
        elif k == key.UP:
            self.u = 4
        elif k == key.DOWN:
            self.u = 3
        elif k == key.TAB:
            self._increment_selected_agent_index()

    def _key_release(self, k, mod):
        from pyglet.window import key

        if k == key.LEFT or k == key.RIGHT or k == key.UP or k == key.DOWN:
            self.u = 0


def render_interactively(scenario_name: str, **kwargs):
    """
    Use this script to interactively play with scenarios

    You can change agent by pressing TAB
    You can move agents with the arrow keys
    """
    InteractiveEnv(
        GymWrapper(
            make_env(
                scenario_name=scenario_name,
                num_envs=1,
                device="cpu",
                continuous_actions=False,
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
    # You can move agents with the arrow keys

    scenario_name = "waterfall"

    # Scenario specific variables
    n_agents = 4

    render_interactively(scenario_name, n_agents=n_agents)
