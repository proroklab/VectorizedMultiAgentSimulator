import torch

from vmas.simulator.dynamics.common import Dynamics


class Composite(Dynamics):

    def __init__(self, *dynamics):
        super().__init__()
        self.dynamics = dynamics

    # Same as Dyanmics.agent, copied just to allow overriding the setter below
    @property
    def agent(self):
        if self._agent is None:
            raise ValueError(
                "You need to add the dynamics to an agent during construction before accessing its properties"
            )
        return self._agent

    @agent.setter
    def agent(self, value):
        if self._agent is not None:
            raise ValueError("Agent in dynamics has already been set")
        self._agent = value
        # Also set the agent in all the sub-dynamics
        for dynamics in self.dynamics:
            dynamics.agent = value

    @property
    def action_nvec(self):
        return torch.cat([dynamics.action_nvec for dynamics in self.dynamics], dim=0)

    def process_action(self):
        # Since we can't pass the action directly to each sub-dynamics, we
        # temporarily set it to each sub-action and call process_action on
        # the right sub-dynamics; then we restore the original action.
        actions = self.agent.action.u
        a = 0
        for dynamics in self.dynamics:
            b = a + dynamics.needed_action_size
            dynamics.agent.action.u = actions[:, a:b]
            dynamics.process_action()
            a = b
        self.agent.action.u = actions