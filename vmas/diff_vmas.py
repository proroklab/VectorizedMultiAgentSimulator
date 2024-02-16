#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import time

import torch
from torch import Tensor

from vmas import make_env
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy


class HeuristicPolicy(BaseHeuristicPolicy):
    def __init__(self, clf_epsilon=0.2, clf_slack=100.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clf_epsilon = clf_epsilon  # Exponential CLF convergence rate
        self.clf_slack = clf_slack  # weights on CLF-QP slack variable

    def compute_action(self, observation: Tensor, u_range: Tensor) -> Tensor:
        """
        QP inputs:
        These values need to computed apriri based on observation before passing into QP

        V: Lyapunov function value
        lfV: Lie derivative of Lyapunov function
        lgV: Lie derivative of Lyapunov function
        CLF_slack: CLF constraint slack variable

        QP outputs:
        u: action
        CLF_slack: CLF constraint slack variable, 0 if CLF constraint is satisfied
        """
        # Install it with: pip install cvxpylayers

        self.n_env = observation.shape[0]
        self.device = observation.device

        goal_pos = -observation[:, 4:6]
        action = goal_pos.clamp(-u_range, u_range)

        return action


if __name__ == "__main__":
    n_steps = 100
    render = True
    save_render = False
    scenario = "navigation_diff"
    num_envs = 1
    device = "cpu"

    env = make_env(
        scenario=scenario,
        num_envs=num_envs,
        device=device,
        continuous_actions=True,
        seed=None,
        # Environment specific variables
        n_agents=2,
        collisions=False,
        shared_rew=True,
        grad_enabled=True,
        observe_all_goals=True,
    )

    init_time = time.time()
    step = 0
    policy = HeuristicPolicy(continuous_action=True)

    optim = torch.optim.Adam([env.scenario.distance_between_goals], 0.1)

    while (env.scenario.distance_between_goals < 0.99).any():
        rew_sum = torch.zeros(num_envs, device=device)
        print(env.scenario.distance_between_goals.sigmoid())
        obs = env.reset()
        for s in range(n_steps):
            step += 1
            # print(f"Step {step}")

            actions = []
            for i, agent in enumerate(env.agents):
                action = policy.compute_action(obs[i], agent.action.u_range)
                actions.append(action)

            obs, rews, dones, info = env.step(actions)
            rew_sum += rews[0]

            if render:
                frame = env.render(mode="human")

        loss = -rew_sum.mean()
        loss.backward()

        optim.step()
        optim.zero_grad()
        env.world.zero_grad()
