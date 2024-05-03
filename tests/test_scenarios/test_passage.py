#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import torch

from vmas import make_env


class TestPassage:
    def setup_env(
        self,
        n_envs,
        **kwargs,
    ) -> None:
        self.n_passages = kwargs.get("n_passages", 4)

        self.continuous_actions = True
        self.env = make_env(
            scenario="passage",
            num_envs=n_envs,
            device="cpu",
            continuous_actions=self.continuous_actions,
            # Environment specific variables
            **kwargs,
        )
        self.env.seed(0)

    def test_heuristic(self, n_envs=4):
        self.setup_env(n_passages=1, shared_reward=True, n_envs=4)

        obs = self.env.reset()
        agent_switched = torch.full((5, n_envs), False)
        all_done = torch.full((n_envs,), False)
        while not all_done.all():
            actions = []

            for i in range(5):
                obs_agent = obs[i]
                dist_to_passage = obs_agent[:, 6:8]
                dist_to_goal = obs_agent[:, 4:6]
                dist_to_passage_is_close = (
                    torch.linalg.vector_norm(dist_to_passage, dim=1) <= 0.025
                )

                action_agent = torch.clamp(
                    2 * dist_to_passage,
                    min=-self.env.agents[i].u_range,
                    max=self.env.agents[i].u_range,
                )
                agent_switched[i] += dist_to_passage_is_close
                action_agent[agent_switched[i]] = torch.clamp(
                    2 * dist_to_goal,
                    min=-self.env.agents[i].u_range,
                    max=self.env.agents[i].u_range,
                )[agent_switched[i]]

                actions.append(action_agent)

            obs, new_rews, dones, _ = self.env.step(actions)

            if dones.any():
                all_done += dones
                for env_index, done in enumerate(dones):
                    if done:
                        agent_switched[:, env_index] = False
                        self.env.reset_at(env_index)
