#  Copyright (c) 2022. Matteo Bettini
#  All rights reserved.
import maps
import torch
from maps.scenarios import balance

if __name__ == "__main__":
    n_envs = 300
    n_steps = 200
    env = maps.make_env(
        scenario_name="give_way",
        num_envs=n_envs,
        device="cpu",
        continuous_actions=True,
        rllib_wrapped=False,
        max_steps=n_steps,
    )
    policy = balance.HeuristicPolicy(True)

    obs = env.reset()
    rews = None
    total_reward = torch.zeros(n_envs)

    for j in range(n_steps):
        actions = []
        idx = 0

        if (obs[1][:, :1] < 0).all():
            action_1 = torch.tensor([0.5, -1]).repeat(n_envs, 1)
        else:
            action_1 = torch.tensor([0.5, 1]).repeat(n_envs, 1)
        action_2 = torch.tensor([-0.4, 0]).repeat(n_envs, 1)
        actions += [action_1, action_2]

        obs, new_rews, dones, _ = env.step(actions)
        env.render()
        new_rews = torch.stack(new_rews, dim=1).mean(dim=1)
        total_reward += new_rews
        if dones.any():
            for env_index, done in enumerate(dones):
                if done:
                    env.reset_at(env_index)
    print(total_reward.mean())
    print(total_reward.std())
