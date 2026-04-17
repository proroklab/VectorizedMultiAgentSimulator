#  Copyright (c) ProrokLab.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import pytest
import torch

from vmas import make_env


class TestRoadTraffic:
    def setup_env(self, n_envs, device="cpu", **kwargs) -> None:
        self.env = make_env(
            scenario="road_traffic",
            num_envs=n_envs,
            device=device,
            continuous_actions=True,
            **kwargs,
        )
        self.env.seed(0)

    def _seed_buffer(self, device):
        """Seed initial_state_buffer with a real state and force it to always be used."""
        scenario = self.env.scenario
        buf = scenario.initial_state_buffer
        buf.add(scenario.state_buffer.get_latest(n=1)[0])
        buf.probability_use_recording = torch.tensor(1.0, device=device)

    @pytest.mark.parametrize("map_type", ["1", "2"])
    def test_map_type_runs(self, map_type, n_envs=4, n_steps=10):
        self.setup_env(n_envs=n_envs, map_type=map_type)
        self.env.reset()
        for _ in range(n_steps):
            actions = [
                torch.zeros(n_envs, agent.action.action_size)
                for agent in self.env.agents
            ]
            obs, rews, dones, _ = self.env.step(actions)
            if dones.any():
                for env_index, done in enumerate(dones):
                    if done:
                        self.env.reset_at(env_index)

    def test_map_type_2_reset_uses_buffer(self, n_envs=4):
        self.setup_env(n_envs=n_envs, map_type="2")
        self.env.reset()
        actions = [
            torch.zeros(n_envs, agent.action.action_size)
            for agent in self.env.agents
        ]
        self.env.step(actions)
        self._seed_buffer(device="cpu")
        self.env.reset_at(0)

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="GPU required to reproduce road_traffic map_type=2 device bugs",
    )
    def test_gpu_map_type_2_rand_device(self, n_envs=4):
        self.setup_env(n_envs=n_envs, device="cuda", map_type="2")
        self.env.reset()
        actions = [
            torch.zeros(n_envs, agent.action.action_size, device="cuda")
            for agent in self.env.agents
        ]
        self.env.step(actions)
        self._seed_buffer(device="cuda")
        self.env.reset_at(0)
