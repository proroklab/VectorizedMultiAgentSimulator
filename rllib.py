#  Copyright (c) 2022. Matteo Bettini
#  All rights reserved.

import os
import platform
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import ray
from ray import tune
from ray.rllib import RolloutWorker, BaseEnv, Policy
from ray.rllib.agents import DefaultCallbacks, MultiCallbacks
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.evaluation import Episode
from ray.rllib.utils.typing import PolicyID
from ray.tune import register_env
from ray.tune.integration.wandb import WandbLoggerCallback

import wandb
from make_env import make_env


def env_creator(config: Dict):
    env = make_env(
        scenario_name=config["scenario_name"],
        num_envs=config["num_envs"],
        device=config["device"],
        continuous_actions=config["continuous_actions"],
        rllib_wrapped=True,
        max_steps=config["max_steps"],
    )
    return env


if not ray.is_initialized():
    ray.init()
    print("Ray init!")
register_env("waterfall", lambda config: env_creator(config))


class RenderingCallbacks(DefaultCallbacks):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frames = []

    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        episode: Episode,
        **kwargs,
    ) -> None:
        self.frames.append(base_env.vector_env.try_render_at(mode="rgb_array"))

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Episode,
        **kwargs,
    ) -> None:
        print(self.frames)
        vid = np.transpose(self.frames, (0, 3, 1, 2))
        episode.media["rendering"] = wandb.Video(vid, fps=10, format="mp4")
        self.frames = []


scratch_dir = (
    Path("/Users/Matteo/scratch/")
    if platform.system() == "Darwin"
    else Path("/local/scratch/mb2389/")
)


def train():
    tune.run(
        PPOTrainer,
        stop={"training_iteration": 5000},
        callbacks=[
            WandbLoggerCallback(
                project=f"maps_test",
                api_key_file=f"{scratch_dir}/wandb_api_key_file",
            )
        ],
        config={
            "seed": 0,
            "framework": "torch",
            "env": "waterfall",
            "kl_coeff": 0.01,
            "kl_target": 0.01,
            "lambda": 0.99,
            "clip_param": 0.2,  # 0.3
            "vf_loss_coeff": 1,
            "vf_clip_param": float("inf"),
            "entropy_coeff": 0,  # 0.01,
            "train_batch_size": 10000,
            "rollout_fragment_length": 200,
            "sgd_minibatch_size": 2000,
            "num_sgd_iter": 32,
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "num_workers": 5,
            "num_gpus_per_worker": 0,
            "num_envs_per_worker": 32,
            "lr": 5e-5,
            "gamma": 0.999,
            "use_gae": True,
            "use_critic": True,
            "batch_mode": "truncate_episodes",
            "env_config": {
                "device": "cpu",
                "num_envs": 32,
                "scenario_name": "waterfall",
                "continuous_actions": True,
                "max_steps": 100,
            },
            "evaluation_interval": 2,
            "evaluation_duration": 1,
            "evaluation_num_workers": 1,
            "evaluation_parallel_to_training": True,
            "evaluation_config": {
                "num_envs_per_worker": 1,
                "callbacks": MultiCallbacks(
                    [
                        RenderingCallbacks,
                    ]
                ),
            },
        },
    )


if __name__ == "__main__":

    train()
