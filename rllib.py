#  Copyright (c) 2022. Matteo Bettini
#  All rights reserved.

import os
from typing import Dict, Optional

import numpy as np
import ray
from ray import tune
from ray.rllib import RolloutWorker, BaseEnv, Policy
from ray.rllib.agents import DefaultCallbacks
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.evaluation import Episode
from ray.rllib.utils.typing import PolicyID
from ray.tune import register_env
from ray.tune.integration.wandb import WandbLoggerCallback

import wandb
from maps.make_env import make_env


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
        vid = np.transpose(self.frames, (0, 3, 1, 2))
        episode.media["rendering"] = wandb.Video(vid, fps=30, format="mp4")
        self.frames = []


def train():

    scenario_name = "waterfall"

    continuous_actions = False
    max_steps = 100
    num_vectorized_envs = 32
    num_workers = 7
    maps_device = "cpu"
    RLLIB_NUM_GPUS = int(os.environ.get("RLLIB_NUM_GPUS", "0"))
    num_gpus = min(RLLIB_NUM_GPUS - 0.5, 0)  # Driver GPU
    num_gpus_per_worker = (1 - num_gpus) / num_workers if maps_device == "cuda" else 0

    tune.run(
        PPOTrainer,
        stop={"training_iteration": 5000},
        callbacks=[
            WandbLoggerCallback(
                project=f"maps_test",
                api_key="",
            )
        ],
        config={
            "seed": 0,
            "framework": "torch",
            "env": "waterfall",
            "kl_coeff": 0.01,
            "kl_target": 0.01,
            "lambda": 0.99,
            "clip_param": 0.2,
            "vf_loss_coeff": 1,
            "vf_clip_param": float("inf"),
            "entropy_coeff": 0,
            "train_batch_size": 10000,
            "rollout_fragment_length": 200,
            "sgd_minibatch_size": 2000,
            "num_sgd_iter": 32,
            "num_gpus": num_gpus,
            "num_workers": num_workers,
            "num_gpus_per_worker": num_gpus_per_worker,
            "num_envs_per_worker": num_vectorized_envs,
            "lr": 5e-5,
            "gamma": 0.99,
            "use_gae": True,
            "use_critic": True,
            "batch_mode": "truncate_episodes",
            "env_config": {
                "device": maps_device,
                "num_envs": num_vectorized_envs,
                "scenario_name": scenario_name,
                "continuous_actions": continuous_actions,
                "max_steps": max_steps,
            },
            "evaluation_interval": 50,
            "evaluation_duration": 1,
            "evaluation_num_workers": 1,
            "evaluation_parallel_to_training": True,
            "evaluation_config": {
                "num_envs_per_worker": 1,
                "callbacks": RenderingCallbacks,
                "env_config": {
                    "num_envs": 1,
                },
            },
        },
    )


if __name__ == "__main__":
    train()
