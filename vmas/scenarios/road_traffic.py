#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import pathlib
import time
import xml.etree.ElementTree as ET  # For reading the map data
from typing import Dict

import torch
from torch import Tensor

from vmas import render_interactively
from vmas.simulator.core import Agent, Box, World

from vmas.simulator.dynamics.kinematic_bicycle import KinematicBicycle
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils


class Scenario(BaseScenario):
    """
    This scenario originally comes from the paper "Xu et al. - 2024 - A Sample Efficient and Generalizable Multi-Agent Reinforcement Learning Framework
    for Motion Planning" (http://dx.doi.org/10.13140/RG.2.2.24505.17769, see also its GitHub repo https://github.com/cas-lab-munich/generalizable-marl/tree/1.0.0),
    which aims to design an MARL framework with efficient observation design to enable fast training and to empower agents the ability to generalize
    to unseen scenarios.

    Six observation design strategies are proposed in the paper. They correspond to six parameters in this file, and their default
    values are True. Setting them to False will impair the observation efficiency in the evaluation conducted in the paper.
        - is_ego_view: Whether to use ego view (otherwise bird view)
        - is_apply_mask: Whether to mask distant agents
        - is_observe_distance_to_agents: Whether to observe the distance to other agents
        - is_observe_distance_to_boundaries: Whether to observe the distance to labelet boundaries (otherwise the points on lanelet boundaries)
        - is_observe_distance_to_center_line: Whether to observe the distance to reference path (otherwise None)
        - is_observe_vertices: Whether to observe the vertices of other agents (otherwise center points)

    In addition, there are some commonly used parameters you may want to adjust to suit your case:
        - n_agents: Number of agents
        - dt: Sample time in seconds
        - map_type: One of {'1', '2', '3'}:
                         1: the entire map will be used
                         2: the entire map will be used ; besides, challenging initial state buffer will be recorded and used when resetting the envs (inspired
                         by Kaufmann et al. - Nature 2023 - Champion-level drone racing using deep reinforcement learning)
                         3: a specific part of the map (intersection, merge-in, or merge-out) will be used for each env when making or resetting it. You can control the probability of using each of them by the parameter `scenario_probabilities`. It is an array with three values. The first value corresponds to the probability of using intersection. The second and the third values correspond to merge-in and merge-out, respectively. If you only want to use one specific part of the map for all parallel envs, you can set the other two values to zero. For example, if you want to train a RL policy only for intersection, they can set `scenario_probabilities` to [1.0, 0.0, 0.0].
        - is_partial_observation: Whether to enable partial observation (to model partially observable MDP)
        - n_nearing_agents_observed: Number of nearing agents to be observed (consider limited sensor range)

        is_testing_mode: Testing mode is designed to test the learned policy.
                         In non-testing mode, once a collision occurs, all agents will be reset with random initial states.
                         To ensure these initial states are feasible, the initial positions are conservatively large (1.2*diagonalLengthOfAgent).
                         This ensures agents are initially safe and avoids putting agents in an immediate dangerous situation at the beginning of a new scenario.
                         During testing, only colliding agents will be reset, without changing the states of other agents, who are possibly interacting with other agents.
                         This may allow for more effective testing.

    For other parameters, see the class Parameter defined in this file.
    """

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.init_params(batch_dim, device, **kwargs)
        world = self.init_world(batch_dim, device)
        self.init_agents(world)
        return world

    def init_params(self, batch_dim, device, **kwargs):
        # Geometry
        self.world_x_dim = kwargs.pop(
            "world_x_dim", 4.5
        )  # The x-dimension of the world in [m]
        self.world_y_dim = kwargs.pop(
            "world_y_dim", 4.0
        )  # The y-dimension of the world in [m]
        self.agent_width = kwargs.pop(
            "agent_width", 0.08
        )  # The width of the agent in [m]
        self.agent_length = kwargs.pop(
            "agent_length", 0.16
        )  # The length of the agent in [m]
        self.l_f = kwargs.pop("l_f", self.agent_length / 2)  # Front wheelbase in [m]
        self.l_r = kwargs.pop(
            "l_r", self.agent_length - self.l_f
        )  # Rear wheelbase in [m]
        lane_width = kwargs.pop(
            "lane_width", 0.15
        )  # The (rough) width of each lane in [m]

        # Reward
        r_p_normalizer = (
            100  # This parameter normalizes rewards and penalties to [-1, 1].
        )
        # This is useful for RL algorithms with an actor-critic architecture where the critic's
        # output is limited to [-1, 1] (e.g., due to tanh activation function).

        reward_progress = (
            kwargs.pop("reward_progress", 10) / r_p_normalizer
        )  # Reward for moving along reference paths
        reward_vel = (
            kwargs.pop("reward_vel", 5) / r_p_normalizer
        )  # Reward for moving in high velocities.
        reward_reach_goal = (
            kwargs.pop("reward_reach_goal", 0) / r_p_normalizer
        )  # Goal-reaching reward

        # Penalty
        threshold_deviate_from_ref_path = kwargs.pop(
            "threshold_deviate_from_ref_path", (lane_width - self.agent_width) / 2
        )  # Use for penalizing of deviating from reference path

        threshold_reach_goal = kwargs.pop(
            "threshold_reach_goal", self.agent_width / 2
        )  # Threshold less than which agents are considered at their goal positions

        threshold_change_steering = kwargs.pop(
            "threshold_change_steering", 10
        )  # Threshold above which agents will be penalized for changing steering too quick [degree]

        threshold_near_boundary_high = kwargs.pop(
            "threshold_near_boundary_high", (lane_width - self.agent_width) / 2 * 0.9
        )  # Threshold beneath which agents will started be
        # Penalized for being too close to lanelet boundaries
        threshold_near_boundary_low = kwargs.pop(
            "threshold_near_boundary_low", 0
        )  # Threshold above which agents will be penalized for being too close to lanelet boundaries

        threshold_near_other_agents_c2c_high = kwargs.pop(
            "threshold_near_other_agents_c2c_high", self.agent_length + self.agent_width
        )  # Threshold beneath which agents will started be
        # Penalized for being too close to other agents (for center-to-center distance)
        threshold_near_other_agents_c2c_low = kwargs.pop(
            "threshold_near_other_agents_c2c_low",
            (self.agent_length + self.agent_width) / 2,
        )  # Threshold above which agents will be penalized (for center-to-center distance,
        # If a c2c distance is less than the half of the agent width, they are colliding, which will be penalized by another penalty)

        threshold_no_reward_if_too_close_to_boundaries = kwargs.pop(
            "threshold_no_reward_if_too_close_to_boundaries", self.agent_width / 10
        )
        threshold_no_reward_if_too_close_to_other_agents = kwargs.pop(
            "threshold_no_reward_if_too_close_to_other_agents", self.agent_width / 6
        )

        # Visualization
        self.resolution_factor = kwargs.pop("resolution_factor", 200)  # Default 200

        # Reference path
        sample_interval_ref_path = kwargs.pop(
            "sample_interval_ref_path", 2
        )  # Integer, sample interval from the long-term reference path for the short-term reference paths
        max_ref_path_points = kwargs.pop(
            "max_ref_path_points", 200
        )  # The estimated maximum points on the reference path

        noise_level = kwargs.pop(
            "noise_level", 0.2 * self.agent_width
        )  # Noise will be generated by the standary normal distribution. This parameter controls the noise level

        n_stored_steps = kwargs.pop(
            "n_stored_steps",
            5,  # The number of steps to store (include the current step). At least one
        )
        n_observed_steps = kwargs.pop(
            "n_observed_steps", 1
        )  # The number of steps to observe (include the current step). At least one, and at most `n_stored_steps`

        self.render_origin = kwargs.pop(
            "render_origin", [self.world_x_dim / 2, self.world_y_dim / 2]
        )

        self.viewer_size = kwargs.pop(
            "viewer_size",
            (
                int(self.world_x_dim * self.resolution_factor),
                int(self.world_y_dim * self.resolution_factor),
            ),
        )

        self.max_steering_angle = kwargs.pop(
            "max_steering_angle",
            torch.deg2rad(torch.tensor(35, device=device, dtype=torch.float32)),
        )
        self.max_speed = kwargs.pop("max_speed", 1.0)

        self.viewer_zoom = kwargs.pop("viewer_zoom", 1.44)

        parameters = Parameters(
            n_agents=kwargs.pop("n_agents", 20),
            is_partial_observation=kwargs.pop("is_partial_observation", True),
            is_testing_mode=kwargs.pop("is_testing_mode", False),
            is_visualize_short_term_path=kwargs.pop(
                "is_visualize_short_term_path", True
            ),
            map_type=kwargs.pop("map_type", "1"),
            n_nearing_agents_observed=kwargs.pop("n_nearing_agents_observed", 2),
            is_real_time_rendering=kwargs.pop("is_real_time_rendering", False),
            n_points_short_term=kwargs.pop("n_points_short_term", 3),
            dt=kwargs.pop("dt", 0.05),
            is_ego_view=kwargs.pop("is_ego_view", True),
            is_apply_mask=kwargs.pop("is_apply_mask", True),
            is_observe_vertices=kwargs.pop("is_observe_vertices", True),
            is_observe_distance_to_agents=kwargs.pop(
                "is_observe_distance_to_agents", True
            ),
            is_observe_distance_to_boundaries=kwargs.pop(
                "is_observe_distance_to_boundaries", True
            ),
            is_observe_distance_to_center_line=kwargs.pop(
                "is_observe_distance_to_center_line", True
            ),
            scenario_probabilities=kwargs.pop(
                "scenario_probabilities", [1.0, 0.0, 0.0]
            ),  # Probabilities of training agents in intersection, merge-in, and merge-out scenario
            is_add_noise=kwargs.pop("is_add_noise", True),
            is_observe_ref_path_other_agents=kwargs.pop(
                "is_observe_ref_path_other_agents", False
            ),
            is_visualize_extra_info=kwargs.pop("is_visualize_extra_info", False),
            render_title=kwargs.pop(
                "render_title",
                "Multi-Agent Reinforcement Learning for Road Traffic (CPM Lab Scenario)",
            ),
            n_steps_stored=kwargs.pop("n_steps_stored", 10),
            n_steps_before_recording=kwargs.pop("n_steps_before_recording", 10),
            n_points_nearing_boundary=kwargs.pop("n_points_nearing_boundary", 5),
        )
        self.parameters = kwargs.pop("parameters", parameters)

        # Ensure parameters meet simulation requirements
        if self.parameters.map_type == "3":
            if (
                self.parameters.scenario_probabilities[1] != 0
                or self.parameters.scenario_probabilities[2] != 0
            ):
                if self.parameters.n_agents > 5:
                    raise ValueError(
                        "For map_type '3', if the second or third value of scenario_probabilities is not zero, a maximum of 5 agents are allowed, as only a merge-in or a merge-out will be used."
                    )
            else:
                if self.parameters.n_agents > 10:
                    raise ValueError(
                        "For map_type '3', if only the first value of scenario_probabilities is not zero, a maximum of 10 agents are allowed, as only an intersection will be used."
                    )

        if self.parameters.n_nearing_agents_observed >= self.parameters.n_agents:
            raise ValueError("n_nearing_agents_observed must be less than n_agents")

        self.n_agents = self.parameters.n_agents

        # Timer for the first env
        self.timer = Timer(
            start=time.time(),
            end=0,
            step=torch.zeros(
                batch_dim, device=device, dtype=torch.int32
            ),  # Each environment has its own time step
            step_begin=time.time(),
            render_begin=0,
        )

        # Get map data
        map_file_path = kwargs.pop("map_file_path", None)
        if map_file_path is None:
            map_file_path = str(
                pathlib.Path(__file__).parent.parent
                / "scenarios_data"
                / "road_traffic"
                / "road_traffic_cpm_lab.xml"
            )
        self.map_data = get_map_data(map_file_path, device=device)
        # Long-term reference path
        (
            reference_paths_all,
            reference_paths_intersection,
            reference_paths_merge_in,
            reference_paths_merge_out,
        ) = get_reference_paths(self.map_data)

        # Determine the maximum number of points on the reference path
        if self.parameters.map_type in ("1", "2"):
            # Train on the entire map
            max_ref_path_points = (
                max([ref_p["center_line"].shape[0] for ref_p in reference_paths_all])
                + self.parameters.n_points_short_term * sample_interval_ref_path
                + 2
            )  # Append a smaller buffer
        else:
            # Train on a partial map (intersection)
            max_ref_path_points = (
                max(
                    [
                        ref_p["center_line"].shape[0]
                        for ref_p in reference_paths_intersection
                        + reference_paths_merge_in
                        + reference_paths_merge_out
                    ]
                )
                + self.parameters.n_points_short_term * sample_interval_ref_path
                + 2
            )  # Append a smaller buffer

        # Get all reference paths
        self.ref_paths_map_related = ReferencePathsMapRelated(
            long_term_all=reference_paths_all,
            long_term_intersection=reference_paths_intersection,
            long_term_merge_in=reference_paths_merge_in,
            long_term_merge_out=reference_paths_merge_out,
            point_extended_all=torch.zeros(
                (
                    len(reference_paths_all),
                    self.parameters.n_points_short_term * sample_interval_ref_path,
                    2,
                ),
                device=device,
                dtype=torch.float32,
            ),  # Not interesting, may be useful in the future
            point_extended_intersection=torch.zeros(
                (
                    len(reference_paths_intersection),
                    self.parameters.n_points_short_term * sample_interval_ref_path,
                    2,
                ),
                device=device,
                dtype=torch.float32,
            ),
            point_extended_merge_in=torch.zeros(
                (
                    len(reference_paths_merge_in),
                    self.parameters.n_points_short_term * sample_interval_ref_path,
                    2,
                ),
                device=device,
                dtype=torch.float32,
            ),
            point_extended_merge_out=torch.zeros(
                (
                    len(reference_paths_merge_out),
                    self.parameters.n_points_short_term * sample_interval_ref_path,
                    2,
                ),
                device=device,
                dtype=torch.float32,
            ),
            sample_interval=torch.tensor(
                sample_interval_ref_path, device=device, dtype=torch.int32
            ),
        )

        # Extended the reference path by several points along the last vector of the center line
        idx_broadcasting_entend = torch.arange(
            1,
            self.parameters.n_points_short_term * sample_interval_ref_path + 1,
            device=device,
            dtype=torch.int32,
        ).unsqueeze(1)
        for idx, i_path in enumerate(reference_paths_all):
            center_line_i = i_path["center_line"]
            direction = center_line_i[-1] - center_line_i[-2]
            self.ref_paths_map_related.point_extended_all[idx, :] = (
                center_line_i[-1] + idx_broadcasting_entend * direction
            )
        for idx, i_path in enumerate(reference_paths_intersection):
            center_line_i = i_path["center_line"]
            direction = center_line_i[-1] - center_line_i[-2]
            self.ref_paths_map_related.point_extended_intersection[idx, :] = (
                center_line_i[-1] + idx_broadcasting_entend * direction
            )
        for idx, i_path in enumerate(reference_paths_merge_in):
            center_line_i = i_path["center_line"]
            direction = center_line_i[-1] - center_line_i[-2]
            self.ref_paths_map_related.point_extended_merge_in[idx, :] = (
                center_line_i[-1] + idx_broadcasting_entend * direction
            )
        for idx, i_path in enumerate(reference_paths_merge_out):
            center_line_i = i_path["center_line"]
            direction = center_line_i[-1] - center_line_i[-2]
            self.ref_paths_map_related.point_extended_merge_out[idx, :] = (
                center_line_i[-1] + idx_broadcasting_entend * direction
            )

        # Initialize agent-specific reference paths, which will be determined in `reset_world_at` function
        self.ref_paths_agent_related = ReferencePathsAgentRelated(
            long_term=torch.zeros(
                (batch_dim, self.n_agents, max_ref_path_points, 2),
                device=device,
                dtype=torch.float32,
            ),  # Long-term reference paths of agents
            long_term_vec_normalized=torch.zeros(
                (batch_dim, self.n_agents, max_ref_path_points, 2),
                device=device,
                dtype=torch.float32,
            ),
            left_boundary=torch.zeros(
                (batch_dim, self.n_agents, max_ref_path_points, 2),
                device=device,
                dtype=torch.float32,
            ),
            right_boundary=torch.zeros(
                (batch_dim, self.n_agents, max_ref_path_points, 2),
                device=device,
                dtype=torch.float32,
            ),
            entry=torch.zeros(
                (batch_dim, self.n_agents, 2, 2), device=device, dtype=torch.float32
            ),
            exit=torch.zeros(
                (batch_dim, self.n_agents, 2, 2), device=device, dtype=torch.float32
            ),
            is_loop=torch.zeros(
                (batch_dim, self.n_agents), device=device, dtype=torch.bool
            ),
            n_points_long_term=torch.zeros(
                (batch_dim, self.n_agents), device=device, dtype=torch.int32
            ),
            n_points_left_b=torch.zeros(
                (batch_dim, self.n_agents), device=device, dtype=torch.int32
            ),
            n_points_right_b=torch.zeros(
                (batch_dim, self.n_agents), device=device, dtype=torch.int32
            ),
            short_term=torch.zeros(
                (batch_dim, self.n_agents, self.parameters.n_points_short_term, 2),
                device=device,
                dtype=torch.float32,
            ),  # Short-term reference path
            short_term_indices=torch.zeros(
                (batch_dim, self.n_agents, self.parameters.n_points_short_term),
                device=device,
                dtype=torch.int32,
            ),
            n_points_nearing_boundary=torch.tensor(
                self.parameters.n_points_nearing_boundary,
                device=device,
                dtype=torch.int32,
            ),
            nearing_points_left_boundary=torch.zeros(
                (
                    batch_dim,
                    self.n_agents,
                    self.parameters.n_points_nearing_boundary,
                    2,
                ),
                device=device,
                dtype=torch.float32,
            ),  # Nearing left boundary
            nearing_points_right_boundary=torch.zeros(
                (
                    batch_dim,
                    self.n_agents,
                    self.parameters.n_points_nearing_boundary,
                    2,
                ),
                device=device,
                dtype=torch.float32,
            ),  # Nearing right boundary
            scenario_id=torch.zeros(
                (batch_dim, self.n_agents), device=device, dtype=torch.int32
            ),  # Which scenarios agents are (1 for intersection, 2 for merge-in, 3 for merge-out)
            path_id=torch.zeros(
                (batch_dim, self.n_agents), device=device, dtype=torch.int32
            ),  # Which paths agents are
            point_id=torch.zeros(
                (batch_dim, self.n_agents), device=device, dtype=torch.int32
            ),  # Which points agents are
        )

        # The shape of each agent is considered a rectangle with 4 vertices.
        # The first vertex is repeated at the end to close the shape.
        self.vertices = torch.zeros(
            (batch_dim, self.n_agents, 5, 2), device=device, dtype=torch.float32
        )

        weighting_ref_directions = torch.linspace(
            1,
            0.2,
            steps=self.parameters.n_points_short_term,
            device=device,
            dtype=torch.float32,
        )
        weighting_ref_directions /= weighting_ref_directions.sum()
        self.rewards = Rewards(
            progress=torch.tensor(reward_progress, device=device, dtype=torch.float32),
            weighting_ref_directions=weighting_ref_directions,  # Progress in the weighted directions (directions indicating by
            # closer short-term reference points have higher weights)
            higth_v=torch.tensor(reward_vel, device=device, dtype=torch.float32),
            reach_goal=torch.tensor(
                reward_reach_goal, device=device, dtype=torch.float32
            ),
        )
        self.rew = torch.zeros(batch_dim, device=device, dtype=torch.float32)

        self.penalties = Penalties(
            deviate_from_ref_path=torch.tensor(
                -2 / 100, device=device, dtype=torch.float32
            ),
            weighting_deviate_from_ref_path=self.map_data["mean_lane_width"] / 2,
            near_boundary=torch.tensor(-20 / 100, device=device, dtype=torch.float32),
            near_other_agents=torch.tensor(
                -20 / 100, device=device, dtype=torch.float32
            ),
            collide_with_agents=torch.tensor(
                -100 / 100, device=device, dtype=torch.float32
            ),
            collide_with_boundaries=torch.tensor(
                -100 / 100, device=device, dtype=torch.float32
            ),
            change_steering=torch.tensor(-2 / 100, device=device, dtype=torch.float32),
            time=torch.tensor(5 / 100, device=device, dtype=torch.float32),
        )

        self.observations = Observations(
            is_partial=torch.tensor(
                self.parameters.is_partial_observation, device=device, dtype=torch.bool
            ),
            n_nearing_agents=torch.tensor(
                self.parameters.n_nearing_agents_observed,
                device=device,
                dtype=torch.int32,
            ),
            noise_level=torch.tensor(noise_level, device=device, dtype=torch.float32),
            n_stored_steps=torch.tensor(
                n_stored_steps, device=device, dtype=torch.int32
            ),
            n_observed_steps=torch.tensor(
                n_observed_steps, device=device, dtype=torch.int32
            ),
            nearing_agents_indices=torch.zeros(
                (batch_dim, self.n_agents, self.parameters.n_nearing_agents_observed),
                device=device,
                dtype=torch.int32,
            ),
        )
        if self.parameters.is_ego_view:
            self.observations.past_pos = CircularBuffer(
                torch.zeros(
                    (n_stored_steps, batch_dim, self.n_agents, self.n_agents, 2),
                    device=device,
                    dtype=torch.float32,
                )
            )
            self.observations.past_rot = CircularBuffer(
                torch.zeros(
                    (n_stored_steps, batch_dim, self.n_agents, self.n_agents),
                    device=device,
                    dtype=torch.float32,
                )
            )
            self.observations.past_vertices = CircularBuffer(
                torch.zeros(
                    (n_stored_steps, batch_dim, self.n_agents, self.n_agents, 4, 2),
                    device=device,
                    dtype=torch.float32,
                )
            )
            self.observations.past_vel = CircularBuffer(
                torch.zeros(
                    (n_stored_steps, batch_dim, self.n_agents, self.n_agents, 2),
                    device=device,
                    dtype=torch.float32,
                )
            )
            self.observations.past_short_term_ref_points = CircularBuffer(
                torch.zeros(
                    (
                        n_stored_steps,
                        batch_dim,
                        self.n_agents,
                        self.n_agents,
                        self.parameters.n_points_short_term,
                        2,
                    ),
                    device=device,
                    dtype=torch.float32,
                )
            )
            self.observations.past_left_boundary = CircularBuffer(
                torch.zeros(
                    (
                        n_stored_steps,
                        batch_dim,
                        self.n_agents,
                        self.n_agents,
                        self.parameters.n_points_nearing_boundary,
                        2,
                    ),
                    device=device,
                    dtype=torch.float32,
                )
            )
            self.observations.past_right_boundary = CircularBuffer(
                torch.zeros(
                    (
                        n_stored_steps,
                        batch_dim,
                        self.n_agents,
                        self.n_agents,
                        self.parameters.n_points_nearing_boundary,
                        2,
                    ),
                    device=device,
                    dtype=torch.float32,
                )
            )
        else:
            # Bird view
            self.observations.past_pos = CircularBuffer(
                torch.zeros(
                    (n_stored_steps, batch_dim, self.n_agents, 2),
                    device=device,
                    dtype=torch.float32,
                )
            )
            self.observations.past_rot = CircularBuffer(
                torch.zeros(
                    (n_stored_steps, batch_dim, self.n_agents),
                    device=device,
                    dtype=torch.float32,
                )
            )
            self.observations.past_vertices = CircularBuffer(
                torch.zeros(
                    (n_stored_steps, batch_dim, self.n_agents, 4, 2),
                    device=device,
                    dtype=torch.float32,
                )
            )
            self.observations.past_vel = CircularBuffer(
                torch.zeros(
                    (n_stored_steps, batch_dim, self.n_agents, 2),
                    device=device,
                    dtype=torch.float32,
                )
            )
            self.observations.past_short_term_ref_points = CircularBuffer(
                torch.zeros(
                    (
                        n_stored_steps,
                        batch_dim,
                        self.n_agents,
                        self.parameters.n_points_short_term,
                        2,
                    ),
                    device=device,
                    dtype=torch.float32,
                )
            )
            self.observations.past_left_boundary = CircularBuffer(
                torch.zeros(
                    (
                        n_stored_steps,
                        batch_dim,
                        self.n_agents,
                        self.parameters.n_points_nearing_boundary,
                        2,
                    ),
                    device=device,
                    dtype=torch.float32,
                )
            )
            self.observations.past_right_boundary = CircularBuffer(
                torch.zeros(
                    (
                        n_stored_steps,
                        batch_dim,
                        self.n_agents,
                        self.parameters.n_points_nearing_boundary,
                        2,
                    ),
                    device=device,
                    dtype=torch.float32,
                )
            )

        self.observations.past_action_vel = CircularBuffer(
            torch.zeros(
                (n_stored_steps, batch_dim, self.n_agents),
                device=device,
                dtype=torch.float32,
            )
        )
        self.observations.past_action_steering = CircularBuffer(
            torch.zeros(
                (n_stored_steps, batch_dim, self.n_agents),
                device=device,
                dtype=torch.float32,
            )
        )
        self.observations.past_distance_to_ref_path = CircularBuffer(
            torch.zeros(
                (n_stored_steps, batch_dim, self.n_agents),
                device=device,
                dtype=torch.float32,
            )
        )
        self.observations.past_distance_to_boundaries = CircularBuffer(
            torch.zeros(
                (n_stored_steps, batch_dim, self.n_agents),
                device=device,
                dtype=torch.float32,
            )
        )
        self.observations.past_distance_to_left_boundary = CircularBuffer(
            torch.zeros(
                (n_stored_steps, batch_dim, self.n_agents),
                device=device,
                dtype=torch.float32,
            )
        )
        self.observations.past_distance_to_right_boundary = CircularBuffer(
            torch.zeros(
                (n_stored_steps, batch_dim, self.n_agents),
                device=device,
                dtype=torch.float32,
            )
        )
        self.observations.past_distance_to_agents = CircularBuffer(
            torch.zeros(
                (n_stored_steps, batch_dim, self.n_agents, self.n_agents),
                device=device,
                dtype=torch.float32,
            )
        )

        self.normalizers = Normalizers(
            pos=torch.tensor(
                [self.agent_length * 10, self.agent_length * 10],
                device=device,
                dtype=torch.float32,
            ),
            pos_world=torch.tensor(
                [self.world_x_dim, self.world_y_dim], device=device, dtype=torch.float32
            ),
            v=torch.tensor(self.max_speed, device=device, dtype=torch.float32),
            rot=torch.tensor(2 * torch.pi, device=device, dtype=torch.float32),
            action_steering=self.max_steering_angle,
            action_vel=torch.tensor(self.max_speed, device=device, dtype=torch.float32),
            distance_lanelet=torch.tensor(
                lane_width * 3, device=device, dtype=torch.float32
            ),
            distance_ref=torch.tensor(
                lane_width * 3, device=device, dtype=torch.float32
            ),
            distance_agent=torch.tensor(
                self.agent_length * 10, device=device, dtype=torch.float32
            ),
        )

        self.distances = Distances(
            agents=torch.zeros(
                batch_dim, self.n_agents, self.n_agents, dtype=torch.float32
            ),
            left_boundaries=torch.zeros(
                (batch_dim, self.n_agents, 1 + 4), device=device, dtype=torch.float32
            ),  # The first entry for the center, the last 4 entries for the four vertices
            right_boundaries=torch.zeros(
                (batch_dim, self.n_agents, 1 + 4), device=device, dtype=torch.float32
            ),
            boundaries=torch.zeros(
                (batch_dim, self.n_agents), device=device, dtype=torch.float32
            ),
            ref_paths=torch.zeros(
                (batch_dim, self.n_agents), device=device, dtype=torch.float32
            ),
            closest_point_on_ref_path=torch.zeros(
                (batch_dim, self.n_agents), device=device, dtype=torch.int32
            ),
            closest_point_on_left_b=torch.zeros(
                (batch_dim, self.n_agents), device=device, dtype=torch.int32
            ),
            closest_point_on_right_b=torch.zeros(
                (batch_dim, self.n_agents), device=device, dtype=torch.int32
            ),
        )

        self.thresholds = Thresholds(
            reach_goal=torch.tensor(
                threshold_reach_goal, device=device, dtype=torch.float32
            ),
            deviate_from_ref_path=torch.tensor(
                threshold_deviate_from_ref_path, device=device, dtype=torch.float32
            ),
            near_boundary_low=torch.tensor(
                threshold_near_boundary_low, device=device, dtype=torch.float32
            ),
            near_boundary_high=torch.tensor(
                threshold_near_boundary_high, device=device, dtype=torch.float32
            ),
            near_other_agents_low=torch.tensor(
                threshold_near_other_agents_c2c_low, device=device, dtype=torch.float32
            ),
            near_other_agents_high=torch.tensor(
                threshold_near_other_agents_c2c_high, device=device, dtype=torch.float32
            ),
            change_steering=torch.tensor(
                threshold_change_steering, device=device, dtype=torch.float32
            ).deg2rad(),
            no_reward_if_too_close_to_boundaries=torch.tensor(
                threshold_no_reward_if_too_close_to_boundaries,
                device=device,
                dtype=torch.float32,
            ),
            no_reward_if_too_close_to_other_agents=torch.tensor(
                threshold_no_reward_if_too_close_to_other_agents,
                device=device,
                dtype=torch.float32,
            ),
            distance_mask_agents=self.normalizers.pos[0],
        )

        self.constants = Constants(
            env_idx_broadcasting=torch.arange(
                batch_dim, device=device, dtype=torch.int32
            ).unsqueeze(-1),
            empty_action_vel=torch.zeros(
                (batch_dim, self.n_agents), device=device, dtype=torch.float32
            ),
            empty_action_steering=torch.zeros(
                (batch_dim, self.n_agents), device=device, dtype=torch.float32
            ),
            mask_pos=torch.tensor(1, device=device, dtype=torch.float32),
            mask_zero=torch.tensor(0, device=device, dtype=torch.float32),
            mask_one=torch.tensor(1, device=device, dtype=torch.float32),
            reset_agent_min_distance=torch.tensor(
                (self.l_f + self.l_r) ** 2 + self.agent_width**2,
                device=device,
                dtype=torch.float32,
            ).sqrt()
            * 1.2,
        )

        # Initialize collision matrix
        self.collisions = Collisions(
            with_agents=torch.zeros(
                (batch_dim, self.n_agents, self.n_agents),
                device=device,
                dtype=torch.bool,
            ),
            with_lanelets=torch.zeros(
                (batch_dim, self.n_agents), device=device, dtype=torch.bool
            ),
            with_entry_segments=torch.zeros(
                (batch_dim, self.n_agents), device=device, dtype=torch.bool
            ),
            with_exit_segments=torch.zeros(
                (batch_dim, self.n_agents), device=device, dtype=torch.bool
            ),
        )

        self.initial_state_buffer = (
            InitialStateBuffer(  # Used only when "map_type == '3'"
                probability_record=torch.tensor(
                    1.0, device=device, dtype=torch.float32
                ),
                probability_use_recording=torch.tensor(
                    kwargs.pop("probability_use_recording", 0.2),
                    device=device,
                    dtype=torch.float32,
                ),
                buffer=torch.zeros(
                    (100, self.n_agents, 8), device=device, dtype=torch.float32
                ),  # [pos_x, pos_y, rot, vel_x, vel_y, scenario_id, path_id, point_id]
            )
        )

        ScenarioUtils.check_kwargs_consumed(kwargs)

        self.state_buffer = StateBuffer(
            buffer=torch.zeros(
                (self.parameters.n_steps_before_recording, batch_dim, self.n_agents, 8),
                device=device,
                dtype=torch.float32,
            ),  # [pos_x, pos_y, rot, vel_x, vel_y, scenario_id, path_id, point_id],
        )

    def init_world(self, batch_dim: int, device: torch.device):
        # Make world
        world = World(
            batch_dim,
            device,
            x_semidim=self.world_x_dim,
            y_semidim=self.world_y_dim,
            dt=self.parameters.dt,
        )
        return world

    def init_agents(self, world, *kwargs):
        # Create agents
        for i in range(self.n_agents):
            agent = Agent(
                name=f"agent_{i}",
                shape=Box(length=self.l_f + self.l_r, width=self.agent_width),
                color=tuple(
                    torch.rand(3, device=world.device, dtype=torch.float32).tolist()
                ),
                collide=False,
                render_action=False,
                u_range=[
                    self.max_speed,
                    self.max_steering_angle,
                ],  # Control command serves as velocity command
                u_multiplier=[1, 1],
                max_speed=self.max_speed,
                dynamics=KinematicBicycle(  # Use the kinematic bicycle model for each agent
                    world,
                    width=self.agent_width,
                    l_f=self.l_f,
                    l_r=self.l_r,
                    max_steering_angle=self.max_steering_angle,
                    integration="rk4",  # one of {"euler", "rk4"}
                ),
            )
            world.add_agent(agent)

    def reset_world_at(self, env_index: int = None, agent_index: int = None):
        """
        This function resets the world at the specified env_index and the specified agent_index.
        If env_index is given as None, the majority part of computation will be done in a vectorized manner.

        Args:
        :param env_index: index of the environment to reset. If None a vectorized reset should be performed
        :param agent_index: index of the agent to reset. If None all agents in the specified environment will be reset.
        """
        agents = self.world.agents

        is_reset_single_agent = agent_index is not None

        for env_i in (
            [env_index] if env_index is not None else range(self.world.batch_dim)
        ):
            # Begining of a new simulation (only record for the first env)
            if env_i == 0:
                self.timer.start = time.time()
                self.timer.step_begin = time.time()
                self.timer.end = 0

            if not is_reset_single_agent:
                # Each time step of a simulation
                self.timer.step[env_i] = 0

            ref_paths_scenario, extended_points = self.reset_scenario_related_ref_paths(
                env_i, is_reset_single_agent, agent_index
            )

            # If map_type is 2, there is a certain probability of using initial state buffer to reset the agents
            if (
                (self.parameters.map_type == "2")
                and (
                    torch.rand(1) < self.initial_state_buffer.probability_use_recording
                )
                and (self.initial_state_buffer.valid_size >= 1)
            ):
                is_use_state_buffer = True
                initial_state = self.initial_state_buffer.get_random()
                self.ref_paths_agent_related.scenario_id[env_i] = initial_state[
                    :, self.initial_state_buffer.idx_scenario
                ]  # Update
                self.ref_paths_agent_related.path_id[env_i] = initial_state[
                    :, self.initial_state_buffer.idx_path
                ]  # Update
                self.ref_paths_agent_related.point_id[env_i] = initial_state[
                    :, self.initial_state_buffer.idx_point
                ]  # Update
            else:
                is_use_state_buffer = False
                initial_state = None

            # Reset initial states of the agents and agent-related reference paths
            for i_agent in (
                range(self.n_agents)
                if not is_reset_single_agent
                else agent_index.unsqueeze(0)
            ):
                ref_path, path_id = self.reset_init_state(
                    env_i,
                    i_agent,
                    is_reset_single_agent,
                    is_use_state_buffer,
                    initial_state,
                    ref_paths_scenario,
                    agents,
                )

                self.reset_agent_related_ref_path(
                    env_i, i_agent, ref_path, path_id, extended_points
                )

            # The operations below can be done for all envs in parallel
            if env_index is None:
                if env_i == (self.world.batch_dim - 1):
                    env_j = slice(None)  # `slice(None)` is equivalent to `:`
                else:
                    continue
            else:
                env_j = env_i

            for i_agent in (
                range(self.n_agents)
                if not is_reset_single_agent
                else agent_index.unsqueeze(0)
            ):

                self.reset_init_distances_and_short_term_ref_path(
                    env_j, i_agent, agents
                )

            # Compute mutual distances between agents
            mutual_distances = get_distances_between_agents(
                self=self, is_set_diagonal=True
            )
            # Reset mutual distances of all envs
            self.distances.agents[env_j, :, :] = mutual_distances[env_j, :, :]

            # Reset the collision matrix
            self.collisions.with_agents[env_j, :, :] = False
            self.collisions.with_lanelets[env_j, :] = False
            self.collisions.with_entry_segments[env_j, :] = False
            self.collisions.with_exit_segments[env_j, :] = False

        # Reset the state buffer
        self.state_buffer.reset()
        state_add = torch.cat(
            (
                torch.stack([a.state.pos for a in agents], dim=1),
                torch.stack([a.state.rot for a in agents], dim=1),
                torch.stack([a.state.vel for a in agents], dim=1),
                self.ref_paths_agent_related.scenario_id[:].unsqueeze(-1),
                self.ref_paths_agent_related.path_id[:].unsqueeze(-1),
                self.ref_paths_agent_related.point_id[:].unsqueeze(-1),
            ),
            dim=-1,
        )
        self.state_buffer.add(state_add)  # Add new state

    def reset_scenario_related_ref_paths(
        self, env_i, is_reset_single_agent, agent_index
    ):
        """
        Resets scenario-related reference paths and scenario IDs for the specified environment and agents.

        This function determines and sets the long-term reference paths based on the current map_type.
        If `is_reset_single_agent` is true, the current paths for the specified agent will be kept.

        Args:
            env_i (int): The index of the environment to reset.
            is_reset_single_agent (bool): Flag indicating whether only a single agent is being reset.
            agent_index (int or None): The index of the agent to reset. If None, all agents in
                                    the specified environment are reset.

        Returns:
            - ref_paths_scenario (list): The list of reference paths for the current scenario.
            - extended_points (tensor): [numOfRefPaths, numExtendedPoints, 2] The extended points for the current scenario.
        """
        # Get the center line and boundaries of the long-term reference path
        if self.parameters.map_type in {"1", "2"}:
            ref_paths_scenario = self.ref_paths_map_related.long_term_all
            extended_points = self.ref_paths_map_related.point_extended_all
            self.ref_paths_agent_related.scenario_id[env_i, :] = 0
        else:
            if is_reset_single_agent:
                scenario_id = self.ref_paths_agent_related.scenario_id[
                    env_i, agent_index
                ]  # Keep the same scenario
            else:
                scenario_id = (
                    torch.multinomial(
                        torch.tensor(
                            self.parameters.scenario_probabilities,
                            device=self.world.device,
                            dtype=torch.float32,
                        ),
                        1,
                        replacement=True,
                    ).item()
                    + 1
                )  # A random interger {1, 2, 3}. 0 for the whole map, 1 for intersection, 2 for merge-in, 3 for merge-out scenario
                self.ref_paths_agent_related.scenario_id[env_i, :] = scenario_id
            if scenario_id == 1:
                # Intersection scenario
                ref_paths_scenario = self.ref_paths_map_related.long_term_intersection
                extended_points = self.ref_paths_map_related.point_extended_intersection
            elif scenario_id == 2:
                # Merge-in scenario
                ref_paths_scenario = self.ref_paths_map_related.long_term_merge_in
                extended_points = self.ref_paths_map_related.point_extended_merge_in
            elif scenario_id == 3:
                # Merge-out scenario
                ref_paths_scenario = self.ref_paths_map_related.long_term_merge_out
                extended_points = self.ref_paths_map_related.point_extended_merge_out
        return ref_paths_scenario, extended_points

    def reset_init_state(
        self,
        env_i,
        i_agent,
        is_reset_single_agent,
        is_use_state_buffer,
        initial_state,
        ref_paths_scenario,
        agents,
    ):
        """
        This function resets the initial position, rotation, and velocity for an agent based on the provided
        initial state buffer if it is used. Otherwise, it randomly generates initial states ensuring they
        are feasible and do not collide with other agents.
        """
        if is_use_state_buffer:
            path_id = initial_state[i_agent, self.initial_state_buffer.idx_path].int()
            ref_path = ref_paths_scenario[path_id]

            agents[i_agent].set_pos(initial_state[i_agent, 0:2], batch_index=env_i)
            agents[i_agent].set_rot(initial_state[i_agent, 2], batch_index=env_i)
            agents[i_agent].set_vel(initial_state[i_agent, 3:5], batch_index=env_i)
        else:
            is_feasible_initial_position_found = False
            # Ramdomly generate initial states for each agent
            while not is_feasible_initial_position_found:
                path_id = torch.randint(
                    0, len(ref_paths_scenario), (1,)
                ).item()  # Select randomly a path
                self.ref_paths_agent_related.path_id[env_i, i_agent] = path_id  # Update
                ref_path = ref_paths_scenario[path_id]

                num_points = ref_path["center_line"].shape[0]
                if (self.parameters.scenario_probabilities[1] == 0) & (
                    self.parameters.scenario_probabilities[2] == 0
                ):  # Train only in intersection scenario
                    random_point_id = torch.randint(6, int(num_points / 2), (1,)).item()
                else:
                    random_point_id = torch.randint(3, num_points - 5, (1,)).item()
                self.ref_paths_agent_related.point_id[
                    env_i, i_agent
                ] = random_point_id  # Update
                position_start = ref_path["center_line"][random_point_id]
                agents[i_agent].set_pos(position_start, batch_index=env_i)

                # Check if the initial position is feasible
                if not is_reset_single_agent:
                    if i_agent == 0:
                        # The initial position of the first agent is always feasible
                        is_feasible_initial_position_found = True
                        continue
                    else:
                        positions = torch.stack(
                            [
                                self.world.agents[i].state.pos[env_i]
                                for i in range(i_agent + 1)
                            ]
                        )
                else:
                    # Check if the initial position of the agent to be reset is collision-free with other agents
                    positions = torch.stack(
                        [
                            self.world.agents[i].state.pos[env_i]
                            for i in range(self.n_agents)
                        ]
                    )

                diff_sq = (
                    positions[i_agent, :] - positions
                ) ** 2  # Calculate pairwise squared differences in positions
                initial_mutual_distances_sq = torch.sum(diff_sq, dim=-1)
                initial_mutual_distances_sq[i_agent] = (
                    torch.max(initial_mutual_distances_sq) + 1
                )  # Set self-to-self distance to a sufficiently high value
                min_distance_sq = torch.min(initial_mutual_distances_sq)

                is_feasible_initial_position_found = min_distance_sq >= (
                    self.constants.reset_agent_min_distance**2
                )

            rot_start = ref_path["center_line_yaw"][random_point_id]
            vel_start_abs = (
                torch.rand(1, dtype=torch.float32, device=self.world.device)
                * agents[i_agent].max_speed
            )  # Random initial velocity
            vel_start = torch.hstack(
                [
                    vel_start_abs * torch.cos(rot_start),
                    vel_start_abs * torch.sin(rot_start),
                ]
            )

            agents[i_agent].set_rot(rot_start, batch_index=env_i)
            agents[i_agent].set_vel(vel_start, batch_index=env_i)

            return ref_path, path_id

    def reset_agent_related_ref_path(
        self, env_i, i_agent, ref_path, path_id, extended_points
    ):
        """
        This function resets the agent-related reference paths and updates various related attributes
        for a specified agent in an environment.
        """
        # Long-term reference paths for agents
        n_points_long_term = ref_path["center_line"].shape[0]

        self.ref_paths_agent_related.long_term[
            env_i, i_agent, 0:n_points_long_term, :
        ] = ref_path["center_line"]
        self.ref_paths_agent_related.long_term[
            env_i,
            i_agent,
            n_points_long_term : (
                n_points_long_term
                + self.parameters.n_points_short_term
                * self.ref_paths_map_related.sample_interval
            ),
            :,
        ] = extended_points[path_id, :, :]
        self.ref_paths_agent_related.long_term[
            env_i,
            i_agent,
            (
                n_points_long_term
                + self.parameters.n_points_short_term
                * self.ref_paths_map_related.sample_interval
            ) :,
            :,
        ] = extended_points[path_id, -1, :]
        self.ref_paths_agent_related.n_points_long_term[
            env_i, i_agent
        ] = n_points_long_term

        self.ref_paths_agent_related.long_term_vec_normalized[
            env_i, i_agent, 0 : n_points_long_term - 1, :
        ] = ref_path["center_line_vec_normalized"]
        self.ref_paths_agent_related.long_term_vec_normalized[
            env_i,
            i_agent,
            (n_points_long_term - 1) : (
                n_points_long_term
                - 1
                + self.parameters.n_points_short_term
                * self.ref_paths_map_related.sample_interval
            ),
            :,
        ] = ref_path["center_line_vec_normalized"][-1, :]

        n_points_left_b = ref_path["left_boundary_shared"].shape[0]
        self.ref_paths_agent_related.left_boundary[
            env_i, i_agent, 0:n_points_left_b, :
        ] = ref_path["left_boundary_shared"]
        self.ref_paths_agent_related.left_boundary[
            env_i, i_agent, n_points_left_b:, :
        ] = ref_path["left_boundary_shared"][-1, :]
        self.ref_paths_agent_related.n_points_left_b[env_i, i_agent] = n_points_left_b

        n_points_right_b = ref_path["right_boundary_shared"].shape[0]
        self.ref_paths_agent_related.right_boundary[
            env_i, i_agent, 0:n_points_right_b, :
        ] = ref_path["right_boundary_shared"]
        self.ref_paths_agent_related.right_boundary[
            env_i, i_agent, n_points_right_b:, :
        ] = ref_path["right_boundary_shared"][-1, :]
        self.ref_paths_agent_related.n_points_right_b[env_i, i_agent] = n_points_right_b

        self.ref_paths_agent_related.entry[env_i, i_agent, 0, :] = ref_path[
            "left_boundary_shared"
        ][0, :]
        self.ref_paths_agent_related.entry[env_i, i_agent, 1, :] = ref_path[
            "right_boundary_shared"
        ][0, :]

        self.ref_paths_agent_related.exit[env_i, i_agent, 0, :] = ref_path[
            "left_boundary_shared"
        ][-1, :]
        self.ref_paths_agent_related.exit[env_i, i_agent, 1, :] = ref_path[
            "right_boundary_shared"
        ][-1, :]

        self.ref_paths_agent_related.is_loop[env_i, i_agent] = ref_path["is_loop"]

    def reset_init_distances_and_short_term_ref_path(self, env_j, i_agent, agents):
        """
        This function calculates the distances from the agent's center of gravity (CG) to its reference path and boundaries,
        and computes the positions of the four vertices of the agent. It also determines the short-term reference paths
        for the agent based on the long-term reference paths and the agent's current position.
        """
        # Distance from the center of gravity (CG) of the agent to its reference path
        (
            self.distances.ref_paths[env_j, i_agent],
            self.distances.closest_point_on_ref_path[env_j, i_agent],
        ) = get_perpendicular_distances(
            point=agents[i_agent].state.pos[env_j, :],
            polyline=self.ref_paths_agent_related.long_term[env_j, i_agent],
            n_points_long_term=self.ref_paths_agent_related.n_points_long_term[
                env_j, i_agent
            ],
        )
        # Distances from CG to left boundary
        (
            center_2_left_b,
            self.distances.closest_point_on_left_b[env_j, i_agent],
        ) = get_perpendicular_distances(
            point=agents[i_agent].state.pos[env_j, :],
            polyline=self.ref_paths_agent_related.left_boundary[env_j, i_agent],
            n_points_long_term=self.ref_paths_agent_related.n_points_left_b[
                env_j, i_agent
            ],
        )
        self.distances.left_boundaries[env_j, i_agent, 0] = center_2_left_b - (
            agents[i_agent].shape.width / 2
        )
        # Distances from CG to right boundary
        (
            center_2_right_b,
            self.distances.closest_point_on_right_b[env_j, i_agent],
        ) = get_perpendicular_distances(
            point=agents[i_agent].state.pos[env_j, :],
            polyline=self.ref_paths_agent_related.right_boundary[env_j, i_agent],
            n_points_long_term=self.ref_paths_agent_related.n_points_right_b[
                env_j, i_agent
            ],
        )
        self.distances.right_boundaries[env_j, i_agent, 0] = center_2_right_b - (
            agents[i_agent].shape.width / 2
        )
        # Calculate the positions of the four vertices of the agents
        self.vertices[env_j, i_agent] = get_rectangle_vertices(
            center=agents[i_agent].state.pos[env_j, :],
            yaw=agents[i_agent].state.rot[env_j, :],
            width=agents[i_agent].shape.width,
            length=agents[i_agent].shape.length,
            is_close_shape=True,
        )
        # Distances from the four vertices of the agent to its left and right lanelet boundary
        for c_i in range(4):
            (
                self.distances.left_boundaries[env_j, i_agent, c_i + 1],
                _,
            ) = get_perpendicular_distances(
                point=self.vertices[env_j, i_agent, c_i, :],
                polyline=self.ref_paths_agent_related.left_boundary[env_j, i_agent],
                n_points_long_term=self.ref_paths_agent_related.n_points_left_b[
                    env_j, i_agent
                ],
            )
            (
                self.distances.right_boundaries[env_j, i_agent, c_i + 1],
                _,
            ) = get_perpendicular_distances(
                point=self.vertices[env_j, i_agent, c_i, :],
                polyline=self.ref_paths_agent_related.right_boundary[env_j, i_agent],
                n_points_long_term=self.ref_paths_agent_related.n_points_right_b[
                    env_j, i_agent
                ],
            )
        # Distance from agent to its left/right lanelet boundary is defined as the minimum distance among five distances (four vertices, CG)
        self.distances.boundaries[env_j, i_agent], _ = torch.min(
            torch.hstack(
                (
                    self.distances.left_boundaries[env_j, i_agent],
                    self.distances.right_boundaries[env_j, i_agent],
                )
            ),
            dim=-1,
        )

        # Get the short-term reference paths
        (
            self.ref_paths_agent_related.short_term[env_j, i_agent],
            _,
        ) = get_short_term_reference_path(
            polyline=self.ref_paths_agent_related.long_term[env_j, i_agent],
            index_closest_point=self.distances.closest_point_on_ref_path[
                env_j, i_agent
            ],
            n_points_to_return=self.parameters.n_points_short_term,
            device=self.world.device,
            is_polyline_a_loop=self.ref_paths_agent_related.is_loop[env_j, i_agent],
            n_points_long_term=self.ref_paths_agent_related.n_points_long_term[
                env_j, i_agent
            ],
            sample_interval=self.ref_paths_map_related.sample_interval,
            n_points_shift=1,
        )

        if not self.parameters.is_observe_distance_to_boundaries:
            # Get nearing points on boundaries
            (
                self.ref_paths_agent_related.nearing_points_left_boundary[
                    env_j, i_agent
                ],
                _,
            ) = get_short_term_reference_path(
                polyline=self.ref_paths_agent_related.left_boundary[env_j, i_agent],
                index_closest_point=self.distances.closest_point_on_left_b[
                    env_j, i_agent
                ],
                n_points_to_return=self.parameters.n_points_nearing_boundary,
                device=self.world.device,
                is_polyline_a_loop=self.ref_paths_agent_related.is_loop[env_j, i_agent],
                n_points_long_term=self.ref_paths_agent_related.n_points_long_term[
                    env_j, i_agent
                ],
                sample_interval=1,
                n_points_shift=1,
            )
            (
                self.ref_paths_agent_related.nearing_points_right_boundary[
                    env_j, i_agent
                ],
                _,
            ) = get_short_term_reference_path(
                polyline=self.ref_paths_agent_related.right_boundary[env_j, i_agent],
                index_closest_point=self.distances.closest_point_on_right_b[
                    env_j, i_agent
                ],
                n_points_to_return=self.parameters.n_points_nearing_boundary,
                device=self.world.device,
                is_polyline_a_loop=self.ref_paths_agent_related.is_loop[env_j, i_agent],
                n_points_long_term=self.ref_paths_agent_related.n_points_long_term[
                    env_j, i_agent
                ],
                sample_interval=1,
                n_points_shift=1,
            )

    def reward(self, agent: Agent):
        """
        Issue rewards for the given agent in all envs.
            Positive Rewards:
                Moving forward (become negative if the projection of the moving direction to its reference path is negative)
                Moving forward with high speed (become negative if the projection of the moving direction to its reference path is negative)
                Reaching goal (optional)

            Negative Rewards (penalties):
                Too close to lane boundaries
                Too close to other agents
                Deviating from reference paths
                Changing steering too quick
                Colliding with other agents
                Colliding with lane boundaries

        Args:
            agent: The agent for which the observation is to be generated.

        Returns:
            A tensor with shape [batch_dim].
        """
        # Initialize
        self.rew[:] = 0

        # Get the index of the current agent
        agent_index = self.world.agents.index(agent)

        # [update] mutual distances between agents, vertices of each agent, and collision matrices
        self.update_state_before_rewarding(agent, agent_index)

        # [reward] forward movement
        latest_state = self.state_buffer.get_latest(n=1)
        move_vec = (agent.state.pos - latest_state[:, agent_index, 0:2]).unsqueeze(
            1
        )  # Vector of the current movement

        ref_points_vecs = self.ref_paths_agent_related.short_term[
            :, agent_index
        ] - latest_state[:, agent_index, 0:2].unsqueeze(
            1
        )  # Vectors from the previous position to the points on the short-term reference path
        move_projected = torch.sum(move_vec * ref_points_vecs, dim=-1)
        move_projected_weighted = torch.matmul(
            move_projected, self.rewards.weighting_ref_directions
        )  # Put more weights on nearing reference points

        reward_movement = (
            move_projected_weighted
            / (agent.max_speed * self.world.dt)
            * self.rewards.progress
        )
        self.rew += reward_movement  # Relative to the maximum possible movement

        # [reward] high velocity
        v_proj = torch.sum(agent.state.vel.unsqueeze(1) * ref_points_vecs, dim=-1).mean(
            -1
        )
        factor_moving_direction = torch.where(
            v_proj > 0, 1, 2
        )  # Get penalty if move in negative direction

        reward_vel = (
            factor_moving_direction * v_proj / agent.max_speed * self.rewards.higth_v
        )
        self.rew += reward_vel

        # [reward] reach goal
        reward_goal = (
            self.collisions.with_exit_segments[:, agent_index] * self.rewards.reach_goal
        )
        self.rew += reward_goal

        # [penalty] close to lanelet boundaries
        penalty_close_to_lanelets = (
            exponential_decreasing_fcn(
                x=self.distances.boundaries[:, agent_index],
                x0=self.thresholds.near_boundary_low,
                x1=self.thresholds.near_boundary_high,
            )
            * self.penalties.near_boundary
        )
        self.rew += penalty_close_to_lanelets

        # [penalty] close to other agents
        mutual_distance_exp_fcn = exponential_decreasing_fcn(
            x=self.distances.agents[:, agent_index, :],
            x0=self.thresholds.near_other_agents_low,
            x1=self.thresholds.near_other_agents_high,
        )
        penalty_close_to_agents = (
            torch.sum(mutual_distance_exp_fcn, dim=1) * self.penalties.near_other_agents
        )
        self.rew += penalty_close_to_agents

        # [penalty] deviating from reference path
        self.rew += (
            self.distances.ref_paths[:, agent_index]
            / self.penalties.weighting_deviate_from_ref_path
            * self.penalties.deviate_from_ref_path
        )

        # [penalty] changing steering too quick
        steering_current = self.observations.past_action_steering.get_latest(n=1)[
            :, agent_index
        ]
        steering_past = self.observations.past_action_steering.get_latest(n=2)[
            :, agent_index
        ]

        steering_change = torch.clamp(
            (steering_current - steering_past).abs() * self.normalizers.action_steering
            - self.thresholds.change_steering,  # Not forget to denormalize
            min=0,
        )
        steering_change_reward_factor = steering_change / (
            2 * agent.u_range[1] - 2 * self.thresholds.change_steering
        )
        penalty_change_steering = (
            steering_change_reward_factor * self.penalties.change_steering
        )
        self.rew += penalty_change_steering

        # [penalty] colliding with other agents
        is_collide_with_agents = self.collisions.with_agents[:, agent_index]
        penalty_collide_other_agents = (
            is_collide_with_agents.any(dim=-1) * self.penalties.collide_with_agents
        )
        self.rew += penalty_collide_other_agents

        # [penalty] colliding with lanelet boundaries
        is_collide_with_lanelets = self.collisions.with_lanelets[:, agent_index]
        penalty_collide_lanelet = (
            is_collide_with_lanelets * self.penalties.collide_with_boundaries
        )
        self.rew += penalty_collide_lanelet

        # [penalty/reward] time
        # Get time reward if moving in positive direction; otherwise get time penalty
        time_reward = (
            torch.where(v_proj > 0, 1, -1)
            * agent.state.vel.norm(dim=-1)
            / agent.max_speed
            * self.penalties.time
        )
        self.rew += time_reward

        # [update] previous positions and short-term reference paths
        self.update_state_after_rewarding(agent_index)

        return self.rew

    def update_state_before_rewarding(self, agent, agent_index):
        """Update some states (such as mutual distances between agents, vertices of each agent, and
        collision matrices) that will be used before rewarding agents.
        """
        if agent_index == 0:  # Avoid repeated computations
            # Timer
            self.timer.step_begin = (
                time.time()
            )  # Set to the current time as the begin of the current time step
            self.timer.step += 1  # Increment step by 1

            # Update distances between agents
            self.distances.agents = get_distances_between_agents(
                self=self, is_set_diagonal=True
            )
            self.collisions.with_agents[:] = False  # Reset
            self.collisions.with_lanelets[:] = False  # Reset
            self.collisions.with_entry_segments[:] = False  # Reset
            self.collisions.with_exit_segments[:] = False  # Reset

            for a_i in range(self.n_agents):
                self.vertices[:, a_i] = get_rectangle_vertices(
                    center=self.world.agents[a_i].state.pos,
                    yaw=self.world.agents[a_i].state.rot,
                    width=self.world.agents[a_i].shape.width,
                    length=self.world.agents[a_i].shape.length,
                    is_close_shape=True,
                )
                # Update the collision matrices
                for a_j in range(a_i + 1, self.n_agents):
                    # Check for collisions between agents using the interX function
                    collision_batch_index = interX(
                        self.vertices[:, a_i], self.vertices[:, a_j], False
                    )
                    self.collisions.with_agents[
                        torch.nonzero(collision_batch_index), a_i, a_j
                    ] = True
                    self.collisions.with_agents[
                        torch.nonzero(collision_batch_index), a_j, a_i
                    ] = True

                # Check for collisions between agents and lanelet boundaries
                collision_with_left_boundary = interX(
                    L1=self.vertices[:, a_i],
                    L2=self.ref_paths_agent_related.left_boundary[:, a_i],
                    is_return_points=False,
                )  # [batch_dim]
                collision_with_right_boundary = interX(
                    L1=self.vertices[:, a_i],
                    L2=self.ref_paths_agent_related.right_boundary[:, a_i],
                    is_return_points=False,
                )  # [batch_dim]
                self.collisions.with_lanelets[
                    (collision_with_left_boundary | collision_with_right_boundary), a_i
                ] = True

                # Check for collisions with entry or exit segments (only need if agents' reference paths are not a loop)
                if not self.ref_paths_agent_related.is_loop[:, a_i].any():
                    self.collisions.with_entry_segments[:, a_i] = interX(
                        L1=self.vertices[:, a_i],
                        L2=self.ref_paths_agent_related.entry[:, a_i],
                        is_return_points=False,
                    )
                    self.collisions.with_exit_segments[:, a_i] = interX(
                        L1=self.vertices[:, a_i],
                        L2=self.ref_paths_agent_related.exit[:, a_i],
                        is_return_points=False,
                    )

        # Distance from the center of gravity (CG) of the agent to its reference path
        (
            self.distances.ref_paths[:, agent_index],
            self.distances.closest_point_on_ref_path[:, agent_index],
        ) = get_perpendicular_distances(
            point=agent.state.pos,
            polyline=self.ref_paths_agent_related.long_term[:, agent_index],
            n_points_long_term=self.ref_paths_agent_related.n_points_long_term[
                :, agent_index
            ],
        )
        # Distances from CG to left boundary
        (
            center_2_left_b,
            self.distances.closest_point_on_left_b[:, agent_index],
        ) = get_perpendicular_distances(
            point=agent.state.pos[:, :],
            polyline=self.ref_paths_agent_related.left_boundary[:, agent_index],
            n_points_long_term=self.ref_paths_agent_related.n_points_left_b[
                :, agent_index
            ],
        )
        self.distances.left_boundaries[:, agent_index, 0] = center_2_left_b - (
            agent.shape.width / 2
        )
        # Distances from CG to right boundary
        (
            center_2_right_b,
            self.distances.closest_point_on_right_b[:, agent_index],
        ) = get_perpendicular_distances(
            point=agent.state.pos[:, :],
            polyline=self.ref_paths_agent_related.right_boundary[:, agent_index],
            n_points_long_term=self.ref_paths_agent_related.n_points_right_b[
                :, agent_index
            ],
        )
        self.distances.right_boundaries[:, agent_index, 0] = center_2_right_b - (
            agent.shape.width / 2
        )
        # Distances from the four vertices of the agent to its left and right lanelet boundary
        for c_i in range(4):
            (
                self.distances.left_boundaries[:, agent_index, c_i + 1],
                _,
            ) = get_perpendicular_distances(
                point=self.vertices[:, agent_index, c_i, :],
                polyline=self.ref_paths_agent_related.left_boundary[:, agent_index],
                n_points_long_term=self.ref_paths_agent_related.n_points_left_b[
                    :, agent_index
                ],
            )
            (
                self.distances.right_boundaries[:, agent_index, c_i + 1],
                _,
            ) = get_perpendicular_distances(
                point=self.vertices[:, agent_index, c_i, :],
                polyline=self.ref_paths_agent_related.right_boundary[:, agent_index],
                n_points_long_term=self.ref_paths_agent_related.n_points_right_b[
                    :, agent_index
                ],
            )
        # Distance from agent to its left/right lanelet boundary is defined as the minimum distance among five distances (four vertices, CG)
        self.distances.boundaries[:, agent_index], _ = torch.min(
            torch.hstack(
                (
                    self.distances.left_boundaries[:, agent_index],
                    self.distances.right_boundaries[:, agent_index],
                )
            ),
            dim=-1,
        )

    def update_state_after_rewarding(self, agent_index):
        """Update some states (such as previous positions and short-term reference paths) after rewarding agents."""
        if agent_index == (self.n_agents - 1):  # Avoid repeated updating
            state_add = torch.cat(
                (
                    torch.stack([a.state.pos for a in self.world.agents], dim=1),
                    torch.stack([a.state.rot for a in self.world.agents], dim=1),
                    torch.stack([a.state.vel for a in self.world.agents], dim=1),
                    self.ref_paths_agent_related.scenario_id[:].unsqueeze(-1),
                    self.ref_paths_agent_related.path_id[:].unsqueeze(-1),
                    self.ref_paths_agent_related.point_id[:].unsqueeze(-1),
                ),
                dim=-1,
            )
            self.state_buffer.add(state_add)

        (
            self.ref_paths_agent_related.short_term[:, agent_index],
            _,
        ) = get_short_term_reference_path(
            polyline=self.ref_paths_agent_related.long_term[:, agent_index],
            index_closest_point=self.distances.closest_point_on_ref_path[
                :, agent_index
            ],
            n_points_to_return=self.parameters.n_points_short_term,
            device=self.world.device,
            is_polyline_a_loop=self.ref_paths_agent_related.is_loop[:, agent_index],
            n_points_long_term=self.ref_paths_agent_related.n_points_long_term[
                :, agent_index
            ],
            sample_interval=self.ref_paths_map_related.sample_interval,
        )

        if not self.parameters.is_observe_distance_to_boundaries:
            # Get nearing points on boundaries
            (
                self.ref_paths_agent_related.nearing_points_left_boundary[
                    :, agent_index
                ],
                _,
            ) = get_short_term_reference_path(
                polyline=self.ref_paths_agent_related.left_boundary[:, agent_index],
                index_closest_point=self.distances.closest_point_on_left_b[
                    :, agent_index
                ],
                n_points_to_return=self.parameters.n_points_nearing_boundary,
                device=self.world.device,
                is_polyline_a_loop=self.ref_paths_agent_related.is_loop[:, agent_index],
                n_points_long_term=self.ref_paths_agent_related.n_points_long_term[
                    :, agent_index
                ],
                sample_interval=1,
                n_points_shift=-2,
            )
            (
                self.ref_paths_agent_related.nearing_points_right_boundary[
                    :, agent_index
                ],
                _,
            ) = get_short_term_reference_path(
                polyline=self.ref_paths_agent_related.right_boundary[:, agent_index],
                index_closest_point=self.distances.closest_point_on_right_b[
                    :, agent_index
                ],
                n_points_to_return=self.parameters.n_points_nearing_boundary,
                device=self.world.device,
                is_polyline_a_loop=self.ref_paths_agent_related.is_loop[:, agent_index],
                n_points_long_term=self.ref_paths_agent_related.n_points_long_term[
                    :, agent_index
                ],
                sample_interval=1,
                n_points_shift=-2,
            )

    def observation(self, agent: Agent):
        """
        Generate an observation for the given agent in all envs.

        Args:
            agent: The agent for which the observation is to be generated.

        Returns:
            The observation for the given agent in all envs, which consists of the observation of this agent itself and possibly the observation of its surrounding agents.
                The observation of this agent itself includes
                    position (in case of using bird view),
                    rotation (in case of using bird view),
                    velocity,
                    short-term reference path,
                    distance to its reference path (optional), and
                    lane boundaries (or distances to them).
                The observation of its surrounding agents includes their
                    vertices (or positions and rotations),
                    velocities,
                    distances to them (optional), and
                    reference paths (optional).
        """
        agent_index = self.world.agents.index(agent)

        self.update_observation_and_normalize(agent, agent_index)

        # Observation of other agents
        obs_other_agents = self.observe_other_agents(agent_index)

        obs_self = self.observe_self(agent_index)

        obs_self.append(obs_other_agents)  # Append the observations of other agents

        obs_all = [o for o in obs_self if o is not None]  # Filter out None values

        obs = torch.hstack(obs_all)  # Convert from list to tensor

        if self.parameters.is_add_noise:
            # Add sensor noise if required
            return obs + (
                self.observations.noise_level
                * torch.rand_like(obs, device=self.world.device, dtype=torch.float32)
            )
        else:
            # Return without sensor noise
            return obs

    def update_observation_and_normalize(self, agent, agent_index):
        """Update observation and normalize them."""
        if agent_index == 0:  # Avoid repeated computations
            positions_global = torch.stack(
                [a.state.pos for a in self.world.agents], dim=0
            ).transpose(0, 1)
            rotations_global = (
                torch.stack([a.state.rot for a in self.world.agents], dim=0)
                .transpose(0, 1)
                .squeeze(-1)
            )
            # Add new observation & normalize
            self.observations.past_distance_to_agents.add(
                self.distances.agents / self.normalizers.distance_lanelet
            )
            self.observations.past_distance_to_ref_path.add(
                self.distances.ref_paths / self.normalizers.distance_lanelet
            )
            self.observations.past_distance_to_left_boundary.add(
                torch.min(self.distances.left_boundaries, dim=-1)[0]
                / self.normalizers.distance_lanelet
            )
            self.observations.past_distance_to_right_boundary.add(
                torch.min(self.distances.right_boundaries, dim=-1)[0]
                / self.normalizers.distance_lanelet
            )
            self.observations.past_distance_to_boundaries.add(
                self.distances.boundaries / self.normalizers.distance_lanelet
            )

            if self.parameters.is_ego_view:
                pos_i_others = torch.zeros(
                    (self.world.batch_dim, self.n_agents, self.n_agents, 2),
                    device=self.world.device,
                    dtype=torch.float32,
                )  # Positions of other agents relative to agent i
                rot_i_others = torch.zeros(
                    (self.world.batch_dim, self.n_agents, self.n_agents),
                    device=self.world.device,
                    dtype=torch.float32,
                )  # Rotations of other agents relative to agent i
                vel_i_others = torch.zeros(
                    (self.world.batch_dim, self.n_agents, self.n_agents, 2),
                    device=self.world.device,
                    dtype=torch.float32,
                )  # Velocities of other agents relative to agent i
                ref_i_others = torch.zeros_like(
                    (self.observations.past_short_term_ref_points.get_latest())
                )  # Reference paths of other agents relative to agent i
                l_b_i_others = torch.zeros_like(
                    (self.observations.past_left_boundary.get_latest())
                )  # Left boundaries of other agents relative to agent i
                r_b_i_others = torch.zeros_like(
                    (self.observations.past_right_boundary.get_latest())
                )  # Right boundaries of other agents relative to agent i
                ver_i_others = torch.zeros_like(
                    (self.observations.past_vertices.get_latest())
                )  # Vertices of other agents relative to agent i

                for a_i in range(self.n_agents):
                    pos_i = self.world.agents[a_i].state.pos
                    rot_i = self.world.agents[a_i].state.rot

                    # Store new observation - position
                    pos_i_others[:, a_i] = transform_from_global_to_local_coordinate(
                        pos_i=pos_i,
                        pos_j=positions_global,
                        rot_i=rot_i,
                    )

                    # Store new observation - rotation
                    rot_i_others[:, a_i] = rotations_global - rot_i

                    for a_j in range(self.n_agents):
                        # Store new observation - velocities
                        rot_rel = rot_i_others[:, a_i, a_j].unsqueeze(1)
                        vel_abs = torch.norm(
                            self.world.agents[a_j].state.vel, dim=1
                        ).unsqueeze(1)
                        vel_i_others[:, a_i, a_j] = torch.hstack(
                            (vel_abs * torch.cos(rot_rel), vel_abs * torch.sin(rot_rel))
                        )

                        # Store new observation - reference paths
                        ref_i_others[
                            :, a_i, a_j
                        ] = transform_from_global_to_local_coordinate(
                            pos_i=pos_i,
                            pos_j=self.ref_paths_agent_related.short_term[:, a_j],
                            rot_i=rot_i,
                        )

                        # Store new observation - left boundary
                        if not self.parameters.is_observe_distance_to_boundaries:
                            l_b_i_others[
                                :, a_i, a_j
                            ] = transform_from_global_to_local_coordinate(
                                pos_i=pos_i,
                                pos_j=self.ref_paths_agent_related.nearing_points_left_boundary[
                                    :, a_j
                                ],
                                rot_i=rot_i,
                            )

                            # Store new observation - right boundary
                            r_b_i_others[
                                :, a_i, a_j
                            ] = transform_from_global_to_local_coordinate(
                                pos_i=pos_i,
                                pos_j=self.ref_paths_agent_related.nearing_points_right_boundary[
                                    :, a_j
                                ],
                                rot_i=rot_i,
                            )

                        # Store new observation - vertices
                        ver_i_others[
                            :, a_i, a_j
                        ] = transform_from_global_to_local_coordinate(
                            pos_i=pos_i,
                            pos_j=self.vertices[:, a_j, 0:4, :],
                            rot_i=rot_i,
                        )

                # Add new observations & normalize
                self.observations.past_pos.add(
                    pos_i_others
                    / (
                        self.normalizers.pos
                        if self.parameters.is_ego_view
                        else self.normalizers.pos_world
                    )
                )
                self.observations.past_rot.add(rot_i_others / self.normalizers.rot)
                self.observations.past_vel.add(vel_i_others / self.normalizers.v)
                self.observations.past_short_term_ref_points.add(
                    ref_i_others
                    / (
                        self.normalizers.pos
                        if self.parameters.is_ego_view
                        else self.normalizers.pos_world
                    )
                )
                self.observations.past_left_boundary.add(
                    l_b_i_others
                    / (
                        self.normalizers.pos
                        if self.parameters.is_ego_view
                        else self.normalizers.pos_world
                    )
                )
                self.observations.past_right_boundary.add(
                    r_b_i_others
                    / (
                        self.normalizers.pos
                        if self.parameters.is_ego_view
                        else self.normalizers.pos_world
                    )
                )
                self.observations.past_vertices.add(
                    ver_i_others
                    / (
                        self.normalizers.pos
                        if self.parameters.is_ego_view
                        else self.normalizers.pos_world
                    )
                )

            else:  # Global coordinate system
                # Store new observations
                self.observations.past_pos.add(
                    positions_global
                    / (
                        self.normalizers.pos
                        if self.parameters.is_ego_view
                        else self.normalizers.pos_world
                    )
                )
                self.observations.past_vel.add(
                    torch.stack([a.state.vel for a in self.world.agents], dim=1)
                    / self.normalizers.v
                )
                self.observations.past_rot.add(
                    rotations_global[:] / self.normalizers.rot
                )
                self.observations.past_vertices.add(
                    self.vertices[:, :, 0:4, :]
                    / (
                        self.normalizers.pos
                        if self.parameters.is_ego_view
                        else self.normalizers.pos_world
                    )
                )
                self.observations.past_short_term_ref_points.add(
                    self.ref_paths_agent_related.short_term[:]
                    / (
                        self.normalizers.pos
                        if self.parameters.is_ego_view
                        else self.normalizers.pos_world
                    )
                )
                self.observations.past_left_boundary.add(
                    self.ref_paths_agent_related.nearing_points_left_boundary
                    / (
                        self.normalizers.pos
                        if self.parameters.is_ego_view
                        else self.normalizers.pos_world
                    )
                )
                self.observations.past_right_boundary.add(
                    self.ref_paths_agent_related.nearing_points_right_boundary
                    / (
                        self.normalizers.pos
                        if self.parameters.is_ego_view
                        else self.normalizers.pos_world
                    )
                )

            # Add new observation - actions & normalize
            if agent.action.u is None:
                self.observations.past_action_vel.add(self.constants.empty_action_vel)
                self.observations.past_action_steering.add(
                    self.constants.empty_action_steering
                )
            else:
                self.observations.past_action_vel.add(
                    torch.stack([a.action.u[:, 0] for a in self.world.agents], dim=1)
                    / self.normalizers.action_vel
                )
                self.observations.past_action_steering.add(
                    torch.stack([a.action.u[:, 1] for a in self.world.agents], dim=1)
                    / self.normalizers.action_steering
                )

    def observe_other_agents(self, agent_index):
        """Observe surrounding agents."""
        if self.observations.is_partial:
            # Each agent observes only a fixed number of nearest agents
            nearing_agents_distances, nearing_agents_indices = torch.topk(
                self.distances.agents[:, agent_index],
                k=self.observations.n_nearing_agents,
                largest=False,
            )

            if self.parameters.is_apply_mask:
                # Nearing agents that are distant will be masked
                mask_nearing_agents_too_far = (
                    nearing_agents_distances >= self.thresholds.distance_mask_agents
                )
            else:
                # Otherwise no agents will be masked
                mask_nearing_agents_too_far = torch.zeros(
                    (self.world.batch_dim, self.parameters.n_nearing_agents_observed),
                    device=self.world.device,
                    dtype=torch.bool,
                )

            indexing_tuple_1 = (
                (self.constants.env_idx_broadcasting,)
                + ((agent_index,) if self.parameters.is_ego_view else ())
                + (nearing_agents_indices,)
            )

            # Positions of nearing agents
            obs_pos_other_agents = self.observations.past_pos.get_latest()[
                indexing_tuple_1
            ]  # [batch_size, n_nearing_agents, 2]
            obs_pos_other_agents[
                mask_nearing_agents_too_far
            ] = self.constants.mask_one  # Position mask

            # Rotations of nearing agents
            obs_rot_other_agents = self.observations.past_rot.get_latest()[
                indexing_tuple_1
            ]  # [batch_size, n_nearing_agents]
            obs_rot_other_agents[
                mask_nearing_agents_too_far
            ] = self.constants.mask_zero  # Rotation mask

            # Velocities of nearing agents
            obs_vel_other_agents = self.observations.past_vel.get_latest()[
                indexing_tuple_1
            ]  # [batch_size, n_nearing_agents]
            obs_vel_other_agents[
                mask_nearing_agents_too_far
            ] = self.constants.mask_zero  # Velocity mask

            # Reference paths of nearing agents
            obs_ref_path_other_agents = (
                self.observations.past_short_term_ref_points.get_latest()[
                    indexing_tuple_1
                ]
            )  # [batch_size, n_nearing_agents, n_points_short_term, 2]
            obs_ref_path_other_agents[
                mask_nearing_agents_too_far
            ] = self.constants.mask_one  # Reference-path mask

            # vertices of nearing agents
            obs_vertices_other_agents = self.observations.past_vertices.get_latest()[
                indexing_tuple_1
            ]  # [batch_size, n_nearing_agents, 4, 2]
            obs_vertices_other_agents[
                mask_nearing_agents_too_far
            ] = self.constants.mask_one  # Reference-path mask

            # Distances to nearing agents
            obs_distance_other_agents = (
                self.observations.past_distance_to_agents.get_latest()[
                    self.constants.env_idx_broadcasting,
                    agent_index,
                    nearing_agents_indices,
                ]
            )  # [batch_size, n_nearing_agents]
            obs_distance_other_agents[
                mask_nearing_agents_too_far
            ] = self.constants.mask_one  # Distance mask

        else:
            obs_pos_other_agents = self.observations.past_pos.get_latest()[
                :, agent_index
            ]  # [batch_size, n_agents, 2]
            obs_rot_other_agents = self.observations.past_rot.get_latest()[
                :, agent_index
            ]  # [batch_size, n_agents, (n_agents)]
            obs_vel_other_agents = self.observations.past_vel.get_latest()[
                :, agent_index
            ]  # [batch_size, n_agents, 2]
            obs_ref_path_other_agents = (
                self.observations.past_short_term_ref_points.get_latest()[
                    :, agent_index
                ]
            )  # [batch_size, n_agents, n_points_short_term, 2]
            obs_vertices_other_agents = self.observations.past_vertices.get_latest()[
                :, agent_index
            ]  # [batch_size, n_agents, 4, 2]
            obs_distance_other_agents = (
                self.observations.past_distance_to_agents.get_latest()[:, agent_index]
            )  # [batch_size, n_agents]
            obs_distance_other_agents[
                :, agent_index
            ] = 0  # Reset self-self distance to zero

        # Flatten the last dimensions to combine all features into a single dimension
        obs_pos_other_agents_flat = obs_pos_other_agents.reshape(
            self.world.batch_dim, self.observations.n_nearing_agents, -1
        )
        obs_rot_other_agents_flat = obs_rot_other_agents.reshape(
            self.world.batch_dim, self.observations.n_nearing_agents, -1
        )
        obs_vel_other_agents_flat = obs_vel_other_agents.reshape(
            self.world.batch_dim, self.observations.n_nearing_agents, -1
        )
        obs_ref_path_other_agents_flat = obs_ref_path_other_agents.reshape(
            self.world.batch_dim, self.observations.n_nearing_agents, -1
        )
        obs_vertices_other_agents_flat = obs_vertices_other_agents.reshape(
            self.world.batch_dim, self.observations.n_nearing_agents, -1
        )
        obs_distance_other_agents_flat = obs_distance_other_agents.reshape(
            self.world.batch_dim, self.observations.n_nearing_agents, -1
        )

        # Observation of other agents
        obs_others_list = [
            obs_vertices_other_agents_flat
            if self.parameters.is_observe_vertices
            else torch.cat(  # [other] vertices
                [
                    obs_pos_other_agents_flat,  # [others] positions
                    obs_rot_other_agents_flat,  # [others] rotations
                ],
                dim=-1,
            ),
            obs_vel_other_agents_flat,  # [others] velocities
            obs_distance_other_agents_flat
            if self.parameters.is_observe_distance_to_agents
            else None,  # [others] mutual distances
            obs_ref_path_other_agents_flat
            if self.parameters.is_observe_ref_path_other_agents
            else None,  # [others] reference paths
        ]
        obs_others_list = [
            o for o in obs_others_list if o is not None
        ]  # Filter out None values
        obs_other_agents = torch.cat(obs_others_list, dim=-1).reshape(
            self.world.batch_dim, -1
        )  # [batch_size, -1]

        return obs_other_agents

    def observe_self(self, agent_index):
        """Observe the given agent itself."""
        indexing_tuple_3 = (
            (self.constants.env_idx_broadcasting,)
            + (agent_index,)
            + ((agent_index,) if self.parameters.is_ego_view else ())
        )
        indexing_tuple_vel = (
            (self.constants.env_idx_broadcasting,)
            + (agent_index,)
            + ((agent_index, 0) if self.parameters.is_ego_view else ())
        )  # In local coordinate system, only the first component is interesting, as the second is always 0

        # Self-observations
        obs_self = [
            None
            if self.parameters.is_ego_view
            else self.observations.past_pos.get_latest()[indexing_tuple_3].reshape(
                self.world.batch_dim, -1
            ),  # [own] position,
            None
            if self.parameters.is_ego_view
            else self.observations.past_rot.get_latest()[indexing_tuple_3].reshape(
                self.world.batch_dim, -1
            ),  # [own] rotation,
            self.observations.past_vel.get_latest()[indexing_tuple_vel].reshape(
                self.world.batch_dim, -1
            ),  # [own] velocity
            self.observations.past_short_term_ref_points.get_latest()[
                indexing_tuple_3
            ].reshape(
                self.world.batch_dim, -1
            ),  # [own] short-term reference path
            self.observations.past_distance_to_ref_path.get_latest()[
                :, agent_index
            ].reshape(self.world.batch_dim, -1)
            if self.parameters.is_observe_distance_to_center_line
            else None,  # [own] distances to reference paths
            self.observations.past_distance_to_left_boundary.get_latest()[
                :, agent_index
            ].reshape(self.world.batch_dim, -1)
            if self.parameters.is_observe_distance_to_boundaries
            else self.observations.past_left_boundary.get_latest()[
                indexing_tuple_3
            ].reshape(
                self.world.batch_dim, -1
            ),  # [own] left boundaries
            self.observations.past_distance_to_right_boundary.get_latest()[
                :, agent_index
            ].reshape(self.world.batch_dim, -1)
            if self.parameters.is_observe_distance_to_boundaries
            else self.observations.past_right_boundary.get_latest()[
                indexing_tuple_3
            ].reshape(
                self.world.batch_dim, -1
            ),  # [own] right boundaries
        ]
        return obs_self

    def done(self):
        """
        This function computes the done flag for each env in a vectorized way.

        Testing mode is designed to test the learned policy. In testing mode, collisions do
        not terminate the current simulation; instead, the colliding agents (not all agents)
        will be reset. Besides, if `map_type` is "3", those agents who leave their entries
        or exits will be reset.
        """
        is_collision_with_agents = self.collisions.with_agents.view(
            self.world.batch_dim, -1
        ).any(
            dim=-1
        )  # [batch_dim]
        is_collision_with_lanelets = self.collisions.with_lanelets.any(dim=-1)

        if self.parameters.map_type == "2":  # Record into the initial state buffer
            if torch.rand(1) > (
                1 - self.initial_state_buffer.probability_record
            ):  # Only a certain probability to record
                for env_collide in torch.where(is_collision_with_agents)[0]:
                    self.initial_state_buffer.add(
                        self.state_buffer.get_latest(n=self.parameters.n_steps_stored)[
                            env_collide
                        ]
                    )

        if self.parameters.is_testing_mode:
            # In testing mode, we do not reset the whole scenario. Instead, we only reset colliding agents.
            is_done = torch.zeros(
                self.world.batch_dim, device=self.world.device, dtype=torch.bool
            )

            # Reset single agent
            agents_reset = (
                self.collisions.with_agents.any(dim=-1)
                | self.collisions.with_lanelets
                | self.collisions.with_entry_segments
                | self.collisions.with_exit_segments
            )
            agents_reset_indices = torch.where(agents_reset)
            for env_idx, agent_idx in zip(
                agents_reset_indices[0], agents_reset_indices[1]
            ):
                self.reset_world_at(env_index=env_idx, agent_index=agent_idx)
        else:
            if self.parameters.map_type == "3":
                is_done = is_collision_with_agents | is_collision_with_lanelets

                # Reset single agnet
                agents_reset = (
                    self.collisions.with_entry_segments
                    | self.collisions.with_exit_segments
                )
                agents_reset_indices = torch.where(agents_reset)
                for env_idx, agent_idx in zip(
                    agents_reset_indices[0], agents_reset_indices[1]
                ):
                    if not is_done[env_idx]:
                        self.reset_world_at(env_index=env_idx, agent_index=agent_idx)

            else:
                is_done = is_collision_with_agents | is_collision_with_lanelets

        return is_done

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        """
        This function computes the info dict for "agent" in a vectorized way
        The returned dict should have a key for each info of interest and the corresponding value should
        be a tensor of shape (n_envs, info_size)

        Implementors can access the world at "self.world"

        To increase performance, tensors created should have the device set, like:
        torch.tensor(..., device=self.world.device)

        :param agent: Agent batch to compute info of
        :return: info: A dict with a key for each info of interest, and a tensor value  of shape (n_envs, info_size)
        """
        agent_index = self.world.agents.index(agent)  # Index of the current agent

        is_action_empty = agent.action.u is None

        is_collision_with_agents = self.collisions.with_agents[:, agent_index].any(
            dim=-1
        )  # [batch_dim]
        is_collision_with_lanelets = self.collisions.with_lanelets.any(dim=-1)

        info = {
            "pos": agent.state.pos / self.normalizers.pos_world,
            "rot": angle_eliminate_two_pi(agent.state.rot) / self.normalizers.rot,
            "vel": agent.state.vel / self.normalizers.v,
            "act_vel": (agent.action.u[:, 0] / self.normalizers.action_vel)
            if not is_action_empty
            else self.constants.empty_action_vel[:, agent_index],
            "act_steer": (agent.action.u[:, 1] / self.normalizers.action_steering)
            if not is_action_empty
            else self.constants.empty_action_steering[:, agent_index],
            "ref": (
                self.ref_paths_agent_related.short_term[:, agent_index]
                / self.normalizers.pos_world
            ).reshape(self.world.batch_dim, -1),
            "distance_ref": self.distances.ref_paths[:, agent_index]
            / self.normalizers.distance_ref,
            "distance_left_b": self.distances.left_boundaries[:, agent_index].min(
                dim=-1
            )[0]
            / self.normalizers.distance_lanelet,
            "distance_right_b": self.distances.right_boundaries[:, agent_index].min(
                dim=-1
            )[0]
            / self.normalizers.distance_lanelet,
            "is_collision_with_agents": is_collision_with_agents,
            "is_collision_with_lanelets": is_collision_with_lanelets,
        }

        return info

    def extra_render(self, env_index: int = 0):
        from vmas.simulator import rendering

        if self.parameters.is_real_time_rendering:
            if self.timer.step[0] == 0:
                pause_duration = 0  # Not sure how long should the simulation be paused at time step 0, so rather 0
            else:
                pause_duration = self.world.dt - (time.time() - self.timer.render_begin)
            if pause_duration > 0:
                time.sleep(pause_duration)

            self.timer.render_begin = time.time()  # Update
        geoms = []

        # Visualize all lanelets
        for i in range(len(self.map_data["lanelets"])):
            lanelet = self.map_data["lanelets"][i]

            geom = rendering.PolyLine(
                v=lanelet["left_boundary"],
                close=False,
            )
            xform = rendering.Transform()
            geom.add_attr(xform)
            geom.set_color(*Color.BLACK.value)
            geoms.append(geom)

            geom = rendering.PolyLine(
                v=lanelet["right_boundary"],
                close=False,
            )
            xform = rendering.Transform()
            geom.add_attr(xform)
            geom.set_color(*Color.BLACK.value)
            geoms.append(geom)

        if self.parameters.is_visualize_extra_info:
            hight_a = -0.10
            hight_b = -0.20
            hight_c = -0.30

            # Title
            geom = rendering.TextLine(
                text=self.parameters.render_title,
                x=0.05 * self.resolution_factor,
                y=(self.world.y_semidim + hight_a) * self.resolution_factor,
                font_size=14,
            )
            xform = rendering.Transform()
            geom.add_attr(xform)
            geoms.append(geom)

            # Time and time step
            geom = rendering.TextLine(
                text=f"t: {self.timer.step[0]*self.parameters.dt:.2f} sec",
                x=0.05 * self.resolution_factor,
                y=(self.world.y_semidim + hight_b) * self.resolution_factor,
                font_size=14,
            )
            xform = rendering.Transform()
            geom.add_attr(xform)
            geoms.append(geom)

            geom = rendering.TextLine(
                text=f"n: {self.timer.step[0]}",
                x=0.05 * self.resolution_factor,
                y=(self.world.y_semidim + hight_c) * self.resolution_factor,
                font_size=14,
            )
            xform = rendering.Transform()
            geom.add_attr(xform)
            geoms.append(geom)

        for agent_i in range(self.n_agents):
            if self.parameters.is_visualize_short_term_path:
                geom = rendering.PolyLine(
                    v=self.ref_paths_agent_related.short_term[env_index, agent_i],
                    close=False,
                )
                xform = rendering.Transform()
                geom.add_attr(xform)
                geom.set_color(*self.world.agents[agent_i].color)
                geoms.append(geom)

                for i_p in self.ref_paths_agent_related.short_term[env_index, agent_i]:
                    circle = rendering.make_circle(radius=0.01, filled=True)
                    xform = rendering.Transform()
                    circle.add_attr(xform)
                    xform.set_translation(i_p[0], i_p[1])
                    circle.set_color(*self.world.agents[agent_i].color)
                    geoms.append(circle)

            # Visualize nearing points on boundaries
            if not self.parameters.is_observe_distance_to_boundaries:
                # Left boundary
                geom = rendering.PolyLine(
                    v=self.ref_paths_agent_related.nearing_points_left_boundary[
                        env_index, agent_i
                    ],
                    close=False,
                )
                xform = rendering.Transform()
                geom.add_attr(xform)
                geom.set_color(*self.world.agents[agent_i].color)
                geoms.append(geom)

                for i_p in self.ref_paths_agent_related.nearing_points_left_boundary[
                    env_index, agent_i
                ]:
                    circle = rendering.make_circle(radius=0.01, filled=True)
                    xform = rendering.Transform()
                    circle.add_attr(xform)
                    xform.set_translation(i_p[0], i_p[1])
                    circle.set_color(*self.world.agents[agent_i].color)
                    geoms.append(circle)

                # Right boundary
                geom = rendering.PolyLine(
                    v=self.ref_paths_agent_related.nearing_points_right_boundary[
                        env_index, agent_i
                    ],
                    close=False,
                )
                xform = rendering.Transform()
                geom.add_attr(xform)
                geom.set_color(*self.world.agents[agent_i].color)
                geoms.append(geom)

                for i_p in self.ref_paths_agent_related.nearing_points_right_boundary[
                    env_index, agent_i
                ]:
                    circle = rendering.make_circle(radius=0.01, filled=True)
                    xform = rendering.Transform()
                    circle.add_attr(xform)
                    xform.set_translation(i_p[0], i_p[1])
                    circle.set_color(*self.world.agents[agent_i].color)
                    geoms.append(circle)

            # Agent IDs
            geom = rendering.TextLine(
                text=f"{agent_i}",
                x=(
                    self.world.agents[agent_i].state.pos[env_index, 0]
                    / self.world.x_semidim
                )
                * self.viewer_size[0],
                y=(
                    self.world.agents[agent_i].state.pos[env_index, 1]
                    / self.world.y_semidim
                )
                * self.viewer_size[1],
                font_size=14,
            )
            xform = rendering.Transform()
            geom.add_attr(xform)
            geoms.append(geom)

            # Lanelet boundaries of agents' reference path
            if self.parameters.is_visualize_lane_boundary:
                if agent_i == 0:
                    # Left boundary
                    geom = rendering.PolyLine(
                        v=self.ref_paths_agent_related.left_boundary[
                            env_index, agent_i
                        ],
                        close=False,
                    )
                    xform = rendering.Transform()
                    geom.add_attr(xform)
                    geom.set_color(*self.world.agents[agent_i].color)
                    geoms.append(geom)
                    # Right boundary
                    geom = rendering.PolyLine(
                        v=self.ref_paths_agent_related.right_boundary[
                            env_index, agent_i
                        ],
                        close=False,
                    )
                    xform = rendering.Transform()
                    geom.add_attr(xform)
                    geom.set_color(*self.world.agents[agent_i].color)
                    geoms.append(geom)
                    # Entry
                    geom = rendering.PolyLine(
                        v=self.ref_paths_agent_related.entry[env_index, agent_i],
                        close=False,
                    )
                    xform = rendering.Transform()
                    geom.add_attr(xform)
                    geom.set_color(*self.world.agents[agent_i].color)
                    geoms.append(geom)
                    # Exit
                    geom = rendering.PolyLine(
                        v=self.ref_paths_agent_related.exit[env_index, agent_i],
                        close=False,
                    )
                    xform = rendering.Transform()
                    geom.add_attr(xform)
                    geom.set_color(*self.world.agents[agent_i].color)
                    geoms.append(geom)

        return geoms


# Utilities


class Parameters:
    """Stores parameter."""

    def __init__(
        self,
        # General parameters
        n_agents: int = 20,  # Number of agents
        dt: float = 0.05,  # Sample time in seconds
        device: str = "cpu",  # Tensor device
        map_type: str = "1",  # One of {'1', '2', '3'}:
        # 1 for the entire map
        # 2 for the entire map with prioritized replay buffer (proposed by Schaul et al. - ICLR 2016 - Prioritized Experience Replay)
        # 3 for the entire map with challenging initial state buffer (inspired
        # by Kaufmann et al. - Nature 2023 - Champion-level drone racing using deep reinforcement learning)
        # 4 for specific scenario (intersection, merge-in, or merge-out)
        is_prb: bool = False,  # Whether to enable prioritized replay buffer
        scenario_probabilities=None,  # Probabilities of training agents in intersection, merge-in, and merge-out scenario
        # Observation
        n_points_short_term: int = 3,  # Number of points that build a short-term reference path
        is_partial_observation: bool = True,  # Whether to enable partial observation (to model partially observable MDP)
        n_nearing_agents_observed: int = 2,  # Number of nearing agents to be observed (consider limited sensor range)
        # Observation design strategies
        is_ego_view: bool = True,  # Whether to use ego view (otherwise bird view)
        is_apply_mask: bool = True,  # Whether to mask distant agents
        is_observe_distance_to_agents: bool = True,  # Whether to observe the distance to other agents
        is_observe_distance_to_boundaries: bool = True,  # Whether to observe the distance to labelet boundaries (otherwise the points on lanelet boundaries)
        is_observe_distance_to_center_line: bool = True,  # Whether to observe the distance to reference path (otherwise None)
        is_observe_vertices: bool = True,  # Whether to observe the vertices of other agents (otherwise center points)
        is_add_noise: bool = True,  # Whether to add noise to observations (to simulate sensor noise)
        is_observe_ref_path_other_agents: bool = False,  # Whether to observe the reference paths of other agents
        # Visu
        is_visualize_short_term_path: bool = True,  # Whether to visualize short-term reference paths
        is_visualize_lane_boundary: bool = False,  # Whether to visualize lane boundary
        is_real_time_rendering: bool = False,  # Simulation will be paused at each time step for a certain duration to enable real-time rendering
        is_visualize_extra_info: bool = True,  # Whether to render extra information such time and time step
        render_title: str = "",  # The title to be rendered
        is_testing_mode: bool = False,  # Testing mode is designed to test the learned policy.
        # In testing mode, collisions do not terminate the current simulation; instead, the colliding agents (not all agents) will be reset.
        # In non-testing mode, once a collision occurs, all agents will be reset.
        n_steps_stored: int = 10,  # Store the states of agents of previous several time steps
        n_steps_before_recording: int = 10,  # The states of agents at time step `current_time_step - n_steps_before_recording` before collisions
        # will be recorded and used later when resetting the envs (used only when map_type = "2")
        n_points_nearing_boundary: int = 5,  # The number of points on nearing boundaries to be observed
    ):

        self.n_agents = n_agents
        self.dt = dt

        self.device = device

        self.map_type = map_type

        # Observation
        self.is_partial_observation = is_partial_observation
        self.n_points_short_term = n_points_short_term
        self.n_nearing_agents_observed = n_nearing_agents_observed
        self.is_observe_distance_to_agents = is_observe_distance_to_agents

        self.is_testing_mode = is_testing_mode
        self.is_visualize_short_term_path = is_visualize_short_term_path
        self.is_visualize_lane_boundary = is_visualize_lane_boundary

        self.is_ego_view = is_ego_view
        self.is_apply_mask = is_apply_mask
        self.is_observe_distance_to_boundaries = is_observe_distance_to_boundaries
        self.is_observe_distance_to_center_line = is_observe_distance_to_center_line
        self.is_observe_vertices = is_observe_vertices
        self.is_add_noise = is_add_noise
        self.is_observe_ref_path_other_agents = is_observe_ref_path_other_agents

        self.is_real_time_rendering = is_real_time_rendering
        self.is_visualize_extra_info = is_visualize_extra_info
        self.render_title = render_title

        self.is_prb = is_prb
        if scenario_probabilities is None:
            scenario_probabilities = [
                1.0,
                0.0,
                0.0,
            ]
        self.scenario_probabilities = scenario_probabilities
        self.n_steps_stored = n_steps_stored
        self.n_steps_before_recording = n_steps_before_recording
        self.n_points_nearing_boundary = n_points_nearing_boundary


class Normalizers:
    """Normalizers for positions, velocities, rotations, etc."""

    def __init__(
        self,
        pos=None,
        pos_world=None,
        v=None,
        rot=None,
        action_steering=None,
        action_vel=None,
        distance_lanelet=None,
        distance_agent=None,
        distance_ref=None,
    ):
        self.pos = pos
        self.pos_world = pos_world
        self.v = v
        self.rot = rot
        self.action_steering = action_steering
        self.action_vel = action_vel
        self.distance_lanelet = distance_lanelet
        self.distance_agent = distance_agent
        self.distance_ref = distance_ref


class Rewards:
    """Rewards for moving forward, moving with high speed, etc."""

    def __init__(
        self,
        progress=None,
        weighting_ref_directions=None,
        higth_v=None,
        reach_goal=None,
        reach_intermediate_goal=None,
    ):
        self.progress = progress
        self.weighting_ref_directions = weighting_ref_directions
        self.higth_v = higth_v
        self.reach_goal = reach_goal
        self.reach_intermediate_goal = reach_intermediate_goal


class Penalties:
    """Penalties for collisions, being too close to other agents or lane boundaries, etc."""

    def __init__(
        self,
        deviate_from_ref_path=None,
        deviate_from_goal=None,
        weighting_deviate_from_ref_path=None,
        near_boundary=None,
        near_other_agents=None,
        collide_with_agents=None,
        collide_with_boundaries=None,
        collide_with_obstacles=None,
        leave_world=None,
        time=None,
        change_steering=None,
    ):
        self.deviate_from_ref_path = (
            deviate_from_ref_path  # Penalty for deviating from reference path
        )
        self.deviate_from_goal = (
            deviate_from_goal  # Penalty for deviating from goal position
        )
        self.weighting_deviate_from_ref_path = weighting_deviate_from_ref_path
        self.near_boundary = (
            near_boundary  # Penalty for being too close to lanelet boundaries
        )
        self.near_other_agents = (
            near_other_agents  # Penalty for being too close to other agents
        )
        self.collide_with_agents = (
            collide_with_agents  # Penalty for colliding with other agents
        )
        self.collide_with_boundaries = (
            collide_with_boundaries  # Penalty for colliding with lanelet boundaries
        )
        self.collide_with_obstacles = (
            collide_with_obstacles  # Penalty for colliding with obstacles
        )
        self.leave_world = leave_world  # Penalty for leaving the world
        self.time = time  # Penalty for losing time
        self.change_steering = (
            change_steering  # Penalty for changing steering direction
        )


class Thresholds:
    """Different thresholds, such as starting from which distance agents are deemed being too close to other agents."""

    def __init__(
        self,
        deviate_from_ref_path=None,
        near_boundary_low=None,
        near_boundary_high=None,
        near_other_agents_low=None,
        near_other_agents_high=None,
        reach_goal=None,
        reach_intermediate_goal=None,
        change_steering=None,
        no_reward_if_too_close_to_boundaries=None,
        no_reward_if_too_close_to_other_agents=None,
        distance_mask_agents=None,
    ):
        self.deviate_from_ref_path = deviate_from_ref_path
        self.near_boundary_low = near_boundary_low
        self.near_boundary_high = near_boundary_high
        self.near_other_agents_low = near_other_agents_low
        self.near_other_agents_high = near_other_agents_high
        self.reach_goal = reach_goal  # Threshold less than which agents are considered at their goal positions
        self.reach_intermediate_goal = reach_intermediate_goal  # Threshold less than which agents are considered at their intermediate goal positions
        self.change_steering = change_steering
        self.no_reward_if_too_close_to_boundaries = no_reward_if_too_close_to_boundaries  # Agents get no reward if they are too close to lanelet boundaries
        self.no_reward_if_too_close_to_other_agents = no_reward_if_too_close_to_other_agents  # Agents get no reward if they are too close to other agents
        self.distance_mask_agents = (
            distance_mask_agents  # Threshold above which nearing agents will be masked
        )


class ReferencePathsMapRelated:
    def __init__(
        self,
        long_term_all=None,
        long_term_intersection=None,
        long_term_merge_in=None,
        long_term_merge_out=None,
        point_extended_all=None,
        point_extended_intersection=None,
        point_extended_merge_in=None,
        point_extended_merge_out=None,
        long_term_vecs_normalized=None,
        point_extended=None,
        sample_interval=None,
    ):
        self.long_term_all = long_term_all  # All long-term reference paths
        self.long_term_intersection = long_term_intersection  # Long-term reference paths for the intersection scenario
        self.long_term_merge_in = (
            long_term_merge_in  # Long-term reference paths for the mergin in scenario
        )
        self.long_term_merge_out = (
            long_term_merge_out  # Long-term reference paths for the merge out scenario
        )
        self.point_extended_all = point_extended_all  # Extend the long-term reference paths by one point at the end
        self.point_extended_intersection = point_extended_intersection  # Extend the long-term reference paths by one point at the end
        self.point_extended_merge_in = point_extended_merge_in  # Extend the long-term reference paths by one point at the end
        self.point_extended_merge_out = point_extended_merge_out  # Extend the long-term reference paths by one point at the end

        self.long_term_vecs_normalized = long_term_vecs_normalized  # Normalized vectors of the line segments on the long-term reference path
        self.point_extended = point_extended  # Extended point for a non-loop reference path (address the oscillations near the goal)
        self.sample_interval = sample_interval  # Integer, sample interval from the long-term reference path for the short-term reference paths


class ReferencePathsAgentRelated:
    def __init__(
        self,
        long_term: torch.Tensor = None,
        long_term_vec_normalized: torch.Tensor = None,
        point_extended: torch.Tensor = None,
        left_boundary: torch.Tensor = None,
        right_boundary: torch.Tensor = None,
        entry: torch.Tensor = None,
        exit: torch.Tensor = None,
        n_points_long_term: torch.Tensor = None,
        n_points_left_b: torch.Tensor = None,
        n_points_right_b: torch.Tensor = None,
        is_loop: torch.Tensor = None,
        n_points_nearing_boundary: torch.Tensor = None,
        nearing_points_left_boundary: torch.Tensor = None,
        nearing_points_right_boundary: torch.Tensor = None,
        short_term: torch.Tensor = None,
        short_term_indices: torch.Tensor = None,
        scenario_id: torch.Tensor = None,
        path_id: torch.Tensor = None,
        point_id: torch.Tensor = None,
    ):
        self.long_term = long_term  # Actual long-term reference paths of agents
        self.long_term_vec_normalized = (
            long_term_vec_normalized  # Normalized vectories on the long-term trajectory
        )
        self.point_extended = point_extended
        self.left_boundary = left_boundary
        self.right_boundary = right_boundary
        self.entry = entry  # [for non-loop path only] Line segment of entry
        self.exit = exit  # [for non-loop path only] Line segment of exit
        self.is_loop = is_loop  # Whether the reference path is a loop
        self.n_points_long_term = (
            n_points_long_term  # The number of points on the long-term reference paths
        )
        self.n_points_left_b = n_points_left_b  # The number of points on the left boundary of the long-term reference paths
        self.n_points_right_b = n_points_right_b  # The number of points on the right boundary of the long-term reference paths
        self.short_term = short_term  # Short-term reference path
        self.short_term_indices = short_term_indices  # Indices that indicate which part of the long-term reference path is used to build the short-term reference path
        self.n_points_nearing_boundary = n_points_nearing_boundary  # Number of points on nearing boundaries to be observed
        self.nearing_points_left_boundary = (
            nearing_points_left_boundary  # Nearing left boundary
        )
        self.nearing_points_right_boundary = (
            nearing_points_right_boundary  # Nearing right boundary
        )

        self.scenario_id = scenario_id  # Which scenarios agents are (current implementation includes (1) intersection, (2) merge-in, and (3) merge-out)
        self.path_id = path_id  # Which paths agents are
        self.point_id = point_id  # Which points agents are


class Distances:
    def __init__(
        self,
        agents=None,
        left_boundaries=None,
        right_boundaries=None,
        boundaries=None,
        ref_paths=None,
        closest_point_on_ref_path=None,
        closest_point_on_left_b=None,
        closest_point_on_right_b=None,
        goal=None,
        obstacles=None,
    ):
        self.agents = agents  # Distances between agents
        self.left_boundaries = left_boundaries  # Distances between agents and the left boundaries of their current lanelets (for each vertex of each agent)
        self.right_boundaries = right_boundaries  # Distances between agents and the right boundaries of their current lanelets (for each vertex of each agent)
        self.boundaries = boundaries  # The minimum distances between agents and the boundaries of their current lanelets
        self.ref_paths = ref_paths  # Distances between agents and the center line of their current lanelets
        self.closest_point_on_ref_path = (
            closest_point_on_ref_path  # Index of the closest point on reference path
        )
        self.closest_point_on_left_b = (
            closest_point_on_left_b  # Index of the closest point on left boundary
        )
        self.closest_point_on_right_b = (
            closest_point_on_right_b  # Index of the closest point on right boundary
        )
        self.goal = goal  # Distances to goal positions
        self.obstacles = obstacles  # Distances to obstacles


class Timer:
    # This class stores the data relevant to static and dynamic obstacles.
    def __init__(
        self,
        step=None,
        start=None,
        end=None,
        step_begin=None,
        render_begin=None,
    ):
        self.step = step  # Count the current time step
        self.start = start  # Time point of simulation start
        self.end = end  # Time point of simulation end
        self.step_begin = step_begin  # Time when the current time step begins
        self.render_begin = (
            render_begin  # Time when the rendering of the current time step begins
        )


class Collisions:
    def __init__(
        self,
        with_obstacles: torch.Tensor = None,
        with_agents: torch.Tensor = None,
        with_lanelets: torch.Tensor = None,
        with_entry_segments: torch.Tensor = None,
        with_exit_segments: torch.Tensor = None,
    ):
        self.with_agents = with_agents  # Whether collide with agents
        self.with_obstacles = with_obstacles  # Whether collide with obstacles
        self.with_lanelets = with_lanelets  # Whether collide with lanelet boundaries
        self.with_entry_segments = (
            with_entry_segments  # Whether collide with entry segments
        )
        self.with_exit_segments = (
            with_exit_segments  # Whether collide with exit segments
        )


class Constants:
    # Predefined constants that may be used during simulations
    def __init__(
        self,
        env_idx_broadcasting: torch.Tensor = None,
        empty_action_vel: torch.Tensor = None,
        empty_action_steering: torch.Tensor = None,
        mask_pos: torch.Tensor = None,
        mask_vel: torch.Tensor = None,
        mask_rot: torch.Tensor = None,
        mask_zero: torch.Tensor = None,
        mask_one: torch.Tensor = None,
        reset_agent_min_distance: torch.Tensor = None,
    ):
        self.env_idx_broadcasting = env_idx_broadcasting
        self.empty_action_vel = empty_action_vel
        self.empty_action_steering = empty_action_steering
        self.mask_pos = mask_pos
        self.mask_zero = mask_zero
        self.mask_one = mask_one
        self.mask_vel = mask_vel
        self.mask_rot = mask_rot
        self.reset_agent_min_distance = reset_agent_min_distance  # The minimum distance between agents when being reset


class CircularBuffer:
    def __init__(
        self,
        buffer: torch.Tensor = None,
    ):
        """Initializes a circular buffer to store initial states."""
        self.buffer = buffer  # Buffer
        self.buffer_size = buffer.shape[0]  # Buffer size
        self.pointer = 0  # Point to the index where the new entry should be stored
        self.valid_size = 0  # Valid size of the buffer, maximum being `buffer_size`

    def add(self, recording: torch.Tensor = None):
        """Adds a new recording to the buffer, overwriting the oldest recording if the buffer is full.

        Args:
            recording: A recording tensor to add to the buffer.
        """
        self.buffer[self.pointer] = recording
        self.pointer = (
            self.pointer + 1
        ) % self.buffer_size  # Increment, loop back to 0 if full
        self.valid_size = min(
            self.valid_size + 1, self.buffer_size
        )  # Increment up to the maximum size

    def get_latest(self, n=1):
        """Returns the n-th latest recording from the buffer.

        Args:
            n: Specifies which latest recording to retrieve (1-based index: 1 is the most recent).

        Return:
            The n-th latest recording. If n is larger than valid_size, returns the first recording.
        """
        if n > self.valid_size:
            index = 0
        else:
            index = (self.pointer - n) % self.buffer_size

        return self.buffer[index]

    def reset(self):
        """Reset the buffer."""
        self.buffer[:] = 0
        self.pointer = 0
        self.valid_size = 0


class StateBuffer(CircularBuffer):
    def __init__(self, buffer: torch.Tensor = None):
        """Initializes a circular buffer to store initial states."""
        super().__init__(buffer=buffer)  # Properly initialize the parent class
        self.idx_scenario = 5
        self.idx_path = 6
        self.idx_point = 7


class InitialStateBuffer(CircularBuffer):
    def __init__(
        self,
        buffer: torch.Tensor = None,
        probability_record: torch.Tensor = None,
        probability_use_recording: torch.Tensor = None,
    ):
        """Initializes a circular buffer to store initial states."""
        super().__init__(buffer=buffer)  # Properly initialize the parent class
        self.probability_record = probability_record  # Probability of recording a collision-event into the buffer
        self.probability_use_recording = probability_use_recording
        self.idx_scenario = 5
        self.idx_path = 6
        self.idx_point = 7

    def get_random(self):
        """Returns a randomly selected recording from the buffer.

        Return:
            A randomly selected recording tensor. If the buffer is empty, returns None.
        """
        if self.valid_size == 0:
            return None
        else:
            # Random index based on the current size of the buffer
            random_index = torch.randint(0, self.valid_size, ())

            return self.buffer[random_index]


class Observations:
    def __init__(
        self,
        is_partial=None,
        n_nearing_agents=None,
        nearing_agents_indices=None,
        noise_level=None,
        n_stored_steps=None,
        n_observed_steps=None,
        past_pos: CircularBuffer = None,
        past_rot: CircularBuffer = None,
        past_vertices: CircularBuffer = None,
        past_vel: CircularBuffer = None,
        past_short_term_ref_points: CircularBuffer = None,
        past_action_vel: CircularBuffer = None,
        past_action_steering: CircularBuffer = None,
        past_distance_to_ref_path: CircularBuffer = None,
        past_distance_to_boundaries: CircularBuffer = None,
        past_distance_to_left_boundary: CircularBuffer = None,
        past_distance_to_right_boundary: CircularBuffer = None,
        past_distance_to_agents: CircularBuffer = None,
        past_left_boundary: CircularBuffer = None,
        past_right_boundary: CircularBuffer = None,
    ):
        self.is_partial = is_partial  # Local observation
        self.n_nearing_agents = n_nearing_agents
        self.nearing_agents_indices = nearing_agents_indices
        self.noise_level = noise_level  # Whether to add noise to observations
        self.n_stored_steps = n_stored_steps  # Number of past steps to store
        self.n_observed_steps = n_observed_steps  # Number of past steps to observe

        self.past_pos = past_pos  # Past positions
        self.past_rot = past_rot  # Past rotations
        self.past_vertices = past_vertices  # Past vertices
        self.past_vel = past_vel  # Past velocites

        self.past_short_term_ref_points = (
            past_short_term_ref_points  # Past short-term reference points
        )
        self.past_left_boundary = past_left_boundary  # Past left lanelet boundary
        self.past_right_boundary = past_right_boundary  # Past right lanelet boundary

        self.past_action_vel = past_action_vel  # Past velocity action
        self.past_action_steering = past_action_steering  # Past steering action
        self.past_distance_to_ref_path = (
            past_distance_to_ref_path  # Past distance to refrence path
        )
        self.past_distance_to_boundaries = (
            past_distance_to_boundaries  # Past distance to lanelet boundaries
        )
        self.past_distance_to_left_boundary = (
            past_distance_to_left_boundary  # Past distance to left lanelet boundary
        )
        self.past_distance_to_right_boundary = (
            past_distance_to_right_boundary  # Past distance to right lanelet boundary
        )
        self.past_distance_to_agents = (
            past_distance_to_agents  # Past mutual distance between agents
        )


def parse_point(element, device):
    """Parses a point element to extract x and y coordinates."""
    x = float(element.find("x").text) if element.find("x") is not None else None
    y = float(element.find("y").text) if element.find("y") is not None else None
    return torch.tensor([x, y], device=device)


def parse_bound(element, device):
    """Parses a bound (left boundary or right boundary) element to extract points and line marking."""
    points = [parse_point(point, device) for point in element.findall("point")]
    points = torch.vstack(points)
    line_marking = (
        element.find("lineMarking").text
        if element.find("lineMarking") is not None
        else None
    )
    return points, line_marking


def get_center_length_yaw_polyline(polyline: torch.Tensor):
    """This function calculates the center points, lengths, and yaws of all line segments of the given polyline."""

    center_points = polyline.unfold(0, 2, 1).mean(dim=2)

    polyline_vecs = polyline.diff(dim=0)
    lengths = polyline_vecs.norm(dim=1)
    yaws = torch.atan2(polyline_vecs[:, 1], polyline_vecs[:, 0])

    return center_points, lengths, yaws, polyline_vecs


def parse_lanelet(element, device):
    """Parses a lanelet element to extract detailed information."""
    lanelet_data = {
        "id": int(element.get("id")),
        "left_boundary": [],
        "left_boundary_center_points": [],
        "left_boundary_lengths": [],
        "left_boundary_yaws": [],
        "left_line_marking": None,
        "right_boundary": [],
        "right_boundary_center_points": [],
        "right_boundary_lengths": [],
        "right_boundary_yaws": [],
        "right_line_marking": None,
        "center_line": [],
        "center_line_center_points": [],
        "center_line_lengths": [],
        "center_line_yaws": [],
        "center_line_marking": "dashed",
        "predecessor": [],
        "successor": [],
        "adjacent_left": None,
        "adjacent_right": None,
        "lanelet_type": None,
    }
    for child in element:
        if child.tag == "leftBound":
            (
                lanelet_data["left_boundary"],
                lanelet_data["left_line_marking"],
            ) = parse_bound(child, device)
        elif child.tag == "rightBound":
            (
                lanelet_data["right_boundary"],
                lanelet_data["right_line_marking"],
            ) = parse_bound(child, device)
        elif child.tag == "predecessor":
            lanelet_data["predecessor"].append(int(child.get("ref")))
        elif child.tag == "successor":
            lanelet_data["successor"].append(int(child.get("ref")))
        elif child.tag == "adjacentLeft":
            lanelet_data["adjacent_left"] = {
                "id": int(child.get("ref")),
                "drivingDirection": child.get("drivingDir"),
            }
        elif child.tag == "adjacentRight":
            lanelet_data["adjacent_right"] = {
                "id": int(child.get("ref")),
                "drivingDirection": child.get("drivingDir"),
            }
        elif child.tag == "lanelet_type":
            lanelet_data["lanelet_type"] = child.text

    lanelet_data["center_line"] = (
        lanelet_data["left_boundary"] + lanelet_data["right_boundary"]
    ) / 2

    (
        lanelet_data["center_line_center_points"],
        lanelet_data["center_line_lengths"],
        lanelet_data["center_line_yaws"],
        _,
    ) = get_center_length_yaw_polyline(polyline=lanelet_data["center_line"])
    (
        lanelet_data["left_boundary_center_points"],
        lanelet_data["left_boundary_lengths"],
        lanelet_data["left_boundary_yaws"],
        _,
    ) = get_center_length_yaw_polyline(polyline=lanelet_data["left_boundary"])
    (
        lanelet_data["right_boundary_center_points"],
        lanelet_data["right_boundary_lengths"],
        lanelet_data["right_boundary_yaws"],
        _,
    ) = get_center_length_yaw_polyline(polyline=lanelet_data["right_boundary"])

    return lanelet_data


def parse_intersections(element):
    """This function parses the lanes of the intersection."""
    intersection_info = []

    for incoming in element.findall("incoming"):
        incoming_info = {
            "incomingLanelet": int(
                incoming.find("incomingLanelet").get("ref")
            ),  # The starting lanelet of a part of the intersection
            "successorsRight": int(
                incoming.find("successorsRight").get("ref")
            ),  # The successor right lanelet of the incoming lanelet
            "successorsStraight": [
                int(s.get("ref")) for s in incoming.findall("successorsStraight")
            ],  # The successor lanelet(s) of the incoming lanelet
            "successorsLeft": int(
                incoming.find("successorsLeft").get("ref")
            ),  # The successor left lanelet of the incoming lanelet
        }
        intersection_info.append(incoming_info)

    return intersection_info


def get_map_data(map_file_path, device=None):
    """This function returns the map data."""
    if device is None:
        device = torch.device("cpu")
    tree = ET.parse(map_file_path)
    root = tree.getroot()
    lanelets = []
    intersection_info = []
    for child in root:
        if child.tag == "lanelet":
            lanelets.append(parse_lanelet(child, device))
        elif child.tag == "intersection":
            intersection_info = parse_intersections(child)

    # Calculate the mean lane width
    mean_lane_width = torch.mean(
        torch.norm(
            torch.vstack([lanelets[i]["left_boundary"] for i in range(len(lanelets))])
            - torch.vstack(
                [lanelets[i]["right_boundary"] for i in range(len(lanelets))]
            ),
            dim=1,
        )
    )

    # Storing all the data in a single tree variable
    map_data = {
        "lanelets": lanelets,
        "intersection_info": intersection_info,
        "mean_lane_width": mean_lane_width,
    }

    return map_data


def get_rectangle_vertices(
    center: torch.Tensor, yaw, width, length, is_close_shape: bool = True
):
    """Compute the vertices of rectangles for a batch of agents given their centers, yaws (rotations),
    widths, and lengths, using PyTorch tensors.

    Args:
        center: [batch_dim, 2] or [2] center positions of the rectangles. In the case of the latter, batch_dim is deemed to be 1.
        yaw: [batch_dim, 1] or [1] or [] Rotation angles in radians.
        width: [scalar] Width of the rectangles.
        length: [scalar] Length of the rectangles.

    Return:
        [batch_dim, 4, 2] vertex points of the rectangles for each agent.
    """
    if center.ndim == 1:
        center = center.unsqueeze(0)

    if yaw.ndim == 0:
        yaw = yaw.unsqueeze(0).unsqueeze(0)
    elif yaw.ndim == 1:
        yaw = yaw.unsqueeze(0)

    batch_dim = center.shape[0]

    width_half = width / 2
    length_half = length / 2

    # vertex points relative to the center
    if is_close_shape:
        vertices = torch.tensor(
            [
                [length_half, width_half],
                [length_half, -width_half],
                [-length_half, -width_half],
                [-length_half, width_half],
                [length_half, width_half],
            ],
            dtype=center.dtype,
            device=center.device,
        )  # Repeat the first vertex to close the shape
    else:
        vertices = torch.tensor(
            [
                [length_half, width_half],
                [length_half, -width_half],
                [-length_half, -width_half],
                [-length_half, width_half],
            ],
            dtype=center.dtype,
            device=center.device,
        )

    # Expand vertices to match batch size
    vertices = vertices.unsqueeze(0).repeat(batch_dim, 1, 1)

    # Create rotation matrices for each agent
    cos_yaw = torch.cos(yaw).squeeze(1)
    sin_yaw = torch.sin(yaw).squeeze(1)

    # Rotation matrix for each agent
    rot_matrix = torch.stack(
        [
            torch.stack([cos_yaw, -sin_yaw], dim=1),
            torch.stack([sin_yaw, cos_yaw], dim=1),
        ],
        dim=1,
    )

    # Apply rotation to vertices
    vertices_rotated = torch.matmul(rot_matrix, vertices.transpose(1, 2)).transpose(
        1, 2
    )

    # Add center positions to the rotated vertices
    vertices_global = vertices_rotated + center.unsqueeze(1)

    return vertices_global


def get_perpendicular_distances(
    point: torch.Tensor, polyline: torch.Tensor, n_points_long_term: torch.Tensor = None
):
    """Calculate the minimum perpendicular distance from the given point(s) to the given polyline.

    Args:
        point: torch.Size([batch_size, 2]) or torch.Size([2]), position of the point. In the case of the latter, the batch_size is deemed to be 1.
        polyline: torch.Size([num_points, 2]) or torch.Size([batch_size, num_points, 2]) x- and y-coordinates of the points on the polyline.
    """

    if point.ndim == 1:
        point = point.unsqueeze(0)

    batch_size = point.shape[0]

    # Expand the polyline points to match the batch size
    if polyline.ndim == 2:
        polyline_expanded = polyline.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [batch_size, n_points, 2]
    else:
        polyline_expanded = polyline

    # Split the polyline into line segments
    line_starts = polyline_expanded[:, :-1, :]
    line_ends = polyline_expanded[:, 1:, :]

    # Create vectors for each line segment and for the point to the start of each segment
    point_expanded = point.unsqueeze(1)  # Shape: [batch_size, 1, 2]
    line_vecs = line_ends - line_starts
    point_vecs = point_expanded - line_starts

    # Project point_vecs onto line_vecs
    line_lens_squared = (
        torch.sum(line_vecs**2, dim=2) + 1e-8
    )  # Adding a small value for numerical stability
    projected_lengths = torch.sum(point_vecs * line_vecs, dim=2) / line_lens_squared

    # Clamp the projections to lie within the line segments
    clamped_lengths = torch.clamp(projected_lengths, 0, 1)

    # Find the closest points on the line segments to the given points
    closest_points = line_starts + (line_vecs * clamped_lengths.unsqueeze(2))

    # Calculate the distances from the given points to these closest points
    distances = torch.norm(closest_points - point_expanded, dim=2)

    if n_points_long_term is not None:
        if n_points_long_term.ndim == 0:
            n_points_long_term = n_points_long_term.unsqueeze(0)
        for env_idx, n_long_term_point in enumerate(n_points_long_term):
            distance_to_end_point = distances[env_idx, n_long_term_point - 2].clone()
            distances[env_idx, n_long_term_point - 1 :] = distance_to_end_point

    perpendicular_distances, indices_closest_points = torch.min(distances, dim=1)

    adjusted_indices = (
        indices_closest_points + 1
    )  # Force the nearest point to lie always in the future

    return perpendicular_distances, adjusted_indices


def get_short_term_reference_path(
    polyline: torch.Tensor,
    index_closest_point: torch.Tensor,
    n_points_to_return: int,
    device=None,
    is_polyline_a_loop: torch.Tensor = False,
    n_points_long_term: torch.Tensor = None,
    sample_interval: int = 2,
    n_points_shift: int = 1,
):
    """
    Args:
        polyline:                   [batch_size, num_points, 2] or [num_points, 2]. In the case of the latter, batch_dim is deemed as 1.
        index_closest_point:        [batch_size, 1] or [1] or []. In the case of the latter, batch_dim is deemed as 1.
        n_points_to_return:         [1] or []. In the case of the latter, batch_dim is deemed as 1.
        is_polyline_a_loop:         [batch_size] or []. In the case of the latter, batch_dim is deemed as 1.
        n_points_long_term:         [batch_size] or []. In the case of the latter, batch_dim is deemed as 1.
        sample_interval:            Sample interval to match specific purposes;
                                    set to 2 when using this function to get the short-term reference path;
                                    set to 1 when using this function to get the nearing boundary points.
        n_points_shift:             Number of points to be shifted to match specific purposes;
                                    set to 1 when using this function to get the short-term reference path to "force" the first point of the short-term reference path being in front of the agent;
                                    set to -2 when using this function to get the nearing boundary points to consider the points behind the agent.
    """
    if polyline.ndim == 2:
        polyline = polyline.unsqueeze(0)
    if index_closest_point.ndim == 1:
        index_closest_point = index_closest_point.unsqueeze(1)
    elif index_closest_point.ndim == 0:
        index_closest_point = index_closest_point.unsqueeze(0).unsqueeze(1)
    if is_polyline_a_loop.ndim == 0:
        is_polyline_a_loop = is_polyline_a_loop.unsqueeze(0)
    if n_points_long_term.ndim == 0:
        n_points_long_term = n_points_long_term.unsqueeze(0)

    if device is None:
        device = torch.device("cpu")

    batch_size = index_closest_point.shape[0]

    future_points_idx = (
        torch.arange(n_points_to_return, device=device) * sample_interval
        + index_closest_point
        + n_points_shift
    )

    if n_points_long_term is None:
        n_points_long_term = polyline.shape[-2]

    for env_i in range(batch_size):
        n_long_term_point = n_points_long_term[env_i]
        if is_polyline_a_loop[env_i]:
            # Apply modulo to handle the case that each agent's reference path is a loop
            future_points_idx[env_i] = torch.where(
                future_points_idx[env_i] >= n_long_term_point - 1,
                (future_points_idx[env_i] + 1) % n_long_term_point,
                future_points_idx[env_i],
            )  # Use "+ 1" to skip the last point since it overlaps with the first point

    # Extract
    short_term_path = polyline[
        torch.arange(batch_size, device=device, dtype=torch.int).unsqueeze(
            1
        ),  # For broadcasting
        future_points_idx,
    ]

    return short_term_path, future_points_idx


def exponential_decreasing_fcn(x, x0, x1):
    """
    Exponential function y(x) = (e^( -(x-x0) / (x1-x0) ) - e^-1) / (1 - e^-1), so that y decreases exponentially from 1 to 0 when x increases from x0 to x1, where
    x = max(min(x, x1), x0),
    x1 = threshold_near_boundary, and
    x0 = agent.shape.width/2.
    """
    x_clamped = torch.clamp(x, min=x0, max=x1)  # x stays inside [x0, x1]
    y = (torch.exp(-(x_clamped - x0) / (x1 - x0)) - 1 / torch.e) / (1 - 1 / torch.e)

    return y


def get_distances_between_agents(self, is_set_diagonal=False):
    """This function calculates the mutual distances between agents.
        Currently, the calculation of two types of distances is supported ('c2c' and 'MTV'):
            c2c: center-to-center distance
            MTV: minimum translation vector (MTV)-based distance
    Args:
        is_set_diagonal: whether to set the diagonal elements (distance from an agent to this agent itself) from zero to a high value
    """
    # Collect positions for all agents across all batches, shape [n_agents, batch_size, 2]
    positions = torch.stack(
        [self.world.agents[i].state.pos for i in range(self.n_agents)]
    )

    # Reshape from [n_agents, batch_size, 2] to [batch_size, n_agents, 2]
    positions_reshaped = positions.transpose(0, 1)

    # Reshape for broadcasting: shape becomes [batch_size, n_agents, 1, 2] and [batch_size, 1, n_agents, 2]
    pos1 = positions_reshaped.unsqueeze(2)
    pos2 = positions_reshaped.unsqueeze(1)

    # Calculate squared differences, shape [batch_size, n_agents, n_agents, 2]
    squared_diff = (pos1 - pos2) ** 2

    # Sum over the last dimension to get squared distances, shape [batch_size, n_agents, n_agents]
    squared_distances = squared_diff.sum(-1)

    # Take the square root to get actual distances, shape [batch_size, n_agents, n_agents]
    mutual_distances = torch.sqrt(squared_distances)

    if is_set_diagonal:
        mutual_distances.diagonal(dim1=-2, dim2=-1).fill_(mutual_distances.max() + 1)

    return mutual_distances


def interX(L1, L2, is_return_points=False):
    """Calculate the intersections of batches of curves.
        Adapted from https://www.mathworks.com/matlabcentral/fileexchange/22441-curve-intersections
    Args:
        L1: [batch_size, num_points, 2]
        L2: [batch_size, num_points, 2]
        is_return_points: bool. Whether to return the intersecting points.
    """
    batch_dim = L1.shape[0]
    collision_index = torch.zeros(batch_dim, dtype=torch.bool)  # Initialize

    # Handle empty inputs
    if L1.numel() == 0 or L2.numel() == 0:
        return torch.empty((0, 2), device=L1.device) if is_return_points else False

    # Extract x and y coordinates
    x1, y1 = L1[..., 0], L1[..., 1]
    x2, y2 = L2[..., 0], L2[..., 1]

    # Compute differences
    dx1, dy1 = torch.diff(x1, dim=1), torch.diff(y1, dim=1)
    dx2, dy2 = torch.diff(x2, dim=1), torch.diff(y2, dim=1)

    # Determine 'signed distances'
    S1 = dx1 * y1[..., :-1] - dy1 * x1[..., :-1]
    S2 = dx2 * y2[..., :-1] - dy2 * x2[..., :-1]

    # Helper function for computing D
    def D(x, y):
        return (x[..., :-1] - y) * (x[..., 1:] - y)

    C1 = (
        D(
            dx1.unsqueeze(2) * y2.unsqueeze(1) - dy1.unsqueeze(2) * x2.unsqueeze(1),
            S1.unsqueeze(2),
        )
        < 0
    )
    C2 = (
        D(
            (
                y1.unsqueeze(2) * dx2.unsqueeze(1) - x1.unsqueeze(2) * dy2.unsqueeze(1)
            ).transpose(1, 2),
            S2.unsqueeze(2),
        )
        < 0
    ).transpose(1, 2)

    # Obtain the segments where an intersection is expected
    batch_indices, i, j = torch.where(C1 & C2)
    batch_indices_pruned = torch.sort(torch.unique(batch_indices))[0]
    collision_index[batch_indices_pruned] = True

    if is_return_points:
        # In case of collisions, return collision points; else return empty points
        if batch_indices.numel() == 0:
            return torch.empty((0, 2), device=L1.device)
        else:
            # Process intersections for each batch item
            intersections = []
            for b in batch_indices.unique():
                L = dy2[b, j] * dx1[b, i] - dy1[b, i] * dx2[b, j]
                nonzero = L != 0
                i_nz, j_nz, L_nz = i[nonzero], j[nonzero], L[nonzero]

                P = torch.stack(
                    (
                        (dx2[b, j_nz] * S1[b, i_nz] - dx1[b, i_nz] * S2[b, j_nz])
                        / L_nz,
                        (dy2[b, j_nz] * S1[b, i_nz] - dy1[b, i_nz] * S2[b, j_nz])
                        / L_nz,
                    ),
                    dim=-1,
                )
                intersections.append(P)
            # Combine intersections from all batches
            return torch.cat(intersections, dim=0)
    else:
        # Simply return whether collisions occur or not
        return collision_index


def transform_from_global_to_local_coordinate(
    pos_i: torch.Tensor, pos_j: torch.Tensor, rot_i
):
    """
    Args:
        pos_i: torch.Size([batch_size, 2])
        pos_j: torch.Size([batch_size, num_points, 2]) or torch.Size([num_points, 2])
        rot_i: torch.Size([batch_size, 1])
    """
    # Prepare for vectorized ccomputation
    if pos_j.ndim == 3:
        pos_i_extended = pos_i.unsqueeze(1)
    else:
        pos_i_extended = pos_i.unsqueeze(1)
        # Check if the last point overlaps with the first point
        if (pos_j[0, :] - pos_j[-1, :]).norm() == 0:
            pos_j = pos_j[0:-1, :].unsqueeze(0)
        else:
            pos_j = pos_j.unsqueeze(0)

    pos_vec = pos_j - pos_i_extended
    pos_vec_abs = pos_vec.norm(dim=2)
    rot_rel = torch.atan2(pos_vec[:, :, 1], pos_vec[:, :, 0]) - rot_i

    pos_rel = torch.stack(
        (
            torch.cos(rot_rel) * pos_vec_abs,
            torch.sin(rot_rel) * pos_vec_abs,
        ),
        dim=2,
    )

    return pos_rel


def angle_eliminate_two_pi(angle):
    """
    Normalize an angle to be within the range -pi to pi.

    Args:
        angle (torch.Tensor): The angle to normalize in radians. Can be a tensor of any shape.

    Returns:
        torch.Tensor: Normalized angle between -pi and pi.
    """
    two_pi = 2 * torch.pi
    angle = angle % two_pi  # Normalize angle to be within 0 and 2*pi
    angle[angle > torch.pi] -= two_pi  # Shift to -pi to pi range
    return angle


def get_reference_lanelet_index(ref_path_id, path_to_loop):
    """
    Get loop of lanelets used for reference_path_struct.

    Args:
    ref_path_id (int): Path ID.

    Returns:
    list: List of lanelets indices.
    """
    # Define loops of paths (successive lanelets)
    reference_lanelets_loops = [
        [4, 6, 8, 60, 58, 56, 54, 80, 82, 84, 86, 34, 32, 30, 28, 2],  # Loop 1
        [1, 3, 23, 10, 12, 17, 43, 38, 36, 49, 29, 27],  # Loop 2
        [64, 62, 75, 55, 53, 79, 81, 101, 88, 90, 95, 69],  # Loop 3
        [40, 45, 97, 92, 94, 100, 83, 85, 33, 31, 48, 42],  # Loop 4
        [5, 7, 59, 57, 74, 68, 66, 71, 19, 14, 16, 22],  # Loop 5
        [41, 39, 20, 63, 61, 57, 55, 67, 65, 98, 37, 35, 31, 29],  # Loop 6
        [3, 5, 9, 11, 72, 91, 93, 81, 83, 87, 89, 46, 13, 15],  # Loop 7
    ]

    # Get loop index and starting lanelet for the given agent
    loop_index, starting_lanelet = path_to_loop.get(ref_path_id, (None, None))

    if loop_index is not None:
        # Take loop from all defined loops
        reference_lanelets_loop = reference_lanelets_loops[
            loop_index - 1
        ]  # Adjust for 0-based index
        # Find index of defined starting lanelet
        index_starting_lanelet = reference_lanelets_loop.index(starting_lanelet)
        # Shift loop according to starting lanelet
        lanelets_index = (
            reference_lanelets_loop[index_starting_lanelet:]
            + reference_lanelets_loop[:index_starting_lanelet]
        )
        return lanelets_index
    else:
        return []  # Return empty list if ref_path_id is not found


def calculate_reference_path(
    reference_lanelets_index, map_data, lanelets_share_same_boundaries_list
):
    # Initialize
    left_boundaries = None
    right_boundaries = None
    left_boundaries_shared = None
    right_boundaries_shared = None
    center_lines = None

    for lanelet in reference_lanelets_index:
        lanelets_share_same_boundaries = next(
            (
                group
                for group in lanelets_share_same_boundaries_list
                if lanelet in group
            ),
            None,
        )

        # Extracting left and right boundaries
        left_bound = map_data["lanelets"][lanelet - 1][
            "left_boundary"
        ]  # Lanelet IDs start from 1, while the index of a list in Python starts from 0
        right_bound = map_data["lanelets"][lanelet - 1]["right_boundary"]
        left_bound_shared = map_data["lanelets"][lanelets_share_same_boundaries[0] - 1][
            "left_boundary"
        ]
        right_bound_shared = map_data["lanelets"][
            lanelets_share_same_boundaries[-1] - 1
        ]["right_boundary"]

        if left_boundaries is None:
            left_boundaries = left_bound
            right_boundaries = right_bound
            left_boundaries_shared = left_bound_shared
            right_boundaries_shared = right_bound_shared
        else:
            # Concatenate boundary data while avoiding duplicated adta at the connection point
            if torch.norm(left_boundaries[-1, :] - left_bound[0, :]) < 1e-4:
                left_boundaries = torch.cat((left_boundaries, left_bound[1:, :]), dim=0)
                left_boundaries_shared = torch.cat(
                    (left_boundaries_shared, left_bound_shared[1:, :]), dim=0
                )
            else:
                left_boundaries = torch.cat((left_boundaries, left_bound), dim=0)
                left_boundaries_shared = torch.cat(
                    (left_boundaries_shared, left_bound_shared), dim=0
                )

            if torch.norm(right_boundaries[-1, :] - right_bound[0, :]) < 1e-4:
                right_boundaries = torch.cat(
                    (right_boundaries, right_bound[1:, :]), dim=0
                )
                right_boundaries_shared = torch.cat(
                    (right_boundaries_shared, right_bound_shared[1:, :]), dim=0
                )
            else:
                right_boundaries = torch.cat((right_boundaries, right_bound), dim=0)
                right_boundaries_shared = torch.cat(
                    (right_boundaries_shared, right_bound_shared), dim=0
                )

    center_lines = (left_boundaries + right_boundaries) / 2

    # Check if the center line is a loop
    is_loop = (center_lines[0, :] - center_lines[-1, :]).norm() <= 1e-4

    center_lines_vec = torch.diff(
        center_lines, dim=0
    )  # Vectors connecting each pair of neighboring points on the center lines
    center_lines_vec_length = torch.norm(
        center_lines_vec, dim=1
    )  # The lengths of the vectors
    center_lines_vec_mean_length = torch.mean(
        center_lines_vec_length
    )  # The mean length of the vectors
    center_lines_vec_normalized = center_lines_vec / center_lines_vec_length.unsqueeze(
        1
    )

    center_line_yaw = torch.atan2(center_lines_vec[:, 1], center_lines_vec[:, 0])

    reference_path = {
        "reference_lanelets": reference_lanelets_index,
        "left_boundary": left_boundaries,
        "right_boundary": right_boundaries,
        "left_boundary_shared": left_boundaries_shared,
        "right_boundary_shared": right_boundaries_shared,
        "center_line": center_lines,  # Center lines are calculated based on left and right boundaries (instead of shared left and right boundaries)
        "center_line_yaw": center_line_yaw,  # Yaw angle of each point on the center lines
        "center_line_vec_normalized": center_lines_vec_normalized,  # Normalized vectors connecting each pair of neighboring points on the center lines
        "center_line_vec_mean_length": center_lines_vec_mean_length,
        "is_loop": is_loop,
    }
    return reference_path


def get_reference_paths(map_data):
    """This function returns the (long-term) reference paths."""
    reference_paths_all = []
    reference_paths_intersection = []
    reference_paths_merge_in = []
    reference_paths_merge_out = []

    path_intersection = [
        [11, 25, 13],
        [11, 26, 52, 37],
        [11, 72, 91],
        [12, 18, 14],
        [12, 17, 43, 38],
        [12, 73, 92],
        [39, 51, 37],
        [39, 50, 102, 91],
        [39, 20, 63],
        [40, 44, 38],
        [40, 45, 97, 92],
        [40, 21, 64],
        [89, 103, 91],
        [89, 104, 78, 63],
        [89, 46, 13],
        [90, 96, 92],
        [90, 95, 69, 64],
        [90, 47, 14],
        [65, 77, 63],
        [65, 76, 24, 13],
        [65, 98, 37],
        [66, 70, 64],
        [66, 71, 19, 14],
        [66, 99, 38],
    ]
    path_merge_in = [
        [34, 32],
        [33, 31],
        [35, 31],
        [36, 49],
    ]
    path_merge_out = [
        [6, 8],
        [5, 7],
        [5, 9],
        [23, 10],
    ]

    # Mapping agent_id to loop index and starting lanelet: agent_id: (loop_index, starting_lanelet)
    path_to_loop = {
        1: (1, 4),
        2: (2, 1),
        3: (3, 64),
        4: (4, 42),
        5: (5, 22),
        6: (6, 39),
        7: (7, 15),
        8: (1, 8),
        9: (2, 10),
        10: (3, 75),
        11: (4, 45),
        12: (5, 59),
        13: (6, 61),
        14: (7, 5),
        15: (1, 58),
        16: (2, 17),
        17: (3, 79),
        18: (4, 92),
        19: (5, 68),
        20: (6, 55),
        21: (7, 11),
        22: (1, 54),
        23: (2, 38),
        24: (3, 88),
        25: (4, 100),
        26: (5, 19),
        27: (6, 65),
        28: (7, 93),
        29: (1, 82),
        30: (2, 49),
        31: (3, 95),
        32: (4, 33),
        33: (5, 14),
        34: (6, 35),
        35: (7, 83),
        36: (1, 86),
        37: (6, 29),
        38: (7, 89),
        39: (1, 32),
        40: (1, 28),
    }

    lanelets_share_same_boundaries_list = [  # Some lanelets share the same boundary (such as adjacent left and adjacent right lanelets)
        [4, 3, 22],
        [6, 5, 23],
        [8, 7],
        [60, 59],
        [58, 57, 75],
        [56, 55, 74],
        [54, 53],
        [80, 79],
        [82, 81, 100],
        [84, 83, 101],
        [86, 85],
        [34, 33],
        [32, 31, 49],
        [30, 29, 48],
        [28, 27],
        [2, 1],  # outer circle (urban)
        [13, 14],
        [15, 16],
        [9, 10],
        [11, 12],  # inner circle (top right)
        [63, 64],
        [61, 62],
        [67, 68],
        [65, 66],  # inner circle (bottom right)
        [91, 92],
        [93, 94],
        [87, 88],
        [89, 90],  # inner circle (bottom left)
        [37, 38],
        [35, 36],
        [41, 42],
        [39, 40],  # inner circle (top left)
        [25, 18],
        [26, 17],
        [52, 43],
        [72, 73],  # intersection: incoming 1 and incoming 2
        [51, 44],
        [50, 45],
        [102, 97],
        [20, 21],  # intersection: incoming 3 and incoming 4
        [103, 96],
        [104, 95],
        [78, 69],
        [46, 47],  # intersection: incoming 5 and incoming 6
        [77, 70],
        [76, 71],
        [24, 19],
        [98, 99],  # intersection: incoming 7 and incoming 8
    ]

    num_paths_all = len(path_to_loop)
    for ref_path_id in range(num_paths_all):
        reference_lanelets_index = get_reference_lanelet_index(
            ref_path_id + 1, path_to_loop
        )  # Path ID starts from 1
        reference_path = calculate_reference_path(
            reference_lanelets_index, map_data, lanelets_share_same_boundaries_list
        )
        reference_paths_all.append(reference_path)

    for reference_lanelets_index in path_intersection:
        reference_path = calculate_reference_path(
            reference_lanelets_index, map_data, lanelets_share_same_boundaries_list
        )
        reference_paths_intersection.append(reference_path)

    for reference_lanelets_index in path_merge_in:
        reference_path = calculate_reference_path(
            reference_lanelets_index, map_data, lanelets_share_same_boundaries_list
        )
        reference_paths_merge_in.append(reference_path)

    for reference_lanelets_index in path_merge_out:
        reference_path = calculate_reference_path(
            reference_lanelets_index, map_data, lanelets_share_same_boundaries_list
        )
        reference_paths_merge_out.append(reference_path)

    return (
        reference_paths_all,
        reference_paths_intersection,
        reference_paths_merge_in,
        reference_paths_merge_out,
    )


if __name__ == "__main__":
    scenario = Scenario()
    render_interactively(
        scenario=scenario,
        control_two_agents=False,
    )
