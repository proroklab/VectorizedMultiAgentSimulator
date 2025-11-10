#  Copyright (c) ProrokLab.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

from vmas.interactive_rendering import render_interactively
from vmas.make_env import make_env
from vmas.simulator.environment import Wrapper

from vmas.simulator.utils import _init_pyglet_device

_init_pyglet_device()

__all__ = [
    "make_env",
    "render_interactively",
    "scenarios",
    "debug_scenarios",
    "mpe_scenarios",
]

__version__ = "1.5.2"

scenarios = sorted(
    [
        "dropout",
        "dispersion",
        "transport",
        "reverse_transport",
        "give_way",
        "wheel",
        "balance",
        "football",
        "discovery",
        "flocking",
        "passage",
        "joint_passage_size",
        "joint_passage",
        "ball_passage",
        "ball_trajectory",
        "buzz_wire",
        "multi_give_way",
        "navigation",
        "sampling",
        "wind_flocking",
        "road_traffic",
    ]
)
"""List of the vmas scenarios (excluding MPE and debug)"""

debug_scenarios = sorted(
    [
        "asym_joint",
        "circle_trajectory",
        "goal",
        "het_mass",
        "line_trajectory",
        "vel_control",
        "waterfall",
        "diff_drive",
        "kinematic_bicycle",
        "pollock",
        "drone",
    ]
)
"""List of the vmas debug scenarios """


mpe_scenarios = sorted(
    [
        "simple",
        "simple_adversary",
        "simple_crypto",
        "simple_push",
        "simple_reference",
        "simple_speaker_listener",
        "simple_spread",
        "simple_tag",
        "simple_world_comm",
    ]
)
"""List of the vmas MPE scenarios """
