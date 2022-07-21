#  Copyright (c) 2022.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

from __future__ import annotations

from typing import List, TYPE_CHECKING, Tuple

import torch

import vmas.simulator.core
import vmas.simulator.utils

if TYPE_CHECKING:
    from vmas.simulator.rendering import Geom

UNCOLLIDABLE_JOINT_RENDERING_WIDTH = 1



class JointConstraint:
    """
    This is an uncollidable constraint that bounds two entities in the specified anchor points at the specified distance
    """

    def __init__(
        self,
        entity_a: vmas.simulator.core.Entity,
        entity_b: vmas.simulator.core.Entity,
        anchor_a: Tuple[float, float] = (0.0, 0.0),
        anchor_b: Tuple[float, float] = (0.0, 0.0),
        dist: float = 0.0,
    ):
        assert entity_a != entity_b, "Cannot join same entity"
        for anchor in (anchor_a, anchor_b):
            assert (
                max(anchor) <= 1 and min(anchor) >= -1
            ), f"Joint anchor points should be between -1 and 1, got {anchor}"
        assert dist >= 0, f"Joint dist must be >= 0, got {dist}"

        self.entity_a = entity_a
        self.entity_b = entity_b
        self.anchor_a = anchor_a
        self.anchor_b = anchor_b
        self.dist = dist

    def get_delta_anchor_a(self):
        return vmas.simulator.core.World._rotate_vector(
            torch.tensor(
                self.entity_a.shape.get_delta_from_anchor(self.anchor_a),
                device=self.entity_a.state.pos.device,
            ),
            self.entity_a.state.rot,
        ).squeeze(-1)

    def get_delta_anchor_b(self):
        return vmas.simulator.core.World._rotate_vector(
            torch.tensor(
                self.entity_b.shape.get_delta_from_anchor(self.anchor_b),
                device=self.entity_b.state.pos.device,
            ),
            self.entity_b.state.rot,
        ).squeeze(-1)

    def pos_point_a(self):
        return self.entity_a.state.pos + self.get_delta_anchor_a()

    def pos_point_b(self):
        return self.entity_b.state.pos + self.get_delta_anchor_b()

    def render(self, env_index: int = 0) -> "List[Geom]":
        if self.dist == 0:
            return []
        from vmas.simulator import rendering

        geoms: List[rendering.Geom] = []
        joint_line = rendering.Line(
            (-self.dist / 2, 0),
            (self.dist / 2, 0),
            width=UNCOLLIDABLE_JOINT_RENDERING_WIDTH,
        )

        pos_point_a = self.pos_point_a()[env_index]
        pos_point_b = self.pos_point_b()[env_index]
        angle = torch.atan2(
            pos_point_b[vmas.simulator.utils.Y] - pos_point_a[vmas.simulator.utils.Y],
            pos_point_b[vmas.simulator.utils.X] - pos_point_a[vmas.simulator.utils.X],
        )

        xform = rendering.Transform()
        xform.set_translation(*((pos_point_a + pos_point_b) / 2))
        xform.set_rotation(angle)
        joint_line.add_attr(xform)

        geoms.append(joint_line)
        return geoms
