#  Copyright (c) 2022.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, List

import torch

import vmas.simulator.core
import vmas.simulator.utils

if TYPE_CHECKING:
    from vmas.simulator.rendering import Geom

UNCOLLIDABLE_JOINT_RENDERING_WIDTH = 1


class Joint(vmas.simulator.utils.Observer):
    def __init__(
        self,
        entity_a: vmas.simulator.core.Entity,
        entity_b: vmas.simulator.core.Entity,
        anchor_a: Tuple[float, float] = (0.0, 0.0),
        anchor_b: Tuple[float, float] = (0.0, 0.0),
        rotate_a: bool = True,
        rotate_b: bool = True,
        dist: float = 0.0,
        collidable: bool = False,
        width: float = 0.0,
        mass: float = 1.0,
    ):
        assert entity_a != entity_b, "Cannot join same entity"
        for anchor in (anchor_a, anchor_b):
            assert (
                max(anchor) <= 1 and min(anchor) >= -1
            ), f"Joint anchor points should be between -1 and 1, got {anchor}"
        assert dist >= 0, f"Joint dist must be >= 0, got {dist}"
        if dist == 0:
            assert not collidable, "Cannot have collidable joint with dist 0"
            assert width == 0, "Cannot have width for joint with dist 0"
        if width > 0:
            assert collidable

        self.entity_a = entity_a
        self.entity_b = entity_b
        self.landmark = None
        self.joint_constraints = []

        if dist == 0:
            self.joint_constraints.append(
                JointConstraint(
                    entity_a,
                    entity_b,
                    anchor_a=anchor_a,
                    anchor_b=anchor_b,
                    dist=dist,
                    rotate=rotate_a and rotate_b,
                ),
            )
        else:
            entity_a.subscribe(self)
            entity_b.subscribe(self)

            self.landmark = vmas.simulator.core.Landmark(
                name=f"joint {entity_a.name} {entity_b.name}",
                collide=collidable,
                movable=True,
                rotatable=rotate_a and rotate_b,
                mass=mass,
                shape=vmas.simulator.core.Box(length=dist, width=width)
                if width != 0
                else vmas.simulator.core.Line(length=dist),
                color=vmas.simulator.utils.Color.BLACK,
                is_joint=True,
            )
            self.joint_constraints += [
                JointConstraint(
                    self.landmark,
                    entity_a,
                    anchor_a=(-1, 0),
                    anchor_b=anchor_a,
                    dist=0.0,
                    rotate=rotate_a,
                ),
                JointConstraint(
                    self.landmark,
                    entity_b,
                    anchor_a=(1, 0),
                    anchor_b=anchor_b,
                    dist=0.0,
                    rotate=rotate_b,
                ),
            ]

    def notify(self, observable, *args, **kwargs):
        pos_a = self.joint_constraints[0].pos_point(self.entity_a)
        pos_b = self.joint_constraints[1].pos_point(self.entity_b)

        self.landmark.set_pos(
            (pos_a + pos_b) / 2,
            batch_index=None,
        )
        self.landmark.set_rot(
            torch.atan2(
                pos_b[:, vmas.simulator.utils.Y] - pos_a[:, vmas.simulator.utils.Y],
                pos_b[:, vmas.simulator.utils.X] - pos_a[:, vmas.simulator.utils.X],
            ).unsqueeze(-1),
            batch_index=None,
        )


# Private class: do not instantiate directly
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
        rotate: bool = True,
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
        self.rotate = rotate

    def get_delta_anchor(self, entity: vmas.simulator.core.Entity):
        if entity == self.entity_a:
            anchor = self.anchor_a
        elif entity == self.entity_b:
            anchor = self.anchor_b
        else:
            assert False
        return vmas.simulator.utils.TorchUtils.rotate_vector(
            torch.tensor(
                entity.shape.get_delta_from_anchor(anchor),
                device=entity.state.pos.device,
            ),
            entity.state.rot,
        ).squeeze(-1)

    def pos_point(self, entity: vmas.simulator.core.Entity):
        return entity.state.pos + self.get_delta_anchor(entity)

    def render(self, env_index: int = 0) -> List[Geom]:
        if self.dist == 0:
            return []
        from vmas.simulator import rendering

        geoms: List[rendering.Geom] = []
        joint_line = rendering.Line(
            (-self.dist / 2, 0),
            (self.dist / 2, 0),
            width=UNCOLLIDABLE_JOINT_RENDERING_WIDTH,
        )
        pos_point_a = self.pos_point(self.entity_a)[env_index]
        pos_point_b = self.pos_point(self.entity_b)[env_index]
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
