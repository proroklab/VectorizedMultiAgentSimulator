#  Copyright (c) ProrokLab.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch

from vmas.simulator.utils import TorchUtils


def _get_inner_point_box(outside_point, surface_point, box_pos):
    v = surface_point - outside_point
    u = box_pos - surface_point
    v_norm = torch.linalg.vector_norm(v, dim=-1).unsqueeze(-1)
    x_magnitude = (v * u).sum(-1).unsqueeze(-1) / v_norm
    x = (v / v_norm) * x_magnitude
    cond = v_norm == 0
    cond_exp = cond.expand(x.shape)
    x = torch.where(cond_exp, surface_point, x)
    x_magnitude = torch.where(cond, 0, x_magnitude)
    return surface_point + x, torch.abs(x_magnitude.squeeze(-1))


def _get_closest_box_box(
    box_pos,
    box_rot,
    box_width,
    box_length,
    box2_pos,
    box2_rot,
    box2_width,
    box2_length,
):
    if not isinstance(box_width, torch.Tensor):
        box_width = torch.tensor(
            box_width, dtype=torch.float32, device=box_pos.device
        ).expand(box_pos.shape[0])
    if not isinstance(box_length, torch.Tensor):
        box_length = torch.tensor(
            box_length, dtype=torch.float32, device=box2_pos.device
        ).expand(box_pos.shape[0])
    if not isinstance(box2_width, torch.Tensor):
        box2_width = torch.tensor(
            box2_width, dtype=torch.float32, device=box2_pos.device
        ).expand(box2_pos.shape[0])
    if not isinstance(box2_length, torch.Tensor):
        box2_length = torch.tensor(
            box2_length, dtype=torch.float32, device=box2_pos.device
        ).expand(box2_pos.shape[0])

    lines_pos, lines_rot, lines_length = _get_all_lines_box(
        torch.stack([box_pos, box2_pos], dim=0),
        torch.stack([box_rot, box2_rot], dim=0),
        torch.stack([box_width, box2_width], dim=0),
        torch.stack([box_length, box2_length], dim=0),
    )
    # Unbind on 1 since _get_all_lines_box adds a dimension at 0
    lines_a_pos, lines_b_pos = lines_pos.unbind(1)
    lines_a_rot, lines_b_rot = lines_rot.unbind(1)
    lines_a_length, lines_b_length = lines_length.unbind(1)

    points_first, points_second = _get_closest_line_box(
        torch.stack(
            [
                box2_pos.unsqueeze(0).expand(lines_a_pos.shape),
                box_pos.unsqueeze(0).expand(lines_b_pos.shape),
            ],
            dim=0,
        ),
        torch.stack(
            [
                box2_rot.unsqueeze(0).expand(lines_a_rot.shape),
                box_rot.unsqueeze(0).expand(lines_b_rot.shape),
            ],
            dim=0,
        ),
        torch.stack(
            [
                box2_width.unsqueeze(0).expand(lines_a_length.shape),
                box_width.unsqueeze(0).expand(lines_b_length.shape),
            ],
            dim=0,
        ),
        torch.stack(
            [
                box2_length.unsqueeze(0).expand(lines_a_length.shape),
                box_length.unsqueeze(0).expand(lines_b_length.shape),
            ],
            dim=0,
        ),
        torch.stack([lines_a_pos, lines_b_pos], dim=0),
        torch.stack([lines_a_rot, lines_b_rot], dim=0),
        torch.stack([lines_a_length, lines_b_length], dim=0),
    )
    points_box2_a, points_box_b = points_first.unbind(0)
    points_box_a, points_box2_b = points_second.unbind(0)

    p1s = points_box_a.unbind(0) + points_box_b.unbind(0)
    p2s = points_box2_a.unbind(0) + points_box2_b.unbind(0)

    closest_point_1 = torch.full(
        box_pos.shape,
        float("inf"),
        device=box_pos.device,
        dtype=torch.float32,
    )
    closest_point_2 = torch.full(
        box_pos.shape,
        float("inf"),
        device=box_pos.device,
        dtype=torch.float32,
    )
    distance = torch.full(
        box_pos.shape[:-1],
        float("inf"),
        device=box_pos.device,
        dtype=torch.float32,
    )
    for p1, p2 in zip(p1s, p2s):
        d = torch.linalg.vector_norm(p1 - p2, dim=-1)
        is_closest = d < distance
        is_closest_exp = is_closest.unsqueeze(-1).expand(p1.shape)
        closest_point_1 = torch.where(is_closest_exp, p1, closest_point_1)
        closest_point_2 = torch.where(is_closest_exp, p2, closest_point_2)
        distance = torch.where(is_closest, d, distance)

    return closest_point_1, closest_point_2


def _get_line_extrema(line_pos, line_rot, line_length):
    line_length = line_length.view(line_rot.shape)
    x = (line_length / 2) * torch.cos(line_rot)
    y = (line_length / 2) * torch.sin(line_rot)
    xy = torch.cat([x, y], dim=-1)

    point_a = line_pos + xy
    point_b = line_pos - xy

    return point_a, point_b


def _get_closest_points_line_line(
    line_pos, line_rot, line_length, line2_pos, line2_rot, line2_length
):
    if not isinstance(line_length, torch.Tensor):
        line_length = torch.tensor(
            line_length, dtype=torch.float32, device=line_pos.device
        ).expand(line_pos.shape[0])
    if not isinstance(line2_length, torch.Tensor):
        line2_length = torch.tensor(
            line2_length, dtype=torch.float32, device=line_pos.device
        ).expand(line_pos.shape[0])

    points_a, points_b = _get_line_extrema(
        torch.stack([line_pos, line2_pos], dim=0),
        torch.stack([line_rot, line2_rot], dim=0),
        torch.stack([line_length, line2_length], dim=0),
    )
    point_a1, point_b1 = points_a.unbind(0)
    point_a2, point_b2 = points_b.unbind(0)

    point_i, d_i = _get_intersection_point_line_line(
        point_a1, point_a2, point_b1, point_b2
    )

    (
        point_a1_line_b,
        point_a2_line_b,
        point_b1_line_a,
        point_b2_line_a,
    ) = _get_closest_point_line(
        torch.stack([line2_pos, line2_pos, line_pos, line_pos], dim=0),
        torch.stack([line2_rot, line2_rot, line_rot, line_rot], dim=0),
        torch.stack([line2_length, line2_length, line_length, line_length], dim=0),
        torch.stack([point_a1, point_a2, point_b1, point_b2], dim=0),
    ).unbind(
        0
    )

    point_pairs = (
        (point_a1, point_a1_line_b),
        (point_a2, point_a2_line_b),
        (point_b1_line_a, point_b1),
        (point_b2_line_a, point_b2),
    )

    closest_point_1 = torch.full(
        line_pos.shape,
        float("inf"),
        device=line_pos.device,
        dtype=torch.float32,
    )
    closest_point_2 = torch.full(
        line_pos.shape,
        float("inf"),
        device=line_pos.device,
        dtype=torch.float32,
    )
    min_distance = torch.full(
        line_pos.shape[:-1],
        float("inf"),
        device=line_pos.device,
        dtype=torch.float32,
    )
    for p1, p2 in point_pairs:
        d = torch.linalg.vector_norm(p1 - p2, dim=-1)
        is_closest = d < min_distance
        is_closest_exp = is_closest.unsqueeze(-1).expand(p1.shape)
        closest_point_1 = torch.where(is_closest_exp, p1, closest_point_1)
        closest_point_2 = torch.where(is_closest_exp, p2, closest_point_2)
        min_distance = torch.where(is_closest, d, min_distance)

    cond = (d_i == 0).unsqueeze(-1).expand(point_i.shape)
    closest_point_1 = torch.where(cond, point_i, closest_point_1)
    closest_point_2 = torch.where(cond, point_i, closest_point_2)

    return closest_point_1, closest_point_2


def _get_intersection_point_line_line(point_a1, point_a2, point_b1, point_b2):
    """
    Taken from:
    https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
    """
    r = point_a2 - point_a1
    s = point_b2 - point_b1
    p = point_a1
    q = point_b1
    cross_q_minus_p_r = TorchUtils.cross(q - p, r)
    cross_q_minus_p_s = TorchUtils.cross(q - p, s)
    cross_r_s = TorchUtils.cross(r, s)
    u = cross_q_minus_p_r / cross_r_s
    t = cross_q_minus_p_s / cross_r_s
    t_in_range = (0 <= t) * (t <= 1)
    u_in_range = (0 <= u) * (u <= 1)

    cross_r_s_is_zero = cross_r_s == 0

    distance = torch.full(
        point_a1.shape[:-1],
        float("inf"),
        device=point_a1.device,
        dtype=torch.float32,
    )
    point = torch.full(
        point_a1.shape,
        float("inf"),
        device=point_a1.device,
        dtype=torch.float32,
    )

    condition = ~cross_r_s_is_zero * u_in_range * t_in_range
    condition_exp = condition.expand(point.shape)

    point = torch.where(condition_exp, p + t * r, point)
    distance = torch.where(condition.squeeze(-1), 0.0, distance)

    return point, distance


def _get_closest_point_box(box_pos, box_rot, box_width, box_length, test_point_pos):
    if not isinstance(box_width, torch.Tensor):
        box_width = torch.tensor(
            box_width, dtype=torch.float32, device=box_pos.device
        ).expand(box_pos.shape[0])
    if not isinstance(box_length, torch.Tensor):
        box_length = torch.tensor(
            box_length, dtype=torch.float32, device=box_pos.device
        ).expand(box_pos.shape[0])

    closest_points = _get_all_points_box(
        box_pos, box_rot, box_width, box_length, test_point_pos
    )
    closest_point = torch.full(
        box_pos.shape,
        float("inf"),
        device=box_pos.device,
        dtype=torch.float32,
    )
    distance = torch.full(
        box_pos.shape[:-1],
        float("inf"),
        device=box_pos.device,
        dtype=torch.float32,
    )
    for p in closest_points:
        d = torch.linalg.vector_norm(test_point_pos - p, dim=-1)
        is_closest = d < distance
        is_closest_exp = is_closest.unsqueeze(-1).expand(p.shape)
        closest_point = torch.where(is_closest_exp, p, closest_point)
        distance = torch.where(is_closest, d, distance)

    return closest_point


def _get_all_lines_box(box_pos, box_rot, box_width, box_length):
    # Rotate normal vector by the angle of the box
    rotated_vector = torch.cat([box_rot.cos(), box_rot.sin()], dim=-1)
    rot_2 = box_rot + torch.pi / 2
    rotated_vector2 = torch.cat([rot_2.cos(), rot_2.sin()], dim=-1)

    expanded_half_box_length = box_length.unsqueeze(-1).expand(rotated_vector.shape) / 2
    expanded_half_box_width = box_width.unsqueeze(-1).expand(rotated_vector.shape) / 2

    # Middle points of the sides
    p1 = box_pos + rotated_vector * expanded_half_box_length
    p2 = box_pos - rotated_vector * expanded_half_box_length
    p3 = box_pos + rotated_vector2 * expanded_half_box_width
    p4 = box_pos - rotated_vector2 * expanded_half_box_width

    ps = []
    rots = []
    lengths = []
    for i, p in enumerate([p1, p2, p3, p4]):
        ps.append(p)
        rots.append(box_rot + torch.pi / 2 if i <= 1 else box_rot)
        lengths.append(box_width if i <= 1 else box_length)

    return (
        torch.stack(ps, dim=0),
        torch.stack(rots, dim=0),
        torch.stack(lengths, dim=0),
    )


def _get_closest_line_box(
    box_pos, box_rot, box_width, box_length, line_pos, line_rot, line_length
):
    if not isinstance(box_width, torch.Tensor):
        box_width = torch.tensor(
            box_width, dtype=torch.float32, device=box_pos.device
        ).expand(box_pos.shape[0])
    if not isinstance(box_length, torch.Tensor):
        box_length = torch.tensor(
            box_length, dtype=torch.float32, device=box_pos.device
        ).expand(box_pos.shape[0])
    if not isinstance(line_length, torch.Tensor):
        line_length = torch.tensor(
            line_length, dtype=torch.float32, device=line_pos.device
        ).expand(line_pos.shape[0])

    lines_pos, lines_rot, lines_length = _get_all_lines_box(
        box_pos, box_rot, box_width, box_length
    )

    closest_point_1 = torch.full(
        box_pos.shape,
        float("inf"),
        device=box_pos.device,
        dtype=torch.float32,
    )
    closest_point_2 = torch.full(
        box_pos.shape,
        float("inf"),
        device=box_pos.device,
        dtype=torch.float32,
    )
    distance = torch.full(
        box_pos.shape[:-1],
        float("inf"),
        device=box_pos.device,
        dtype=torch.float32,
    )
    ps_box, ps_line = _get_closest_points_line_line(
        lines_pos,
        lines_rot,
        lines_length,
        line_pos.unsqueeze(0).expand(lines_pos.shape),
        line_rot.unsqueeze(0).expand(lines_rot.shape),
        line_length.unsqueeze(0).expand(lines_length.shape),
    )

    for p_box, p_line in zip(ps_box.unbind(0), ps_line.unbind(0)):
        d = torch.linalg.vector_norm(p_box - p_line, dim=-1)
        is_closest = d < distance
        is_closest_exp = is_closest.unsqueeze(-1).expand(closest_point_1.shape)
        closest_point_1 = torch.where(is_closest_exp, p_box, closest_point_1)
        closest_point_2 = torch.where(is_closest_exp, p_line, closest_point_2)
        distance = torch.where(is_closest, d, distance)
    return closest_point_1, closest_point_2


def _get_all_points_box(box_pos, box_rot, box_width, box_length, test_point_pos):
    lines_pos, lines_rot, lines_length = _get_all_lines_box(
        box_pos, box_rot, box_width, box_length
    )

    closest_points = _get_closest_point_line(
        lines_pos,
        lines_rot,
        lines_length,
        test_point_pos.unsqueeze(0).expand(lines_pos.shape),
    ).unbind(0)

    return closest_points


def _get_closest_point_line(
    line_pos,
    line_rot,
    line_length,
    test_point_pos,
    limit_to_line_length: bool = True,
):
    assert line_rot.shape[-1] == 1
    if not isinstance(line_length, torch.Tensor):
        line_length = torch.tensor(
            line_length, dtype=torch.float32, device=line_pos.device
        ).expand(line_rot.shape)
    # Rotate it by the angle of the line
    rotated_vector = torch.cat([line_rot.cos(), line_rot.sin()], dim=-1)
    # Get distance between line and sphere
    delta_pos = line_pos - test_point_pos
    # Dot product of distance and line vector
    dot_p = (delta_pos * rotated_vector).sum(-1).unsqueeze(-1)
    # Coordinates of the closes point
    sign = torch.sign(dot_p)
    distance_from_line_center = (
        torch.minimum(
            torch.abs(dot_p),
            (line_length / 2).view(dot_p.shape),
        )
        if limit_to_line_length
        else torch.abs(dot_p)
    )
    closest_point = line_pos - sign * distance_from_line_center * rotated_vector
    return closest_point
