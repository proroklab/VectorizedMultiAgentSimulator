#  Copyright (c) ProrokLab.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
import importlib
import os
import typing
import warnings
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom

_has_matplotlib = importlib.util.find_spec("matplotlib") is not None

X = 0
Y = 1
Z = 2
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
VIEWER_DEFAULT_ZOOM = 1.2
INITIAL_VIEWER_SIZE = (700, 700)
LINE_MIN_DIST = 4 / 6e2
COLLISION_FORCE = 100
JOINT_FORCE = 130
TORQUE_CONSTRAINT_FORCE = 1

DRAG = 0.25
LINEAR_FRICTION = 0.0
ANGULAR_FRICTION = 0.0

DEVICE_TYPING = Union[torch.device, str, int]


AGENT_OBS_TYPE = Union[Tensor, Dict[str, Tensor]]
AGENT_INFO_TYPE = Dict[str, Tensor]
AGENT_REWARD_TYPE = Tensor

OBS_TYPE = Union[List[AGENT_OBS_TYPE], Dict[str, AGENT_OBS_TYPE]]
INFO_TYPE = Union[List[AGENT_INFO_TYPE], Dict[str, AGENT_INFO_TYPE]]
REWARD_TYPE = Union[List[AGENT_REWARD_TYPE], Dict[str, AGENT_REWARD_TYPE]]
DONE_TYPE = Tensor


class Color(Enum):
    RED = (0.75, 0.25, 0.25)
    GREEN = (0.25, 0.75, 0.25)
    BLUE = (0.25, 0.25, 0.75)
    LIGHT_GREEN = (0.45, 0.95, 0.45)
    WHITE = (0.75, 0.75, 0.75)
    GRAY = (0.25, 0.25, 0.25)
    BLACK = (0.15, 0.15, 0.15)
    ORANGE = (1.00, 0.50, 0)
    PINK = (0.97, 0.51, 0.75)
    PURPLE = (0.60, 0.31, 0.64)
    YELLOW = (0.87, 0.87, 0)


def override(cls):
    """Decorator for documenting method overrides."""

    def check_override(method):
        if method.__name__ not in dir(cls):
            raise NameError("{} does not override any method of {}".format(method, cls))
        return method

    return check_override


def _init_pyglet_device():
    available_devices = os.getenv("CUDA_VISIBLE_DEVICES")
    if available_devices is not None and len(available_devices) > 0:
        os.environ["PYGLET_HEADLESS_DEVICE"] = (
            available_devices.split(",")[0]
            if len(available_devices) > 1
            else available_devices
        )


class Observable:
    def __init__(self):
        self._observers = []

    def subscribe(self, observer):
        self._observers.append(observer)

    def notify_observers(self, *args, **kwargs):
        for obs in self._observers:
            obs.notify(self, *args, **kwargs)

    def unsubscribe(self, observer):
        self._observers.remove(observer)


class Observer(ABC):
    @abstractmethod
    def notify(self, observable, *args, **kwargs):
        raise NotImplementedError


def save_video(name: str, frame_list: List[np.array], fps: int):
    """Requres cv2"""
    import cv2

    video_name = name + ".mp4"

    # Produce a video
    video = cv2.VideoWriter(
        video_name,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,  # FPS
        (frame_list[0].shape[1], frame_list[0].shape[0]),
    )
    for img in frame_list:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(f"{name}.png", img)
        # break
        video.write(img)
    video.release()


def x_to_rgb_colormap(
    x: np.ndarray,
    low: float = None,
    high: float = None,
    alpha: float = 1.0,
    cmap_name: str = "viridis",
    cmap_res: int = 10,
):
    from matplotlib import cm

    colormap = cm.get_cmap(cmap_name, cmap_res)(range(cmap_res))[:, :-1]
    if low is None:
        low = np.min(x)
    if high is None:
        high = np.max(x)
    x = np.clip(x, low, high)
    if high - low > 1e-5:
        x = (x - low) / (high - low) * (cmap_res - 1)
    x_c0_idx = np.floor(x).astype(int)
    x_c1_idx = np.ceil(x).astype(int)
    x_c0 = colormap[x_c0_idx, :]
    x_c1 = colormap[x_c1_idx, :]
    t = x - x_c0_idx
    rgb = t[:, None] * x_c1 + (1 - t)[:, None] * x_c0
    colors = np.concatenate([rgb, alpha * np.ones((rgb.shape[0], 1))], axis=-1)
    return colors


def extract_nested_with_index(data: Union[Tensor, Dict[str, Tensor]], index: int):
    if isinstance(data, Tensor):
        return data[index]
    elif isinstance(data, Dict):
        return {
            key: extract_nested_with_index(value, index) for key, value in data.items()
        }
    else:
        raise NotImplementedError(f"Invalid type of data {data}")


class TorchUtils:
    @staticmethod
    def clamp_with_norm(tensor: Tensor, max_norm: float):
        norm = torch.linalg.vector_norm(tensor, dim=-1)
        new_tensor = (tensor / norm.unsqueeze(-1)) * max_norm
        cond = (norm > max_norm).unsqueeze(-1).expand(tensor.shape)
        tensor = torch.where(cond, new_tensor, tensor)
        return tensor

    @staticmethod
    def rotate_vector(vector: Tensor, angle: Tensor):
        if len(angle.shape) == len(vector.shape):
            angle = angle.squeeze(-1)

        assert vector.shape[:-1] == angle.shape
        assert vector.shape[-1] == 2

        cos = torch.cos(angle)
        sin = torch.sin(angle)
        return torch.stack(
            [
                vector[..., X] * cos - vector[..., Y] * sin,
                vector[..., X] * sin + vector[..., Y] * cos,
            ],
            dim=-1,
        )

    @staticmethod
    def cross(vector_a: Tensor, vector_b: Tensor):
        return (
            vector_a[..., X] * vector_b[..., Y] - vector_a[..., Y] * vector_b[..., X]
        ).unsqueeze(-1)

    @staticmethod
    def compute_torque(f: Tensor, r: Tensor) -> Tensor:
        return TorchUtils.cross(r, f)

    @staticmethod
    def to_numpy(data: Union[Tensor, Dict[str, Tensor], List[Tensor]]):
        if isinstance(data, Tensor):
            return data.cpu().detach().numpy()
        elif isinstance(data, Dict):
            return {key: TorchUtils.to_numpy(value) for key, value in data.items()}
        elif isinstance(data, Sequence):
            return [TorchUtils.to_numpy(value) for value in data]
        else:
            raise NotImplementedError(f"Invalid type of data {data}")

    @staticmethod
    def recursive_clone(value: Union[Dict[str, Tensor], Tensor]):
        if isinstance(value, Tensor):
            return value.clone()
        else:
            return {key: TorchUtils.recursive_clone(val) for key, val in value.items()}

    @staticmethod
    def recursive_require_grad_(value: Union[Dict[str, Tensor], Tensor, List[Tensor]]):
        if isinstance(value, Tensor) and torch.is_floating_point(value):
            value.requires_grad_(True)
        elif isinstance(value, Dict):
            for val in value.values():
                TorchUtils.recursive_require_grad_(val)
        else:
            for val in value:
                TorchUtils.recursive_require_grad_(val)

    @staticmethod
    def where_from_index(env_index, new_value, old_value):
        mask = torch.zeros_like(old_value, dtype=torch.bool, device=old_value.device)
        mask[env_index] = True
        return torch.where(mask, new_value, old_value)


class ScenarioUtils:
    @staticmethod
    def spawn_entities_randomly(
        entities,
        world,
        env_index: int,
        min_dist_between_entities: float,
        x_bounds: Tuple[int, int],
        y_bounds: Tuple[int, int],
        occupied_positions: Tensor = None,
        disable_warn: bool = False,
    ):
        batch_size = world.batch_dim if env_index is None else 1

        if occupied_positions is None:
            occupied_positions = torch.zeros(
                (batch_size, 0, world.dim_p), device=world.device
            )

        for entity in entities:
            pos = ScenarioUtils.find_random_pos_for_entity(
                occupied_positions,
                env_index,
                world,
                min_dist_between_entities,
                x_bounds,
                y_bounds,
                disable_warn,
            )
            occupied_positions = torch.cat([occupied_positions, pos], dim=1)
            entity.set_pos(pos.squeeze(1), batch_index=env_index)

    @staticmethod
    def find_random_pos_for_entity(
        occupied_positions: torch.Tensor,
        env_index: int,
        world,
        min_dist_between_entities: float,
        x_bounds: Tuple[int, int],
        y_bounds: Tuple[int, int],
        disable_warn: bool = False,
    ):
        batch_size = world.batch_dim if env_index is None else 1

        pos = None
        tries = 0
        while True:
            proposed_pos = torch.cat(
                [
                    torch.empty(
                        (batch_size, 1, 1),
                        device=world.device,
                        dtype=torch.float32,
                    ).uniform_(*x_bounds),
                    torch.empty(
                        (batch_size, 1, 1),
                        device=world.device,
                        dtype=torch.float32,
                    ).uniform_(*y_bounds),
                ],
                dim=2,
            )
            if pos is None:
                pos = proposed_pos
            if occupied_positions.shape[1] == 0:
                break

            dist = torch.cdist(occupied_positions, pos)
            overlaps = torch.any((dist < min_dist_between_entities).squeeze(2), dim=1)
            if torch.any(overlaps, dim=0):
                pos[overlaps] = proposed_pos[overlaps]
            else:
                break
            tries += 1
            if tries > 50_000 and not disable_warn:
                warnings.warn(
                    "It is taking many iterations to spawn the entity, make sure the bounds or "
                    "the min_dist_between_entities are not too tight to fit all entities."
                    "You can disable this warning by setting disable_warn=True"
                )
        return pos

    @staticmethod
    def check_kwargs_consumed(dictionary_of_kwargs: Dict, warn: bool = True):
        if len(dictionary_of_kwargs) > 0:
            message = f"Scenario kwargs: {dictionary_of_kwargs} passed but not used by the scenario."
            if warn:
                warnings.warn(
                    message + " This will turn into an error in future versions."
                )
            else:
                raise ValueError(message)

    @staticmethod
    def render_agent_indices(
        scenario, env_index: int, start_from: int = 0, exclude: List = None
    ) -> "List[Geom]":
        from vmas.simulator import rendering

        aspect_r = scenario.viewer_size[X] / scenario.viewer_size[Y]
        if aspect_r > 1:
            dimensional_ratio = (aspect_r, 1)
        else:
            dimensional_ratio = (1, 1 / aspect_r)

        geoms = []
        for i, entity in enumerate(scenario.world.agents):
            if exclude is not None and entity in exclude:
                continue
            i = i + start_from
            line = rendering.TextLine(
                text=str(i),
                font_size=15,
                x=(
                    (entity.state.pos[env_index, X] * scenario.viewer_size[X])
                    / (scenario.viewer_zoom**2 * dimensional_ratio[X] * 2)
                    + scenario.viewer_size[X] / 2
                ),
                y=(
                    (entity.state.pos[env_index, Y] * scenario.viewer_size[Y])
                    / (scenario.viewer_zoom**2 * dimensional_ratio[Y] * 2)
                    + scenario.viewer_size[Y] / 2
                ),
            )
            geoms.append(line)
        return geoms

    @staticmethod
    def plot_entity_rotation(entity, env_index: int, length: float = 0.15) -> "Geom":
        from vmas.simulator import rendering

        color = entity.color
        line = rendering.Line(
            (0, 0),
            (length, 0),
            width=2,
        )
        xform = rendering.Transform()
        xform.set_rotation(entity.state.rot[env_index])
        xform.set_translation(*entity.state.pos[env_index])
        line.add_attr(xform)
        line.set_color(*color)
        return line
