"""
2D rendering framework
"""

#  Copyright (c) ProrokLab.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import division

import math
import os
import sys
from itertools import chain
from typing import Callable, Optional, Tuple, Union

import numpy as np
import pyglet
import six
import torch

from vmas.simulator.utils import TorchUtils, x_to_rgb_colormap

try:
    from pyglet.gl import (
        GL_BLEND,
        GL_LINE_LOOP,
        GL_LINE_SMOOTH,
        GL_LINE_SMOOTH_HINT,
        GL_LINE_STIPPLE,
        GL_LINE_STRIP,
        GL_LINES,
        GL_NICEST,
        GL_ONE_MINUS_SRC_ALPHA,
        GL_POINTS,
        GL_POLYGON,
        GL_QUADS,
        GL_SRC_ALPHA,
        GL_TRIANGLES,
        glBegin,
        glBlendFunc,
        glClearColor,
        glColor4f,
        glDisable,
        glEnable,
        glEnd,
        glHint,
        glLineStipple,
        glLineWidth,
        glPopMatrix,
        glPushMatrix,
        glRotatef,
        glScalef,
        glTranslatef,
        gluOrtho2D,
        glVertex2f,
        glVertex3f,
    )
except ImportError:
    raise ImportError(
        "Error occurred while running `from pyglet.gl import *`, HINT: make sure you have OpenGL installed. "
        "On Ubuntu, you can run 'apt-get install python3-opengl'. If you're running on a server, you may need a "
        "virtual frame buffer; something like this should work:"
        "'xvfb-run -s \"-screen 0 1400x900x24\" python <your_script.py>'"
    )


if "Apple" in sys.version:
    if "DYLD_FALLBACK_LIBRARY_PATH" in os.environ:
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] += ":/usr/lib"
        # (JDS 2016/04/15): avoid bug on Anaconda 2.3.0 / Yosemite

RAD2DEG = 57.29577951308232


def get_display(spec):
    """Convert a display specification (such as :0) into an actual Display
    object.

    Pyglet only supports multiple Displays on Linux.
    """
    if spec is None:
        return None
    elif isinstance(spec, six.string_types):
        return pyglet.canvas.Display(spec)
    else:
        raise RuntimeError(
            "Invalid display specification: {}. (Must be a string like :0 or None.)".format(
                spec
            )
        )


class Viewer(object):
    def __init__(self, width, height, display=None, visible=True):
        display = get_display(display)

        self.width = width
        self.height = height

        self.window = pyglet.window.Window(
            width=width, height=height, display=display, visible=visible
        )
        self.window.on_close = self.window_closed_by_user

        self.geoms = []
        self.onetime_geoms = []
        self.transform = Transform()
        self.bounds = None

        glEnable(GL_BLEND)
        # glEnable(GL_MULTISAMPLE)
        glEnable(GL_LINE_SMOOTH)
        # glHint(GL_LINE_SMOOTH_HINT, GL_DONT_CARE)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glLineWidth(2.0)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def close(self):
        self.window.close()

    def window_closed_by_user(self):
        self.close()

    def set_bounds(self, left, right, bottom, top):
        assert right > left and top > bottom
        self.bounds = torch.tensor([left, right, bottom, top], device=left.device)
        scalex = self.width / (right - left)
        scaley = self.height / (top - bottom)
        self.transform = Transform(
            translation=(-left * scalex, -bottom * scaley),
            scale=(scalex, scaley),
        )

    def add_geom(self, geom):
        self.geoms.append(geom)

    def add_onetime(self, geom):
        self.onetime_geoms.append(geom)

    def add_onetime_list(self, geoms):
        self.onetime_geoms.extend(geoms)

    def render(self, return_rgb_array=False):
        glClearColor(1, 1, 1, 1)

        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()

        self.transform.enable()

        text_lines = []
        for geom in chain(self.geoms, self.onetime_geoms):
            if isinstance(geom, TextLine):
                text_lines.append(geom)
            else:
                geom.render()

        self.transform.disable()

        for text in text_lines:
            text.render()

        pyglet.gl.glMatrixMode(pyglet.gl.GL_PROJECTION)
        pyglet.gl.glLoadIdentity()
        gluOrtho2D(0, self.width, 0, self.height)

        arr = None
        if return_rgb_array:
            arr = self.get_array()
        self.window.flip()
        self.onetime_geoms = []
        return arr

    def get_array(self):
        buffer = pyglet.image.get_buffer_manager().get_color_buffer()
        image_data = buffer.get_image_data()
        arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
        # In https://github.com/openai/gym-http-api/issues/2, we
        # discovered that someone using Xmonad on Arch was having
        # a window of size 598 x 398, though a 600 x 400 window
        # was requested. (Guess Xmonad was preserving a pixel for
        # the boundary.) So we use the buffer height/width rather
        # than the requested one.
        arr = arr.reshape((buffer.height, buffer.width, 4))
        arr = arr[::-1, :, 0:3]
        return arr


class Geom(object):
    def __init__(self):
        self._color = Color((0, 0, 0, 1.0))
        self.attrs = [self._color]

    def render(self):
        for attr in reversed(self.attrs):
            attr.enable()
        self.render1()
        for attr in self.attrs:
            attr.disable()

    def render1(self):
        raise NotImplementedError

    def add_attr(self, attr):
        self.attrs.append(attr)

    def set_color(self, r, g, b, alpha=1):
        self._color.vec4 = (r, g, b, alpha)


class Attr(object):
    def enable(self):
        raise NotImplementedError

    def disable(self):
        pass


class Transform(Attr):
    def __init__(self, translation=(0.0, 0.0), rotation=0.0, scale=(1, 1)):
        self.set_translation(*translation)
        self.set_rotation(rotation)
        self.set_scale(*scale)

    def enable(self):
        glPushMatrix()
        glTranslatef(
            self.translation[0], self.translation[1], 0
        )  # translate to GL loc ppint
        glRotatef(RAD2DEG * self.rotation, 0, 0, 1.0)
        glScalef(self.scale[0], self.scale[1], 1)

    def disable(self):
        glPopMatrix()

    def set_translation(self, newx, newy):
        self.translation = (float(newx), float(newy))

    def set_rotation(self, new):
        self.rotation = float(new)

    def set_scale(self, newx, newy):
        self.scale = (float(newx), float(newy))


class Color(Attr):
    def __init__(self, vec4):
        self.vec4 = vec4

    def enable(self):
        glColor4f(*self.vec4)


class LineStyle(Attr):
    def __init__(self, style):
        self.style = style

    def enable(self):
        glEnable(GL_LINE_STIPPLE)
        glLineStipple(1, self.style)

    def disable(self):
        glDisable(GL_LINE_STIPPLE)


class LineWidth(Attr):
    def __init__(self, stroke):
        self.stroke = stroke

    def enable(self):
        glLineWidth(self.stroke)


class TextLine(Geom):
    def __init__(
        self,
        text: str = "",
        font_size: int = 15,
        x: float = 0.0,
        y: float = 0.0,
    ):
        super().__init__()

        if pyglet.font.have_font("Courier"):
            font = "Courier"
        elif pyglet.font.have_font("Secret Code"):
            font = "Secret Code"
        else:
            font = None

        self.label = pyglet.text.Label(
            text,
            font_name=font,
            font_size=font_size,
            color=(0, 0, 0, 255),
            x=x,
            y=y,
            anchor_x="left",
            anchor_y="bottom",
        )

    def render1(self):
        if self.label is not None:
            self.label.draw()

    def set_text(self, text, font_size: Optional[int] = None):
        self.label.text = text
        if font_size is not None:
            self.label.font_size = font_size


class Point(Geom):
    def __init__(self):
        Geom.__init__(self)

    def render1(self):
        glBegin(GL_POINTS)  # draw point
        glVertex3f(0.0, 0.0, 0.0)
        glEnd()


class Image(Geom):
    def __init__(self, img, x, y, scale):
        super().__init__()
        self.x = x
        self.y = y
        self.scale = scale
        img_shape = img.shape
        img = img.astype(np.uint8).reshape(-1)
        tex_data = (pyglet.gl.GLubyte * img.size)(*img)
        pyg_img = pyglet.image.ImageData(
            img_shape[1],
            img_shape[0],
            "RGBA",
            tex_data,
            pitch=img_shape[1] * img_shape[2] * 1,  # width x channels x bytes per pixel
        )
        self.img = pyg_img
        self.sprite = pyglet.sprite.Sprite(
            img=self.img, x=self.x, y=self.y, subpixel=True
        )
        self.sprite.update(scale=self.scale)

    def render1(self):
        self.sprite.draw()


class FilledPolygon(Geom):
    def __init__(self, v, draw_border: float = True):
        Geom.__init__(self)
        self.draw_border = draw_border
        self.v = v

    def render1(self):
        if len(self.v) == 4:
            glBegin(GL_QUADS)
        elif len(self.v) > 4:
            glBegin(GL_POLYGON)
        else:
            glBegin(GL_TRIANGLES)
        for p in self.v:
            glVertex3f(p[0], p[1], 0)  # draw each vertex
        glEnd()

        if self.draw_border:
            color = (
                self._color.vec4[0] * 0.5,
                self._color.vec4[1] * 0.5,
                self._color.vec4[2] * 0.5,
                self._color.vec4[3] * 0.5,
            )
            glColor4f(*color)
            glBegin(GL_LINE_LOOP)
            for p in self.v:
                glVertex3f(p[0], p[1], 0)  # draw each vertex
            glEnd()


class Compound(Geom):
    def __init__(self, gs):
        Geom.__init__(self)
        self.gs = gs
        for g in self.gs:
            g.attrs = [a for a in g.attrs if not isinstance(a, Color)]

    def render1(self):
        for g in self.gs:
            g.render()


class PolyLine(Geom):
    def __init__(self, v, close):
        Geom.__init__(self)
        self.v = v
        self.close = close
        self.linewidth = LineWidth(1)
        self.add_attr(self.linewidth)

    def render1(self):
        glBegin(GL_LINE_LOOP if self.close else GL_LINE_STRIP)
        for p in self.v:
            glVertex3f(p[0], p[1], 0)  # draw each vertex
        glEnd()

    def set_linewidth(self, x):
        self.linewidth.stroke = x


class Line(Geom):
    def __init__(self, start=(0.0, 0.0), end=(0.0, 0.0), width: float = 1):
        Geom.__init__(self)
        self.start = start
        self.end = end
        self.linewidth = LineWidth(width)
        self.add_attr(self.linewidth)

    def set_linewidth(self, x):
        self.linewidth.stroke = x

    def render1(self):
        glBegin(GL_LINES)
        glVertex2f(*self.start)
        glVertex2f(*self.end)
        glEnd()


class Grid(Geom):
    def __init__(self, spacing: float = 0.1, length: float = 50, width: float = 0.5):
        Geom.__init__(self)
        self.spacing = spacing
        self.linewidth = LineWidth(width)
        self.length = length
        self.add_attr(self.linewidth)

    def set_linewidth(self, x):
        self.linewidth.stroke = x

    def render1(self):
        for point in np.arange(-self.length / 2, self.length / 2, self.spacing):
            glBegin(GL_LINES)
            glVertex2f(point, -self.length / 2)
            glVertex2f(point, self.length / 2)
            glEnd()
            glBegin(GL_LINES)
            glVertex2f(-self.length / 2, point)
            glVertex2f(self.length / 2, point)
            glEnd()


def render_function_util(
    f: Callable,
    plot_range: Union[
        float,
        Tuple[float, float],
        Tuple[Tuple[float, float], Tuple[float, float]],
    ],
    precision: float = 0.01,
    cmap_range: Optional[Tuple[float, float]] = None,
    cmap_alpha: float = 1.0,
    cmap_name: str = "viridis",
):
    if isinstance(plot_range, int) or isinstance(plot_range, float):
        x_min = -plot_range
        y_min = -plot_range
        x_max = plot_range
        y_max = plot_range
    elif len(plot_range) == 2:
        if isinstance(plot_range[0], int) or isinstance(plot_range[0], float):
            x_min = -(plot_range[0])
            y_min = -(plot_range[1])
            x_max = plot_range[0]
            y_max = plot_range[1]
        else:
            x_min = plot_range[0][0]
            y_min = plot_range[1][0]
            x_max = plot_range[0][1]
            y_max = plot_range[1][1]

    xpoints = np.arange(x_min, x_max, precision)
    ypoints = np.arange(y_min, y_max, precision)

    ygrid, xgrid = np.meshgrid(ypoints, xpoints)
    pos = np.stack((xgrid, ygrid), axis=-1).reshape(-1, 2)
    pos_shape = pos.shape

    outputs = f(pos)

    if isinstance(outputs, torch.Tensor):
        outputs = TorchUtils.to_numpy(outputs)

    assert isinstance(outputs, np.ndarray)
    assert outputs.shape[0] == pos_shape[0]
    assert outputs.ndim <= 2

    if outputs.ndim == 2 and outputs.shape[1] == 1:
        outputs = outputs.squeeze(-1)
    elif outputs.ndim == 2:
        assert outputs.shape[1] == 4

    # Output is an alpha
    if outputs.ndim == 1:
        if cmap_range is None:
            cmap_range = [None, None]
        outputs = x_to_rgb_colormap(
            outputs,
            low=cmap_range[0],
            high=cmap_range[1],
            alpha=cmap_alpha,
            cmap_name=cmap_name,
        )

    img = outputs.reshape(xgrid.shape[0], xgrid.shape[1], outputs.shape[-1])

    img = img * 255
    img = np.transpose(img, (1, 0, 2))
    geom = Image(img, x=x_min, y=y_min, scale=precision)

    return geom


def make_circle(radius=10, res=30, filled=True, angle=2 * math.pi):
    return make_ellipse(
        radius_x=radius, radius_y=radius, res=res, filled=filled, angle=angle
    )


def make_ellipse(radius_x=10, radius_y=5, res=30, filled=True, angle=2 * math.pi):
    points = []
    for i in range(res):
        ang = -angle / 2 + angle * i / res
        points.append((math.cos(ang) * radius_x, math.sin(ang) * radius_y))
    if angle % (2 * math.pi) != 0:
        points.append((0, 0))
    if filled:
        return FilledPolygon(points)
    else:
        return PolyLine(points, True)


def make_polygon(v, filled=True, draw_border: float = True):
    if filled:
        return FilledPolygon(v, draw_border=draw_border)
    else:
        return PolyLine(v, True)


def make_polyline(v):
    return PolyLine(v, False)


def make_capsule(length, width):
    l, r, t, b = 0, length, width / 2, -width / 2
    box = make_polygon([(l, b), (l, t), (r, t), (r, b)])
    circ0 = make_circle(width / 2)
    circ1 = make_circle(width / 2)
    circ1.add_attr(Transform(translation=(length, 0)))
    geom = Compound([box, circ0, circ1])
    return geom


# ================================================================
