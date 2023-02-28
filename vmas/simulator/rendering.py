"""
2D rendering framework
"""
#  Copyright (c) 2022-2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

from __future__ import division

import math
import os
import sys
from typing import Callable, Tuple, Optional, Union

import numpy as np
import pyglet
import six
import torch

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
        "On Ubuntu, you can run 'apt-get install python-opengl'. If you're running on a server, you may need a "
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
        self.text_lines = []
        self.onetime_geoms = []
        self.transform = Transform()
        self.bounds = np.array([0.,0.,0.,0.])

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
        self.bounds = np.array([left, right, bottom, top])
        scalex = self.width / (right - left)
        scaley = self.height / (top - bottom)
        self.transform = Transform(
            translation=(-left * scalex, -bottom * scaley), scale=(scalex, scaley)
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
        for geom in self.geoms:
            geom.render()
        for geom in self.onetime_geoms:
            geom.render()
        self.transform.disable()

        pyglet.gl.glMatrixMode(pyglet.gl.GL_PROJECTION)
        pyglet.gl.glLoadIdentity()
        gluOrtho2D(0, self.width, 0, self.height)
        for geom in self.text_lines:
            geom.render()

        arr = None
        if return_rgb_array:
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

        self.window.flip()
        self.onetime_geoms = []
        return arr

    # Convenience
    def draw_circle(self, radius=10, res=30, filled=True, **attrs):
        geom = make_circle(radius=radius, res=res, filled=filled)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_polygon(self, v, filled=True, **attrs):
        geom = make_polygon(v=v, filled=filled)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_polyline(self, v, **attrs):
        geom = make_polyline(v=v)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_line(self, start, end, **attrs):
        geom = Line(start, end)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def get_array(self):
        self.window.flip()
        image_data = (
            pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        )
        self.window.flip()
        arr = np.fromstring(image_data.data, dtype=np.uint8, sep="")
        arr = arr.reshape((self.height, self.width, 4))
        return arr[::-1, :, 0:3]


def _add_attrs(geom, attrs):
    if "color" in attrs:
        geom.set_color(*attrs["color"])
    if "linewidth" in attrs:
        geom.set_linewidth(attrs["linewidth"])


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


class TextLine:
    def __init__(self, window, idx):
        self.idx = idx
        self.window = window
        pyglet.font.add_file(os.path.join(os.path.dirname(__file__), "secrcode.ttf"))
        self.label = None
        self.set_text("")

    def render(self):
        if self.label is not None:
            self.label.draw()

    def set_text(self, text, font_size: int = 20):
        if pyglet.font.have_font("Courier"):
            font = "Courier"
        elif pyglet.font.have_font("Secret Code"):
            font = "Secret Code"
        else:
            return

        self.label = pyglet.text.Label(
            text,
            font_name=font,
            color=(0, 0, 0, 255),
            font_size=font_size,
            x=0,
            y=self.idx * 40 + 20,
            anchor_x="left",
            anchor_y="bottom",
        )

        self.label.draw()


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
        self.sprite = pyglet.sprite.Sprite(img=self.img, x=self.x, y=self.y)
        # self.sprite.update(scale_x=self.wx, scale_y=self.wy)
        self.sprite.update(scale=self.scale)
        # breakpoint()

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


def render_function_util(
        f: Callable,
        env,
        precision: Optional[float] = None,
        plot_range: Optional[Union[
            float,
            Tuple[float, float],
            Tuple[Tuple[float, float], Tuple[float, float]]]
        ] = None,
        cmap_range: Optional[Tuple[float, float]] = None
):

    geoms = []

    if plot_range is None:
        x_min, x_max, y_min, y_max = env.viewer.bounds.tolist()
        x_min -= precision
        y_min -= precision
        x_max += precision
        y_max += precision
    elif isinstance(plot_range, int) or isinstance(plot_range, float):
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
    # The width and height of a Pyglet image must be integers, so we do floor/ceil
    xpoints = np.arange(np.floor(x_min), np.ceil(x_max), precision)
    ypoints = np.arange(np.floor(y_min), np.ceil(y_max), precision)

    xgrid, ygrid = np.meshgrid(xpoints, ypoints)
    x = np.stack((xgrid, ygrid), axis=-1).reshape(-1,2)
    outputs = f(x, env=env)

    if isinstance(outputs, torch.Tensor):
        outputs = outputs.detach().cpu().numpy()

    if outputs.ndim==1 or outputs.shape[-1] == 1:
        if outputs.ndim > 1:
            outputs = outputs.squeeze(axis=1)
        if cmap_range is None:
            minimum_range = 0.1
            low = np.min(outputs)
            high = np.max(outputs)
            high = max(high, low+minimum_range)
            cmap_range = [low, high]
        outputs = colormap(outputs, low=cmap_range[0], high=cmap_range[1], alpha=1)

    img = outputs.reshape(xgrid.shape[0], xgrid.shape[1], outputs.shape[-1])

    # To account for the larger plotting area, we set the added regions to be blank
    x_left_rounding = int(np.floor((x_min - np.floor(x_min)) / precision))
    x_right_rounding = int(np.floor((np.ceil(x_max) - x_max) / precision))
    y_bottom_rounding = int(np.floor((y_min - np.floor(y_min)) / precision))
    y_top_rounding = int(np.floor((np.ceil(y_max) - y_max) / precision))
    if x_left_rounding > 0:
        img[:x_left_rounding,:,:] = 0
    if x_right_rounding > 0:
        img[-x_right_rounding:, :, :] = 0
    if x_right_rounding > 0:
        img[:,:y_bottom_rounding, :] = 0
    if y_top_rounding > 0:
        img[:,-y_top_rounding:, :] = 0

    img = img * 255
    img = np.transpose(img, (1,0,2))
    geom = Image(img, x=np.floor(x_min), y=np.floor(y_min), scale=precision)
    geoms.append(geom)

    # ### OLD METHOD ###
    # l, r, t, b = (
    #     0,
    #     precision,
    #     precision,
    #     0,
    # )
    # poly_points = [(l, b), (l, t), (r, t), (r, b)]
    # xpoints = np.arange(x_min, x_max, precision)
    # ypoints = np.arange(y_min, y_max, precision)
    # for ix, x in enumerate(xpoints):
    #     for iy, y in enumerate(ypoints):
    #         color = f(x, y)
    #         color = np.array(color).clip(0, 1)
    #         img[ix, iy, :] = color
    #         box = make_polygon(poly_points, draw_border=False)
    #         transform = Transform()
    #         transform.set_translation(x, y)
    #         box.add_attr(transform)
    #         box.set_color(*color)
    #         geoms.append(box)

    return geoms


def colormap(x, low=None, high=None, alpha=1.):
    cmap = np.array([[0.267004, 0.004874, 0.329415],
                     [0.278826, 0.17549, 0.483397],
                     [0.229739, 0.322361, 0.545706],
                     [0.172719, 0.448791, 0.557885],
                     [0.127568, 0.566949, 0.550556],
                     [0.157851, 0.683765, 0.501686],
                     [0.369214, 0.788888, 0.382914],
                     [0.678489, 0.863742, 0.189503]])
    res = cmap.shape[0]
    if low is None:
        low = np.min(x)
    if high is None:
        high = np.max(x)
    x = np.clip(x, low, high)
    x = (x - low) / (high - low) * (res-1)
    x_c0_idx = np.floor(x).astype(int)
    x_c1_idx = np.ceil(x).astype(int)
    x_c0 = cmap[x_c0_idx,:]
    x_c1 = cmap[x_c1_idx,:]
    t = (x - x_c0_idx)
    rgb = t[:,None] * (x_c1) + (1-t)[:,None] * x_c0
    colors = np.concatenate([rgb, alpha * np.ones((rgb.shape[0],1))], axis=-1)
    return colors


def make_circle(radius=10, res=30, filled=True):
    points = []
    for i in range(res):
        ang = 2 * math.pi * i / res
        points.append((math.cos(ang) * radius, math.sin(ang) * radius))
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


# ================================================================


class SimpleImageViewer(object):
    def __init__(self, display=None):
        self.window = None
        self.isopen = False
        self.display = display

    def imshow(self, arr):
        if self.window is None:
            height, width, channels = arr.shape
            self.window = pyglet.window.Window(
                width=width, height=height, display=self.display
            )
            self.width = width
            self.height = height
            self.isopen = True
        assert arr.shape == (
            self.height,
            self.width,
            3,
        ), "You passed in an image with the wrong number shape"
        image = pyglet.image.ImageData(
            self.width, self.height, "RGB", arr.tobytes(), pitch=self.width * -3
        )
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        image.blit(0, 0)
        self.window.flip()

    def close(self):
        if self.isopen:
            self.window.close()
            self.isopen = False

    def __del__(self):
        self.close()
