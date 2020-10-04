from typing import Iterable, Sequence, Tuple

import cairo
import numpy as np


TAU = 2 * np.pi


def remove_prefix(text: str, prefixes: Iterable[str]) -> str:
    for prefix in prefixes:
        if text.startswith(prefix):
            return text[len(prefix):]
    return text


class Color:
    BLACK = (0, 0, 0)
    WHITE = (1, 1, 1)
    def __init__(self, r: float, g: float, b: float):
        self.r = r
        self.g = g
        self.b = b

    @staticmethod
    def parse(raw: str):
        r, g, b = bytes.fromhex(remove_prefix(raw, ['#', '0x']))
        return Color(r / 255, g / 255, b / 255)



def clear(target: cairo.ImageSurface, color: Color=Color.WHITE) -> None:
    ctx = cairo.Context(target)
    ctx.rectangle(0, 0, target.get_width(), target.get_height())
    ctx.set_source_rgb(color.r, color.g, color.b)
    ctx.fill()


class RenderProperties:
    @staticmethod
    def radial_pattern(color: Color) -> cairo.RadialGradient:
        pattern = cairo.RadialGradient(
            0.5, 0.5, 0.05,
            0.5, 0.5, 0.25)
        r, g, b = color.r, color.g, color.b
        pattern.add_color_stop_rgba(0, r, g, b, 0.5)
        pattern.add_color_stop_rgba(1, r, g, b, 0)
        return pattern

    def __init__(self, color, radius, z=0, label=None):
        self.radius = radius
        self.z = z
        self.pattern = RenderProperties.radial_pattern(color)
        self.color = color
        

class Rectangle:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def __str__(self) -> str:
        return "{}, {}, {}, {}".format(self.x, self.y, self.w, self.h)

    def left(self) -> float:
        return self.x
    
    def right(self) -> float:
        return self.x + self.w

    def top(self) -> float:
        return self.y

    def bottom(self) -> float:
        return self.y + self.h
    
    def overlaps(self, other) -> bool:
        return self.left() < other.right() and self.right() > other.left() and self.top() < other.bottom() and self.bottom() > other.top()


def mid_y(label: Tuple[str, Rectangle]) -> float:
    _, r = label
    return r.y + r.h / 2

def layout(rectangles: Sequence[Tuple[str, Rectangle]]) -> Sequence[Tuple[str, Rectangle]]:
    # Find overlapping rectangles
    for _, a in sorted(rectangles, key=mid_y):
        for _, b in rectangles:
            if a != b and a.overlaps(b):
                diff = a.bottom() - b.top()
                a.y -= diff / 2                
                b.y += diff / 2
    return rectangles


class Renderer:
    default_properties = RenderProperties(Color(0.42, 0.22, 1), 3)
    
    def __init__(self, output, simulation, labels, resolution, fg=Color(1, 1, 1), bg=Color(0, 0, 0), legend=None):
        self.output = output
        self.simulation = simulation
        self.labels = labels
        width, height = resolution
        self.surface = cairo.ImageSurface(cairo.Format.RGB24, width, height)
        self.fg = fg
        self.bg = bg
        self.legend = legend
        self.properties = {}
    
    def render(self):
        surface = self.surface

        ctx = cairo.Context(surface)
        w, h = surface.get_width(), surface.get_height()

        clear(surface, color=self.bg)
        
        mx, my = w // 2, h // 2
        scale = min(w, h)
        positions = self.simulation.positions()
        items = [(identifier, self.properties.get(identifier, self.default_properties), positions[index]) for identifier, index in self.simulation.bodies.items()]

        labels = []
        for identifier, properties, (x, y) in sorted(items, key=lambda item: item[1].z, reverse=True):
            ctx.save()
            ctx.translate(mx + x * scale, my + y * scale)
            ctx.scale(2 * properties.radius, 2 * properties.radius)
            ctx.rectangle(0, 0, 1, 1)
            ctx.clip()
            ctx.set_source(properties.pattern)
            ctx.mask(properties.pattern)
            ctx.restore()
        
            if identifier in self.labels:
                label = identifier
                extents = ctx.text_extents(label)
                rectangle = Rectangle(mx + x * scale - extents.x_bearing - (extents.width / 2) + properties.radius, my + y * scale + extents.height + properties.radius, extents.width, extents.height)
                labels.append((label, rectangle))

        ctx.set_source_rgb(self.fg.r, self.fg.g, self.fg.b)
        for label, rectangle in layout(labels):
            ctx.move_to(rectangle.x, rectangle.y)
            ctx.show_text(label)

        # overlay
        ctx.set_source_rgb(self.fg.r, self.fg.g, self.fg.b)
        ctx.move_to(8, 24)
        ctx.set_font_size(16)
        ctx.show_text(self.simulation.get_time().date().isoformat())

        # legend
        y = 48
        for color, label in self.legend():
            ctx.set_source_rgb(color.r, color.g, color.b)
            ctx.arc(16, y - 5, 6, 0, TAU)
            ctx.fill()
            ctx.move_to(24, y)
            ctx.set_source_rgb(self.fg.r, self.fg.g, self.fg.b)
            ctx.show_text(label)
            y += 16

        self.output.write(surface.get_data())