import cairo

import numpy as np


TAU = 2 * np.pi


def clear(target: cairo.ImageSurface, color=(1, 1, 1)) -> None:
    r, g, b = color
    ctx = cairo.Context(target)
    ctx.rectangle(0, 0, target.get_width(), target.get_height())
    ctx.set_source_rgb(r, g, b)
    ctx.fill()


class RenderProperties:
    def __init__(self, color, radius):
        self.color = color
        self.radius = radius


class Renderer:
    default_properties = RenderProperties((0.42, 0.22, 1), 3)

    def __init__(self, output, simulation, resolution, bg=(0, 0, 0)):
        self.output = output
        self.simulation = simulation
        width, height = resolution
        self.surface = cairo.ImageSurface(cairo.Format.RGB24, width, height)
        self.bg = bg
        self.properties = {}
    
    def render(self):
        surface = self.surface
        
        ctx = cairo.Context(surface)
        w, h = surface.get_width(), surface.get_height()

        clear(surface, color=self.bg)
        
        mx, my = w // 2, h // 2
        scale = min(w, h)
        for identifier, (x, y) in zip(self.simulation.identifiers, self.simulation.positions):
            properties = self.properties.get(identifier, self.default_properties)
            ctx.set_source_rgb(*properties.color)
            ctx.arc(mx + x * scale, my + y * scale, properties.radius, 0, TAU)
            ctx.fill()
        
        # overlay
        ctx.set_source_rgb(0, 0, 0)
        ctx.move_to(8, 24)
        ctx.set_font_size(16)
        ctx.show_text(self.simulation.get_time().date().isoformat())

        self.output.write(surface.get_data())