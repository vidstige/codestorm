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
    def __init__(self, color, radius, z=0, label=None):
        self.color = color
        self.radius = radius
        self.z = z
        self.label = label


class Renderer:
    default_properties = RenderProperties((0.42, 0.22, 1), 3)
    text_color = (0, 0, 0)
    
    def __init__(self, output, simulation, resolution, bg=(0, 0, 0)):
        self.output = output
        self.simulation = simulation
        width, height = resolution
        self.surface = cairo.ImageSurface(cairo.Format.RGB24, width, height)
        self.bg = bg
        self.properties = {}
    
    def render(self):
        surface = self.surface
        
        pattern = cairo.RadialGradient(
            0.5, 0.5, 0,
            0.5, 0.5, 0.25)
        pattern.add_color_stop_rgba(0, 0, 0, 0, 1)
        pattern.add_color_stop_rgba(1, 0, 0, 0, 0)

        ctx = cairo.Context(surface)
        w, h = surface.get_width(), surface.get_height()

        clear(surface, color=self.bg)
        
        mx, my = w // 2, h // 2
        scale = min(w, h)
        items = [(self.properties.get(identifier, self.default_properties), position) for identifier, position in zip(self.simulation.identifiers, self.simulation.positions)]
        for properties, (x, y) in sorted(items, key=lambda item: item[0].z, reverse=True):
            ctx.save()
            ctx.translate(mx + x * scale, my + y * scale)
            ctx.scale(2*properties.radius, 2*properties.radius)
            ctx.rectangle(0, 0, 1, 1)
            ctx.clip()
            ctx.set_source(pattern)
            ctx.mask(pattern)
            ctx.restore()

            if properties.label:
                extents = ctx.text_extents(properties.label)
                ctx.move_to(mx + x * scale - (extents.width / 2), my + y * scale + extents.height + properties.radius)
                ctx.set_source_rgb(*self.text_color)
                ctx.show_text(properties.label)
                
        
        # overlay
        ctx.set_source_rgb(*self.text_color)
        ctx.move_to(8, 24)
        ctx.set_font_size(16)
        ctx.show_text(self.simulation.get_time().date().isoformat())

        self.output.write(surface.get_data())