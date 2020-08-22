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
    @staticmethod
    def radial_pattern(color) -> cairo.RadialGradient:
        pattern = cairo.RadialGradient(
            0.5, 0.5, 0,
            0.5, 0.5, 0.25)
        r, g, b = color
        pattern.add_color_stop_rgba(0, r, g, b, 0.75)
        pattern.add_color_stop_rgba(1, r, g, b, 0)
        return pattern

    def __init__(self, color, radius, z=0, label=None):
        self.radius = radius
        self.z = z
        self.label = label
        self.pattern = RenderProperties.radial_pattern(color)
        

class Renderer:
    default_properties = RenderProperties((0.42, 0.22, 1), 3)
    
    def __init__(self, output, simulation, resolution, fg=(1, 1, 1), bg=(0, 0, 0)):
        self.output = output
        self.simulation = simulation
        width, height = resolution
        self.surface = cairo.ImageSurface(cairo.Format.RGB24, width, height)
        self.fg = fg
        self.bg = bg
        self.properties = {}
    
    def render(self):
        surface = self.surface

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
            ctx.set_source(properties.pattern)
            ctx.mask(properties.pattern)
            ctx.restore()

            if properties.label:
                extents = ctx.text_extents(properties.label)
                ctx.move_to(mx + x * scale - extents.x_bearing - (extents.width / 2) + properties.radius, my + y * scale + extents.height + properties.radius)
                ctx.set_source_rgb(*self.fg)
                ctx.show_text(properties.label)
                
        
        # overlay
        ctx.set_source_rgb(*self.fg)
        ctx.move_to(8, 24)
        ctx.set_font_size(16)
        ctx.show_text(self.simulation.get_time().date().isoformat())

        self.output.write(surface.get_data())