import itertools

from github import Github
import cairo
import numpy as np


TAU = 2 * np.pi


class Simulation:
    def __init__(self):
        self.t = 0
        self.positions = np.empty((0, 2))
        self.springs = []

    def velocities(self, positions, t: float):
        return 0

    def step(self, dt: float):
        p = self.positions
        v = self.velocities
        t = self.t

        k1 = dt * v(p, t)
        k2 = dt * v(p + 0.5 * k1, t + 0.5 * dt)
        k3 = dt * v(p + 0.5 * k2, t + 0.5 * dt)
        k4 = dt * v(p + k3, t + dt)
        self.positions = p + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        self.t += dt


def clear(target: cairo.ImageSurface, color=(1, 1, 1)) -> None:
    r, g, b = color
    ctx = cairo.Context(target)
    ctx.rectangle(0, 0, target.get_width(), target.get_height())
    ctx.set_source_rgb(r, g, b)
    ctx.fill()


class Renderer:
    def __init__(self, simulation, resolution, bg=(0, 0, 0)):
        self.simulation = simulation
        width, height = resolution
        self.surface = cairo.ImageSurface(cairo.Format.RGB24, width, height)
        self.bg = bg
    
    def render(self):
        surface = self.surface
        
        ctx = cairo.Context(surface)
        ctx.set_source_rgb(0.42, 0.22, 4)
        #w, h = surface.get_width(), surface.get_height()

        clear(surface, color=self.bg)
        
        for x, y in self.simulation.positions:
            ctx.move_to(x, y)
            ctx.arc(x, y, 1, 0, TAU)
            ctx.fill()

        f.write(surface.get_data())


# or using an access token
with open('.token') as f:
    token = f.read().strip()

g = Github(token)

#volumental = g.get_organization("Volumental")
#for repo in volumental.get_repos():
#    print(repo)
reconstruction = g.get_repo("Volumental/Reconstruction")
commits = reconstruction.get_commits()

simulation = Simulation()
renderer = Renderer(simulation, (320, 200))

for commit in itertools.islice(commits, 10):
    #print(commit.files)
