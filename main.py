import itertools
import sys

from github import Github
import cairo
import numpy as np


TAU = 2 * np.pi


class Simulation:
    def __init__(self):
        self.t = 0
        #self.positions = np.empty((0, 2))
        n = 10
        self.positions = np.random.rand(n, 2) * 100
        self.springs = []

    def velocities(self, positions, t: float):
        # compute spring forces

        v = np.zeros(positions.shape)
        # find indices annd lengths
        ii = [i for i, _, _ in self.springs]
        jj = [j for _, j, _ in self.springs]
        ll = [l for _, _, l in self.springs]
        stiffness = 1
        mass = 1
    
        # compute deltas
        di = positions[jj] - positions[ii]
        dj = positions[ii] - positions[jj]

        # compute lengths
        norms = np.linalg.norm(di, axis=-1)

        # compute force magnitudes
        forces = (norms - ll) * stiffness

        # compute accelerations and accumulate
        v[ii] += (di / norms[:, None]) * forces[:, None] / mass
        v[jj] += (dj / norms[:, None]) * forces[:, None] / mass
        return v

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
    def __init__(self, output, simulation, resolution, bg=(0, 0, 0)):
        self.output = output
        self.simulation = simulation
        width, height = resolution
        self.surface = cairo.ImageSurface(cairo.Format.RGB24, width, height)
        self.bg = bg
    
    def render(self):
        surface = self.surface
        
        ctx = cairo.Context(surface)
        ctx.set_source_rgb(0.42, 0.22, 1)
        #w, h = surface.get_width(), surface.get_height()

        clear(surface, color=self.bg)
        
        for x, y in self.simulation.positions:
            ctx.move_to(x, y)
            ctx.arc(x, y, 5, 0, TAU)
            ctx.fill()

        self.output.write(surface.get_data())


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
simulation.springs.append((3, 4, 20))
simulation.springs.append((1, 2, 20))
simulation.springs.append((5, 6, 20))
renderer = Renderer(
    sys.stdout.buffer,
    simulation,
    (320, 200),
    bg=(1, 1, 1))

for _ in range(200):
    simulation.step(dt=0.1)
    renderer.render()

#for commit in itertools.islice(commits, 10):
#    for f in commit.files:
#        print(dir(f))
