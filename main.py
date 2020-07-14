import itertools
import sys

from github import Github
import cairo
import numpy as np


TAU = 2 * np.pi


class Simulation:
    def __init__(self):
        self.t = 0
        self.positions = np.empty((0, 2))
        self.identifiers = []
        self.masses = np.empty((0, 1))
        self.springs = []

    def add_body(self, position, identity, mass=1):
        assert position.shape[1] == self.positions.shape[1]
        self.identifiers.append(identity)
        self.positions = np.vstack((self.positions, position))
        self.masses = np.vstack((self.masses, mass))
        return identity
    
    def index_of(self, identity) -> int:
        return self.identifiers.index(identity)

    def remove_body(self, identity) -> None:
        i = self.index_of(identity)
        del i
        # 1. assert no springs are used
        # 2. swap last and i
        # 3. remove last element

    def add_spring(self, a, b, length, stiffness=1, t=None) -> None:
        """Adds a spring"""
        i = self.index_of(a)
        j = self.index_of(b)
        self.springs.append((i, j, length, stiffness, t or self.t))

    def velocities(self, positions, t: float):
        # compute spring forces

        v = np.zeros(positions.shape)
        # find indices annd lengths
        ii = [spring[0] for spring in self.springs]
        jj = [spring[1] for spring in self.springs]
        ll = [spring[2] for spring in self.springs]
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


def codestorm(commits):
    simulation = Simulation()
    for i in range(20):
        simulation.add_body(np.random.rand(1, 2) * 200, i)
    
    simulation.add_spring(1, 2, 20)
    simulation.add_spring(3, 4, 20)
    simulation.add_spring(5, 6, 20)

    renderer = Renderer(
        sys.stdout.buffer,
        simulation,
        (320, 200),
        bg=(1, 1, 1))
    
    # maps filenames to tuples of simulation index and timestamp
    files = {}
    # maps filenames to tuples of simulation index and timestamp
    authors = {}

    for commit in commits:
        pass
        # if not initialized:
        # initialize to commit.date - 1 day
        # simulate and render until commit.date
        # add files or update timestamp
        # add authors or update timestamps
        # add spring forces or update timestamps
        # prune old spring forces
        # prune old files
        # prune old authors

    for _ in range(200):
        simulation.step(dt=0.05)
        renderer.render()


import pickle
import os

def load_commits(directory):
    import os

    for path in sorted(directory.iterdir(), key=os.path.getmtime):
        with path.open('rb') as f:
            commit = pickle.load(f)
        yield commit

from datetime import datetime

def download_commits(directory, repo_slug):
    with open('.token') as f:
        token = f.read().strip()

    g = Github(token)

    #volumental = g.get_organization("Volumental")
    #for repo in volumental.get_repos():
    #    print(repo)

    repo = g.get_repo(repo_slug)
    commits = repo.get_commits()

    for commit in commits:
        path = directory / commit.sha
        print(path)

        if not path.exists():
            with path.open('wb') as f:
                pickle.dump(commit, f, protocol=pickle.HIGHEST_PROTOCOL)
            date = datetime.strptime(
                commit.last_modified,
                '%a, %d %b %Y %H:%M:%S %Z')
            ts = date.timestamp()
            os.utime(str(path), (ts, ts))


def main():
    from pathlib import Path
    commit_cache = Path("commit-cache/")
    download_commits(commit_cache, "Volumental/Reconstruction")
    #commits = load_commits(commit_cache)
    #codestorm(commits)
    

if __name__ == "__main__":
    main()
