from datetime import datetime, timedelta
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

    def set_time(self, t):
        self.t = t

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
        # find indices and lengths
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

    def __contains__(self, identifier) -> bool:
        try:
            self.index_of(identifier)
            return True
        except ValueError:
            return False

class TickSimulation(Simulation):
    def __init__(self, tick_length: timedelta):
        self.tick_length = tick_length
        super().__init__()

    def set_time(self, time: datetime):
        super().set_time(time.timestamp() / self.tick_length.total_seconds())

    def get_time(self) -> datetime:
        return datetime.fromtimestamp(self.t * self.tick_length.total_seconds())

    def step(self, dt: timedelta):
        super().step(dt.total_seconds() / self.tick_length.total_seconds())


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
            #ctx.move_to(160 + x * 160, 100 + y * 100)
            ctx.arc(160 + x * 160, 100 + y * 100, 5, 0, TAU)
            ctx.fill()
        
        # overlay
        ctx.set_source_rgb(0, 0, 0)
        ctx.move_to(8, 24)
        ctx.set_font_size(16)
        ctx.show_text(self.simulation.get_time().isoformat(' ', timespec='seconds'))

        self.output.write(surface.get_data())


def lazy_merge(a, b):
    """Lazily merges two iterables and produces a new one without"""
    ai = iter(a)
    bi = iter(b)

    aa = next(ai)
    bb = next(bi)

    while True:        
        while aa <= bb:
            yield aa
            aa = next(ai)
        
        while bb <= aa:
            yield bb
            bb = next(bi)


def steady(start, timestep):
    """Returns infinite iterator with evenly spaced timestamps"""
    t = start
    while True:
        yield t
        t += timestep


def codestorm(commits):
    simulation = TickSimulation(timedelta(days=1))
    #for i in range(20):
    #    simulation.add_body(np.random.rand(1, 2) * 200, i)
    
    #simulation.add_spring(1, 2, 20)
    #simulation.add_spring(3, 4, 20)
    #simulation.add_spring(5, 6, 20)

    renderer = Renderer(
        sys.stdout.buffer,
        simulation,
        (320, 200),
        bg=(1, 1, 1))
    
    # maps identifiers to timestamps
    files = {}
    authors = {}
    springs = {}

    # find start time
    commit_iterator = iter(commits)
    first = next(commit_iterator)
    commit_iterator = itertools.chain([first], commit_iterator)

    # create frame generator
    start_time = last_modified(first) - timedelta(days=1)
    commit_timestamps = ((last_modified(c), c) for c in commits)
    frame_timestamps = ((timestamp, None) for timestamp in steady(start_time, timedelta(days=8)))

    simulation.set_time(start_time)

    timestep = timedelta(days=0.1)
    for timestamp, commit in lazy_merge(commit_timestamps, frame_timestamps):            
        # simulate until timestamp
        while timestamp - simulation.get_time() > timestep:
            simulation.step(dt=timestep)
        # time left is smaller than timestep
        simulation.step(dt=timestamp - simulation.get_time())

        if commit:
            # Add body for author (if needed) and update timestamp
            author = commit.author
            if author not in simulation:
                simulation.add_body(np.random.rand(1, 2) - 0.5, author)
            authors[author] = simulation.get_time()
            
            # Add body for file (if needed) and update timestamp
            for phile in commit.files or tuple():
                filename = phile.filename
                if author not in simulation:
                    simulation.add_body(np.random.rand(1, 2) - 0.5, filename)
                    files[filename] = simulation.get_time()

                # spring id
                sid = '{author}-{filename}'.format(author=author, filename=filename)
                
        
            # add spring forces or update timestamps
            # prune old spring forces
            # prune old files
            # prune old authors
        else:
            # this is a frame
            renderer.render()


import pickle
import os

def load_commits(directory):
    """Returns all commits in date order from commit repository"""
    for path in sorted(directory.iterdir(), key=os.path.getmtime):
        with path.open('rb') as f:
            commit = pickle.load(f)
        yield commit


def last_modified(commit) -> datetime:
    return datetime.strptime(
        commit.last_modified,
        '%a, %d %b %Y %H:%M:%S %Z')


def download_commits(directory, repo_slug):
    """Downloads all commits into commit-cache"""
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

        if not path.exists():
            print(path)
            list(commit.files)  # trigger loading of files
            with path.open('wb') as f:
                pickle.dump(commit, f, protocol=pickle.HIGHEST_PROTOCOL)
            ts = last_modified(commit).timestamp()
            os.utime(str(path), (ts, ts))


def main():
    from pathlib import Path
    commit_cache = Path("commit-cache/")
    commits = load_commits(commit_cache)
    #for commit in commits:
    #    print(commit.sha)
    codestorm(commits)


if __name__ == "__main__":
    main()
