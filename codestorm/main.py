import argparse
from copy import copy
from datetime import datetime, timedelta
import itertools
import json
import os
from pathlib import Path
import sys
from typing import Dict, Iterable

import numpy as np

from codestorm.fetch import GithubAPI, Cloning, Slug
from codestorm.storage import DirectoryStorage, SQLiteStorage, Commit
from codestorm.renderer import RenderProperties, Renderer


def constant(stiffness, age):
    return stiffness


def replace_index(spring, a, b):
    i, j, length, stiffness, timestamp = spring
    if i == a:
        i = b
    if j == a:
        j = b
    return i, j, length, stiffness, timestamp


class Simulation:
    def __init__(self, stiffness=constant):
        self.t = 0
        self.positions = np.empty((0, 2))
        self.identifiers = []
        self.masses = np.empty((0, 1))
        self.springs = {}
        self.stiffness = stiffness

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
        # 1. remove springs attached to body
        to_remove = [sid for sid, spring in self.springs.items() if i == spring[0] or i == spring[1]]
        for sid in to_remove:
            self.remove_spring(sid)

        # 2. Update all springs refering the last element to now refer i instead
        last = len(self.identifiers) - 1
        self.springs = {sid: replace_index(spring, last, i) for sid, spring in self.springs.items()}

        # 3. swap last and i
        self.identifiers[i], self.identifiers[-1] = self.identifiers[-1], self.identifiers[i]
        self.positions[i], self.positions[-1] = self.positions[-1], self.positions[i]
        self.masses[i], self.masses[-1] = self.masses[-1], self.masses[i]

        # 4. remove last element
        del self.identifiers[-1]
        self.positions = self.positions[:-1]
        self.masses = self.masses[:-1]

    def add_spring(self, identifier, a, b, length, stiffness=1) -> None:
        """Adds a spring"""
        i = self.index_of(a)
        j = self.index_of(b)
        self.springs[identifier] = (i, j, length, stiffness, self.t)

    def remove_spring(self, identifier):
        del self.springs[identifier]
    
    def iter_springs(self):
        """Iterate all spring ids and age"""
        t = self.t
        for spring_id, spring in self.springs.items():
            yield spring_id, t - spring[4]

    def velocities(self, positions, t: float):
        # compute spring forces

        v = np.zeros(positions.shape)
        # find indices and lengths
        ii = [spring[0] for spring in self.springs.values()]
        jj = [spring[1] for spring in self.springs.values()]
        ll = [spring[2] for spring in self.springs.values()]
        ss = [spring[3] for spring in self.springs.values()]
        tt = [spring[4] for spring in self.springs.values()]
    
        # compute deltas
        di = positions[jj] - positions[ii]
        dj = positions[ii] - positions[jj]

        # compute lengths
        norms = np.linalg.norm(di, axis=-1)

        # compute spring force magnitudes
        age = self.t - np.array(tt)
        forces = (norms - ll) * self.stiffness(np.array(ss), age)

        # compute accelerations and accumulate
        mass = 1
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
    def __init__(self, tick_length: timedelta, stiffness):
        self.tick_length = tick_length
        self.stiffness_function = stiffness
        super().__init__(stiffness=self._stiffness)
    
    def _stiffness(self, stiffness, age):
        return self.stiffness_function(stiffness, age * self.tick_length.total_seconds())

    def iter_springs(self):
        for spring_id, age in super().iter_springs():
            yield spring_id, age * self.tick_length

    def set_time(self, time: datetime):
        super().set_time(time.timestamp() / self.tick_length.total_seconds())

    def get_time(self) -> datetime:
        return datetime.fromtimestamp(self.t * self.tick_length.total_seconds())

    def step(self, dt: timedelta):
        super().step(dt.total_seconds() / self.tick_length.total_seconds())


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


def steady(start: datetime, timestep: timedelta) -> Iterable[datetime]:
    """Returns infinite iterator with evenly spaced timestamps"""
    t = start
    while True:
        yield t
        t += timestep



# tween
def sin_inout(nt):
    return (np.cos(np.pi * nt) + 1) / 2


Intensity = float

def intensity_at(age: timedelta, intensity: Intensity, duration: timedelta):
    nt = np.clip(age.total_seconds() / duration.total_seconds(), 0, 1)
    return intensity * sin_inout(nt)


def size(commit: Commit) -> float:
    return sum(f.additions + f.changes + f.deletions for f in commit.files)


def from_intensity(original: RenderProperties, intensity: Intensity, label: str=None) -> RenderProperties:
    """Compute render properties from original + intensity"""
    properties = copy(original)
    properties.radius = np.clip(original.radius * np.log(intensity), 1, 128)
    properties.label = label
    return properties


class Config:
    def __init__(self, seed=None, file_types={}):
        self.seed = seed
        self.file_types = file_types
        self._properties_cache = {}

    def render_properties_for(self, filename: str):
        color = self.file_types.get(Path(filename).suffix, self.file_types[None])
        properties = self._properties_cache.get(color)
        if not properties:
            properties = RenderProperties(color, radius=4)
            self._properties_cache[color] = properties
        return properties


import os

def codestorm(commits: Iterable[Commit], config: Config):
    # the duration a force is active
    spring_duration = timedelta(weeks=52)
    file_duration = timedelta(weeks=28)
    author_duration = timedelta(weeks=11)

    def stiffness(stiffness, age):
        # normalized time (0..1)
        nt = np.clip(age / spring_duration.total_seconds(), 0, 1)
        return stiffness * sin_inout(nt)

    np.random.seed(config.seed)

    simulation = TickSimulation(timedelta(days=1), stiffness=stiffness)

    renderer = Renderer(
        sys.stdout.buffer,
        simulation,
        (640, 480),
        bg=(1, 1, 1))
    
    # maps identifiers to timestamps, intensity tuples
    files = {}
    authors = {}

    # render properties
    author_properties = RenderProperties(color=(1, 0, 0), radius=2, z=-1)

    # find start time
    commit_iterator = iter(commits)
    first = next(commit_iterator)
    commit_iterator = itertools.chain([first], commit_iterator)

    # create frame generator
    render_instruction = Commit(None, None, None, None)
    start_time = last_modified(first) - timedelta(days=2)
    commit_timestamps = ((last_modified(c), c) for c in commits)
    frame_timestamps = ((timestamp, render_instruction) for timestamp in steady(start_time, timedelta(days=0.5)))

    simulation.set_time(start_time)

    timestep = timedelta(days=0.1)
    for timestamp, commit in lazy_merge(commit_timestamps, frame_timestamps):            
        #print(timestamp, commit, file=sys.stderr)
        # simulate until timestamp
        while timestamp - simulation.get_time() > timestep:
            simulation.step(dt=timestep)
        # time left is smaller than timestep
        simulation.step(dt=timestamp - simulation.get_time())

        if commit == render_instruction:
            # this is a frame, just render it
            
            # update render properties
            t = simulation.get_time()
            for author, (timestamp, intensity) in authors.items():
                i = intensity_at(t - timestamp, intensity, author_duration)
                renderer.properties[author] = from_intensity(author_properties, i, label=author)

            for filename, (timestamp, intensity) in files.items():
                i = intensity_at(t - timestamp, intensity, file_duration)
                render_properties = config.render_properties_for(filename)
                renderer.properties[filename] = from_intensity(render_properties, i)
            
            renderer.render()
        else:
            # Add body for author (if needed) and update timestamp
            author = commit.committer.login
            if author not in simulation:
                simulation.add_body(np.random.rand(1, 2) - 0.5, author)

                # add springs between all other authors
                for peer in authors:
                    sid = '-'.join(sorted([peer, author]))
                    simulation.add_spring(sid, peer, author, 0.3, 0.01)                

            # update timestamp and intensity
            pt, pi = authors.get(author, (simulation.get_time(), 0))
            spillover = intensity_at(age=simulation.get_time() - pt, intensity=pi, duration=author_duration)
            intensity = spillover + size(commit)
            authors[author] = simulation.get_time(), intensity

            # Add body for file (if needed) and update timestamp
            for phile in commit.files or tuple():
                filename = os.path.basename(phile.filename)
                if filename not in simulation:
                    simulation.add_body(np.random.rand(1, 2) - 0.5, filename)

                pt, pi = files.get(filename, (simulation.get_time(), 0))
                spillover = intensity_at(age=simulation.get_time() - pt, intensity=pi, duration=author_duration)
                intensity = spillover + phile.additions + phile.changes + phile.deletions
                files[filename] = simulation.get_time(), intensity

                # add spring force
                # spring id
                sid = '{author}-{filename}'.format(author=author, filename=filename)
                # add spring forces or update timestamps
                simulation.add_spring(sid, author, filename, 0.2, 0.02)
                      
            # remove old spring forces
            to_remove = [spring_id for spring_id, age in simulation.iter_springs() if age > spring_duration]
            for spring_id in to_remove:
                simulation.remove_spring(spring_id)

            t = simulation.get_time()
            to_remove = [f for f, (timestamp, _) in files.items() if t - timestamp > file_duration]
            for f in to_remove:
                simulation.remove_body(f)
                del files[f]

            to_remove = [author for author, (timestamp, _) in authors.items() if t - timestamp > author_duration]
            for f in to_remove:
                simulation.remove_body(f)
                del authors[f]


def last_modified(commit) -> datetime:
    return datetime.strptime(
        commit.last_modified,
        #'%a, %d %b %Y %H:%M:%S %Z')
        '%Y-%m-%d %H:%M:%S')


def remove_prefix(text: str, prefixes: Iterable[str]) -> str:
    for prefix in prefixes:
        if text.startswith(prefix):
            return text[len(prefix):]
    return text


class Color:
    def __init__(self, raw: str):
        r, g, b = bytes.fromhex(remove_prefix(raw, ['#', '0x']))
        self.color = (r / 256, g / 256, b / 256)


def update_from_json(config: Config, keys: Dict):
    config.seed = keys.get('seed', config.seed)

    for filetype in keys.get('filetypes', []):
        for extension in filetype['extensions']:
            config.file_types[extension] = Color(filetype['color']).color
        


def add_bool_arg(parser, name: str, default=False):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--{}'.format(name), dest=name, action='store_true')
    group.add_argument('--no-{}'.format(name), dest=name, action='store_false')
    parser.set_defaults(**{name: default})


def main():
    parser = argparse.ArgumentParser(description='codestorm')
    parser.add_argument(
        '--cache', type=Path, default=Path("commit-cache/"))
    parser.add_argument(
        '--download', metavar='repos', type=str, nargs='*',
        help='downloads')
    parser.add_argument('--config', type=Path, default=Path('codestorm.json'))

    add_bool_arg(parser, 'render', True)

    args = parser.parse_args()

    #storage = DirectoryStorage(args.cache)
    storage = SQLiteStorage('commits.db')
    #fetcher = GithubAPI()
    fetcher = Cloning()
    for repo_slug in args.download or []:
        slug = Slug.from_string(repo_slug)
        for commit in fetcher.commits(slug):
            if commit not in storage:
                print(commit.sha)
                storage.store(commit)
    
    #commits = storage.commits()
    #everything = []
    #for commit in commits:
    #    for f in commit.files:
    #        suffix = Path(f.filename).suffix
    #        everything.append(suffix)
    #from collections import Counter
    #print(Counter(everything))
    config = Config()
    config_file = args.config
    with config_file.open() as f:
        update_from_json(config, json.load(f))

    if args.render:
        commits = storage.commits()
        codestorm(commits, config)
        #for commit in commits:
        #    print(commit.sha)


if __name__ == "__main__":
    main()
