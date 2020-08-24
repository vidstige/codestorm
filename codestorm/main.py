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
from codestorm.simulation import Simulation


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


def from_intensity(original: RenderProperties, intensity: Intensity) -> RenderProperties:
    """Compute render properties from original + intensity"""
    properties = copy(original)
    properties.radius = np.clip(original.radius * np.log(intensity), 1, 128)
    return properties


class Config:
    def __init__(self, seed=None, file_types={}):
        self.seed = seed
        self.foreground = None
        self.background = None
        self.file_types = file_types
        self.author_properties = None
        self._properties_cache = {}

    def render_properties_for(self, filename: str):
        color = self.file_types.get(Path(filename).suffix, self.file_types[None])
        properties = self._properties_cache.get(color)
        if not properties:
            properties = RenderProperties(color, radius=4)
            self._properties_cache[color] = properties
        return properties


def remove_prefix(text: str, prefixes: Iterable[str]) -> str:
    for prefix in prefixes:
        if text.startswith(prefix):
            return text[len(prefix):]
    return text


class Color:
    BLACK = (0, 0, 0)
    WHITE = (1, 1, 1)
    def __init__(self, raw: str):
        r, g, b = bytes.fromhex(remove_prefix(raw, ['#', '0x']))
        self.color = (r / 255, g / 255, b / 255)


def parse_value(key, value):
    if key == "color":
        return Color(value).color
    return value


def parse(keys: Dict):
    """Parses colors and more"""
    return  {key: parse_value(key, value) for key, value in keys.items()}


def update_from_json(config: Config, keys: Dict):
    config.seed = keys.get('seed', config.seed)
    config.author_properties = RenderProperties(**parse(keys.get('author')))
    config.foreground = parse(keys.get('foreground')).get('color', Color.WHITE)
    config.background = parse(keys.get('background')).get('color', Color.BLACK)

    for filetype in keys.get('filetypes', []):
        for extension in filetype['extensions']:
            config.file_types[extension] = Color(filetype['color']).color
        


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


    # maps identifiers to timestamps, intensity tuples
    files = {}
    authors = {}

    labels = authors

    renderer = Renderer(
        sys.stdout.buffer,
        simulation,
        labels,
        (640, 480),
        bg=config.background,
        fg=config.foreground)


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
                renderer.properties[author] = from_intensity(config.author_properties, i)

            for filename, (timestamp, intensity) in files.items():
                i = intensity_at(t - timestamp, intensity, file_duration)
                render_properties = config.render_properties_for(filename)
                renderer.properties[filename] = from_intensity(render_properties, i)
            
            renderer.render()
        else:
            # Add body for author (if needed) and update timestamp
            author = commit.committer.login
            if author not in simulation.bodies:
                simulation.add_body(np.random.rand(1, 2) - 0.5, author)

                # add springs between all other authors
                #for peer in authors:
                #    sid = '-'.join(sorted([peer, author]))
                #    simulation.add_spring(sid, peer, author, 0.3, 1.1)                

            # update timestamp and intensity
            pt, pi = authors.get(author, (simulation.get_time(), 0))
            spillover = intensity_at(age=simulation.get_time() - pt, intensity=pi, duration=author_duration)
            intensity = spillover + size(commit)
            authors[author] = simulation.get_time(), intensity

            # Add body for file (if needed) and update timestamp
            for phile in commit.files or tuple():
                filename = os.path.basename(phile.filename)
                if filename not in simulation.bodies:
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
        import itertools
        commits = storage.commits()
        codestorm(itertools.islice(commits, 1000, None), config)
        #for commit in commits:
        #    print(commit.sha)


if __name__ == "__main__":
    main()
