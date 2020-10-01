from copy import copy
from datetime import datetime, timedelta
import itertools
import json
import os
from pathlib import Path
import sys
from typing import BinaryIO, Dict, Iterable, Optional, Tuple

import click
import click_config_file
from github import Github
import numpy as np

from codestorm.fetch import Fetcher, Cloning, Slug
from codestorm.ffmpeg import FFmpeg, Resolution, Raw, VideoFormat, H264
from codestorm.mailmap import Mailmap
from codestorm.storage import Commit, DirectoryStorage, SQLiteStorage, Storage
from codestorm.renderer import Color, RenderProperties, Renderer
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
    def __init__(
            self, seed: int, resolution: Resolution, foreground: Color, background: Color):
        self.seed = seed
        self.resolution = resolution
        self.foreground = foreground
        self.background = background
        self.author_properties = RenderProperties(
            color=Color.parse("#47a276"), radius=2, z=-1
        )
        self.file_properties = RenderProperties(
            color=Color.parse("#9c74c2"), radius=2,
        )


    def render_properties_for(self, filename: str):
        #color = self.file_types.get(Path(filename).suffix, self.file_types[None])
        #properties = self._properties_cache.get(color)
        #if not properties:
        #    properties = RenderProperties(color, radius=4)
        #    self._properties_cache[color] = properties
        #return properties
        return self.file_properties


import os

def codestorm(commits: Iterable[Commit], config: Config, target: BinaryIO):
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
        target,
        simulation,
        labels,
        config.resolution.pair(),
        bg=config.background,
        fg=config.foreground)


    # find start time
    commit_iterator = iter(commits)
    first = next(commit_iterator)
    commit_iterator = itertools.chain([first], commit_iterator)

    # create frame generator
    render_instruction = Commit(None, None, None, None, None)
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
                simulation.add_body(np.random.rand(1, 2) - 0.5, author, mass=10)

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
                    simulation.add_body(np.random.rand(1, 2) - 0.5, filename, mass=1)

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


def _video_format_for(path: Optional[Path]) -> Optional[VideoFormat]:
    """Returns suitable video format for the given path"""
    if path is None:
        return None
    if path.suffix in ('.mkv',):
        return H264(pixel_format='yuv420p', crf=18, preset='slow')
    return None


def _make_session(resolution: Resolution, output: Optional[Path]):
    video_format = Raw('rgb32', resolution, 30)
    if output:
        return FFmpeg().convert(
            video_format=video_format,
            target=output,
            target_format=_video_format_for(output))
    return FFmpeg('ffplay').convert(video_format=video_format)


@click.group(name="codestorm")
@click.option('--cache', type=click.Path(), default=Path('~/.cache/codestorm').expanduser(), help="Where to store commit cache and repositories")
@click.option('--mailmap', type=click.Path(exists=True), help="Mailmap file")
def cli(cache: Path, mailmap: Optional[Path]):
    pass


@cli.command()
@click.option('--seed', type=click.INT, help="Random seed to use")
@click.option('--framerate', type=click.FLOAT, help="Framerate")
@click.option('--resolution', type=Resolution.parse, help="Resolution to render to")
@click.option('--foreground', type=Color.parse, help="Foreground color")
@click.option('--background', type=Color.parse, help="Background color")
@click.option('--output', type=click.Path(), help="File to render to, if omitted video will be displayed instead")
@click.argument('repositories', type=Slug.from_string, nargs=-1)
@click_config_file.configuration_option(exists=True, cmd_name='codestorm')
@click.pass_context
def render(
        ctx,
        seed: int, framerate: float, resolution: Resolution,
        foreground: Optional[Color], background: Optional[Color],
        output: Optional[Path], repositories: Iterable[Slug]):

    cache = ctx.parent.params['cache']
    storage = SQLiteStorage(cache / 'codestorm.db')
    storage.mailmap = Mailmap.load(ctx.parent.params['mailmap'])

    commits = storage.commits()
    config = Config(
        seed=seed,
        resolution=resolution,
        foreground=foreground,
        background=background
    )
    with _make_session(resolution, output) as session:
        codestorm(commits, config, session.buffer)


def _fetch_from(fetcher: Fetcher, storage: Storage, slug: Slug) -> None:
    print(slug)
    for commit in fetcher.commits(slug):
        print(commit.sha)
        storage.store(commit)


@cli.command()
@click.argument('repositories', type=Slug.from_string, nargs=-1)
@click_config_file.configuration_option(exists=True, cmd_name='codestorm')
@click.pass_context
def fetch(ctx, repositories: Iterable[Slug]):
    cache = ctx.parent.params['cache']
    cache.mkdir(exist_ok=True)
    storage = SQLiteStorage(cache / 'codestorm.db')
    storage.mailmap = Mailmap.load(ctx.parent.params['mailmap'])

    fetcher = Cloning(cache / 'repositories')

    for slug in repositories:
        if slug.repository == "*":
            with open('.token') as f:
                token = f.read().strip()

            g = Github(token)
            owner = g.get_organization(slug.owner)
            for repository in owner.get_repos():
                _fetch_from(fetcher, storage, Slug(slug.owner, repository.name))
        else:
            _fetch_from(fetcher, storage, slug)

        
    #
    #commits = storage.commits()
    #everything = []
    #for commit in commits:
    #    everything.append(commit.committer.login)

    #from collections import Counter
    #for value, count in Counter(everything).most_common():
    #    print("{value}: {count}".format(value=value, count=count))
    #import itertools; itertools.islice(commits, 1000, 5000)


def condition(s) -> Tuple[str, str]:
    return s.split('=', 2)


@cli.command()
@click.argument('what', type=str, default='commits')
@click.argument('conditions', type=condition, nargs=-1)
@click_config_file.configuration_option(exists=True, cmd_name='codestorm')
@click.pass_context
def list(ctx, what, conditions):
    cache = ctx.parent.params['cache']
    cache.mkdir(exist_ok=True)
    storage = SQLiteStorage(cache / 'codestorm.db')
    storage.mailmap = Mailmap.load(ctx.parent.params['mailmap'])
    
    if what == 'commits':
        query = dict(conditions)
        for commit in storage.commits(query):
            print(commit.sha, commit.slug, commit.committer)

    if what == 'users':
        for user in storage.users():
            print(user)


@cli.command()
@click.argument('repositories', type=Slug.from_string, nargs=-1)
@click_config_file.configuration_option(exists=True, cmd_name='codestorm')
@click.pass_context
def delete(ctx, repositories: Iterable[Slug]):
    cache = ctx.parent.params['cache']
    cache.mkdir(exist_ok=True)
    storage = SQLiteStorage(cache / 'codestorm.db')
    storage.mailmap = Mailmap.load(ctx.parent.params['mailmap'])
    
    for slug in repositories:
        print(slug)
        storage.delete(slug)


if __name__ == "__main__":
    cli()
