from datetime import datetime
import os
from pathlib import Path
import pickle
from typing import Iterable


def last_modified(commit) -> datetime:
    return datetime.strptime(
        commit.last_modified,
        '%a, %d %b %Y %H:%M:%S %Z')


# To store commits
class Storage:
    pass


class DirectoryStorage(Storage):
    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)

    def __contains__(self, commit) -> bool:
        path = self.directory / commit.sha
        return path.exists()

    def store(self, commit):
        path = self.directory / commit.sha

        list(commit.files)  # trigger loading of files
        with path.open('wb') as f:
            pickle.dump(commit, f, protocol=pickle.HIGHEST_PROTOCOL)
        ts = last_modified(commit).timestamp()
        os.utime(str(path), (ts, ts))

    def commits(self) -> Iterable:
        """Returns all commits in date order from commit repository"""
        for path in sorted(self.directory.iterdir(), key=os.path.getmtime):
            with path.open('rb') as f:
                commit = pickle.load(f)
            yield commit

    

