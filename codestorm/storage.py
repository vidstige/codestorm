import os
from pathlib import Path
import pickle
from typing import Dict, Iterable, Optional

from codestorm.mailmap import Mailmap
from codestorm.commit import NamedUser, File, Commit, last_modified, Slug


# To store commits
class Storage:
    def commits(self) -> Iterable[Commit]:
        return []

    def store(self, commit: Commit) -> None:
        pass

    def delete(self, slug: Slug) -> None:
        pass


import sqlite3


def serialize_files(files):
    return "\\0".join('{}:{}:{}:{}'.format(f.filename, f.additions, f.changes, f.deletions) for f in files)


def parse_size(size: str) -> int:
    if size == 'None':
        return 0
    return int(size)


def deserialize_file(raw):
    try:
        filename, additions, changes, deletions = raw.rsplit(':', 3)
        return File(filename, parse_size(additions), parse_size(changes), parse_size(deletions))
    except ValueError:
        print(raw)
        raise


def deserialize_files(raw):
    if not raw:
        return []
    files = raw.split("\\0")
    return [deserialize_file(f) for f in files]


class SQLiteStorage(Storage):
    def __init__(self, path: Path):
        self.connection = sqlite3.connect(str(path))
        self._create_tables()
        self.mailmap = Mailmap({})
    
    def _create_tables(self):
        cursor = self.connection.cursor()
        # commit table
        cursor.execute(
            '''CREATE TABLE IF NOT EXISTS commits (owner TEXT, repository TEXT, sha TEXT, timestamp DATETIME, committer TEXT, files TEXT)''')
        # sha index
        cursor.execute(
            '''CREATE UNIQUE INDEX IF NOT EXISTS sha_index ON commits (owner, repository, sha)''')
        # timestamp index
        cursor.execute(
            '''CREATE INDEX IF NOT EXISTS timestamp_index ON commits (timestamp)''')
        # committer index
        cursor.execute(
            '''CREATE INDEX IF NOT EXISTS committer_index ON commits (committer)''')

        self.connection.commit()

    def __contains__(self, commit) -> bool:
        return False    

    def store(self, commit: Commit):
        cursor = self.connection.cursor()
        cursor.execute("INSERT INTO commits (owner, repository, sha, timestamp, committer, files) values (?, ?, ?, ?, ?, ?)", (
            commit.slug.owner,
            commit.slug.repository,
            commit.sha,
            commit.last_modified,
            commit.committer.login if commit.committer else None,
            serialize_files(commit.files),
        ))
        self.connection.commit()

    def commits(self, query: Optional[Dict[str, str]]={}) -> Iterable[Commit]:
        cursor = self.connection.cursor()
        where = ''
        if query:
            where = "WHERE {}".format(' AND '.join('{}="{}"'.format(k ,v) for k, v in query.items()))
        rows = cursor.execute('SELECT owner, repository, sha, timestamp, committer, files FROM commits {where}ORDER BY timestamp'.format(where=where))
        for row in rows:
            try:
                owner, repository, sha, timestamp, comitter_login, files_raw = row
                commit = Commit(
                    slug=Slug(owner, repository),
                    last_modified=timestamp,
                    sha=sha,
                    committer=NamedUser(login=self.mailmap.lookup(comitter_login)),
                    files=deserialize_files(files_raw))
                if commit.last_modified:
                    yield commit
            except ValueError:
                print(sha, repository)
                raise
    
    def delete(self, slug: Slug) -> None:
        cursor = self.connection.cursor()
        cursor.execute('DELETE FROM commits WHERE owner="{owner}" AND repository="{repository}"'.format(owner=slug.owner, repository=slug.repository))
        self.connection.commit()
    
    def users(self) -> Iterable[str]:
        cursor = self.connection.cursor()
        rows = cursor.execute('SELECT DISTINCT committer FROM commits')
        users = set()
        for row in rows:
            comitter_login, = row
            users.add(self.mailmap.lookup(comitter_login))
        return sorted(users)

    def repositories(self) -> Iterable[Slug]:
        cursor = self.connection.cursor()
        rows = cursor.execute('SELECT DISTINCT owner, repository FROM commits')
        return [Slug(owner, repo) for owner, repo in rows]


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

    

## text file storage stuff
def split(s: str, separators: str='\t\n ', escape='\\'):
    """Splits a string by separators"""
    result = []
    token = ''
    in_escape = False
    for c in s:
        if not in_escape:
            if c == escape:
                in_escape = True
            elif c in separators:
                result.append(token)
                token = ''
            else:
                token += c
        elif in_escape:
            token += c
            in_escape = False
    result.append(token)
    return result


def esacpe(data: str):
    translation = str.maketrans({
        "-":  r"\-",
        "]":  r"\]",
        "\\": r"\\",
        "^":  r"\^",
        "$":  r"\$",
        "*":  r"\*",
        ".":  r"\."})
    return data.translate(translation)


def unesacpe(data: str):
    translation = {
        r"\-": "-",
        r"\]": "]",
        r"\\": "\\",
        r"\^": "^",
        r"\$": "$",
        r"\*": "*",
        r"\.": "."}
    for old, new in translation.items():
        data = data.replace(old, new)
    return data

