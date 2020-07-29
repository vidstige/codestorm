from datetime import datetime
import os
from pathlib import Path
import pickle
from typing import Iterable


class NamedUser:
    def __init__(self, login):
        self.login = login


class File:
    def __init__(self, filename, additions, changes, deletions):
        self.filename = filename
        self.additions = additions
        self.changes = changes
        self.deletions = deletions
        

class Commit:
    def __init__(self, sha, last_modified, committer, files):
        self.sha = sha
        self.last_modified = last_modified
        self.committer = committer
        self.files = files

    def __le__(self, other):
        if other is None:
            return False
        if other.last_modified is None:
            return False
        if self.last_modified is None:
            return True
        return other.last_modified <= self.last_modified


def last_modified(commit) -> datetime:
    return datetime.strptime(
        commit.last_modified,
        '%a, %d %b %Y %H:%M:%S %Z')


# To store commits
class Storage:
    pass


import sqlite3


def serialize_files(files):
    return '|'.join('{}:{}:{}:{}'.format(f.filename, f.additions, f.changes, f.deletions) for f in files)


def deserialize_file(raw):
    return File(*raw.split(':'))


def deserialize_files(raw):
    if not raw:
        return []
    files = raw.split('|')
    return [deserialize_file(f) for f in files]


class SQLiteStorage(Storage):
    def __init__(self, path: Path):
        self.connection = sqlite3.connect(path)
        self._create_tables()
    
    def _create_tables(self):
        cursor = self.connection.cursor()
        cursor.execute(
            '''CREATE TABLE IF NOT EXISTS commits (timestamp DATETIME, sha TEXT, committer TEXT, files TEXT)''')
        self.connection.commit()

    def __contains__(self, commit) -> bool:
        return False    

    def store(self, commit):
        cursor = self.connection.cursor()
        cursor.execute("INSERT INTO commits (timestamp, sha, committer, files) values ('{timestamp}', '{sha}', '{committer}', '{files}')".format(
            timestamp=commit.last_modified,
            sha=commit.sha,
            committer=commit.committer.login if commit.committer else None,
            files=serialize_files(commit.files),
        ))
        self.connection.commit()

    def commits(self) -> Iterable:
        cursor = self.connection.cursor()
        rows = cursor.execute('SELECT timestamp, sha, committer, files FROM commits ORDER BY timestamp')
        for row in rows:
            timestamp, sha, comitter_login, files_raw = row
            commit = Commit(
                last_modified=timestamp,
                sha=sha,
                committer=NamedUser(login=comitter_login),
                files=deserialize_files(files_raw))
            if commit.last_modified:
                yield commit


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

