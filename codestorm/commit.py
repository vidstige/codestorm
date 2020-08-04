from datetime import datetime


class NamedUser:
    def __init__(self, login):
        self.login = login


class File:
    def __init__(self, filename, additions, changes, deletions):
        self.filename = filename
        self.additions = additions
        self.changes = changes
        self.deletions = deletions

    def __repr__(self) -> str:
        return 'File({filename}, {additions}, {changes}, {deletions})'.format(
            filename=self.filename,
            additions=self.additions,
            changes=self.changes,
            deletions=self.deletions)
        

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