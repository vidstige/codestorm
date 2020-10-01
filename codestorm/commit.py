from datetime import datetime


class Slug:
    def __init__(self, owner: str, repository: str):
        self.owner = owner
        self.repository = repository

    def __str__(self) -> str:
        return '{owner}/{repository}'.format(
            owner=self.owner,
            repository=self.repository)

    def __repr__(self) -> str:
        return 'Slug({}, {})'.format(self.owner, self.repository)

    @staticmethod
    def from_string(s: str):
        parts = s.split('/')
        if len(parts) == 2:
            return Slug(*parts)
        raise ValueError('Invalid slug: {}'.format(s))


class NamedUser:
    def __init__(self, login):
        self.login = login

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.login)


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
    def __init__(self, slug: Slug, sha, last_modified, committer, files):
        self.slug = slug
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

    def __repr__(self) -> str:
        return "{}({}, {}, {}, {}, {})".format(
            self.__class__.__name__,
            self.slug,
            self.sha,
            self.last_modified,
            self.committer,
            self.files)


def last_modified(commit) -> datetime:
    return datetime.strptime(
        commit.last_modified,
        '%a, %d %b %Y %H:%M:%S %Z')