import json
from typing import Iterable

from codestorm.commit import Commit, NamedUser
from codestorm.storage import Storage


class Mailmap(Storage):
    def __init__(self, mailmap: str, inner: Storage):
        self.inner = inner
        self.mailmap = {}
        with open(mailmap) as f:
            mailmap = json.load(f)
            for email, aliases in mailmap.items():
                for alias in aliases:
                    self.mailmap[alias] = email
    
    def _map(self, committer: NamedUser) -> NamedUser:
        replaced = self.mailmap.get(committer.login)
        if replaced:
            return NamedUser(replaced)
        return committer

    def commits(self) -> Iterable[Commit]:
        for commit in self.inner.commits():
            yield Commit(
                commit.sha,
                commit.last_modified,
                self._map(commit.committer),
                commit.files)

    def store(self, commit: Commit) -> None:
        self.inner.store(commit)
