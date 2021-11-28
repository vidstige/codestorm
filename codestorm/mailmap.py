import json
from typing import Dict, Iterable

from codestorm.commit import Commit, NamedUser


class Mailmap:
    def __init__(self, mailmap: Dict[str, str]):
        self.mailmap = mailmap

    @staticmethod
    def load(path: str) -> 'Mailmap':
        mailmap = {}  # type: Dict[str, str]
        if not path:
            return Mailmap(mailmap)

        with open(path) as f:
            inverse_map = json.load(f)

        for email, aliases in inverse_map.items():
            for alias in aliases:
                mailmap[alias] = email

        return Mailmap(mailmap)
    
    def lookup(self, email: str) -> str:
        return self.mailmap.get(email, email)
