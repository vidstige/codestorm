import json
from typing import Dict, Iterable

from codestorm.commit import Commit, NamedUser


class Mailmap:
    def __init__(self, mailmap: Dict[str, str]):
        self.mailmap = mailmap

    @staticmethod
    def load(path: str) -> 'Mailmap':
        mailmap = {}
        if not path:
            return Mailmap(mailmap)

        with open(mailmap) as f:
            mailmap = json.load(f)

        for email, aliases in mailmap.items():
            for alias in aliases:
                mailmap[alias] = email

        return Mailmap(mailmap)
    
    def lookup(self, email: str) -> str:
        return self.mailmap.get(email, email)
