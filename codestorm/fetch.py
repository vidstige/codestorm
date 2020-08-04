from datetime import datetime, timezone, tzinfo
from typing import Iterable, Optional
import subprocess
from pathlib import Path

from github import Github

from codestorm.commit import Commit, File, NamedUser


class Slug:
    def __init__(self, owner: str, repository: str):
        self.owner = owner
        self.repository = repository

    def __str__(self) -> str:
        return '{owner}/{repository}'.format(
            owner=self.owner,
            repository=self.repository)

    @staticmethod
    def from_string(s: str):
        return Slug(*s.split('/'))


class Fetcher:
    def commits(self, slug: Slug):
        return []


class GithubAPI(Fetcher):
    @staticmethod
    def to_commit(commit) -> Commit:
        return Commit(
            commit.sha,
            commit.last_modified,
            committer=NamedUser(commit.committer.login),
            files=[File(f.filename, f.additions, f.changes, f.deletions) for f in commit.files])

    def commits(self, slug: Slug):
        with open('.token') as f:
            token = f.read().strip()

        g = Github(token)

        repo = g.get_repo(slug)
        return (GithubAPI.to_commit(c) for c in repo.get_commits())


class Cloning(Fetcher):
    @staticmethod
    def parse_linecount(s: str) -> Optional[int]:
        if s == '-':
            return None
        return int(s)

    def commits(self, slug: Slug) -> Iterable[Commit]:
        path = Path('.repos')
        git_directory = path / '{}.git'.format(slug.repository)
        if git_directory.exists():
            ## update
            #command = ['git', 'pull', 'origin', 'master']
            #subprocess.check_call(command, cwd=path)
            pass
        else:
            # clone
            command = [
                'git', 'clone', '--bare',
                'git@github.com:{slug}.git'.format(slug=slug)
            ]
            path.mkdir(exist_ok=True)
            subprocess.check_call(command, cwd=path)

        # fetch commits

        command = ['git', 'log', '--no-merges', '--date=unix', '--pretty=========%n%H,%aE,%cd', '--numstat']
        process = subprocess.Popen(
            command, cwd=git_directory, stdout=subprocess.PIPE,
            env=dict(TZ='UTC'))
        lines = iter(process.stdout)
        next(lines)  # skip first boundary
        for line in lines:
            line = line.decode().strip()
            sha = line[:40]
            rest = line[41:]
            email, timestamp_raw = rest.split(',')

            # TODO: git prints these in local timestamp
            timestamp = datetime.fromtimestamp(int(timestamp_raw))

            files = []
            for line in lines:
                line = line.decode().strip()
                if not line:
                    # skip empty line
                    continue

                parts = line.split('\t')
                if len(parts) == 3:
                    added, removed, filename = parts
                    files.append(File(
                        filename,
                        additions=Cloning.parse_linecount(added),
                        changes=0,
                        deletions=Cloning.parse_linecount(removed)))
                else:
                    break
            yield Commit(
                sha=sha,
                last_modified=timestamp,
                committer=NamedUser(login=email),
                files=files)
        