from datetime import datetime, timezone, tzinfo
from typing import Iterable, Optional
import subprocess
from pathlib import Path

from github import Github

from codestorm.commit import Commit, File, NamedUser, Slug



class Fetcher:
    def commits(self, slug: Slug, since: Optional[str]=None):
        return []


class GithubAPI(Fetcher):
    @staticmethod
    def to_commit(slug: Slug, commit) -> Commit:
        return Commit(
            slug,
            commit.sha,
            commit.last_modified,
            committer=NamedUser(commit.committer.login),
            files=[File(f.filename, f.additions, f.changes, f.deletions) for f in commit.files])

    def commits(self, slug: Slug,since: Optional[str]=None):
        del since
        with open('.token') as f:
            token = f.read().strip()

        g = Github(token)

        repo = g.get_repo(str(slug))
        return (GithubAPI.to_commit(slug, c) for c in repo.get_commits())


class Cloning(Fetcher):
    def __init__(self, path: Path):
        super()
        self.path = path

    @staticmethod
    def parse_linecount(s: str) -> Optional[int]:
        if s == '-':
            return None
        return int(s)

    def commits(self, slug: Slug, since: Optional[str]=None) -> Iterable[Commit]:
        git_directory = self.path / '{}.git'.format(slug.repository)
        if git_directory.exists():
            ## update
            command = ['git', 'fetch']
            subprocess.check_call(command, cwd=git_directory)
        else:
            # clone
            command = [
                'git', 'clone', '--bare',
                'git@github.com:{slug}.git'.format(slug=slug)
            ]
            self.path.mkdir(exist_ok=True)
            subprocess.check_call(command, cwd=self.path)

        # fetch commits
        command = ['git', 'log', '--reverse', '--no-merges', '--date=unix', '--pretty=========%n%H,%aE,%cd', '--numstat']
        if since:
            command += ['{}..HEAD'.format(since)]

        process = subprocess.Popen(
            command, cwd=git_directory, stdout=subprocess.PIPE,
            env=dict(TZ='UTC'))
        assert process.stdout
        lines = iter(process.stdout)
        next(lines)  # skip first boundary
        for raw_line in lines:
            line = raw_line.decode().strip()
            sha = line[:40]
            rest = line[41:]
            email, timestamp_raw = rest.split(',')

            # TODO: git prints these in local timestamp
            timestamp = datetime.fromtimestamp(int(timestamp_raw))

            files = []
            for raw_line in lines:
                line = raw_line.decode().strip()
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
                slug=slug,
                sha=sha,
                last_modified=timestamp,
                committer=NamedUser(login=email),
                files=files)
        