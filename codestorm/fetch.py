from github import Github

from codestorm.commit import Commit, File, NamedUser

class Fetcher:
    def commits(self, repo_slug):
        return []


class GithubAPI:
    @staticmethod
    def to_commit(commit) -> Commit:
        return Commit(
            commit.sha,
            commit.last_modified,
            committer=NamedUser(commit.committer.login),
            files=[File(f.filename, f.additions, f.changes, f.deletions) for f in commit.files])

    def commits(self, repo_slug):
        with open('.token') as f:
            token = f.read().strip()

        g = Github(token)

        repo = g.get_repo(repo_slug)
        return repo.get_commits()

