from github import Github

class Fetcher:
    def commits(self, repo_slug):
        return []


class GithubAPI:
    def commits(self, repo_slug):
        with open('.token') as f:
            token = f.read().strip()

        g = Github(token)

        repo = g.get_repo(repo_slug)
        return repo.get_commits()

