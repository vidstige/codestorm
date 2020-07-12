from github import Github

import itertools

# or using an access token
with open('.token') as f:
    token = f.read().strip()

g = Github(token)

#volumental = g.get_organization("Volumental")
#for repo in volumental.get_repos():
#    print(repo)
reconstruction = g.get_repo("Volumental/Reconstruction")
commits = reconstruction.get_commits()

for commit in itertools.islice(commits, 10):
    print(commit.files)
