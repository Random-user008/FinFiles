import requests

def get_top_starred_repos():
    url = 'https://api.github.com/search/repositories'
    params = {'q': 'stars:>100000', 'sort': 'stars', 'order': 'desc'}
    response = requests.get(url, params=params)
    response_dict = response.json()
    top_repos = []
    for repo in response_dict['items'][:5]:
        top_repos.append(repo['full_name'])
    return top_repos

def get_user_repo(username):
    url = f'https://api.github.com/users/{username}/repos'
    response = requests.get(url)
    response_dict = response.json()
    user_repo = response_dict[0]['full_name']
    return user_repo

def get_common_topics(user_repo, top_repos):
    url = f'https://api.github.com/repos/{user_repo}'
    response = requests.get(url)
    response_dict = response.json()
    user_topics = response_dict['topics']
    common_topics = []
    for repo in top_repos:
        url = f'https://api.github.com/repos/{repo}'
        response = requests.get(url)
        response_dict = response.json()
        repo_topics = response_dict['topics']
        for topic in user_topics:
            if topic in repo_topics:
                common_topics.append(topic)
    return common_topics

username = input('Enter your GitHub username: ')
print(f'Recommended repositories for {username}:')
top_repos = get_top_starred_repos()
for repo in top_repos:
    print('- ' + repo)
user_repo = get_user_repo(username)
common_topics = get_common_topics(user_repo, top_repos)
print(f'Your repository ({user_repo}) has common topics with recommended repositories:')
for topic in common_topics:
    print('- ' + topic)
