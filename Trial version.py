import requests
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.neighbors import NearestNeighbors

def extract_topic_distributions(texts, num_topics=10):
    # Convert text data to document-term matrix
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    dtm = vectorizer.fit_transform(texts)

    # Extract topics from document-term matrix
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=0)
    lda.fit(dtm)
    topic_distributions = lda.transform(dtm)

    return topic_distributions

def get_top_topics(descriptions, num_topics=10):
    # Convert text data to document-term matrix
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    dtm = vectorizer.fit_transform(descriptions)

    # Extract topics from document-term matrix
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=0)
    lda.fit(dtm)
    
    # Get the top topics
    top_topics = []
    for i in range(num_topics):
        top_topic_words = [vectorizer.get_feature_names()[index] for index in lda.components_[i].argsort()[-5:]]
        top_topics.append(', '.join(top_topic_words))
        
    return top_topics



def get_top_starred_repos(num_repos=100):
    # Fetch top starred repositories data from Github API
    url = f"https://api.github.com/search/repositories?q=stars:>1&sort=stars&order=desc&per_page={num_repos}"
    response = requests.get(url)
    top_repos = response.json()['items']
    
    # Extract relevant data from repositories
    repo_data = []
    for repo in top_repos:
        name = repo['name']
        description = repo['description']
        url = repo['html_url']
        repo_data.append({'name': name, 'description': description, 'url': url})
        
    top_repos_data = pd.DataFrame(repo_data)
    return top_repos_data


def get_top_n_indices(topic_distributions, top_n=5):
    # Get top n indices with highest topic probability sum
    topic_probs = topic_distributions.sum(axis=1)
    top_n_indices = topic_probs.argsort()[-top_n:][::-1]
    
    return top_n_indices



def recommend_repos():
    top_repos_data = get_top_starred_repos()
    top_repo_descriptions = top_repos_data['description'].tolist()

    # Remove np.nan values from descriptions
    top_repo_descriptions = [desc for desc in top_repo_descriptions if isinstance(desc, str)]
    
    # Fetch user repository data from Github API
    user_repos = requests.get(f'https://api.github.com/users/Random-user008/repos').json()
    user_repo_descriptions = [repo['description'] if repo['description'] is not None else '' for repo in user_repos]
    
    # Remove np.nan values from descriptions
    user_repo_descriptions = [desc for desc in user_repo_descriptions if isinstance(desc, str)]

    # Concatenate user repository descriptions with top repository descriptions
    texts = user_repo_descriptions + top_repo_descriptions
    
    topic_distributions = extract_topic_distributions(texts)
    top_n = 5
    top_n_indices = get_top_n_indices(topic_distributions, top_n)
    if len(top_repos_data) < top_n:
        top_n_repos = top_repos_data
    else:
        top_n_repos = top_repos_data.iloc[top_n_indices]

    print(f"Top {top_n} recommended repos:")
    print(top_n_repos[['name', 'description', 'url']])

    # Find common words in user and top starred repo descriptions
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    dtm = vectorizer.fit_transform(texts)
    common_words = vectorizer.get_feature_names()
    print(f"Common words in user and top starred repo descriptions: {common_words}")




recommend_repos()
