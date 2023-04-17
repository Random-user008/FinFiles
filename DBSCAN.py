import requests
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Define vectorizer with desired parameters
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
nn = NearestNeighbors(n_neighbors=5, algorithm='auto')


def extract_topic_distributions(texts, num_topics=10):
    # Convert text data to document-term matrix
    dtm = vectorizer.fit_transform(texts)

    # Extract topics from document-term matrix
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=0)
    lda.fit(dtm)
    topic_distributions = lda.transform(dtm)

    return topic_distributions


def get_top_topics(descriptions, num_topics=10):
    # Convert text data to document-term matrix
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


def get_top_starred_repos():
    top_repos_data = pd.read_csv('top_starred_repos.csv')
    return top_repos_data


def get_top_n_indices(topic_distributions, top_n=5):
    # Get indices of data points that belong to a cluster
    db = DBSCAN(eps=0.3, min_samples=5)
    db_labels = db.fit_predict(topic_distributions)
    cluster_indices = [np.where(db_labels == i)[0] for i in range(np.max(db_labels) + 1)]

    # Get top n indices from each cluster
    top_n_indices = []
    for indices in cluster_indices:
        if len(indices) > top_n:
            topic_probs = topic_distributions[indices].sum(axis=1)
            top_n_indices.extend(indices[topic_probs.argsort()[-top_n:][::-1]])

    return top_n_indices  


from sklearn.cluster import DBSCAN

def recommend_repos():
    # Get user repositories and their descriptions
    user_repos = requests.get(f'https://api.github.com/users/Random-user008/repos').json()
    user_repo_descriptions = [repo['description'] for repo in user_repos if repo['description']]
    user_repo_languages = [repo['language'] for repo in user_repos if repo['language']]

    # Get top starred repositories and their descriptions
    top_repos_data = get_top_starred_repos()
    top_repo_descriptions = top_repos_data['description'].tolist()
    top_repo_languages = top_repos_data['language'].tolist()

    # Remove np.nan values from descriptions
    top_repo_descriptions = [desc for desc in top_repo_descriptions if isinstance(desc, str)]
    top_repo_languages = [lang for lang in top_repo_languages if isinstance(lang, str)]

    # Concatenate user repository descriptions with top repository descriptions
    texts = user_repo_descriptions + top_repo_descriptions
    languages = user_repo_languages + top_repo_languages

    # Check if there are any repository descriptions available
    if len(texts) == 0:
        print("No repository descriptions available for recommendation")
        return

    # Vectorize the text data using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    vectorizer.fit(texts)

    # Extract topic distributions and language encodings for the text data
    topic_distributions = extract_topic_distributions(texts)
    language_encodings = vectorizer.transform(languages)
    n = max(topic_distributions.shape[0], language_encodings.shape[0],)
    topic_distributions_pad = np.pad(topic_distributions, ((0, n - topic_distributions.shape[0]), (0, 0)))
    language_encodings_pad = np.pad(language_encodings.toarray(), ((0, n - language_encodings.shape[0]), (0, 0)))
    # Concatenate topic distributions and language encodings into a single feature matrix
    feature_matrix = np.concatenate([topic_distributions_pad, language_encodings_pad], axis=1)

    # Cluster the repositories using DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    cluster_labels = dbscan.fit_predict(feature_matrix)
    silhouette_avg = silhouette_score(feature_matrix, cluster_labels)
    print(silhouette_avg)
    # Get the indices of the top recommended repositories
    top_n_indices = get_top_n_indices(topic_distributions, top_n=5)

    # Print the top recommended repositories
    print("Top Recommended Repositories:")
    for index in top_n_indices:
        if(index<len(top_repos_data)):
            if cluster_labels[index] == -1:
                continue
            print(f"{top_repos_data.iloc[index]['name']} ({top_repos_data.iloc[index]['description']}) - {top_repos_data.iloc[index]['url']}")


recommend_repos()