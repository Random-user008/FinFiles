import requests
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
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

def elbow_method_plot(data):
    nbrs = NearestNeighbors(n_neighbors=len(data), algorithm='ball_tree').fit(data)
    distances, indices = nbrs.kneighbors(data)

    # Calculate the within-cluster sum of squares (inertia) for each value of k
    k_range = range(2, 11)
    inertias = []
    sil_scores = []
    for k in k_range:
        centroids = []
        for i in range(k):
            centroid_index = indices[i*len(data)//k][0]
            centroids.append(data[centroid_index])
        centroid_distances, centroid_indices = nbrs.kneighbors(centroids)

        labels = []
        for i in range(len(data)):
            distances = [((data[i] - centroids[j])**2).sum() for j in range(k)]
            labels.append(distances.index(min(distances)))

        inertias.append(sum([min([((data[i] - centroids[j])**2).sum() for j in range(k)]) for i in range(len(data))]))

        sil_scores.append(silhouette_score(data, labels))

    # Plot the elbow curve
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.plot(k_range, inertias, marker='o', color=color)
    ax1.set_xlabel('Number of clusters')
    ax1.set_ylabel('Inertia', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.plot(k_range, sil_scores, marker='o', color=color)
    ax2.set_ylabel('Silhouette Score', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Elbow Method Plot')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

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

def recommend_repos():
    # Get user repositories
    user_repos = requests.get(f'https://api.github.com/users/Random-user008/repos').json()
    user_repo_descriptions = [repo['description'] for repo in user_repos if repo['description']]
    user_repo_languages = [repo['language'] for repo in user_repos if repo['language']]

    # Get top starred repositories
    top_repos_data = pd.read_csv('top_starred_repos.csv')
    top_repo_descriptions = top_repos_data['description'].tolist()
    top_repo_languages = top_repos_data['language'].tolist()

    # Remove np.nan values from descriptions and languages
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
    n = max(topic_distributions.shape[0], language_encodings.shape[0])
    topic_distributions_pad = np.pad(topic_distributions, ((0, n - topic_distributions.shape[0]), (0, 0)))
    language_encodings_pad = np.pad(language_encodings.toarray(), ((0, n - language_encodings.shape[0]), (0, 0)))

    # Concatenate topic distributions and language encodings
    features = np.concatenate((topic_distributions_pad, language_encodings_pad), axis=1)

    # Fit nearest neighbors model
    nn.fit(features)
    elbow_method_plot(features)
    # Get recommendations for each user repository
    for i, repo in enumerate(user_repos):
        if i >= len(user_repo_descriptions) or not user_repo_descriptions[i]:
            continue

    # Get the feature vector for the current user repository
        repo_features = features[:len(user_repo_descriptions), :][i, :].reshape(1, -1)

        # Find the k nearest neighbors to the current user repository
        _, indices = nn.kneighbors(repo_features)
        distances = nn.kneighbors_graph(repo_features).toarray()
        print(f"\nRecommended repositories for '{repo['name']}' repository:")
        for idx in indices[0][1:]:
            # Ignore user's own repository and repositories with missing descriptions
            if idx < len(user_repo_descriptions) or not top_repo_descriptions[idx - len(user_repo_descriptions)]:
                continue
            print(f"{top_repos_data.loc[idx - len(user_repo_descriptions), 'name']} - {top_repos_data.loc[idx - len(user_repo_descriptions), 'description']}")



recommend_repos()