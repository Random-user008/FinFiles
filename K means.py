import requests
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
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
    # Get top n indices with highest topic probability sum
    topic_probs = topic_distributions.sum(axis=1)
    top_n_indices = topic_probs.argsort()[-top_n:][::-1]

    return top_n_indices  


def elbow_method_plot(model, data):
    model.fit(data)

    # Calculate the within-cluster sum of squares (inertia) for each value of k
    k_range = range(1, 11)
    inertias = []
    sil_scores = []
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42)
        model.fit(data)
        inertias.append(model.inertia_)
        if k > 1:
            sil_scores.append(silhouette_score(data, model.labels_))

    # Plot the elbow curve
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.plot(k_range, inertias, marker='o', color=color)
    ax1.set_xlabel('Number of clusters')
    ax1.set_ylabel('Inertia', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.plot(k_range[1:], sil_scores, marker='o', color=color)
    ax2.set_ylabel('Silhouette Score', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Elbow Method Plot')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

from sklearn.metrics.pairwise import cosine_similarity

def recommend_repos():

    user_repos = requests.get(f'https://api.github.com/users/kaayush71/repos').json()
    user_repo_descriptions = [repo['description'] for repo in user_repos if repo['description']]
    user_repo_languages = [repo['language'] for repo in user_repos if repo['language']]
    # user_repo_topics = [",".join(repo['topics']) for repo in user_repos if repo['topics']]
    print()
    top_repos_data = get_top_starred_repos()
    top_repo_descriptions = top_repos_data['description'].tolist()
    top_repo_languages = top_repos_data['language'].tolist()
    # top_repo_topics = top_repos_data['topics'].tolist()
    # Remove np.nan values from descriptions
    top_repo_descriptions = [desc for desc in top_repo_descriptions if isinstance(desc, str)]
    top_repo_languages = [lang for lang in top_repo_languages if isinstance(lang, str)]
    # top_repo_topics = [topic for topic in top_repo_topics if isinstance(topic,str)]
    # Concatenate user repository descriptions with top repository descriptions
    texts = user_repo_descriptions + top_repo_descriptions
    languages = user_repo_languages + top_repo_languages
    # topics  = user_repo_topics +  top_repo_topics
    # Check if there are any repository descriptions available
    if len(texts) == 0:
        print("No repository descriptions available for recommendation")
        return
    
    # Vectorize the text data using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    vectorizer.fit(texts)
    
    # Extract topic distributions and language encodings for the text data
    topic_distributions = extract_topic_distributions(texts)
    # topic_distributions = get_top_topics(texts)
    language_encodings = vectorizer.transform(languages)
    # topics_encoding = vectorizer.transform(topics)
    n = max(topic_distributions.shape[0], language_encodings.shape[0],)
    topic_distributions_pad = np.pad(topic_distributions, ((0, n - topic_distributions.shape[0]), (0, 0)))
    language_encodings_pad = np.pad(language_encodings.toarray(), ((0, n - language_encodings.shape[0]), (0, 0)))
    # topics_pad = np.pad(topics_encoding.toarray(),((0, n - topics_encoding.shape[0]), (0, 0)))
    data = np.concatenate((topic_distributions_pad, language_encodings_pad), axis=1)

    # Combine topic distributions and language encodings
    # data = np.concatenate((topic_distributions, language_encodings.toarray()), axis=1)
    
    # Use the elbow method to find the optimal number of clusters for K-Means
    kmeans = KMeans(random_state=42)
    # elbow_method_plot(kmeans, data)
    
    # Cluster the data using K-Means
    n_clusters = int(input("Enter the number of clusters: "))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data)
    silhouette_avg = silhouette_score(data, kmeans.labels_)
    print("The Silhouette score is :", silhouette_avg)
    # for i in range(n_clusters):
    #     cluster_indices = np.where(kmeans.labels_ == i)[0]
    #     cluster_data = data[cluster_indices]
    #     similarities = cosine_similarity(cluster_data)
    #     user_repo_indices = np.where(np.array(user_repo_descriptions) != '')[0]
    #     for j, user_repo_idx in enumerate(user_repo_indices):
    #         similarities_j = similarities[j,:]
    #         sim_ranked_idx = np.argsort(similarities_j)[::-1]
    #         for sim_idx in sim_ranked_idx:
    #             if user_repo_idx != cluster_indices[sim_idx]:
    #                 repo_desc = texts[cluster_indices[sim_idx]]
    #                 if repo_desc not in user_repo_descriptions:
    #                     print(f"Recommendation {j+1} for cluster {i+1}: {repo_desc[:50]} ... (similarity: {similarities_j[sim_idx]:.2f})")
    #                     break

    # Find the nearest neighbors of the user's repositories within each cluster
    for i in range(n_clusters):
        cluster_indices = np.where(kmeans.labels_ == i)[0]
        cluster_data = data[cluster_indices]
        similarities = cosine_similarity(cluster_data)
        user_repo_indices = np.where(np.array(user_repo_descriptions) != None)[0]
        for j, user_repo_index in enumerate(user_repo_indices):
            similarities_to_user_repo = similarities[j]
            sorted_indices = np.argsort(similarities_to_user_repo)[::-1][:5]
            print(f"\nTop {5} recommendations for user repository {j+1} in cluster {i+1}:")                 
            for index in sorted_indices:
                if index < len(cluster_indices):
                    print(f"Name: {top_repos_data.iloc[cluster_indices[index]]['name']}")
                    print(f"Language: {top_repos_data.iloc[cluster_indices[index]]['language']}")
                    print(f"Description: {top_repos_data.iloc[cluster_indices[index]]['description']}")
                    # print(f"Topics: {top_repos_data.iloc[cluster_indices[index]]['topics']}")
                    # print(f"Cosine similarity level: {similarities_to_user_repo[index]}")
                    print()
                print()

recommend_repos()