import pandas as pd
import matplotlib.pyplot as plt
from .models.k_means import KMeans
from .models.hierarchical_clustering import HierarchicalClustering
from .models.kohonen import Kohonen
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns
from .models.unsupervised_classifier import UnsupervisedClassifier

def variables_plot(data_df):
    data_df.hist(edgecolor='black', linewidth=1.0,
                 xlabelsize=10, ylabelsize=10, grid=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    random_state = 1

    movies_df = pd.read_csv(
        "machine-learning/data/movie_data.csv", header=0, sep=';')
    # TODO: release_date should be discretized instead of dropped?

    # drop duplicates
    movies_df.drop_duplicates(subset=["original_title"], keep="first", inplace=True)
    movies_df.drop_duplicates(subset=["imdb_id"], keep="first", inplace=True)

    # remove non numerical columns
    movies_df = movies_df.drop(
        columns=["original_title", "imdb_id", "overview", "release_date"])

    # TODO: borrar o hacer otra cosa con los registros que tienen algun dato faltante?
    movies_df.dropna(inplace=True)

    GENRES_TO_ANALYZE = ["Adventure", "Comedy", "Drama"]
    movies_df = movies_df[movies_df["genres"].isin(GENRES_TO_ANALYZE)]

    # TODO: SACAR ESTO, ES PARA TESTING
    movies_df = movies_df.sample(frac=0.05, random_state=random_state)
    
    only_genres_df = movies_df["genres"]
    # once removed not interesting genres, we remove the column for the grouping process over numerical variables
    movies_df = movies_df.drop(columns=["genres"])

    # standarize data
    movies_df = pd.DataFrame(StandardScaler().fit_transform(
        movies_df), columns=movies_df.columns)

    # ------ Analisis univariado de variables ------
    # variables_plot(movies_df)

    # ------- K-Means -------
    k_means = KMeans(K=3, max_iter=100, random_state=random_state)
    k_means.fit(movies_df.values)
    print("kmeans clusters: ")
    for cluster in k_means.clusters:
        print(cluster)


    # ------- Hierarchical clustering -------
    # hierarchical_clustering = HierarchicalClustering()
    # hierarchical_clustering.fit(movies_df.values)

    # print("hierarchical clusters evolution: ")
    # for i, clusters in enumerate(hierarchical_clustering.clusters_evolution):
    #     print("Evolution : ", i)
    #     for cluster in clusters:
    #         print(cluster)
            
    # print("hierarchical distance evolution: ", hierarchical_clustering.distance_evolution)

    # ------- Kohonen clustering -------
    # kohonen = Kohonen(max_iter=100, random_state=random_state, initial_radius=4, initial_lr=0.1, K=10)
    # kohonen.fit(movies_df.values)

    # # For every feature, plot the heatmap with its weights
    # # We have 9 features
    # rows = 3
    # cols = 3
    # fig, axes = plt.subplots(rows, cols) 
    # for index, feature in enumerate(movies_df.columns):
    #     row_index = index // cols
    #     col_index = index % cols
    #     feature_weights = kohonen.get_feature_weights(index)
    #     sns.heatmap(feature_weights, cmap="YlGnBu", xticklabels=False, yticklabels=False, ax=axes[row_index, col_index])
    #     axes[row_index, col_index].set_title(feature)
    # plt.show()

    # # Plot the number of elements per cluster
    # cluster_matrix = kohonen.clusters_to_matrix()
    # # Plot heatmap without x and y ticks
    # sns.heatmap(cluster_matrix, annot=True, cmap="YlGnBu", linewidths=0.5, xticklabels=False, yticklabels=False)
    # plt.show()

    # # Plot U-Matrix
    # u_matrix = kohonen.get_u_matrix()
    # # Plot heatmap
    # sns.heatmap(u_matrix, cmap="Greys_r", annot=True, fmt=".2f", linewidths=0.5, xticklabels=False, yticklabels=False)
    # plt.show()

    # ------- Unsupervised classifier -------
    unsupervised_classifier = UnsupervisedClassifier(k_means)
    unsupervised_classifier.fit(movies_df.values, only_genres_df.values, movies_df.columns, only_genres_df.name)
    predictions = unsupervised_classifier.predict(movies_df.values[:5])
    print("predictions: ", predictions)
    print("real values: ", only_genres_df.values[:5])
