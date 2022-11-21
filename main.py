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

    # remove non numerical columns
    movies_df = movies_df.drop(
        columns=["original_title", "imdb_id", "overview", "release_date"])

    # TODO: borrar o hacer otra cosa con los registros que tienen algun dato faltante?
    movies_df.dropna(inplace=True)

    GENRES_TO_ANALYZE = ["Adventura", "Comedia", "Drama"]
    movies_df = movies_df[movies_df["genres"].isin(GENRES_TO_ANALYZE)]

    # once removed not interesting genres, we remove the column for the grouping process over numerical variables
    movies_df = movies_df.drop(columns=["genres"])

    # TODO: SACAR ESTO, ES PARA TESTING
    movies_df = movies_df.sample(frac=0.05, random_state=random_state)

    # standarize data
    movies_df = pd.DataFrame(StandardScaler().fit_transform(
        movies_df), columns=movies_df.columns)

    # ------ Analisis univariado de variables ------
    # variables_plot(movies_df)

    # ------- K-Means -------
    # k_means = KMeans(K=3, max_iter=10, random_state=random_state)
    # print("kmeans clusters: ", k_means.fit(movies_df.values))


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
    kohonen = Kohonen(max_iter=100, random_state=random_state, initial_radius=4, initial_lr=0.1, K=10)
    kohonen.fit(movies_df.values)
    
    # Plot the number of elements per cluster
    cluster_matrix = kohonen.clusters_to_matrix()
    # Plot heatmap
    sns.heatmap(cluster_matrix, annot=True, cmap="YlGnBu")
    plt.show()

    # Plot U-Matrix
    u_matrix = kohonen.get_u_matrix()
    # Plot heatmap
    sns.heatmap(u_matrix, cmap="Greys_r", annot=True, fmt=".2f")
    plt.show()
    