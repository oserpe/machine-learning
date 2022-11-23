import pandas as pd
import matplotlib.pyplot as plt

from ..models.cluster import ClusteringDistance

from .plot_2d_clusters import plot_2d_clusters_scatter

from .pca import apply_pca_to_data
from ..models.k_means import KMeans
from ..models.hierarchical_clustering import HierarchicalClustering
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from ..models.unsupervised_classifier import UnsupervisedClassifier
from ..utils.plots import plot_n_k_fold_cv_eval, plot_cf_matrix, plot_kohonen_matrix_predictions, plot_curves_with_legend, plot_metrics_heatmap
from ..data.generate_dataset import generate_dataset


def k_means_elbow_method(Ks, times, initial_random_state):
    movies_df, only_genres_df = generate_dataset()
    
    X = movies_df.values
    Ws = []
    legends = []

    for i in range(times):
        random_state = initial_random_state + i
        Ws_k = []
        print("Training ", i+1, " of ", times)
        for K in Ks:
            print("\tK = ", K)
            k_means = KMeans(K, max_iter=100, random_state=random_state)
            k_means.fit(X)
            Ws_k.append(k_means.variations[-1])

        legends.append("Random state = " + str(random_state))
        Ws.append(Ws_k)

    plot_curves_with_legend(Ks, Ws, legends=list(
        range(times)), X_label="K", Y_label="W")



def pca_plot(X, model, title):
    pca, pca_data = apply_pca_to_data(data=X, n_components=2)

    # fit the model
    model.fit(pca_data)

    pca_data_df = pd.DataFrame(pca_data, columns=["PC1", "PC2"])

    # get the labels
    df_clusters = pd.DataFrame()
    for i, cluster in enumerate(model.get_clusters()):
        df_points = pd.DataFrame(cluster.points, columns=pca_data_df.columns)
        df_points["cluster"] = i
        df_merged = pd.merge(pca_data_df, df_points)
        df_clusters = pd.concat([df_clusters, df_merged])

    # separate labels from data
    labels = df_clusters["cluster"].to_list()
    df_clusters = df_clusters.drop(columns=["cluster"])

    x_data = df_clusters["PC1"].to_list()
    y_data = df_clusters["PC2"].to_list()
    # plot the data
    plot_2d_clusters_scatter(x_data, y_data, labels, x_label=f"PC1 {pca.explained_variance_ratio_[0]*100:.2f}%",
                             y_label=f"PC2 {pca.explained_variance_ratio_[1]*100:.2f}%", title=title)


if __name__ == "__main__":
    random_state = 1

    movies_df, only_genres_df = generate_dataset()

    # movies_df = movies_df.head(100)
    # only_genres_df = only_genres_df.head(100)

    # ------- PLOTS ------- #

    # Plot metodo del codo para elegir K en K_means
    # k_means_elbow_method([1,2,3,4,6,8], 5, random_state)

    # # Plot "model" n k fold
    # model = "kohonen"
    # # n_k_fold(model, movies_df, only_genres_df)

    # plot_kohonen_clustering(movies_df, annotations=False)

    # kohonen_matrix_predictions(movies_df, only_genres_df)

    # Plot de PCA de KMeans
    k_means = KMeans(K=3, max_iter=100, random_state=random_state)
    pca_plot(movies_df, k_means, "KMeans Clustering with PCA")

    # Plot de PCA de Kohonen
    kohonen = Kohonen(K=5, max_iter=200, initial_lr=0.1,
                      initial_radius=5, random_state=random_state)
    pca_plot(movies_df, kohonen, "Kohonen Clustering with PCA")

    # Plot de PCA de jerarquico
    hierarchical = HierarchicalClustering(
        K=3, distance_metric=ClusteringDistance.CENTROID)
    pca_plot(movies_df, hierarchical, "Hierarchical Clustering with PCA")
