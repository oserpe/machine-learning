import pandas as pd
import matplotlib.pyplot as plt

from .plot_2d_clusters import plot_2d_clusters_scatter

from .pca import apply_pca_to_data
from ..models.k_means import KMeans
from ..models.hierarchical_clustering import HierarchicalClustering
from ..models.kohonen import Kohonen
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns
from ..models.unsupervised_classifier import UnsupervisedClassifier
from ..utils.plots import plot_n_k_fold_cv_eval, plot_cf_matrix, plot_kohonen_matrix_predictions, plot_curves_with_legend, plot_metrics_heatmap
from ..data.generate_dataset import generate_dataset


def elbow_method(X, Ks, times, initial_random_state):
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


def plot_kohonen_clustering(movies_df, annotations=True):
    kohonen = Kohonen(max_iter=100, random_state=random_state,
                      initial_radius=4, initial_lr=0.1, K=5)
    kohonen.fit(movies_df.values)

    # For every feature, plot the heatmap with its weights
    # We have 10 features
    rows = 4
    cols = 3
    fig, axes = plt.subplots(rows, cols)
    for index, feature in enumerate(movies_df.columns):
        row_index = index // cols
        col_index = index % cols
        feature_weights = kohonen.get_feature_weights(index)
        sns.heatmap(feature_weights, cmap="YlGnBu", xticklabels=False,
                    yticklabels=False, ax=axes[row_index, col_index])
        axes[row_index, col_index].set_title(feature)
    plt.show()

    # Plot the number of elements per cluster
    cluster_matrix = kohonen.clusters_to_matrix()
    # Plot heatmap without x and y ticks
    sns.heatmap(cluster_matrix, annot=annotations, cmap="YlGnBu",
                linewidths=0.5, xticklabels=False, yticklabels=False)
    plt.show()

    # Plot U-Matrix
    u_matrix = kohonen.get_u_matrix()
    # Plot heatmap
    sns.heatmap(u_matrix, cmap="Greys_r", annot=annotations, fmt=".2f",
                linewidths=0.5, xticklabels=False, yticklabels=False)
    plt.show()


def kohonen_matrix_predictions(movies_df, only_genres_df):
    classes = only_genres_df[only_genres_df.columns[0]].unique().tolist()
    unsupervised_classifier = UnsupervisedClassifier(
        "kohonen", K=5, max_iter=100)

    unsupervised_classifier.fit(
        movies_df, only_genres_df)
    y_predictions = unsupervised_classifier.predict(
        movies_df)
    y = only_genres_df.values.flatten()

    # Plot confusion matrix
    cf_matrix = plot_cf_matrix(y, y_predictions, labels=classes)

    # Plot metrics heatmap
    metrics_dict, metrics_df = plot_metrics_heatmap(cf_matrix)

    # Plot kohonen matrix with clusters
    kohonen_predictions = unsupervised_classifier._model.predict(
        movies_df.values)
    plot_kohonen_matrix_predictions(
        unsupervised_classifier._model, y, kohonen_predictions, classes)


def n_k_fold(model, movies_df, only_genres_df):
    classes = only_genres_df[only_genres_df.columns[0]].unique().tolist()
    X_features = movies_df.columns.tolist()
    y_features = only_genres_df.columns.tolist()
    unsupervised_classifier = UnsupervisedClassifier(model, K=3, max_iter=100)

    # Plot N-K-Fold
    n = 5
    k = 5
    plot_n_k_fold_cv_eval(movies_df.values, only_genres_df.values, n=n, model=unsupervised_classifier,
                          k=k, X_features=X_features, y_features=y_features, classes=classes)


def pca_plot(X, model, title):
    pca, pca_data = apply_pca_to_data(data=X, n_components=2)

    # fit the model
    model.fit(pca_data)

    pca_data_df = pd.DataFrame(pca_data, columns=["PC1", "PC2"])

    # get the labels
    df_clusters = []
    for i, cluster in enumerate(model.get_clusters()):
        df_points = pd.DataFrame(cluster.points, columns=pca_data_df.columns)
        df_points["cluster"] = i
        df_merged = pd.merge(pca_data_df, df_points)
        df_clusters.append(df_merged)

    # flatten the list
    df_clusters = [item for sublist in df_clusters for item in sublist]

    # separate labels from data
    df_clusters = pd.DataFrame(df_clusters)
    labels = df_clusters["cluster"].to_numpy()
    df_clusters = df_clusters.drop(columns=["cluster"])

    # plot the data
    plot_2d_clusters_scatter(df_clusters, labels, title=title)


if __name__ == "__main__":
    random_state = 1

    movies_df, only_genres_df = generate_dataset()

    movies_df = movies_df.head(100)
    only_genres_df = only_genres_df.head(100)

    # ------- PLOTS ------- #
    # Plot metodo del codo para elegir K en K_means
    # elbow_method(movies_df.values, [1, 2, 3, 4, 6, 8], 5, random_state)

    # Plot "model" n k fold
    model = "kohonen"
    # n_k_fold(model, movies_df, only_genres_df)
    
    plot_kohonen_clustering(movies_df, annotations=False)

    kohonen_matrix_predictions(movies_df, only_genres_df)


    # Plot de PCA de KMeans
    k_means = KMeans(K=3, max_iter=100, random_state=random_state)
    pca_plot(movies_df, k_means, "KMeans Clustering with PCA")
