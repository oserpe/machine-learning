import pandas as pd
import matplotlib.pyplot as plt
from ..models.k_means import KMeans
from ..models.hierarchical_clustering import HierarchicalClustering
from ..models.kohonen import Kohonen
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns
from ..models.unsupervised_classifier import UnsupervisedClassifier
from ..utils.plots import plot_n_k_fold_cv_eval, plot_cf_matrix, plot_kohonen_matrix_predictions, plot_curves_with_legend
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


def plot_kohonen_clustering(movies_df):
    kohonen = Kohonen(max_iter=100, random_state=random_state,
                      initial_radius=4, initial_lr=0.1, K=5)
    kohonen.fit(movies_df.values)

    # For every feature, plot the heatmap with its weights
    # We have 9 features
    rows = 3
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
    sns.heatmap(cluster_matrix, annot=True, cmap="YlGnBu",
                linewidths=0.5, xticklabels=False, yticklabels=False)
    plt.show()

    # Plot U-Matrix
    u_matrix = kohonen.get_u_matrix()
    # Plot heatmap
    sns.heatmap(u_matrix, cmap="Greys_r", annot=True, fmt=".2f",
                linewidths=0.5, xticklabels=False, yticklabels=False)
    plt.show()


def kohonen_matrix_predictions(movies_df, only_genres_df):
    classes = only_genres_df.unique().tolist()
    X_features = movies_df.columns.tolist()
    y_feature = only_genres_df.name
    unsupervised_classifier = UnsupervisedClassifier(
        "kohonen", K=5, max_iter=100)

    test_samples = 250
    unsupervised_classifier.fit(
        movies_df.values, only_genres_df.values, X_features, y_feature)
    y_predictions = unsupervised_classifier.predict(
        movies_df.values[:test_samples])
    y = only_genres_df.values[:test_samples]
    # Plot kohonen matrix with clusters
    kohonen_predictions = unsupervised_classifier.model.predict(
        movies_df.values)
    plot_kohonen_matrix_predictions(
        unsupervised_classifier._model, y, kohonen_predictions, classes)


def n_k_fold(model, movies_df, only_genres_df):
    classes = only_genres_df.unique().tolist()
    X_features = movies_df.columns.tolist()
    y_feature = only_genres_df.name
    unsupervised_classifier = UnsupervisedClassifier(model, K=5, max_iter=100)

    # Plot N-K-Fold
    n = 5
    k = 5
    plot_n_k_fold_cv_eval(movies_df.values, only_genres_df.values, n=n, model=unsupervised_classifier,
                          k=k, X_features=X_features, y_feature=y_feature, classes=classes)


if __name__ == "__main__":
    random_state = 1

    movies_df, only_genres_df = generate_dataset_all_genres_dataset()
    # movies_df, only_genres_df = generate_dataset()

    # ------- PLOTS ------- #
    # Plot metodo del codo para elegir K en K_means
    elbow_method(movies_df.values, [1,2,3,4,6,8], 5, random_state)

    # Plot "model" n k fold
    # model = "kohonen"
    # n_k_fold(model, movies_df, only_genres_df)

    # kohonen_matrix_predictions(movies_df, only_genres_df)

    # plot_kohonen_clustering(movies_df)
