import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from ..models.unsupervised_classifier import UnsupervisedClassifier
from ..models.cluster import ClusteringDistance
from ..data.generate_dataset import generate_dataset
from ..data.data_categorization import categorize_data_with_equal_width
from matplotlib import pyplot as plt
from ..models.hierarchical_clustering import HierarchicalClustering
from matplotlib.ticker import FormatStrFormatter
from ..utils.plots import plot_n_k_fold_cv_eval, plot_cf_matrix, plot_kohonen_matrix_predictions, plot_curves_with_legend, plot_metrics_heatmap
from ..models.kohonen import Kohonen
import seaborn as sns


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


def find_hyperparameters_kmeans(X_train, y_train, X_test, y_test):
    grid = GridSearchCV(UnsupervisedClassifier(model="kmeans", K=1, max_iter=100, random_state=0, verbose=False), param_grid={
        'K': range(3, 6),
        'max_iter': range(100, 500, 100),
        'random_state': range(0, 3),
        'model': ['kmeans']
    }, cv=5, n_jobs=-1, verbose=3)

    # fitting the model for grid search
    grid.fit(X_train, y_train)

    # print best parameter after tuning
    print(grid.best_params_)
    grid_predictions = grid.predict(X_test)
    print(classification_report(y_test, grid_predictions))

    # save params and classification report to file
    with open("machine-learning/models/kmeans_params.txt", "w") as f:
        f.write(str(grid.best_params_))
        f.write("\n")
        f.write(str(classification_report(y_test, grid_predictions)))


def find_hyperparameters_kohonen(X_train, y_train, X_test, y_test):
    grid = GridSearchCV(UnsupervisedClassifier(model="kohonen", K=1, max_iter=100, random_state=0, verbose=False), param_grid={
        'K': [5],
        'max_iter': [200],
        'random_state': [1],
        'kohonen_initial_lr': [0.01, 0.1, 1],
        'kohonen_initial_radius': [1, 2, 3, 4, 5],
        'model': ['kohonen']
    }, cv=5, n_jobs=-1, verbose=3)

    # fitting the model for grid search
    grid.fit(X_train, y_train)

    # print best parameter after tuning
    print(grid.best_params_)
    grid_predictions = grid.predict(X_test)
    print(classification_report(y_test, grid_predictions))

    # save params and classification report to file
    with open("machine-learning/models/kohonen_params.txt", "w") as f:
        f.write(str(grid.best_params_))
        f.write("\n")
        f.write(str(classification_report(y_test, grid_predictions)))


def find_hyperparameters_hierarchical(X_train, y_train, X_test, y_test):
    grid = GridSearchCV(UnsupervisedClassifier(model="hierarchical", K=1, max_iter=None, random_state=0, verbose=False), param_grid={
        'K': range(2, 5),
        'hierarchical_distance_metric': [ClusteringDistance.CENTROID, ClusteringDistance.MAXIMUM, ClusteringDistance.MINIMUM, ClusteringDistance.AVERAGE],
        'model': ['hierarchical']
    }, cv=5, n_jobs=-1, verbose=3)

    # fitting the model for grid search
    grid.fit(X_train, y_train)

    # print best parameter after tuning
    print(grid.best_params_)
    grid_predictions = grid.predict(X_test)
    print(classification_report(y_test, grid_predictions))

    # save params and classification report to file
    with open("machine-learning/models/hierarchical_params.txt", "w") as f:
        f.write(str(grid.best_params_))
        f.write("\n")
        f.write(str(classification_report(y_test, grid_predictions)))


def plot_genres_impact():
    # iterate over all columns
    movies_df, only_genres_df = generate_dataset(standardize=False)
    data_df = pd.concat([movies_df, only_genres_df], axis=1)
    key_column = "genres"
    round_decimals_columns = ["vote_average"]
    round_1_unit_columns = ["year", "popularity", "runtime"]
    round_3_unit_columns = ["vote_count"]
    round_8_unit_columns = ["budget", "revenue"]

    for column in movies_df.columns.drop(round_1_unit_columns, round_3_unit_columns, round_8_unit_columns):
        display_data = data_df.groupby([column, key_column])[
            column].count().unstack(key_column)
        display_data.plot(kind='bar', rot=0, stacked=True,
                          ylabel="Cantidad de ejemplares")
        plt.show(block=True)

    # round decimals
    for column in round_decimals_columns:
        _round_column_plot(data_df, column, key_column, 0)

    # round 1 unit
    for column in round_1_unit_columns:
        _round_column_plot(data_df, column, key_column, 1)

    # round 3 units
    for column in round_3_unit_columns:
        _round_column_plot(data_df, column, key_column, 3)

    # round 8 units
    for column in round_8_unit_columns:
        _round_column_plot(data_df, column, key_column, 8)


def _round_column_plot(data_df, column, key_column, round_units):
    display_data = data_df
    display_data[column] = data_df[column].apply(
        lambda x: round(round(x, -round_units)))
    display_data = display_data.groupby([column, key_column])[
        column].count().unstack(key_column)
    display_data.plot(kind='bar', rot=0, stacked=True,
                      ylabel="Cantidad de ejemplares")

    if column == "revenue":
        plt.xticks(rotation=20, ha="right")
    elif column == "year":
        plt.xticks(rotation=90, ha="right")
    plt.show(block=True)


def hierarchichal_matrix_predictions(X_train, X_test, y_train, y_test):
    classes = y_test[y_test.columns[0]].unique().tolist()
    unsupervised_classifier = UnsupervisedClassifier(
        "hierarchical", K=3, max_iter=100)

    unsupervised_classifier.fit(
        X_train, y_train)

    y_predictions = unsupervised_classifier.predict(
        X_test)
    y = y_test.values.flatten()

    # Plot confusion matrix
    cf_matrix = plot_cf_matrix(y, y_predictions, labels=classes)

    # Plot metrics heatmap
    plot_metrics_heatmap(cf_matrix)

def plot_hierarchical_elbow_method(X_train, y_train):
    unsupervised_classifier = UnsupervisedClassifier(
        "hierarchical", K=1, max_iter=100)

    unsupervised_classifier.fit(
        X_train, y_train)
    # plot elbow method 
    # !!! CHECK THAT HIERARCHICAL IS SAVING VARIATIONS !!!
    Ks = list(range(1,21))
    Ws = []
    print("Training hierarchical")
    for K in Ks:
        Ws.append(unsupervised_classifier._model.variations[-K])

    plt.plot(Ks, Ws)
    plt.xlabel("K")
    plt.ylabel("W")
    plt.yscale("log")
    plt.show()

    # plot distance evolution
    plt.plot(unsupervised_classifier._model.distance_evolution[-100:])
    plt.xlabel("K")
    plt.ylabel("Distancia minima")
    plt.show()
    
def k_means_matrix_predictions(X_train, X_test, y_train, y_test):
    classes = y_test[y_test.columns[0]].unique().tolist()
    unsupervised_classifier = UnsupervisedClassifier(
        "kmeans", K=5, max_iter=100, random_state=0)

    unsupervised_classifier.fit(
        X_train, y_train)

    y_predictions = unsupervised_classifier.predict(
        X_test)
    y = y_test.values.flatten()

    # Plot confusion matrix
    cf_matrix = plot_cf_matrix(y, y_predictions, labels=classes)

    # Plot metrics heatmap
    plot_metrics_heatmap(cf_matrix)


def plot_kohonen_clustering(kohonen, X_train, annotations=True):
    # kohonen = Kohonen(max_iter=200, random_state=1,
    #                   initial_radius=2, initial_lr=0.1, K=5, )
    # kohonen.fit(X_train.values)

    # For every feature, plot the heatmap with its weights
    # We have 10 features
    rows = 4
    cols = 3
    fig, axes = plt.subplots(rows, cols)
    for index, feature in enumerate(X_train.columns):
        row_index = index // cols
        col_index = index % cols
        feature_weights = kohonen.get_feature_weights(index)
        sns.heatmap(feature_weights, cmap="coolwarm", xticklabels=False,
                    yticklabels=False, ax=axes[row_index, col_index])
        axes[row_index, col_index].set_title(feature)
    plt.show()

    # Plot the number of elements per cluster
    cluster_matrix = kohonen.clusters_to_matrix()
    # Plot heatmap without x and y ticks
    sns.heatmap(cluster_matrix, annot=annotations, cmap="coolwarm",
                linewidths=0.5, xticklabels=False, yticklabels=False)
    plt.show()

    # Plot U-Matrix
    u_matrix = kohonen.get_u_matrix()
    # Plot heatmap
    sns.heatmap(u_matrix, cmap="Greys_r", annot=annotations, fmt=".2f",
                linewidths=0.5, xticklabels=False, yticklabels=False)
    plt.show()




def kohonen_matrix_predictions(X_train, X_test, y_train, y_test):
    classes = y_test[y_test.columns[0]].unique().tolist()
    unsupervised_classifier = UnsupervisedClassifier(
        "kohonen", K=5, max_iter=200, kohonen_initial_radius=2, kohonen_initial_lr=0.1, random_state=1)

    unsupervised_classifier.fit(
        X_train, y_train)

    y_predictions = unsupervised_classifier.predict(
        X_test)
    y = y_test.values.flatten()

    plot_kohonen_clustering(unsupervised_classifier._model, X_train, annotations=False)

    # Plot confusion matrix
    cf_matrix = plot_cf_matrix(y, y_predictions, labels=classes)

    # Plot metrics heatmap
    plot_metrics_heatmap(cf_matrix)

    # Plot kohonen matrix with clusters
    kohonen_predictions = unsupervised_classifier._model.predict(
        X_test.values)
    plot_kohonen_matrix_predictions(
        unsupervised_classifier._model, y, kohonen_predictions, classes)



if __name__ == "__main__":
    # plot_genres_impact()
    random_state = 1

    movies_df, only_genres_df = generate_dataset(
        # n_samples=100,
        random_state=random_state)

    X_features = movies_df.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        movies_df[X_features], only_genres_df, test_size=0.2, random_state=random_state)

    # hierarchichal_matrix_predictions(X_train, X_test, y_train, y_test)
    # k_means_matrix_predictions(X_train, X_test, y_train, y_test)
    kohonen_matrix_predictions(X_train, X_test, y_train, y_test)
    exit()

    # Plot "model" n k fold
    # model = "kohonen"
    # n_k_fold(model, X_train, y_train)

    # find_hyperparameters_kmeans(
    #     X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    find_hyperparameters_kohonen(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    # find_hyperparameters_hierarchical(
    #     X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
