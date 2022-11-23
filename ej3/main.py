import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from ..models.unsupervised_classifier import UnsupervisedClassifier
from ..models.cluster import ClusteringDistance
from ..data.generate_dataset import generate_dataset
from ..data.data_categorization import categorize_data_with_equal_width
from matplotlib import pyplot as plt


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
        'K': [3, 4, 5],
        'max_iter': [100, 200],
        'random_state': [0, 1],
        'kohonen_initial_lr': [0.01, 0.1],
        'kohonen_initial_radius': [1, 2, 3],
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
    discrete_columns = ["budget", "revenue", "popularity"]
    for column in movies_df.columns.drop(discrete_columns):
        display_data = data_df.groupby([column, key_column])[
            column].count().unstack(key_column)

        display_data.plot(kind='bar', rot=0, stacked=True,
                          ylabel="Cantidad de ejemplares")

        plt.show(block=True)

    for column in discrete_columns:
        display_data = categorize_data_with_equal_width(
            data_df, {column: 25000})
        display_data = display_data.groupby([column, key_column])[
            column].count().unstack(key_column)

        plt.hist(display_data, edgecolor='black', stacked=True)
        plt.legend(display_data.columns.values)
        # set title and labels
        plt.xlabel(column)
        plt.ylabel("Cantidad de ejemplares")
        plt.show(block=True)


if __name__ == "__main__":
    # plot_genres_impact()
    random_state = 1

    movies_df, only_genres_df = generate_dataset(
        n_samples=1000, random_state=random_state)

    X_features = movies_df.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        movies_df[X_features], only_genres_df, test_size=0.2, random_state=random_state)

    # find_hyperparameters_kmeans(
    #     X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    # find_hyperparameters_kohonen(
    #     X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    find_hyperparameters_hierarchical(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
