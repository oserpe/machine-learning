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
    round_decimals_columns = ["vote_average"]
    round_1_unit_columns = ["year", "popularity","runtime"]
    round_3_unit_columns = ["vote_count"]
    round_8_unit_columns = ["budget", "revenue"]
    
    for column in movies_df.columns.drop(round_1_unit_columns, round_3_unit_columns, round_8_unit_columns):
        display_data = data_df.groupby([column, key_column])[column].count().unstack(key_column)
        display_data.plot(kind='bar', rot=0, stacked=True, ylabel="Cantidad de ejemplares")
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
    display_data[column] = data_df[column].apply(lambda x: round(round(x, -round_units)))
    display_data = display_data.groupby([column, key_column])[column].count().unstack(key_column)
    display_data.plot(kind='bar', rot=0, stacked=True, ylabel="Cantidad de ejemplares")
    
    if column == "revenue":
        plt.xticks(rotation=20, ha="right")
    elif column == "year":
        plt.xticks(rotation=90, ha="right")
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
