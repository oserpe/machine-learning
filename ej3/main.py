import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from ..models.unsupervised_classifier import UnsupervisedClassifier
from ..models.cluster import ClusteringDistance


def find_hyperparameters_kmeans(X_train, y_train, X_test, y_test):
    grid = GridSearchCV(UnsupervisedClassifier(model="kmeans", K=1, max_iter=100, random_state=0, verbose=False), param_grid={
        'K': range(3, 10),
        'max_iter': range(100, 500, 100),
        'random_state': range(0, 5),
        'model': ['kmeans']
    }, cv=5, n_jobs=-1, verbose=3)

    # fitting the model for grid search
    grid.fit(X_train, y_train)

    # print best parameter after tuning
    print(grid.best_params_)
    grid_predictions = grid.predict(X_test)
    print(classification_report(y_test, grid_predictions))


def find_hyperparameters_kohonen(X_train, y_train, X_test, y_test):
    grid = GridSearchCV(UnsupervisedClassifier(model="kohonen", K=1, max_iter=100, random_state=0, verbose=False), param_grid={
        'K': [3, 4, 5, 6],
        'max_iter': [100, 200, 300],
        'random_state': [0, 1, 2],
        'kohonen_initial_lr': [0.01, 0.1, 0.5, 1],
        'kohonen_initial_radius': [1, 2, 3],
        'model': ['kohonen']
    }, cv=5, n_jobs=-1, verbose=3)

    # fitting the model for grid search
    grid.fit(X_train, y_train)

    # print best parameter after tuning
    print(grid.best_params_)
    grid_predictions = grid.predict(X_test)
    print(classification_report(y_test, grid_predictions))


def find_hyperparameters_hierarchical(X_train, y_train, X_test, y_test):
    grid = GridSearchCV(UnsupervisedClassifier(model="hierarchical", K=1, max_iter=None, random_state=0, verbose=False), param_grid={
        'K': range(3, 10),
        'random_state': range(0, 5),
        'hierarchical_distance_metric': [ClusteringDistance.CENTROID, ClusteringDistance.MAXIMUM, ClusteringDistance.MINIMUM, ClusteringDistance.AVERAGE],
        'model': ['hierarchical']
    }, cv=5, n_jobs=-1, verbose=3)

    # fitting the model for grid search
    grid.fit(X_train, y_train)

    # print best parameter after tuning
    print(grid.best_params_)
    grid_predictions = grid.predict(X_test)
    print(classification_report(y_test, grid_predictions))


if __name__ == "__main__":
    random_state = 1
    movies_df = pd.read_csv(
        "machine-learning/data/movie_data.csv", header=0, sep=';')
    # TODO: release_date should be discretized instead of dropped?

    # drop duplicates
    movies_df.drop_duplicates(
        subset=["original_title"], keep="first", inplace=True)
    movies_df.drop_duplicates(subset=["imdb_id"], keep="first", inplace=True)

    # remove non numerical columns
    movies_df = movies_df.drop(
        columns=["original_title", "imdb_id", "overview", "release_date"])

    # TODO: borrar o hacer otra cosa con los registros que tienen algun dato faltante?
    movies_df.dropna(inplace=True)

    GENRES_TO_ANALYZE = ["Adventure", "Comedy", "Drama"]
    movies_df = movies_df[movies_df["genres"].isin(GENRES_TO_ANALYZE)]

    movies_df = movies_df.sample(frac=0.2, random_state=random_state)

    only_genres_df = movies_df[["genres"]]
    # once removed not interesting genres, we remove the column for the grouping process over numerical variables
    movies_df = movies_df.drop(columns=["genres"])

    # standardize data
    movies_df = pd.DataFrame(StandardScaler().fit_transform(
        movies_df), columns=movies_df.columns)

    X_features = movies_df.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        movies_df[X_features], only_genres_df, test_size=0.2, random_state=random_state)

    # find_hyperparameters_kmeans(
    #     X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    find_hyperparameters_kohonen(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    # find_hyperparameters_hierarchical(
    #     X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
