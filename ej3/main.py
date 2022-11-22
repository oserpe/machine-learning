import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from ..models.unsupervised_classifier import UnsupervisedClassifier


def find_hyperparameters_kmeans(k_range, max_iter_range, random_state_range, X_train, y_train, X_test, y_test):
    grid = GridSearchCV(UnsupervisedClassifier(model="kmeans", K=1, max_iter=100, random_state=0, verbose=False), param_grid={
        'K': k_range,
        'max_iter': max_iter_range,
        'random_state': random_state_range,
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
        'max_iter': [100, 200, 300, 400, 500],
        'random_state': [0, 1, 2, 3, 4, 5],
        'kohonen_initial_lr': [0.01, 0.1, 0.5, 1],
        'kohonen_initial_radius': [1, 2, 3, 4, 5],
        'model': ['kohonen']
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

    movies_df = movies_df.sample(frac=0.1, random_state=random_state)

    only_genres_df = movies_df[["genres"]]
    # once removed not interesting genres, we remove the column for the grouping process over numerical variables
    movies_df = movies_df.drop(columns=["genres"])

    # standardize data
    movies_df = pd.DataFrame(StandardScaler().fit_transform(
        movies_df), columns=movies_df.columns)

    X_features = movies_df.columns.tolist()
    y_feature = only_genres_df.columns.tolist()

    find_hyperparameters_kmeans(k_range=range(3, 5), max_iter_range=range(100, 1000, 200), random_state_range=range(
        0, 5), X_train=movies_df[X_features], y_train=only_genres_df, X_test=movies_df[X_features], y_test=only_genres_df)

    # unsupervised_classifier = UnsupervisedClassifier(
    #     "hierarchical", K=10, max_iter=100, random_state=random_state)
    # unsupervised_classifier.fit(
    #     movies_df[X_features], only_genres_df)

    # print(unsupervised_classifier.score(
    #     movies_df[X_features], only_genres_df))
    # predictions = unsupervised_classifier.predict(movies_df.values[:5])

    # print("predictions: ", predictions)
