import pandas as pd
from sklearn.preprocessing import StandardScaler
from ..models.unsupervised_classifier import UnsupervisedClassifier


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

    movies_df = movies_df.sample(frac=0.05, random_state=random_state)

    only_genres_df = movies_df["genres"]
    # once removed not interesting genres, we remove the column for the grouping process over numerical variables
    movies_df = movies_df.drop(columns=["genres"])

    # standardize data
    movies_df = pd.DataFrame(StandardScaler().fit_transform(
        movies_df), columns=movies_df.columns)

    X_features = movies_df.columns.tolist()
    y_feature = only_genres_df.name

    unsupervised_classifier = UnsupervisedClassifier(
        "kohonen", K=5, max_iter=100, random_state=random_state)
    unsupervised_classifier.fit(
        movies_df.values, only_genres_df.values, X_features, y_feature)
    predictions = unsupervised_classifier.predict(movies_df.values[:5])

    print("predictions: ", predictions)
