import pandas as pd
from sklearn.preprocessing import StandardScaler


def generate_dataset(genres_to_analyze=["Adventure", "Comedy", "Drama"], standardize=True):
    movies_df = pd.read_csv(
        "machine-learning/data/movie_data.csv", header=0, sep=';')
    # TODO: release_date should be discretized instead of dropped?

    # drop duplicates by id
    movies_df.drop_duplicates(subset=["imdb_id"], keep="first", inplace=True)

    movies_df['release_date'] = pd.to_datetime(
        movies_df['release_date'], format='%Y-%m-%d')

    # bin release_date by year in new column
    movies_df['year'] = pd.DatetimeIndex(movies_df['release_date']).year

    # remove non numerical columns
    movies_df = movies_df.drop(
        columns=["original_title", "imdb_id", "overview", "release_date"])

    # TODO: borrar o hacer otra cosa con los registros que tienen algun dato faltante?
    movies_df.dropna(inplace=True)

    if genres_to_analyze:
        # get only genres to analyze
        movies_df = movies_df[movies_df['genres'].isin(genres_to_analyze)]

    # TODO: SACAR ESTO, ES PARA TESTING
    # movies_df = movies_df.sample(frac=0.25, random_state=random_state)

    only_genres_df = movies_df[["genres"]]
    # once removed not interesting genres, we remove the column for the grouping process over numerical variables
    movies_df = movies_df.drop(columns=["genres"])

    # standarize data
    if standardize:
        movies_df = pd.DataFrame(StandardScaler().fit_transform(
            movies_df), columns=movies_df.columns)

    return movies_df, only_genres_df
