import pandas as pd
from sklearn.preprocessing import StandardScaler

def generate_dataset():
    movies_df = pd.read_csv(
        "machine-learning/data/movie_data.csv", header=0, sep=';')
    # TODO: release_date should be discretized instead of dropped?

    # drop duplicates by id
    movies_df.drop_duplicates(subset=["imdb_id"], keep="first", inplace=True)

    # remove non numerical columns
    movies_df = movies_df.drop(
        columns=["original_title", "imdb_id", "overview", "release_date"])

    # TODO: borrar o hacer otra cosa con los registros que tienen algun dato faltante?
    movies_df.dropna(inplace=True)

    GENRES_TO_ANALYZE = ["Adventure", "Comedy", "Drama"]
    movies_df = movies_df[movies_df["genres"].isin(GENRES_TO_ANALYZE)]

    # TODO: SACAR ESTO, ES PARA TESTING
    # movies_df = movies_df.sample(frac=0.25, random_state=random_state)

    only_genres_df = movies_df["genres"]
    # once removed not interesting genres, we remove the column for the grouping process over numerical variables
    movies_df = movies_df.drop(columns=["genres"])

    # standarize data
    movies_df = pd.DataFrame(StandardScaler().fit_transform(
        movies_df), columns=movies_df.columns)
    
    return movies_df, only_genres_df

