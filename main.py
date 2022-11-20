import pandas as pd
import matplotlib.pyplot as plt
from .models.k_means import KMeans
from sklearn.preprocessing import StandardScaler


def variables_plot(data_df):
    data_df.hist(edgecolor='black', linewidth=1.0,
                 xlabelsize=10, ylabelsize=10, grid=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    random_state = 1

    movies_df = pd.read_csv(
        "machine-learning/data/movie_data.csv", header=0, sep=';')
    # TODO: release_date should be discretized instead of dropped?

    # remove non numerical columns
    movies_df = movies_df.drop(
        columns=["genres", "original_title", "imdb_id", "overview", "release_date"])

    # TODO: que hacer con registros que tienen algun dato faltante?
    movies_df.dropna(inplace=True)

    # TODO: SACAR ESTO, ES PARA TESTING
    movies_df = movies_df.sample(frac=0.01, random_state=random_state)

    # standarize data
    movies_df = pd.DataFrame(StandardScaler().fit_transform(
        movies_df), columns=movies_df.columns)

    # ------ Analisis univariado de variables ------
    # variables_plot(movies_df)

    # ------- K-Means -------
    k_means = KMeans(K=3, max_iter=10, random_state=random_state)
    print("kmeans clusters: ", k_means.fit(movies_df.values))
