import pandas as pd
import matplotlib.pyplot as plt


def variables_plot(data_df):
    data_df.hist(edgecolor='black', linewidth=1.0,
                    xlabelsize=10, ylabelsize=10, grid=False)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    random_state = 1
    # TODO: que hacer con registros que tienen algun dato faltante?

    movies_df = pd.read_csv("machine-learning/data/movie_data.csv", header=0, sep=';')
    # TODO: release_date should be discretized instead of dropped?

    # remove non numerical columns
    movies_df = movies_df.drop(columns=["genres", "original_title", "imdb_id", "overview", "release_date"])


    # ------ Analisis univariado de variables ------
    variables_plot(movies_df)
