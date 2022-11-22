from matplotlib import pyplot as plt
import pandas as pd


def numeric_variables_histogram(data_df):
    data_df.hist(edgecolor='black', linewidth=1,
                 xlabelsize=10, ylabelsize=10, grid=False)
    plt.tight_layout()
    plt.show()


def genre_hbarplot(data_df):
    data_df['genres'].value_counts().plot(kind='barh')
    plt.tight_layout()
    plt.show()


def date_year_hbarplot(data_df):
    # group by datetime column by year
    data_df['year'] = pd.DatetimeIndex(data_df['release_date']).year
    data_df['year'].value_counts().plot(kind='barh')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    movies_df = pd.read_csv(
        "machine-learning/data/movie_data.csv", header=0, sep=';')

    # dataset length
    print("Dataset length: ", len(movies_df))
    # number of features
    print("Number of features: ", len(movies_df.columns))
    # feature types
    print("Feature types: ", movies_df.dtypes)

    unique_df = movies_df.drop_duplicates(subset=["imdb_id"], keep="first")

    # number of duplicated rows
    print("Number of duplicated rows: ", len(
        movies_df) - len(unique_df))

    unique_df = unique_df.drop(
        columns=["original_title", "imdb_id", "overview"])

    # convert release_date to datetime
    unique_df['release_date'] = pd.to_datetime(
        unique_df['release_date'], format='%Y-%m-%d')

    numeric_variables_histogram(unique_df)
    genre_hbarplot(unique_df)
    # date_year_hbarplot(unique_df)
