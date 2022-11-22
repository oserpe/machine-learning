import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from ..data.generate_dataset import generate_dataset

def categorize_data_with_equal_width(data_df: pd.DataFrame, columns: dict[str, int]) -> pd.DataFrame:
    for column, bins in columns.items():
        data_df[column] = pd.cut(
            data_df[column], bins, labels=False)

    return data_df

def plot_discretized_attribute_histogram(dataset: pd.DataFrame, attribute: str, bin_count):

    display_data = categorize_data_with_equal_width(
        dataset, {attribute: bin_count})

    plt.hist(display_data[attribute], color='lightblue', edgecolor='black')
    # set title and labels
    plt.title(f"Discretización de {attribute}")
    plt.xlabel(attribute)
    plt.ylabel("Frecuencia")

if __name__ == "__main__":    
    movies_df, only_genres_df = generate_dataset(standardize=False)

    plot_discretized_attribute_histogram(movies_df, "budget", 25000)
    plt.show()
    plot_discretized_attribute_histogram(movies_df, "popularity", 25000)
    plt.show()
    plot_discretized_attribute_histogram(movies_df, "revenue", 25000)
    plt.show()

