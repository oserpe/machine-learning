import pandas as pd
import matplotlib.pyplot as plt
from ..main import categorize_data_with_equal_frequency, categorize_data_with_equal_width
import numpy as np
# plot histogram of creditability, using a red bin for 0 and blue bin for 1


def plot_creditability_histogram(dataset: pd.DataFrame):
    # change 0 to "No" and 1 to "Yes"
    display_data = dataset[["Creditability"]].replace(
        {0: "No", 1: "Yes"})

    # plot histogram
    display_data.apply(pd.value_counts).T.plot(kind='bar', rot=0)


def plot_discretized_attribute_histogram(dataset: pd.DataFrame, attribute: str, bin_count):

    display_data, bins = categorize_data_with_equal_frequency(
        dataset, {attribute: bin_count})

    # add small fraction to each bin for correct visualization
    bins = np.add(bins, 0.000001)

    plt.hist(dataset[attribute],
             bins=bins, color='lightblue', edgecolor='black')
    # set title and labels
    plt.title(f"Discretización de {attribute}")
    plt.xlabel(attribute)
    plt.ylabel("Frecuencia")
    # set xticks
    plt.xticks(bins, rotation=90)


def plot_account_balance_histogram(dataset: pd.DataFrame):
    # change 0 to "No" and 1 to "Yes"
    display_data = dataset[["Account Balance"]].replace(
        {1: "Sin cuenta", 2: "Poco balanceada", 3: "Balanceada", 4: "Bien balanceada"})

    display_data["Creditability"] = dataset[["Creditability"]].replace(
        {0: "No", 1: "Yes"})

    display_data = display_data.groupby(
        ["Account Balance", "Creditability"])["Account Balance"].count().unstack("Creditability")

    display_data.plot(kind='bar', rot=0, stacked=True,
                      title="Creditability dado Account Balance", ylabel="Cantidad de ejemplares")


def plot_general_histogram(dataset: pd.DataFrame):
    dataset.drop(["Creditability"], axis=1).hist(edgecolor='black', linewidth=1.0,
                                                 xlabelsize=10, ylabelsize=10, grid=False)
    plt.tight_layout()


if __name__ == "__main__":
    data_df = pd.read_csv(
        "./machine-learning/ej1/dataset/german_credit.csv", header=0, sep=',')

    # plot_general_histogram(data_df)
    # plot_discretized_attribute_histogram(
    #     data_df, "Duration of Credit (month)", 6)
    plot_discretized_attribute_histogram(data_df, "Credit Amount", 10)
    # plot_discretized_attribute_histogram(data_df, "Age (years)", 7)
    plt.show(block=True)

    # plot_account_balance_histogram(data_df)
    # plot_creditability_histogram(data_df)
    # dataset[["Creditability"]].plot(
    #     kind='hist', bins=2, xticks=[0, 1])
    # plt.show(block=True)
