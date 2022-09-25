import pandas as pd
import matplotlib.pyplot as plt

# plot histogram of creditability, using a red bin for 0 and blue bin for 1


def plot_creditability_histogram(dataset: pd.DataFrame):
    # change 0 to "No" and 1 to "Yes"
    display_data = dataset[["Creditability"]].replace(
        {0: "No", 1: "Yes"})

    # plot histogram
    display_data.apply(pd.value_counts).T.plot(kind='bar', rot=0)
    plt.show(block=True)


if __name__ == "__main__":
    data_df = pd.read_csv(
        "./machine-learning/ej1/dataset/german_credit.csv", header=0, sep=',')

    plot_creditability_histogram(data_df)
    # dataset[["Creditability"]].plot(
    #     kind='hist', bins=2, xticks=[0, 1])
    # plt.show(block=True)
