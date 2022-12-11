from matplotlib import pyplot as plt


def plot_dataset_hist(data_df):
    data_df.hist(edgecolor='black', linewidth=1,
                 xlabelsize=10, ylabelsize=10, grid=False)

    plt.tight_layout()

    # reduce space between subplots
    plt.subplots_adjust(wspace=0.2, hspace=0.5)

    plt.show()
