import pandas as pd
from matplotlib import pyplot as plt
from ..plots.dataset import plot_dataset_hist

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def apply_pca_to_data(data, n_components=2):
    data = data.drop("diagnosis", axis=1)
    # standardize data
    scaled_data = StandardScaler().fit_transform(data)

    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(scaled_data)

    print("Explained variance ratio: ", pca.explained_variance_ratio_)
    print("Explained variance: ", pca.explained_variance_)

    return pca, pca_data

def pca_loadings_hbar_plot(loadings, features, component_label):
    component = pd.DataFrame(loadings, columns=[component_label])
    component.index = features
    # plot sorted histogram by value of n-component
    component.sort_values(by=component_label, inplace=True)
    # decrease space between bars
    component.plot(kind="bar", figsize=(20, 5), width=0.5)
    plt.tight_layout()
    plt.yticks(fontsize=8.25)
    plt.xticks(fontsize=8.25)
    plt.show()

def pca_plot(pca, pca_data, data_df):
    pca_data_df = pd.DataFrame(data=pca_data, columns=["PC1", "PC2"])

    # merge data_df with pca_data_df
    pca_data_df = pd.concat([pca_data_df, data_df[["diagnosis"]]], axis=1)

    # plot all data with scatter where each point is a class depending on its color (M = 1, B = 0)
    # M is malignant, B is benign, so M red, B blue
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel(f'PC1 {pca.explained_variance_ratio_[0]*100:.2f}%', fontsize=13)
    ax.set_ylabel(f'PC2 {pca.explained_variance_ratio_[1]*100:.2f}%', fontsize=13)

    targets = [1, 0]
    colors = ["r", "b"]
    for target, color in zip(targets, colors):
        indices_to_keep = pca_data_df["diagnosis"] == target
        ax.scatter(pca_data_df.loc[indices_to_keep, "PC1"],
                   pca_data_df.loc[indices_to_keep, "PC2"],
                   c=color,
                   s=50,
                   alpha=0.35)
    ax.legend(["M", "B"])
    plt.show()

    data_df.drop("diagnosis", inplace=True, axis=1)

    pca_loadings_hbar_plot(pca.components_[0], data_df.columns, "First component loadings")
    pca_loadings_hbar_plot(pca.components_[1], data_df.columns, "Second component loadings")


def dataset_analysis(data_df):
    # dataset length
    print("Dataset length: ", len(data_df))
    # number of features
    print("Number of features: ", len(data_df.columns))
    # feature types
    print("Feature types: ", data_df.dtypes)

    unique_df = data_df.drop_duplicates(subset=["id"], keep="first")

    # number of duplicated rows
    print("Number of duplicated rows: ", len(
        data_df) - len(unique_df))

    unique_withoutna_df = unique_df.dropna()
    # number of rows with at least one null value
    print("Number of rows with at least one null value: ", len(
        unique_df) - len(unique_withoutna_df))

    print("Final dataset length: ", len(unique_withoutna_df))

    # ------ HISTOGRAM ------

    # drop non-interesting columns
    data_df.drop("id", inplace=True, axis=1)

    # modify non-numerical columns (M = 1, B = 0)
    data_df["diagnosis"] = data_df["diagnosis"].map({"M": 1, "B": 0})

    # separate df into two df's, first with 15 columns, second with 16 columns
    data_df_15 = data_df.iloc[:, 0:15]
    data_df_16 = data_df.iloc[:, 15:31]

    plot_dataset_hist(data_df_15)
    plot_dataset_hist(data_df_16)

    pca, pca_data = apply_pca_to_data(data_df)
    pca_plot(pca, pca_data, data_df)