from ..plots.dataset import plot_dataset_hist


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
