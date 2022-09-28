import pandas as pd


def categorize_data_with_equal_frequency(data_df: pd.DataFrame, columns: dict[str, int]) -> pd.DataFrame:
    categorized_data = data_df.copy()

    for column, q in columns.items():
        categorized_data[column], range = pd.qcut(
            data_df[column], q=q, labels=False, retbins=True)
        # print(range) # print ranges of current discretization, starting from minimum value until the last range that ends on the maximum value
    return categorized_data, range


def categorize_data_with_equal_width(data_df: pd.DataFrame, columns: dict[str, int]) -> pd.DataFrame:
    for column, bins in columns.items():
        data_df[column] = pd.cut(
            data_df[column], bins, labels=False)

    return data_df
