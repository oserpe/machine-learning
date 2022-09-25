from copy import deepcopy
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

class KNN:
    def __init__(self, train_dataset: pd.DataFrame, class_column: str, start_k: int):
        self.train_dataset = train_dataset
        self.class_column = class_column
        self.start_k = start_k

    def get_euclidean_distance(self, source: pd.DataFrame, dest: pd.DataFrame) -> float:
        return np.linalg.norm(source.values - dest.values, axis=1)

    def test(self, sample, weighted = False):
        std_train_dataset_without_class = self.train_dataset.drop([self.class_column], axis=1)

        # add sample to dataset before standarizing
        std_train_dataset_without_class = pd.concat([std_train_dataset_without_class, sample], axis = 0, sort = False)

        # standarize values that will be used by KNN
        numerical_value_columns = [col for col in self.train_dataset.columns if col != self.class_column]
        std_train_dataset_without_class[numerical_value_columns] = \
                    StandardScaler().fit_transform(std_train_dataset_without_class[numerical_value_columns])

        # Once it is standarized with the train data, get the std sample
        std_sample_df = std_train_dataset_without_class.tail(1)
        std_train_dataset_without_class = std_train_dataset_without_class.drop(std_sample_df.index)

        k = self.start_k
        class_found = False
        while k < len(std_train_dataset_without_class) and not class_found:
            distances = []
            for i in range(len(std_train_dataset_without_class)):
                row = std_train_dataset_without_class.iloc[[i]]
                distance = self.get_euclidean_distance(row, std_sample_df)

                distances.append((
                    distance,
                    row.index[0]
                    ))   # (distance, id)

            k_nearest_neighbors_indexes = list(map(lambda x: x[1], sorted(
                distances, key=lambda distance: distance[0])[:k]))

            if weighted:
                # add inv_distance column before grouping
                k_nearest_neighbors_inverse_distances = list(map(lambda x: 1/(x[0])**2, sorted(
                    distances, key=lambda distance: distance[0])[:k]))
                k_nearest_neighbors = self.train_dataset.loc[k_nearest_neighbors_indexes]
                k_nearest_neighbors["inv_distance"] = k_nearest_neighbors_inverse_distances

                classes_by_appearances_sorted = k_nearest_neighbors\
                    .groupby(self.class_column)["inv_distance"].sum().sort_values(ascending=False)

            else:
                # Frequency of the classes of the k nearest neighbors ordered from highest to lowest without weight
                classes_by_appearances_sorted = self.train_dataset.loc[k_nearest_neighbors_indexes]\
                    .groupby(self.class_column).size().sort_values(ascending=False)

            # On tie, increase k and try again
            if len(classes_by_appearances_sorted) == 1 \
                or classes_by_appearances_sorted.iloc[0] != classes_by_appearances_sorted.iloc[1]:
                class_found = True
            else:
                k += 1


        return classes_by_appearances_sorted.iloc[[0]].index[0]
