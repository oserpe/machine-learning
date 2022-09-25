from copy import deepcopy
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd


class KNN:
    def __init__(self, X: pd.DataFrame, Y: pd.DataFrame, k_neighbors: int, weighted: bool, classes = [1,2,3,4,5], classes_column_name = "Star Rating", predicted_class_column_name = "Classification"):
        self.X = X
        self.Y = Y
        self.class_column = classes_column_name # Y.columns[0]
        self.k_neighbors = k_neighbors
        if weighted:
            self.get_ranked_classes = self.get_ranked_classes_by_weight
        else:
            self.get_ranked_classes = self.get_ranked_classes_by_appearances

        self.classes = classes
        self.classes_column_name = classes_column_name
        self.predicted_class_column_name = predicted_class_column_name

    def get_euclidean_distance(self, source: pd.DataFrame, dest: pd.DataFrame) -> float:
        return np.linalg.norm(source.values - dest.values, axis=1)

    # Dummy method, only to be used by Metrics class
    def train(self, dataset: pd.DataFrame):
        self.X = dataset.drop(self.class_column, axis=1)
        self.Y = dataset[[self.class_column]]

    def classify(self, sample):
        # add sample to dataset before standardizing
        std_X = pd.concat(
            [self.X, sample], axis=0, sort=False)

        # standarize values that will be used by KNN
        columns = std_X.columns
        std_X[columns] = \
            StandardScaler().fit_transform(
                std_X[columns])

        # Once it is standarized with the train data, remove and retrieve the std sample
        std_sample_df = std_X.tail(1)
        std_X = std_X.drop(
            std_sample_df.index)

        k = self.k_neighbors
        class_found = False
        while not class_found and k < len(std_X):
            distances = []
            for i in range(len(std_X)):
                row = std_X.iloc[[i]]
                distance = self.get_euclidean_distance(row, std_sample_df)

                distances.append((
                    distance,
                    row.index[0]
                ))   # (distance, id)

            k_nearest_neighbors_indexes = list(map(lambda x: x[1], sorted(
                distances, key=lambda distance: distance[0])[:k]))

            classes_by_appearances_sorted = self.get_ranked_classes(
                k_nearest_neighbors_indexes, k, distances)

            # On tie, increase k and try again
            if len(classes_by_appearances_sorted) == 1 \
                    or classes_by_appearances_sorted.iloc[0] != classes_by_appearances_sorted.iloc[1]:
                class_found = True
            else:
                k += 1

        if not class_found:
            return None

        return classes_by_appearances_sorted.iloc[[0]].index[0]

    def get_ranked_classes_by_weight(self, k_nearest_neighbors_indexes, k, distances):
        zero_distance_neighbours = list(map(lambda x: x[1],
                                            list(filter(lambda x: x[0] == 0, distances))))
        if len(zero_distance_neighbours) > 0:
            # If there are neighbours with zero distance, return the class of the most popular between them
            return self.get_ranked_classes_by_appearances(zero_distance_neighbours)

        # add inv_distance column before grouping
        k_nearest_neighbors_inverse_distances = list(map(lambda x: 1/(x[0])**2, sorted(
            distances, key=lambda distance: distance[0])[:k]))

        # keep only the k nearest neighbours from the original dataset, with their class and inv_distance
        k_nearest_neighbors = self.Y.loc[k_nearest_neighbors_indexes]
        k_nearest_neighbors["inv_distance"] = k_nearest_neighbors_inverse_distances

        classes_by_appearances_sorted = k_nearest_neighbors\
            .groupby(self.class_column)["inv_distance"].sum().sort_values(ascending=False)

        return classes_by_appearances_sorted

    def get_ranked_classes_by_appearances(self, k_nearest_neighbors_indexes, k, distances):
        # Frequency of the classes of the k nearest neighbors ordered from highest to lowest without weight
        classes_by_appearances_sorted = self.Y.loc[k_nearest_neighbors_indexes]\
            .groupby(self.class_column).size().sort_values(ascending=False)

        return classes_by_appearances_sorted

    def test(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset_copy = dataset.copy()
        dataset_copy[self.predicted_class_column_name] = dataset.apply(self.classify, axis=1)
        return dataset_copy