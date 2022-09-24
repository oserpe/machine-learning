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

    def test(self, sample):
        std_train_dataset_without_class = self.train_dataset.drop([self.class_column], axis=1)

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
                distances.append((
                    self.get_euclidean_distance(std_train_dataset_without_class.iloc[[i]], std_sample_df),
                    std_train_dataset_without_class.iloc[[i]].index[0])
                )   # (distance, id)

            k_nearest_neighbors_indexes = list(map(lambda x: x[1], sorted(
                distances, key=lambda distance: distance[0])[:k]))

            # Frequency of the classes of the k nearest neighbors ordered from highest to lowest
            classes_by_appearances_sorted = self.train_dataset.loc[k_nearest_neighbors_indexes][self.class_column].value_counts()

            # On tie, increase k and try again
            if len(classes_by_appearances_sorted) == 1 \
                or classes_by_appearances_sorted.iloc[0] != classes_by_appearances_sorted.iloc[1]:
                class_found = True
            else:
                k += 1


        return classes_by_appearances_sorted.iloc[[0]].index[0]
