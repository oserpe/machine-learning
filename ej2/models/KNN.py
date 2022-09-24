from copy import deepcopy
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

class KNN:
    def __init__(self, train_dataset: pd.DataFrame, class_column: str, k: int):
        self.train_dataset = train_dataset
        self.class_column = class_column
        self.k = k

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
        std_train_dataset_without_class.drop(std_train_dataset_without_class.tail(1).index)

        distances = []

        for i in range(len(std_train_dataset_without_class)):
            distances.append((self.get_euclidean_distance(std_train_dataset_without_class.iloc[[i]], std_sample_df), i))

        # Calculamos los k vecinos más cercanos
        k_nearest_neighbors_indexes = map(lambda x: x[1], sorted(
            distances, key=lambda distance: distance[0])[:self.k])

        print(list(k_nearest_neighbors_indexes))
        print(self.train_dataset.loc[[197]])
        print(self.train_dataset.loc[[230]])

        print(self.train_dataset.loc[k_nearest_neighbors_indexes][self.class_column])
        
        return self.train_dataset.loc[k_nearest_neighbors_indexes][self.class_column].value_counts()
