import math

from .DecisionTree import DecisionTree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class RandomForest:
    def __init__(self, n_estimators=10, samples_per_bag_frac=1, max_node_count=math.inf, tree_type = None):
        self.n_estimators = n_estimators
        self.samples_per_bag_frac = samples_per_bag_frac
        self.max_node_count = max_node_count

        self.trees = [DecisionTree(max_node_count=self.max_node_count) for _ in range(self.n_estimators)]

        self.tree_type = tree_type



    def bagging(self, dataset: pd.DataFrame):
        return dataset.sample(frac=self.samples_per_bag_frac, replace=True)

    def train(self, dataset: pd.DataFrame, class_column: str):
        for i in range(self.n_estimators):
            bag = self.bagging(dataset)
            print("####################")
            print(f"Training tree {i+1} of {self.n_estimators}")

            tree = self.trees[i]
            tree.train(bag, class_column)

            print(f"Tree {i+1} of {self.n_estimators} trained")
            print("####################")
            self.trees.append(tree)

    def classify(self, sample: pd.DataFrame):
        votes = [tree.classify(sample) for tree in self.trees]
        return max(set(votes), key=votes.count)

    def test(self, dataset: pd.DataFrame, prediction_column: str) -> pd.DataFrame:
        print("####################")
        print("Classifying with random forest...")

        dataset[prediction_column] = dataset.apply(self.classify, axis=1)

        print("Classification finished...")
        print("####################")

        return dataset

    def test_every_tree(self, dataset: pd.DataFrame, prediction_column: str) -> list[pd.DataFrame]:
        results = []
        for tree in self.trees:
            results.append(tree.test(dataset, prediction_column))
        return results

    def s_precision_per_node_count(self, train_dataset: pd.DataFrame, test_dataset: pd.DataFrame, initial_node_count: int = 10, max_node_count = 100) -> dict:
        # for every estimator get the associated precision
        results = {}
        for i in range(self.n_estimators):
            tree = self.trees[i]
            results[i] = tree.s_precision_per_node_count(
                train_dataset, test_dataset, initial_node_count, max_node_count)

        # get the average precision for every node count with their respective standard deviation
        average_results = {}
        for node_count in results[0].keys():
            train_precisions = []
            test_precisions = []
            depths = []

            for i in range(self.n_estimators):
                train_precisions.append(
                    results[i][node_count]["train_s_precision"])
                test_precisions.append(
                    results[i][node_count]["test_s_precision"])
                depths.append(results[i][node_count]["depth"])

            average_results[node_count] = {
                "train_s_precision_avg": np.mean(train_precisions),
                "test_s_precision_avg": np.mean(test_precisions),
                "depth_avg": np.mean(depths),
                "train_s_precision_std": np.std(train_precisions),
                "test_s_precision_std": np.std(test_precisions),
                "depth_std": np.std(depths)
            }

        return average_results

    def plot_precision_per_node_count(self, results):
        # get the x axis
        x = list(results.keys())

        # get the y axis
        train_y = [results[node_count]["train_s_precision_avg"]
                   for node_count in x]
        test_y = [results[node_count]["test_s_precision_avg"]
                  for node_count in x]

        # get the standard deviation
        train_std = [results[node_count]["train_s_precision_std"]
                     for node_count in x]
        test_std = [results[node_count]["test_s_precision_std"]
                    for node_count in x]

        # plot the results
        plt.errorbar(x, train_y, label="Train",
                  yerr=train_std, ecolor='blue', marker='o', color="red", elinewidth=0.5, capsize=5, linestyle='--')
        plt.errorbar(x, test_y, label="Test",
                  yerr=test_std, ecolor='red', marker='o', color="blue", elinewidth=0.5, capsize=5, linestyle='--')
        plt.legend()
        plt.xlabel("Node count")
        plt.ylabel("Precision")
        plt.title("Precision per node count")
        plt.ylim(top=1.1)
        plt.show()
