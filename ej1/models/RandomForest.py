from .DecisionTree import DecisionTree
import pandas as pd

class RandomForest:
    def __init__(self, n_estimators=10, samples_per_bag=0.8):
        self.n_estimators = n_estimators
        self.samples_per_bag = samples_per_bag
        self.trees = []

    def bagging(self, dataset: pd.DataFrame):
        return dataset.sample(frac=self.samples_per_bag, replace=True)

    def train(self, dataset: pd.DataFrame, class_column: str):
        for i in range(self.n_estimators):
            bag = self.bagging(dataset)
            print("####################")
            print(f"Training tree {i+1} of {self.n_estimators}")
            tree = DecisionTree()
            tree.train(bag, class_column)

            print(f"Tree {i+1} of {self.n_estimators} trained")
            self.trees.append(tree)

    def classify(self, sample: pd.DataFrame):
        votes = [tree.classify(sample) for tree in self.trees]
        return max(set(votes), key=votes.count)

    def test(self, dataset: pd.DataFrame, prediction_column: str) -> list[pd.DataFrame]:
        results = []
        for tree in self.trees:
            results.append(tree.test(dataset, prediction_column))
            
        return results