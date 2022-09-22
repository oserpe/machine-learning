from .DecisionTree import DecisionTree
import pandas as pd

class RandomForest:
    def __init__(self, n_estimators=10, samples_per_bag_frac=1):
        self.n_estimators = n_estimators
        self.samples_per_bag_frac = samples_per_bag_frac
        self.trees = []

    def bagging(self, dataset: pd.DataFrame):
        return dataset.sample(frac=self.samples_per_bag_frac, replace=True)

    def train(self, dataset: pd.DataFrame, class_column: str):
        for i in range(self.n_estimators):
            bag = self.bagging(dataset)
            print("####################")
            print(f"Training tree {i+1} of {self.n_estimators}")

            tree = DecisionTree()
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
        
