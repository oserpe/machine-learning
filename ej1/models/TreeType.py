from enum import Enum
from .DecisionTree import DecisionTree
from .RandomForest import RandomForest


class TreeType(Enum):
    DECISION_TREE = 0
    RANDOM_FOREST = 1

    def get_tree(self):
        if self == TreeType.DECISION_TREE:
            return DecisionTree(max_node_count=1000, tree_type=self)
        elif self == TreeType.RANDOM_FOREST:
            return RandomForest(n_estimators=3, tree_type=self)
