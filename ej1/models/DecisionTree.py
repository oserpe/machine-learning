import math
import pandas as pd
from .Node import Node, NodeType
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
from pyvis.network import Network

def relative_frequency(count, total, classes_count, laplace_correction=False):
    """Calculates the relative frequency of a given count."""
    if laplace_correction:
        # apply laplace correction
        return (count + 1) / (total + classes_count)
    else:
        return count / total


def calculate_probabilities_by_class(dataset, class_column, class_values):
    """Calculates the probabilities of a given attribute value for a given class."""
    probabilities = {}

    # calculate relative frequencies for n_class_value/N
    for class_value in class_values:
        probabilities[class_value] = relative_frequency(len(dataset[(
            dataset[class_column] == class_value)]), len(dataset), len(class_values))   # apply laplace correction

    return probabilities


def shannon_entropy(probabilities):
    """Calculates the Shannon entropy of a list of probabilities."""
    return sum(-p * math.log2(p) for p in probabilities if p != 0)    


def gini_index(probabilities):
    """Calculates the Gini index of a list of probabilities."""
    return 1 - sum(p ** 2 for p in probabilities)


def information_gain(dataset, parent_entropy, attribute_column, attribute_values, class_column, class_values):
    """Calculates the information gain of a dataset for a given attribute."""
    children_entropy = 0

    for attribute_value in attribute_values:
        # calculate relative frequencies for n_attribute_value/N
        dataset_given_attribute_value = dataset[(
            dataset[attribute_column] == attribute_value)]

        # TODO: maybe apply laplace correction
        if len(dataset_given_attribute_value) == 0:
            continue

        attribute_probabilities = calculate_probabilities_by_class(
            dataset_given_attribute_value, class_column, class_values)

        children_entropy += shannon_entropy(
            attribute_probabilities.values()) * (len(dataset_given_attribute_value) / len(dataset))
        # DUDA: hace falta aplicar laplace aca? Con qué denominador?

    return parent_entropy - children_entropy

class DecisionTree:
    def __init__(self, max_depth=math.inf, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        # TODO: define constraints

    def get_attribute_values(self, dataset: pd.DataFrame, attribute_name):
        return dataset[attribute_name].unique()

    def get_all_attribute_values(self, dataset: pd.DataFrame):
        return {attribute: self.get_attribute_values(dataset, attribute) for attribute in dataset.columns}

    def build_tree_recursive(self, dataset: pd.DataFrame, attributes: list[str], depth = 0):
        # check if class column value is unique
        class_values = dataset[self.class_column].unique()
        if len(class_values) == 1:
            # return leaf node with class value
            leaf_node = Node(NodeType.LEAF, value=class_values[0], depth=depth)
            self.tree.add_node(leaf_node.id, label=str(leaf_node), level=depth)
            
            return leaf_node

        elif len(attributes) == 0:
            print("No remaining attributes...")
            print("Choosing most common class value...")
            
            class_mode = dataset[self.class_column].mode()[0]

            print(f"Most common class value: {class_mode}")
            
            leaf_node = Node(NodeType.LEAF, value=class_mode, depth=depth)
            self.tree.add_node(leaf_node.id, label=str(leaf_node), level=depth)
            
            return leaf_node

        parent_probabilities = calculate_probabilities_by_class(
            dataset, self.class_column, class_values)

        parent_entropy = shannon_entropy(parent_probabilities.values())

        # compute information gain for each attribute
        # keep the maximum information gain and the attribute
        max_gain = 0
        max_gain_attribute = None

        for attribute in attributes:
            attribute_values = self.values_by_attribute[attribute]
            attribute_gain = information_gain(dataset, parent_entropy, attribute, attribute_values, self.class_column,
                                              class_values)
                
            if attribute_gain > max_gain:
                max_gain = attribute_gain
                max_gain_attribute = attribute

        # creamos el nodo "atributo"
        max_gain_attribute_node = Node(NodeType.ATTRIBUTE, value=max_gain_attribute, depth=depth)
        
        self.tree.add_node(max_gain_attribute_node.id, label=str(max_gain_attribute_node), level=depth)

        # update attributes list
        attributes.remove(max_gain_attribute)
        
        # sus hijos se llaman como sus valores
        for attribute_value in self.values_by_attribute[max_gain_attribute]:
            dataset_by_attribute_value = dataset[(dataset[max_gain_attribute] == attribute_value)]
            
            if len(dataset_by_attribute_value) == 0:
                continue
            
            attribute_value_node = Node(NodeType.ATTRIBUTE_VALUE, value=attribute_value, depth=depth+1)
            self.tree.add_node(attribute_value_node.id, label=str(attribute_value_node), level=depth+1)

            self.tree.add_edge(max_gain_attribute_node.id, attribute_value_node.id)

            # creamos el subarbol
            attribute_child_node = self.build_tree_recursive(
                dataset_by_attribute_value, attributes.copy(), depth=depth+2)
                
            self.tree.add_edge(attribute_value_node.id, attribute_child_node.id)

        return max_gain_attribute_node

    def train(self, dataset: pd.DataFrame, class_column: str):
        self.class_column = class_column

        self.values_by_attribute = self.get_all_attribute_values(
            dataset[dataset.columns.drop(class_column)])

        attributes = dataset.columns.drop(class_column).tolist()

        self.tree = nx.Graph()

        self.root_node = self.build_tree_recursive(dataset, attributes)

        print(self.root_node)

        self.tree.add_node(self.root_node.id, label=str(self.root_node), level=0)

    # def get_next_node(self, node: nx.Node, attribute_value):
    #     child_node_value = self.tree.neighbors(node)[attribute_value]

    #     if child_node_value["type"] == "leaf":
    #         return child_node_value

    #     next_node_successor = self.tree.nodes[child_node_value]

    #     return next_node_successor

    def draw(self):

        g = Network(height='100%', width='100%',
                    notebook=True, layout='hierarchical')

        g.from_nx(self.tree)

        g.show('tree.html')


    # def classify(self, sample: pd.DataFrame):
    #     # get root node and its attribute
    #     current_node = self.root_node
    #     current_attribute = current_node["attribute"]

    #     # get attribute value from sample
    #     attribute_value = sample[current_attribute]

    #     # for every other attribute, get the next node until we reach a leaf node
    #     while current_node["type"] != "leaf":
    #         current_node = self.get_next_node(current_node, attribute_value)
    #         current_attribute = current_node["attribute"]
    #         attribute_value = sample[current_attribute]

    #     return current_node["value"]