import math
import pandas as pd
from .Node import Node, NodeType
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
from pyvis.network import Network
import copy

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

    def create_and_set_node(self, node_type: NodeType, value=None, depth=0):
        node = Node(node_type, value, depth)
        self.tree.add_node(node.id, label=str(node), level=depth)
        properties = {
            "value": str(value),
            "id": node.id,
            "type": str(node_type),
        }
        nx.set_node_attributes(self.tree, {node.id: properties})
        return node

    def build_tree_recursive(self, dataset: pd.DataFrame, attributes: list[str], depth = 0):
        # check if class column value is unique
        class_values = dataset[self.class_column].unique()
        if len(class_values) == 1:
            # return leaf node with class value
            leaf_node = self.create_and_set_node(NodeType.LEAF, value=class_values[0], depth=depth)
            return leaf_node

        elif len(attributes) == 0:
            print("No remaining attributes...")
            print("Choosing most common class value...")
            
            class_mode = dataset[self.class_column].mode()[0]

            print(f"Most common class value: {class_mode}")
            
            leaf_node = self.create_and_set_node(NodeType.LEAF, value=class_mode, depth=depth)
            
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
        max_gain_attribute_node = self.create_and_set_node(NodeType.ATTRIBUTE, value=max_gain_attribute, depth=depth)

        # update attributes list
        attributes.remove(max_gain_attribute)

        # sus hijos se llaman como sus valores
        for attribute_value in self.values_by_attribute[max_gain_attribute]:
            dataset_by_attribute_value = dataset[(dataset[max_gain_attribute] == attribute_value)]
            
            if len(dataset_by_attribute_value) == 0:

                print("No remaining samples...\nChoosing most common class value...")
                class_mode = dataset[self.class_column].mode()[0]

                print(f"Most common class value: {class_mode}")

                attribute_child_node = self.create_and_set_node(NodeType.ATTRIBUTE_VALUE, value=attribute_value, depth=depth+1)
                self.tree.add_edge(max_gain_attribute_node.id, attribute_child_node.id)

                leaf_node = self.create_and_set_node(NodeType.LEAF, value=class_mode, depth=depth+2)
                self.tree.add_edge(attribute_child_node.id, leaf_node.id)
                continue
            
            attribute_value_node = self.create_and_set_node(NodeType.ATTRIBUTE_VALUE, value=attribute_value, depth=depth+1)

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

        self.tree = nx.DiGraph()

        self.build_tree_recursive(dataset, attributes)

    def get_next_node(self, node, attribute_value, tree: nx.DiGraph):

        successors = tree.successors(node["id"])

        for successor_id in successors:
            successor_node = tree.nodes[successor_id]
            if successor_node["value"] == str(attribute_value):
                # get the successor of the successor as an attribute value node its 
                # connected to another attribute node or a leaf node
                successor_of_successor_id = list(tree.successors(successor_id))[0]
                return tree.nodes[successor_of_successor_id]
        
        # if no successor is found, return the leaf node associated to the node
        # the value wasn't even present in the training set
        

    def draw(self):

        g = Network(height='100%', width='100%',
                    notebook=True, layout='hierarchical')

        g.from_nx(self.tree)

        g.show('tree3.html')


    def get_root_node(self):
        return self.get_root_node_from_tree(self.tree)

    def get_root_node_from_tree(self, tree: nx.DiGraph):
        return tree.nodes[0]

    def classify(self, sample: pd.DataFrame):
        return self.classify_from_tree(sample, self.tree)

    def classify_from_tree(self, sample: pd.DataFrame, tree: nx.DiGraph):
        # get root node and its attribute
        current_node = self.get_root_node_from_tree(tree)
        current_attribute = current_node["value"]

        # get attribute value from sample
        attribute_value = sample[current_attribute]

        # for every other attribute, get the next node until we reach a leaf node
        while current_node["type"] != str(NodeType.LEAF):
            current_node = self.get_next_node(current_node, attribute_value, tree)

            if current_node["type"] == str(NodeType.LEAF):
                return current_node["value"]

            current_attribute = current_node["value"]
            attribute_value = sample[current_attribute]

        return current_node["value"]


    def prune(self, dataset: pd.DataFrame):
        root_node = self.get_root_node()
        self.prune_recursive(dataset, root_node["id"])

    def prune_recursive(self, dataset: pd.DataFrame, node_id: int):
        # BOTTOM-UP PRUNING

        current_node = self.tree.nodes[node_id]

        if current_node["type"] == str(NodeType.LEAF):
            return

        current_attribute = current_node["value"]

        attribute_value_nodes = self.tree.successors()

        # iterate over attribute value nodes 
        for value_node_id in attribute_value_nodes:
            value_node = self.tree.nodes[value_node_id]
            
            next_node_id = list(self.tree.successors(value_node_id))[0]
            # next node is either a leaf node or an attribute node (only one)

            dataset_given_attribute_value = dataset[(dataset[current_attribute] == value_node["value"])]
            self.prune_recursive(dataset_given_attribute_value, next_node_id)
            
            current_error = self.calculate_error(dataset_given_attribute_value, self.tree)
            
            pruned_tree = copy.deepcopy(self.tree)
            pruned_tree.remove_node(next_node_id)

            class_mode = dataset_given_attribute_value[self.class_column].mode()[0]

            pruned_tree.add_node(next_node_id, type=str(NodeType.LEAF), value=class_mode, level=value_node["depth"]+1)
            pruned_tree.add_edge(value_node_id, next_node_id)

            new_error = self.calculate_error(dataset_given_attribute_value, pruned_tree)
            if new_error < current_error:
                self.tree = pruned_tree

    def calculate_error(self, dataset: pd.DataFrame, tree: nx.DiGraph):
        incorrect_predictions = 0
        for index, sample in dataset.iterrows():
            if self.classify_from_tree(sample, tree) != sample[self.class_column]:
                incorrect_predictions += 1
        
        return incorrect_predictions/len(dataset)
        
        