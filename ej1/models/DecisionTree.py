import math
import pandas as pd

from .Node import Node, NodeType
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import copy


def relative_frequency(count, total, classes_count, laplace_correction=False):
    """Calculates the relative frequency of a given count."""
    if laplace_correction:
        return (count + 1) / (total + classes_count)
    else:
        return count / total


def calculate_probabilities_by_class(dataset, class_column, class_values):
    """Calculates the probabilities of a given attribute value for a given class."""
    probabilities = {}

    # calculate relative frequencies for n_class_value/N
    for class_value in class_values:
        probabilities[class_value] = relative_frequency(len(dataset[(
            dataset[class_column] == class_value)]), len(dataset), len(class_values))

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

    return parent_entropy - children_entropy


class DecisionTree:
    def __init__(self, max_depth=math.inf, min_samples_split=-1, min_samples_leaf=-1, classes=[0, 1],
                 classes_column_name="Creditability", predicted_class_column_name="Classification",
                 max_node_count=math.inf, tree_type=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.classes = classes
        self.classes_column_name = classes_column_name
        self.predicted_class_column_name = predicted_class_column_name
        self.max_node_count = max_node_count
        self.tree_type = tree_type

        self.i = 0

    def get_attribute_values(self, dataset: pd.DataFrame, attribute_name):
        return dataset[attribute_name].unique()

    def get_all_attribute_values(self, dataset: pd.DataFrame):
        return {attribute: self.get_attribute_values(dataset, attribute) for attribute in dataset.columns}

    def create_and_set_node(self, node_type: NodeType, value=None, depth=0, id=None, tree=None, most_common_attribute_value=None):
        node = Node(node_type=node_type, value=value, depth=depth, id=id)
        target_tree = tree if tree else self.tree
        target_tree.add_node(node.id, label=str(node), level=depth)
        properties = {
            "node_value": str(value),
            "node_id": node.id,
            "node_type": str(node_type),
            "node_depth": depth,
            "node_most_common_attribute_value": str(most_common_attribute_value)
        }
        nx.set_node_attributes(target_tree, {node.id: properties})
        return node

    def count_nodes(self):
        return len(self.tree.nodes)

    def build_tree_recursive(self, dataset: pd.DataFrame, attributes: list[str], depth=0):
        if depth > self.max_depth_by_attribute:
            # print(f'Updating depth from {self.max_depth_by_attribute} to {depth}')
            self.max_depth_by_attribute = depth
        # print(f"Building tree at depth {self.max_depth}")
        # check if class column value is unique
        class_values = dataset[self.classes_column_name].unique()
        if len(class_values) == 1:
            # return leaf node with class value
            leaf_node = self.create_and_set_node(
                NodeType.LEAF, value=class_values[0], depth=depth)
            return leaf_node

        elif len(attributes) == 0 or depth == self.max_depth \
                or len(dataset) < self.min_samples_split or self.node_count == self.max_node_count:
            # print("No remaining attributes...")
            # print("Choosing most common class value...")

            class_mode = dataset[self.classes_column_name].mode()[0]

            # print(f"Most common class value: {class_mode}")

            leaf_node = self.create_and_set_node(
                NodeType.LEAF, value=class_mode, depth=depth)

            return leaf_node

        parent_probabilities = calculate_probabilities_by_class(
            dataset, self.classes_column_name, class_values)

        parent_entropy = shannon_entropy(parent_probabilities.values())

        # compute information gain for each attribute
        # keep the maximum information gain and the attribute
        max_gain = 0
        max_gain_attribute = None

        for attribute in attributes:
            attribute_values = self.values_by_attribute[attribute]
            attribute_gain = information_gain(dataset, parent_entropy, attribute, attribute_values,
                                              self.classes_column_name, class_values)

            if attribute_gain > max_gain:
                max_gain = attribute_gain
                max_gain_attribute = attribute

        # get most frequent attribute value
        attribute_value_mode = dataset[max_gain_attribute].mode()[0]

        # creamos el nodo "atributo"
        max_gain_attribute_node = self.create_and_set_node(
            NodeType.ATTRIBUTE, value=max_gain_attribute, depth=depth, most_common_attribute_value=attribute_value_mode)
        # update attributes list
        attributes.remove(max_gain_attribute)

        self.node_count += 1

        # sus hijos se llaman como sus valores
        for attribute_value in self.values_by_attribute[max_gain_attribute]:
            dataset_by_attribute_value = dataset[(
                dataset[max_gain_attribute] == attribute_value)]
            if len(dataset_by_attribute_value) == 0:
                continue

            # Create subtree
            attribute_child_node = self.build_tree_recursive(
                dataset_by_attribute_value, attributes.copy(), depth=depth+1)

            self.tree.add_edge(max_gain_attribute_node.id,
                               attribute_child_node.id, label=str(attribute_value))

        return max_gain_attribute_node

    def train(self, dataset: pd.DataFrame):
        Node.id = -1

        self.values_by_attribute = self.get_all_attribute_values(
            dataset[dataset.columns.drop(self.classes_column_name)])

        attributes = dataset.columns.drop(self.classes_column_name).tolist()

        self.tree = nx.DiGraph()

        self.max_depth_by_attribute = 0
        self.node_count = 0

        self.build_tree_recursive(dataset, attributes, depth=0)

    def get_next_node(self, node, attribute_value, tree: nx.DiGraph):
        out_edges = tree.out_edges(node["node_id"])
        most_common_attribute_value = node["node_most_common_attribute_value"]
        for edge in out_edges:
            edge_value = tree.get_edge_data(*edge)["label"]

            if edge_value == str(attribute_value):
                successor_node_id = edge[1]
                return tree.nodes[successor_node_id]

            if edge_value == str(most_common_attribute_value):
                most_frequent_path_node_id = edge[1]
        return tree.nodes[most_frequent_path_node_id]

    def draw(self):
        g = Network(height='100%', width='100%',
                    notebook=True, layout='hierarchical')

        g.from_nx(self.tree, node_size_transf=lambda x: 20)

        g.show('tree.html')

    def get_root_node(self):
        return self.get_root_node_from_tree(self.tree)

    def get_root_node_from_tree(self, tree: nx.DiGraph):
        return tree.nodes[0]

    def classify(self, sample: pd.Series):
        return self.classify_from_tree(sample, self.tree)

    def classify_from_tree(self, sample: pd.Series, tree: nx.DiGraph):
        # get root node and its attribute
        current_node = self.get_root_node_from_tree(tree)
        current_attribute = current_node["node_value"]

        # get attribute value from sample
        attribute_value = sample[current_attribute]

        # for every other attribute, get the next node until we reach a leaf node
        while current_node["node_type"] != str(NodeType.LEAF):
            current_node = self.get_next_node(
                current_node, attribute_value, tree)

            if current_node["node_type"] == str(NodeType.LEAF):
                return int(current_node["node_value"])

            current_attribute = current_node["node_value"]
            attribute_value = sample[current_attribute]

        return int(current_node["node_value"])

    def prune(self, dataset: pd.DataFrame):
        root_node = self.get_root_node()
        self.prune_recursive(dataset, root_node["node_id"])

    def prune_recursive(self, dataset: pd.DataFrame, node_id: int, previous_node_id: int = None):
        # BOTTOM-UP PRUNING
        current_node = self.tree.nodes[node_id]

        if current_node["node_type"] == str(NodeType.LEAF):
            return

        current_attribute = current_node["node_value"]

        out_edges = self.tree.out_edges(node_id)

        for edge in out_edges:
            next_node_id = edge[1]
            edge_value = self.tree.get_edge_data(*edge)["label"]
            dataset_given_attribute_value = dataset[dataset[current_attribute] == int(
                edge_value)]
            if len(dataset_given_attribute_value) != 0:
                # call prune on the child node
                self.prune_recursive(
                    dataset_given_attribute_value, next_node_id, node_id)

        if previous_node_id is None:
            return  # we are at the root node

        current_error = self.calculate_error(dataset, self.tree)

        pruned_tree = copy.deepcopy(self.tree)

        self.remove_node_and_children(node_id, pruned_tree)

        class_mode = dataset[self.classes_column_name].mode()[0]

        self.create_and_set_node(node_type=NodeType.LEAF, value=class_mode,
                                depth=current_node["node_depth"], id=node_id, tree=pruned_tree)
        edge_label = self.tree.get_edge_data(previous_node_id, node_id)["label"]
        pruned_tree.add_edge(previous_node_id, node_id, label=edge_label
        # , color="red"
        )

        new_error = self.calculate_error(dataset, pruned_tree)
        if new_error < current_error:
            self.i += 1
            self.tree = pruned_tree
            print("PRUNED!")

    def remove_node_and_children(self, node_id, tree):
        node = tree.nodes[node_id]
        if node["node_type"] == str(NodeType.LEAF):
            tree.remove_node(node_id)
            return

        out_edges = tree.out_edges(node_id)
        for edge in list(out_edges):
            next_node_id = edge[1]
            self.remove_node_and_children(next_node_id, tree)

        tree.remove_node(node_id)

    def calculate_error(self, dataset: pd.DataFrame, tree: nx.DiGraph):
        incorrect_predictions = 0
        for index, sample in dataset.iterrows():
            if self.classify_from_tree(sample, tree) != sample[self.classes_column_name]:
                incorrect_predictions += 1

        return incorrect_predictions/len(dataset)

    def test(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset_copy = dataset.copy()
        dataset_copy[self.predicted_class_column_name] = dataset.apply(
            self.classify, axis=1)
        return dataset_copy

    def s_precision(self, dataset: pd.DataFrame) -> float:
        return (dataset[self.predicted_class_column_name] == dataset[self.classes_column_name]).sum() / len(dataset)

    def s_precision_per_depth(self, train_dataset: pd.DataFrame, test_dataset: pd.DataFrame, initial_depth=1, max_depth=7) -> dict:
        results = {}
        for depth in range(initial_depth, max_depth + 1):
            # set max depth
            self.max_depth = depth
            self.min_samples_split = -1

            # build tree
            self.train(train_dataset)

            # test with train dataset
            train_dataset_predictions = self.test(train_dataset)

            # test with test dataset
            test_dataset_predictions = self.test(test_dataset)

            # get number of nodes
            node_count = self.count_nodes()
            print(f'Node count: {node_count} for depth {depth}')

            # get s-precision for train predictions
            train_s_precision = self.s_precision(train_dataset_predictions)

            # get s-precision for test predictions
            test_s_precision = self.s_precision(test_dataset_predictions)

            # add to results
            if node_count not in results:
                results[node_count] = {
                    "train_s_precision": train_s_precision,
                    "test_s_precision": test_s_precision,
                    "depth": depth
                }

        return results

    def s_precision_per_node_count(self, train_dataset: pd.DataFrame, test_dataset: pd.DataFrame,
                                   initial_node_count: int = 5, max_node_count=500, prune=False) -> dict:
        results = {}
        step_size = 30
        for node_count in range(initial_node_count, max_node_count + step_size + 1, step_size):
            self.min_samples_split = -1
            self.max_node_count = node_count

            # build tree
            self.train(train_dataset)

            if prune:
                self.prune(test_dataset)

            # test with train dataset
            train_dataset_predictions = self.test(train_dataset)

            # test with test dataset
            test_dataset_predictions = self.test(test_dataset)

            # get s-precision for train predictions
            train_s_precision = self.s_precision(train_dataset_predictions)

            # get s-precision for test predictions
            test_s_precision = self.s_precision(test_dataset_predictions)

            # add to results
            if node_count not in results:
                results[node_count] = {
                    "train_s_precision": train_s_precision,
                    "test_s_precision": test_s_precision,
                    "depth": self.max_depth_by_attribute
                }

        return results

    def plot_precision_per_node_count_multiple_results(self, results_list: list, method_names: list):
        annotation_position_multiplier = -1
        
        train_s_precisions_by_method = {}
        test_s_precisions_by_method = {}
        node_counts_by_method = {}
        # colors for methods
        colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]
        for index, method in enumerate(method_names):
            train_s_precisions_by_method[method] = []
            test_s_precisions_by_method[method] = []
            node_counts_by_method[method] = []
            results = results_list[index]
            for node_count in results:
                node_counts_by_method[method].append(node_count)
                train_s_precisions_by_method[method].append(results[node_count]["train_s_precision"])
                test_s_precisions_by_method[method].append(results[node_count]["test_s_precision"])
                if "depth" in method:
                    # add depth value to both points
                    plt.annotate(f'd: {results[node_count]["depth"]}', (node_count ,
                                results[node_count]["train_s_precision"] + 0.015 * -annotation_position_multiplier))
                    plt.annotate(f'd: {results[node_count]["depth"]}', (node_count ,
                                results[node_count]["test_s_precision"] + 0.015 * annotation_position_multiplier))
                annotation_position_multiplier *= -1

            plt.plot(node_counts_by_method[method], train_s_precisions_by_method[method], label="Train "+ method,
                    linestyle='--', marker='o', color=colors[(index*2) % len(colors)])
            plt.plot(node_counts_by_method[method], test_s_precisions_by_method[method], label="Test "+ method,
                    linestyle='--', marker='o', color=colors[(index*2 +1) % len(colors)])

        plt.legend()
        plt.xlabel("Node count")
        plt.ylabel("Precision")
        plt.ylim(top=1.1)
        plt.show()


    def plot_precision_per_node_count(self, results: dict):
        train_s_precisions = []
        test_s_precisions = []
        node_counts = []
        depths = []

        annotation_position_multiplier = 1

        for node_count in results:
            node_counts.append(node_count)
            train_s_precisions.append(results[node_count]["train_s_precision"])
            test_s_precisions.append(results[node_count]["test_s_precision"])
            depths.append(results[node_count]["depth"])

            # add depth value to both points
            plt.annotate(f'd: {results[node_count]["depth"]}', (node_count - 20,
                         results[node_count]["train_s_precision"] + 0.015 * annotation_position_multiplier))
            plt.annotate(f'd: {results[node_count]["depth"]}', (node_count - 20,
                         results[node_count]["test_s_precision"] + 0.015 * annotation_position_multiplier))

            annotation_position_multiplier *= -1

        plt.plot(node_counts, train_s_precisions, label="Train",
                 linestyle='--', marker='o', color='r')
        plt.plot(node_counts, test_s_precisions, label="Test",
                 linestyle='--', marker='o', color='b')
        plt.legend()
        plt.xlabel("Node count")
        plt.ylabel("Precision")
        plt.ylim(top=1.1)
        plt.show()
