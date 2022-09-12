import math


def relative_frequency(count, total, classes_count, laplace_correction=True):
    """Calculates the relative frequency of a given count."""
    if laplace_correction:
        # apply laplace correction
        return (count + 1) / (total + len(classes_count))
    else:
        return count / total


def calculate_probabilities(dataset, class_column, class_values, attribute_column, attribute_value):
    """Calculates the probabilities of a given attribute value for a given class."""
    probabilities = {}

    attribute_appereances = len(
        dataset[(dataset[attribute_column] == attribute_value)])

    for class_value in class_values:
        probabilities[class_value] = relative_frequency(len(dataset[(dataset[attribute_column] == attribute_value) & (
            dataset[class_column] == class_value)]), attribute_appereances, class_values)   # apply laplace correction

    return probabilities


def shannon_entropy(probabilities):
    """Calculates the Shannon entropy of a list of probabilities."""
    return sum(-p * math.log(p, 2) for p in probabilities if p)


def gini_index(probabilities):
    """Calculates the Gini index of a list of probabilities."""
    return 1 - sum(p ** 2 for p in probabilities)


def information_gain(dataset, parent_column, parent_value, attribute_column, attribute_values, class_column, class_values):
    """Calculates the information gain of a dataset for a given attribute."""
    parent_probabilities = calculate_probabilities(
        dataset, class_column, class_values, parent_column, parent_value)

    parent_entropy = shannon_entropy(parent_probabilities.values())

    dataset_filtered_by_parent_value = dataset[dataset[parent_column]
                                               == parent_value]

    children_entropy = 0
    for attribute_value in attribute_values:
        attribute_probabilities = calculate_probabilities(
            dataset_filtered_by_parent_value, class_column, class_values, attribute_column, attribute_value)

        children_entropy += shannon_entropy(
            attribute_probabilities.values()) * dataset_filtered_by_parent_value[attribute_column].value_counts()[attribute_value] / len(dataset_filtered_by_parent_value)
        # DUDA: hace falta aplicar laplace aca? Con qué denominador?

    return parent_entropy - children_entropy


class DecisionTree:
    # def __init__(self, ):
