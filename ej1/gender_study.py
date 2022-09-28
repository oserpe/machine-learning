import pandas as pd
from ..metrics import Metrics
from .models.TreeType import TreeType
from .models.DecisionTree import DecisionTree
import networkx as nx

WOMAN_CLASSIFICATION_VALUE = 4
SEX_AND_MARITAL_STATUS_COLUMN = "Sex & Marital Status"


PATH_COLOR = '#FC3631'
NORMAL_COLOR = '#000000'
GOAL_COLOR = '#FFF388'


def color_solution_path(tree, goal_node):
    current_id = goal_node["node_id"]

    tree.nodes[current_id]['color'] = GOAL_COLOR
    tree.nodes[current_id]['shape'] = 'star'

    predecessor_ids = list(tree.predecessors(goal_node["node_id"]))

    while len(predecessor_ids) != 0:
        predecessor_id = predecessor_ids[0]

        attrs = {(predecessor_id, current_id): {"color": PATH_COLOR}}
        nx.set_edge_attributes(tree, attrs)
        tree.nodes[current_id]['shape'] = 'star'

        current_id = predecessor_id
        predecessor_ids = list(tree.predecessors(predecessor_ids[0]))


def gender_study(data_df: pd.DataFrame):
    gender_study_analytics(data_df)

    decision_tree = DecisionTree(
        classes=[1, 2, 3, 4], classes_column_name=SEX_AND_MARITAL_STATUS_COLUMN)

    # get train and test datasets
    train_dataset, test_dataset = Metrics.holdout(data_df, test_size=0.3)

    decision_tree.train(train_dataset)

    # prune
    decision_tree.prune(test_dataset)
    tree = decision_tree.tree
    leaf_nodes = [tree.nodes[x]
                  for x in tree.nodes() if tree.out_degree(x) == 0]

    for leaf in leaf_nodes:
        if leaf["node_value"] == str(WOMAN_CLASSIFICATION_VALUE):
            color_solution_path(decision_tree.tree, leaf)

    # pos = nx.circular_layout(G)

    # # default
    # plt.figure(1)
    # nx.draw(G,pos)

    # # smaller nodes and fonts
    # plt.figure(2)
    # nx.draw(decision_tree.tree,node_size=60,font_size=8)

    decision_tree.draw()

    return


def gender_study_analytics(data_df: pd.DataFrame):
    gender_column = "Sex & Marital Status"
    female_data_df = data_df[data_df[gender_column] == 4]
    male_data_df = data_df[data_df[gender_column] != 4]

    print("Porcentaje de hombres que piden un prestamo sobre el total: ",
          round(len(male_data_df)/len(data_df), 2))

    occupation_column = "Occupation"
    executive_occupation_id = 4
    print("Porcentaje de mujeres en puestos ejecutivos sobre el total de ejecutivos: ",
          round(female_data_df.groupby(occupation_column).size()[executive_occupation_id] /
                data_df.groupby(occupation_column).size()[executive_occupation_id], 2))
    print("En las mujeres el porcentaje que ocupa puestos ejecutivos es: ",
          round(female_data_df.groupby(occupation_column).size()[executive_occupation_id] /
                len(female_data_df), 2))
    print("En los hombres el porcentaje que ocupa puestos ejecutivos es: ",
          round(male_data_df.groupby(occupation_column).size()[executive_occupation_id] /
                len(male_data_df), 2))

    amount_column = "Credit Amount"
    amount_separator = 4
    print(f"En las mujeres el porcentaje de las que piden prestamos menores a la categoria {amount_separator} es: ",
          round(len(female_data_df[female_data_df[amount_column] < amount_separator]) /
                len(female_data_df), 2))
    print(f"En los hombres el porcentaje de los que piden prestamos menores a la categoria {amount_separator} es: ",
          round(len(male_data_df[male_data_df[amount_column] < amount_separator]) /
                len(male_data_df), 2))

    most_valuable_asset_column = "Most valuable available asset"
    print(f"En las mujeres el porcentaje de las que no tienen most valuable asset es: ",
          round(len(female_data_df[female_data_df[most_valuable_asset_column] <= 1]) /
                len(female_data_df), 2))
    print(f"En los hombres el porcentaje de los que no tienen most valuable asset es: ",
          round(len(male_data_df[male_data_df[most_valuable_asset_column] <= 1]) /
                len(male_data_df), 2))
