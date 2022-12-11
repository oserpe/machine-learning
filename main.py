from enum import Enum
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier 
from .metrics import Metrics

result_column = "diagnosis"
result_column_labels = ["M", "B"]

class Estimator(Enum):
    SVC_LINEAR = {"estimator": SVC(kernel="linear"), "algorithm": "SAMME"}
    DECISION_TREE = {"estimator": DecisionTreeClassifier(max_depth=1), "algorithm": "SAMME.R"}
    KNN = {"estimator": KNeighborsClassifier(), "algorithm": "SAMME"}
    PERCEPTRON = {"estimator": Perceptron(), "algorithm": "SAMME"}
    

def adaboost_classify(estimator: Estimator, X_train, X_test, y_train, random_state):
    adaboost = AdaBoostClassifier(estimator=estimator.value["estimator"], random_state=random_state, algorithm=estimator.value["algorithm"])

    adaboost.fit(X_train, y_train)
    
    return adaboost.predict(X_test)
    
def metrics(y_test, y_pred):
    cf_matrix = Metrics.get_confusion_matrix(y_test, y_pred, result_column_labels)
    Metrics.plot_confusion_matrix_heatmap(cf_matrix)
    
    metrics, metrics_df = Metrics.get_metrics_per_class(cf_matrix)
    Metrics.plot_metrics_heatmap(metrics)

def main(data_df):

    # drop duplicates by id
    data_df.drop_duplicates(subset=["id"], keep="first", inplace=True)

    # drop non-interesting columns
    data_df.drop("id", inplace=True, axis=1)
    
    # drop rows with missing values
    data_df.dropna(inplace=True)

    y = data_df[result_column]
    X = data_df.drop(result_column, axis=1)

    # standardize
    X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)

    random_state = 13
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state)

    y_pred = adaboost_classify(Estimator.SVC_LINEAR, X_train, X_test, y_train, random_state)
    metrics(y_test, y_pred)
        
def plot_hist(data_df):
    data_df.hist(edgecolor='black', linewidth=1,
                 xlabelsize=10, ylabelsize=10, grid=False)

    plt.tight_layout()

    # reduce space between subplots
    plt.subplots_adjust(wspace=0.2, hspace=0.5)

    plt.show()
    
def dataset_analysis(data_df):
    # dataset length
    print("Dataset length: ", len(data_df))
    # number of features
    print("Number of features: ", len(data_df.columns))
    # feature types
    print("Feature types: ", data_df.dtypes)

    unique_df = data_df.drop_duplicates(subset=["id"], keep="first")

    # number of duplicated rows
    print("Number of duplicated rows: ", len(
        data_df) - len(unique_df))

    unique_withoutna_df = unique_df.dropna()
    # number of rows with at least one null value
    print("Number of rows with at least one null value: ", len(
        unique_df) - len(unique_withoutna_df))
    
    print("Final dataset length: ", len(unique_withoutna_df))


    # ------ HISTOGRAM ------

    # drop non-interesting columns
    data_df.drop("id", inplace=True, axis=1)

    # modify non-numerical columns (M = 1, B = 0)
    data_df["diagnosis"] = data_df["diagnosis"].map({"M": 1, "B": 0})

    # separate df into two df's, first with 15 columns, second with 16 columns
    data_df_15 = data_df.iloc[:, 0:15]
    data_df_16 = data_df.iloc[:, 15:31]
    
    plot_hist(data_df_15)
    plot_hist(data_df_16)
    

if __name__ == "__main__":
    data_df = pd.read_csv("./machine-learning/breast_cancer_wisconsin_data.csv", header=0, sep=',')
    main(data_df)
    # dataset_analysis(data_df)
