import imp
from ..metrics import *
import pandas as pd

def main():
    data_df = pd.read_csv("./machine-learning/ej1/dataset/german_credit.csv", header=0, sep=',')
    Metrics.k_fold_cross_validation()


if __name__ == "__main__":
    main()