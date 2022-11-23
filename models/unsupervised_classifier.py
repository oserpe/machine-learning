

import numpy as np
import pandas as pd
from .cluster import ClusteringDistance
from .decay_functions import exp_decay
from sklearn.base import BaseEstimator


class UnsupervisedClassifier(BaseEstimator):
    def __init__(self, model, K, max_iter, random_state=0, kohonen_initial_lr=0.01,
                 kohonen_initial_radius=None, kohonen_lr_fn=exp_decay, kohonen_radius_fn=exp_decay,
                 hierarchical_distance_metric=ClusteringDistance.CENTROID, verbose=True):
        self.K = K
        self.max_iter = max_iter
        self.random_state = random_state
        self.kohonen_initial_lr = kohonen_initial_lr
        self.kohonen_initial_radius = kohonen_initial_radius
        self.kohonen_lr_fn = kohonen_lr_fn
        self.kohonen_radius_fn = kohonen_radius_fn
        self.hierarchical_distance_metric = hierarchical_distance_metric
        self.verbose = verbose
        self.model = model

    def get_feature_index(self, feature):
        return self.features.index(feature)

    def fit(self, X, y):
        if self.model == 'kmeans':
            from .k_means import KMeans
            self._model = KMeans(K=self.K, max_iter=self.max_iter,
                                 random_state=self.random_state, verbose=self.verbose)
        elif self.model == 'hierarchical':
            from .hierarchical_clustering import HierarchicalClustering
            self._model = HierarchicalClustering(
                K=self.K, verbose=self.verbose, distance_metric=self.hierarchical_distance_metric)
        elif self.model == 'kohonen':
            from .kohonen import Kohonen
            if self.kohonen_initial_radius is None:
                self.kohonen_initial_radius = self.K
            self._model = Kohonen(K=self.K, max_iter=self.max_iter, random_state=self.random_state, verbose=self.verbose,
                                  initial_lr=self.kohonen_initial_lr, initial_radius=self.kohonen_initial_radius, lr_fn=self.kohonen_lr_fn, radius_fn=self.kohonen_radius_fn)
        else:
            raise ValueError('Invalid model')

        self.X_features = X.columns
        self.y_feature = y.columns[0]

        self._model.fit(X.to_numpy())

        # Create dataframe from X and y

        df = pd.concat([X.reset_index(), y.reset_index()], axis=1)

        # Change from object to float64 the X_features columns
        df = df.astype({feature: 'float64' for feature in self.X_features})
        self.clusters = []

        # For each cluster, get their points and add the remaining features (y's)
        # iteration = -1
        for cluster in self._model.get_clusters():
            df_points = pd.DataFrame(cluster.points, columns=self.X_features)
            df_merged = pd.merge(df, df_points)
            self.clusters.append(df_merged)

    def predict(self, X):

        X = X.to_numpy()

        # Get the cluster that x belongs to
        cluster_indexes = self._model.predict(X)

        return np.array([self.clusters[cluster_index][self.y_feature].mode().loc[0] if len(self.clusters[cluster_index].values) > 0 else None for cluster_index in cluster_indexes])

    def score(self, X, y):
        predictions = self.predict(X)
        y = y.to_numpy()
        correct = 0

        for i in range(len(predictions)):
            if predictions[i] == y[i]:
                correct += 1
        return correct / len(predictions)
