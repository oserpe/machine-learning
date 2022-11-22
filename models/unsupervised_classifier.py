

import numpy as np
import pandas as pd
from .cluster import ClusteringDistance
from .decay_functions import exp_decay
from sklearn.base import BaseEstimator


class UnsupervisedClassifier(BaseEstimator):
    def __init__(self, model, K, max_iter, random_state, kohonen_initial_lr=0.01,
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

        if model == 'kmeans':
            from .k_means import KMeans
            self._model = KMeans(K=K, max_iter=max_iter,
                                 random_state=random_state, verbose=verbose)
        elif model == 'hierarchical':
            from .hierarchical_clustering import HierarchicalClustering
            self._model = HierarchicalClustering(
                K=K, verbose=verbose, distance_metric=hierarchical_distance_metric)
        elif model == 'kohonen':
            from .kohonen import Kohonen
            if kohonen_initial_radius is None:
                kohonen_initial_radius = K
            self._model = Kohonen(K=K, max_iter=max_iter, random_state=random_state, verbose=verbose, initial_lr=kohonen_initial_lr,
                                  initial_radius=kohonen_initial_radius, lr_fn=kohonen_lr_fn, radius_fn=kohonen_radius_fn)
        else:
            raise ValueError('Invalid model')

    def get_feature_index(self, feature):
        return self.features.index(feature)

    def fit(self, X, y):
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
        predictions = []
        X = X.to_numpy()

        for x in X:
            # Get the cluster that x belongs to
            cluster_index = self._model.predict([x])[0]

            # Get the points in that cluster
            cluster_points = self.clusters[cluster_index]

            if len(cluster_points) == 0:
                predictions.append(None)
            else:
                # Get the y's from those points and get the first mode
                mode = cluster_points[self.y_feature
                                      ].mode()[0]
                predictions.append(mode)

        return predictions

    def score(self, X, y):
        predictions = self.predict(X)
        y = y.to_numpy()
        correct = 0

        for i in range(len(predictions)):
            if predictions[i] == y[i]:
                correct += 1
        return correct / len(predictions)
