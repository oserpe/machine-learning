

import numpy as np
import pandas as pd
from .cluster import ClusteringDistance
from .decay_functions import exp_decay


class UnsupervisedClassifier():
    def __init__(self, model, K, max_iter, random_state, kohonen_initial_lr=0.01,
                 kohonen_initial_radius=None, kohonen_lr_fn=exp_decay, kohonen_radius_fn=exp_decay,
                 hierarchical_distance_metric=ClusteringDistance.CENTROID, verbose=True):
        if model == 'kmeans':
            from .k_means import KMeans
            self.model = KMeans(K=K, max_iter=max_iter,
                                random_state=random_state, verbose=verbose)
        elif model == 'hierarchical':
            from .hierarchical_clustering import HierarchicalClustering
            self.model = HierarchicalClustering(
                K=K, max_iter=max_iter, random_state=random_state, verbose=verbose, distance_metric=hierarchical_distance_metric)
        elif model == 'kohonen':
            from .kohonen import Kohonen
            if kohonen_initial_radius is None:
                kohonen_initial_radius = K
            self.model = Kohonen(K=K, max_iter=max_iter, random_state=random_state, verbose=verbose, initial_lr=kohonen_initial_lr,
                                 initial_radius=kohonen_initial_radius, lr_fn=kohonen_lr_fn, radius_fn=kohonen_radius_fn)
        else:
            raise ValueError('Invalid model')

    def get_feature_index(self, feature):
        return self.features.index(feature)

    def fit(self, X, y, X_features, y_feature):
        self.model.fit(X)
        self.X_features = X_features
        self.y_feature = y_feature

        # Create dataframe from X and y
        full_data = np.concatenate((X, y.reshape(-1, 1)), axis=1)
        df = pd.DataFrame(full_data, columns=[*X_features, y_feature])
        # Change from object to float64 the X_features columns
        df = df.astype({feature: 'float64' for feature in X_features})
        self.clusters = []

        # For each cluster, get their points and add the remaining features (y's)
        iteration = -1
        for cluster in self.model.get_clusters(iteration):
            df_points = pd.DataFrame(cluster.points, columns=X_features)
            df_merged = pd.merge(df, df_points)
            self.clusters.append(df_merged)

    def predict(self, X):
        predictions = []

        for x in X:
            # Get the cluster that x belongs to
            cluster_index = self.model.predict([x])[0]

            # Get the points in that cluster
            cluster_points = self.clusters[cluster_index]

            if len(cluster_points) == 0:
                predictions.append(None)
            else:
                # Get the y's from those points and get the mode
                mode = cluster_points[[self.y_feature]
                                      ].mode().iloc[0].values[0]
                predictions.append(mode)

        return predictions
