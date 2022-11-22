

import numpy as np
import pandas as pd

class UnsupervisedClassifier():
    def __init__(self, model):
        self.model = model

    def get_feature_index(self, feature):
        return self.features.index(feature)

    def fit(self, X, y, X_features, y_feature):
        self.model.fit(X)
        self.X_features = X_features
        self.y_feature = y_feature

        # Create dataframe from X and y
        full_data = np.concatenate((X, y.reshape(-1, 1)), axis=1)
        df = pd.DataFrame(full_data, columns = [*X_features, y_feature])
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
                mode = cluster_points[[self.y_feature]].mode().iloc[0].values[0]
                predictions.append(mode)

        return predictions
