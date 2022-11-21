

import numpy as np
import pandas as pd

class UnsupervisedClassifier():
    def __init__(self, model, features):
        self.model = model
        self.features = features
        groups = {}
        # find 
        for x in self.X_train:
            # remove requested features
            x_without_features = np.delete(x, [self.get_feature_index(f) for f in predict_features])
            prediction = self.model.predict(x_without_features)
            groups[hash(prediction)].append(x)
            # kohonen -> (5,3)
            # k-means -> 10

    def get_feature_index(self, feature):
        return self.features.index(feature)
    
    def fit(self, X, Y):
        
        for x, y in zip(X, y):

        # dataset = pd.DataFrame(concat(X,y))
        # [1,1,5,2,1] = self.model.fit(X)
        x_per_group = dataset.append("group", self.model.fit(X))

        # join data in clusters with y 
        self.clustered_data = pd.merge()#..... si si el equals shh mimo mimo


    def predict(self, X):
        # 

        dataset["group" == g].mode("genres")

        groups = {}
        # find 
        for x in self.X_train:
            # remove requested features
            x_without_features = np.delete(x, [self.get_feature_index(f) for f in predict_features])

            w.x
            w.all_features
            prediction = self.model.predict(x_without_features)
            groups[hash(prediction)].append(x)
            # kohonen -> (5,3)
            # k-means -> 10

            w = wrapper de (X, y)
            
            model.predict()

            


        
        # sacar el genero de x
        group = groups[self.model.predict(X_sin_genero)] #tengo todos los ejemplares del mismo grupo CON genero
        moda(group[genero])
