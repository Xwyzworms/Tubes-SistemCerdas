#%%

import pandas as pd
import tensorflow as tf
from sklearn.datasets import load_iris
import numpy as np

print(f"Pandas Version {pd.__version__}")
print(f"Tensorflow Version {tf.__version__}")
#%%

iris = load_iris()
X = iris["data"]
Y = iris["target"]

def shuffle(X,Y,test_size):
    shuffleIndx = int(len(X) * test_size)
    X_train = X[shuffleIndx:]
    Y_train = Y[shuffleIndx:]

    X_test = X[:shuffleIndx]
    Y_test = Y[:shuffleIndx]

    return X_train,X_test,Y_train,Y_test
x_train,x_test,y_train,y_test = shuffle(X,Y,0.2)

#%%
class KNN():
    """
    Class for using 

    """
    def __init__(self,n_neighbors):
        self.n_neighbors =  n_neighbors
        self.x_train = []
        self.y_train = []

    def euclidean_distance(self, x_train_obs, x_test_obs):
        return np.sqrt(tf.reduce_sum( (x_train_obs - x_test_obs)**2,axis=0)) 

    def fit(self, x_train, y_train):
        self.x_train = tf.constant(x_train,dtype=tf.float32)
        self.y_train = tf.constant(y_train,dtype=tf.float32)
    
    def counterTheMost(self, labels):
        counterItM8 = {key:0 for key in set(labels)}
        for label in labels:
            if label in counterItM8.keys():
                counterItM8[label] += 1
        
        return sorted(counterItM8.items(),key=lambda x : x[1])[0][0]

    def predict(self,x_test):
        y_pred = [self.do_calculation(x) for x in x_test]
        return tf.cast(tf.Variable(y_pred, dtype=tf.float32),tf.int32).numpy()

    def do_calculation(self,x_test_obs):
        distance = [self.euclidean_distance(x_train_obs,x_test_obs) for x_train_obs in self.x_train]   
        #Get The Lowest Distances index
        lowestDistances = tf.argsort(distance)[:self.n_neighbors]
        # get the labels :
        labels = [self.y_train[indx].numpy() for indx in lowestDistances]
        return self.counterTheMost(labels)


#%%
knnOBJ = KNN(2)
knnOBJ.fit(tf.constant(x_train,dtype=tf.float32), tf.constant(y_train,dtype=tf.float32))
ypred = knnOBJ.predict(tf.constant(x_train,dtype=tf.float32))

#%%

def getAcc(y_true,y_pred):
    return np.sum(y_true == ypred) / len(y_true)
getAcc(y_train ,ypred)
#%%
ypred = knnOBJ.predict(tf.constant(x_test,dtype=tf.float32))
print(getAcc(y_test,ypred))