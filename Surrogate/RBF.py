#from ML.KMeans import KMeans
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from utils import get_distance
import numpy as np

def rbf(x, c, s):
    distance = get_distance(x, c)
    #return 1 / np.exp(-distance / s ** 2)
    return np.exp(-1 / (2 * s**2) * distance)

class RBFNet(object):
    """Implementation of a Radial Basis Function Network"""
    def __init__(self, k=2, lr=0.01, epochs=100, rbf=rbf):
        self.k = k
        self.lr = lr
        self.epochs = epochs
        self.rbf = rbf
        #self.inferStds = inferStds
        self.w = np.random.randn(k)
        self.b = np.random.randn(1)
        #self.kmeans = KMeans(self.k)
        self.kmeans = KMeans(n_clusters=k, random_state=0)
        
    def fit(self, X, y):
        self.kmeans.fit(X)
        self.centers = self.kmeans.cluster_centers_
        #if self.inferStds:
            # compute stds from data
         #   self.stds = self.kmeans.stds
        #else:
            # use a fixed std 
        dMax = np.max([np.abs(c1 - c2) for c1 in self.centers for c2 in self.centers])
        self.stds = np.repeat(dMax / np.sqrt(2*self.k), self.k)
        # training
        for epoch in range(self.epochs):
            for i in range(X.shape[0]):
                # forward pass
                a = np.array([self.rbf(X[i], c, s) for c, s, in zip(self.centers, self.stds)])
                F = a.T.dot(self.w) + self.b
                #loss = (y[i] - F).flatten() ** 2
                #print('Loss: {0:.8f}'.format(loss[0]))
                
                # backward pass
                error = -(y[i] - F).flatten()
                # online update
                self.w = self.w - self.lr * a * error
                self.b = self.b - self.lr * error 
    
    def predict_set(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            #a = np.array([self.rbf(X[i], c, s) for c, s, in zip(self.centers, self.stds)])
            #F = a.T.dot(self.w) + self.b
            y_pred.append(self.predict_row(X[i]))
        return np.array(y_pred)
    
    def predict_row(self, X):
        a = np.array([self.rbf(X, c, s) for c, s, in zip(self.centers, self.stds)])
        F = a.T.dot(self.w) + self.b
        y_pred = F.item()
        return y_pred
    
    def classifier(self, X_test, y_test):
        y_pred = self.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        return mse
    
    def copy(self):
        rbfnet = RBFNet(lr=self.lr, k=self.k)
        rbfnet.w = self.w.copy()
        rbfnet.b = self.b
        rbfnet.centers = self.centers
        rbfnet.stds = self.stds
        return rbfnet
    
""" from utils import kmeans, get_distance

import numpy as np

#from sklearn.cluster import KMeans

class RBF:
    
    def __init__(self, X, y, tX, ty, num_of_classes,
                 k, std_from_clusters=True):
        self.X = X
        self.y = y

        self.tX = tX
        self.ty = ty

        self.number_of_classes = num_of_classes
        self.k = k
        self.std_from_clusters = std_from_clusters

    def convert_to_one_hot(self, x, num_of_classes):
        arr = np.zeros((len(x), num_of_classes))
        for i in range(len(x)):
            c = int(x[i])
            arr[i][c] = 1
        return arr

    def rbf(self, x, c, s):
        distance = get_distance(x, c)
        return 1 / np.exp(-distance / s ** 2)

    def rbf_list(self, X, centroids, std_list):
        RBF_list = []
        for x in X:
            RBF_list.append([self.rbf(x, c, s) for (c, s) in zip(centroids, std_list)])
        return np.array(RBF_list)
    
    def fit(self):
        self.centroids, self.std_list = kmeans(self.X, self.k, max_iters=1000)
        #kmeans = KMeans(n_clusters=self.k, max_iter=1000, random_state=0).fit(self.X)
        #self.centroids = kmeans.cluster_centers_
        if not self.std_from_clusters:
            dMax = np.max([get_distance(c1, c2) for c1 in self.centroids for c2 in self.centroids])
            self.std_list = np.repeat(dMax / np.sqrt(2 * self.k), self.k)
        RBF_X = self.rbf_list(self.X, self.centroids, self.std_list)
        self.w = np.linalg.pinv(RBF_X.T @ RBF_X) @ RBF_X.T @ self.convert_to_one_hot(self.y, self.number_of_classes)
        RBF_list_tst = self.rbf_list(self.tX, self.centroids, self.std_list)
        self.pred_ty = RBF_list_tst @ self.w
        self.pred_ty = np.array([np.argmax(x) for x in self.pred_ty])
        diff = self.pred_ty - self.ty
        print(diff, type(self.ty))
        a = len(np.where(diff == 0)[0])
        accuracy =  a / len(diff)
        return accuracy
        #print('Accuracy: ', len(np.where(diff == 0)[0]) / len(diff)) """