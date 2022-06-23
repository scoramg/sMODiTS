import numpy as np
from statistics import mean
from sklearn.metrics import mean_squared_error
#from Surrogate.timeseries_learn.metrics.dtw_variants import to_time_series, dtw, lcss
#from Surrogate.fastdtw import fastdtw
from dtaidistance import dtw
#from Surrogate.distance_metrics import DTW

class KNNDTW:
    def __init__(self, n_neighbors = 0, is_regression=False):
        self.n_neighbors = n_neighbors
        #self.knn = None
        self.x = []
        self.y = []
        self.is_regression = is_regression
        
    def set_n_neighbors(self, n_neighbors):
        self.n_neighbors = n_neighbors
    
    def get_neighbors(self, test_row):
        distances = list()
        for i in range(0, len(self.x)):
            #dist = dtw(to_time_series(test_row), to_time_series(np.array(self.x[0])))
            dist = dtw.distance_fast(test_row,  np.array(self.x[i]), use_pruning=True)
            #dist = DTW(test_row, np.array(self.x[0]))
            #dist = fastdtw(test_row,  np.array(self.x[0]))
            distances.append((self.y[i], dist))
        distances.sort(key=lambda tup: tup[1])
        neighbors = list()
        for i in range(self.n_neighbors):
            neighbors.append(distances[i][0])
        return neighbors
    
    
    def fit(self, X_train, Y_train):
        #parameters = {'n_neighbors':[self.n_neighbors]}
        #self.knn = GridSearchCV(KNeighborsClassifier(metric=self.DTW), parameters, cv=10)
        #self.knn.fit(X_train, Y_train)
        self.x = X_train
        self.y = Y_train
        
    def predict_set(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            #print(type(X))
            y_pred.append(self.predict_row(np.array(X[i])))
        return np.array(y_pred)
        
    def predict_row(self, X_test):
        #y_pred = self.knn.predict(X_test)
        #print(type(X_test))
        neighbors = self.get_neighbors(X_test)
        if not self.is_regression:
            output_values = [row for row in neighbors]
            y_pred = max(set(output_values), key=output_values.count)
        else: 
            y_pred = mean(neighbors)
        return y_pred
        #print(classification_report(y_test, y_pred))
    
    def classifier(self, X_test, y_test):
        y_pred = self.predict_set(X_test)
        correct = 0
        if not self.is_regression:
            for i in range(0,len(y_pred)):
                if y_pred[i] == y_test[i]:
                    correct += 1
            error_rate = 1 - (correct/len(y_pred))
        else:
            error_rate = mean_squared_error(y_test, y_pred)
        return  error_rate #misclassification error
        
    def copy(self):
        knndtw = KNNDTW(self.n_neighbors)
        #knndtw.knn = self.knn
        knndtw.x = self.x.copy()
        knndtw.y = self.y.copy()
        return knndtw