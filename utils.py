import ast
import os
import math
import numpy as np
#import asyncio

#def paralelizar(f):
#    def wrapped(*args, **kwargs):
#        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)
#    return wrapped

def to_paired_value(value_list):
    return [[value_list[i], value_list[i + 1]] for i in range(len(value_list)-1)]

def key_to_list(key):
    return ast.literal_eval(key)

def init_dictionary(keys):
    conteo = {}
    for i in range(0, len(keys)):
        conteo[keys[i]]=0
    return conteo

def find_directory(name):
    """ funcion que busca una carpeta y devuelve la ruta de esa carpeta """
    encontro = False
    dir_path = __file__
    dir_ds = ""
    while encontro is False:
        start = os.path.dirname(os.path.realpath(dir_path))
        for dirpath, dirnames, _ in os.walk(start):
            for dirname in dirnames:
                if dirname == name:
                    encontro = True
                    dir_ds = os.path.join(dirpath, dirname)
                    break;
            if encontro:
                break
        dir_path = os.path.dirname(dir_path)
    return dir_ds

def minmax_normalize(df):
    minimum = df.min()
    maximum = df.max()
    norm = 0
    if (maximum - minimum) == 0:
        norm = df/maximum
    else: 
        norm = (df - minimum) / (maximum - minimum)
    return norm

def normalize_matrix(matrix):
    bd_norm = np.empty([len(matrix), len(matrix[0,:])], dtype=float)
    for i in range(0,len(matrix)):
        bd_norm[i,0] = matrix[i,0]
        np.copyto(bd_norm[i,1:],minmax_normalize(np.array(matrix[i,1:])))
    return bd_norm

def getNoOpponents(pop_size):
    return math.floor(pop_size * 0.1);

def compare_by_rank_crowdistance(scheme1, scheme2):
    if scheme1 is None:
        return 1;
    elif scheme2 is None:
        return -1;
    if scheme1.rank < scheme2.rank:
        return -1;
    elif scheme1.rank > scheme2.rank:
        return 1;
    elif scheme1.crowding_distance > scheme2.crowding_distance:
        return -1;
    elif scheme1.crowding_distance < scheme2.crowding_distance:
        return 1;
    return 0;

def get_distance(x1, x2):
    sum = 0
    for i in range(len(x1)):
        sum += (x1[i] - x2[i]) ** 2
    return np.sqrt(sum)

def str_to_list(string):
    l = []
    n = len(string)
    a = string[1:n-1]
    a = a.split(',')
    for i in a:
        l.append(int(i))
    return l

""" 
def eucl_dist(x, y):
    
    #Usage
    #-----
    #L2-norm between point x and y
    #Parameters
    #----------
    #param x : numpy_array
    #param y : numpy_array
    #Returns
    #-------
    #dist : float
    #       L2-norm between x and y
    
    dist = np.linalg.norm(x - y)
    return dist


    
def kmeans(X, k, max_iters):
    centroids = X[np.random.choice(range(len(X)), k)]
    #centroids = X[np.random.choice(range(len(X)), k, replace=False)]
    converged = False
    current_iter = 0
    while (not converged) and (current_iter < max_iters):
        cluster_list = [[] for i in range(len(centroids))]
        for x in X:  # Go through each data point
            distances_list = []
            for c in centroids:
                distances_list.append(get_distance(c, x))
            cluster_list[int(np.argmin(distances_list))].append(x)
        cluster_list = list((filter(None, cluster_list)))
        prev_centroids = centroids.copy()
        centroids = []
        for j in range(len(cluster_list)):
            centroids.append(np.mean(cluster_list[j], axis=0))
        pattern = np.abs(np.sum(prev_centroids) - np.sum(centroids))
        #print('K-MEANS: ', int(pattern))
        converged = (pattern == 0)
        current_iter += 1
    return np.array(centroids), [np.std(x) for x in cluster_list] """