import numpy as np
from utils import init_dictionary

class ConfusionMatrix:

    def __init__(self, bd, str_discrete=[]):
        self.matrix = {}
        self.class_total = {}
        self.row_total = {}
        self.bd = bd
        self.clases = self.bd.clases
        
        if len(str_discrete)>0:
            self.create(str_discrete)
    
    def copy(self):
        mycopy = ConfusionMatrix(self.bd)
        mycopy.matrix = self.matrix.copy()
        mycopy.class_total = self.class_total.copy()
        mycopy.row_total = self.row_total.copy()
        return mycopy
    
    def create(self, str_discrete):  
        ds_strings_unique = np.unique(str_discrete[:,1])
        for i in range(0,len(ds_strings_unique)):
            conteo = init_dictionary(self.clases)
            for j in range(0, len(str_discrete)):
                if ds_strings_unique[i] == str_discrete[j,1]:
                    conteo[str_discrete[j,0]] += 1
            self.matrix[ds_strings_unique[i]] = conteo

        self.class_total = init_dictionary(self.clases)  
        
        for string, counts in self.matrix.items():
            class_sum = 0
            for klass, value in counts.items():
                self.class_total[klass] += int(value)
                class_sum += int(value)
            self.row_total[string] = class_sum
        