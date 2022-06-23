from Surrogate.RBF import RBFNet
from Surrogate.KNNDTW import KNNDTW
from EvolutionaryMethods.nsga2 import NSGA2
from sklearn.model_selection import train_test_split

class Model:
    def __init__(self, ds, model_type=0, n_neighbors=0):
        self.models = []
        self.train_set = []
        self.classes = []
        self.model_type = model_type
        self.needsFilled = False
        self.n_neighbors = n_neighbors
        self.instances = NSGA2(ds)
        
    def create(self, population):
        self.instances.set_population(population)
        self.train()
    
    def update(self):
        self.instances.FastNonDominatedSort()
        #if not self.instances.non_ranked():
        #    domination_count = self.instances.get_minimum_domination_count()
        #    print(domination_count)
        self.instances.get_new_population()
        self.train()
        
    def train(self):
        self.models = []
        self.train_set, self.classes = self.instances.population.to_train_set(self.needsFilled)
        if self.model_type == 1: #RBF
            self.needsFilled = True
            for i in range(0,len(self.classes[0])):
                rbfnet = RBFNet(lr=1e-2, k=2)
                rbfnet.fit(self.train_set, self.classes[:,i])
                self.models.append(rbfnet.copy())
        if self.model_type == 2: #KNN DTW
            self.needsFilled = False
            #self.train_set, self.classes = self.instances.population.to_train_set(self.needsFilled)
            for i in range(0,len(self.classes[0])):
                train_x, test_x, train_y, test_y = train_test_split(self.train_set, self.classes[:,i], test_size=0.34, random_state=42)
                neighbors = [1,3,5,7,9]
                min_score = 1;
                min_n = 0;
                knndtw = KNNDTW(is_regression=True)
                knndtw.fit(train_x, train_y)
                for n in neighbors:
                    knndtw.set_n_neighbors(n)
                    error_rate = knndtw.classifier(test_x, test_y)
                    if error_rate < min_score:
                        min_score = error_rate
                        min_n = n
                best_knndtw = KNNDTW(min_n, is_regression=True)
                best_knndtw.fit(self.train_set, self.classes[:,i])
                self.models.append(best_knndtw.copy())
    
    #def add_row(self, data, classes):
    #    self.train.vstack()