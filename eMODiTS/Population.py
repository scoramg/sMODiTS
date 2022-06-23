import eMODiTS.Scheme as sch
#import Surrogate.Model as model
import math, random
from utils import compare_by_rank_crowdistance
import numpy

class Population:
    def __init__(self, ds, pop_size = 0, idfunctionsconf=0):
        self.individuals = []
        self.ds = ds
        self.size = 0
        self.idfunctionsconf = idfunctionsconf
        
        if pop_size > 0:
            self.create(pop_size=pop_size)
            
    def add_individual(self, scheme):
        self.individuals.append(scheme.copy())
        self.size += 1
        
    def create(self, pop_size):
        for i in range(0,pop_size):
            self.add_individual(sch.Scheme(self.ds, cuts={}, idfunctionsconf=self.idfunctionsconf))
            
    def evaluate(self, model=None):
        no_eval = 0
        for i in range(0,self.size):
            no_eval += self.individuals[i].evaluate(model)
        return no_eval
    
    def crossover(self, pc):
        offsprings = Population(self.ds)
        k = self.size-1
        for i in range(0,self.size):
            if i < k:
                if random.random() <= pc:
                    off1, off2 = self.individuals[i].crossover(self.individuals[k])
                    offsprings.add_individual(off1)
                    offsprings.add_individual(off2)
                else:
                    offsprings.add_individual(self.individuals[i])
                    offsprings.add_individual(self.individuals[k])
            k-=1
        return offsprings
    
    def mutate(self, pm):
        for i in range(0,self.size):
            self.individuals[i].mutate(pm)
    
    def copy(self):
        mycopy = Population(self.ds)
        for i in range(0,len(self.individuals)):
            mycopy.add_individual(self.individuals[i].copy())
        return mycopy
    
    def join(self, other):
        for i in range(0, other.size):
            self.add_individual(other.individuals[i])
    
    def tournament_selection(self):
        parents = Population(self.ds)
        no_opponents = math.floor(self.size * 0.1)
        victories = {}
        for i in range(0,self.size):
            victories[i] = 0
            opponents = []
            opponents.append(i)
            for op in range(0,no_opponents):
                j = random.choice([r for r in range(0,self.size) if r not in opponents])
                if compare_by_rank_crowdistance(self.individuals[i], self.individuals[j]) < 0:
                    victories[i] += 1
                opponents.append(j)
        victories = {k: v for k, v in sorted(victories.items(), key=lambda item: item[1], reverse=True)}
        for i in list(victories.keys())[:self.size]:
            parents.add_individual(self.individuals[i])
        return parents
    
    def to_train_set(self, filled=True):
        train = []
        ff = []
        for i in range(0,self.size):
            data, ff_values = self.individuals[i].to_vector(filled)
            train.append(data)
            ff.append(ff_values)
        return numpy.array(train, dtype=object), numpy.array(ff, dtype=float)
    
    def export_matlab(self):
        data = {}
        fitness = []
        for i in range(0,len(self.individuals)):
            cuts, fits = self.individuals[i].matlab_format()
            data["FrontIndividual"+str(i)] = cuts
            fitness.append(fits)
        data["AccumulatedFrontFitness"] = fitness
        return data