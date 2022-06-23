import math, functools
import eMODiTS.Population as pop
from utils import compare_by_rank_crowdistance#, paralelizar
import numpy as np
#from Surrogate.Model import Model
#from joblib import Parallel, delayed
#import multiprocessing as mp
#import asyncio



class NSGA2:
    def __init__(self, ds, pop_size=0, idfunctionsconf=0, model=None):
        self.fronts = {}
        self.fronts[0] = []
        self.ds = ds
        self.pop_size = pop_size
        self.model = model
        self.population = pop.Population(ds, idfunctionsconf=idfunctionsconf)
        self.no_evaluations = 0
        
    def set_population(self, population):
        self.population = population.copy()
        self.pop_size = self.population.size
        
    def non_ranked(self):
        ranked = True
        for ind in self.population.individuals:
            if ind.rank == -1:
                ranked = False
                break
        return ranked
    
    def get_minimum_domination_count(self):
        count_min = 10000000
        for ind in self.population.individuals:
            if ind.domination_count < count_min and ind.domination_count > 0:
                count_min = ind.domination_count
        return count_min
    
    def FastNonDominatedSort(self):  
        self.fronts = {} 
        self.fronts[0] = [] 
        #for k in range(0,self.population.size):
        #    self.population.individuals[k].dominated_set = set()
        #    self.population.individuals[k].domination_count = 0
        #    self.population.individuals[k].rank = -1
            
        for p in range(0,self.population.size):
            self.population.individuals[p].dominated_set = []
            self.population.individuals[p].domination_count = 0
            self.population.individuals[p].rank = -1
            for q in range(0,self.population.size):
                #if p!=q:
                if self.population.individuals[p].dominates(self.population.individuals[q]):
                    #if self.population.individuals[q] not in self.population.individuals[p].dominated_set: 
                    self.population.individuals[p].dominated_set.append(self.population.individuals[q]) #S
                    #self.population.individuals[q].domination_count += 1
                elif self.population.individuals[q].dominates(self.population.individuals[p]):
                    #self.population.individuals[q].dominated_set.add(self.population.individuals[p]) #S
                    self.population.individuals[p].domination_count += 1
            if self.population.individuals[p].domination_count == 0:
                self.population.individuals[p].rank = 0
                self.fronts[0].append(self.population.individuals[p])
            
        i = 0;
        while self.fronts[i]:
            Q = []
            for j in range(0,len(self.fronts[i])):
                for dom in self.fronts[i][j].dominated_set:
                    dom.domination_count -= 1
                    if dom.domination_count == 0:
                        dom.rank = i + 1
                        if dom not in Q:
                            Q.append(dom)
            i += 1
            self.fronts[i] = Q
        
        if not self.non_ranked():
            domination_count = self.get_minimum_domination_count()
            print(domination_count)
            
        #del self.fronts[len(self.fronts)-1]

    def get_crowding_distance(self):
        for f in range(0, len(self.fronts)):
            if self.fronts[f]:
                if len(self.fronts[f]) in [1,2]:
                    for i in range(0,len(self.fronts[f])):
                        self.fronts[f][i].crowding_distance = math.inf
                else:
                    for m in range(0,len(self.fronts[f][0].fitness_functions.values)):
                        self.fronts[f].sort(key=lambda x: x.fitness_functions.values[m], reverse=False)     
                        self.fronts[f][0].crowding_distance = math.inf
                        self.fronts[f][len(self.fronts[f])-1].crowding_distance = math.inf
                        maximum = max(front.fitness_functions.values[m] for front in self.fronts[f])
                        minimum = min(front.fitness_functions.values[m] for front in self.fronts[f])
                        FmaxFminDiff = maximum-minimum
                        if FmaxFminDiff == 0:
                            FmaxFminDiff = 1
                        for k in range(1,len(self.fronts[f])-1):
                            self.fronts[f][k].crowding_distance = self.fronts[f][k].crowding_distance + ((self.fronts[f][k+1].fitness_functions.values[m] - self.fronts[f][k-1].fitness_functions.values[m])/FmaxFminDiff)
    
    def get_new_population(self):
        new_population = pop.Population(ds=self.population.ds)
        t=0
        for f in range(0, len(self.fronts)):
            if (t+len(self.fronts[f])) < self.pop_size:
                for i in range(0,len(self.fronts[f])):
                    new_population.add_individual(self.fronts[f][i])
                    t+=1
            else:
                j=0
                last = self.fronts[f];
                last.sort(key=functools.cmp_to_key(compare_by_rank_crowdistance))
                while t < self.pop_size:
                    new_population.add_individual(last[j].copy())
                    t+=1
                    j+=1
                break
        self.population = new_population.copy()
        del new_population
        
    def get_first_front_as_population(self):
        population = pop.Population(self.ds)
        for sc in self.fronts[0]:
            population.add_individual(sc.copy())
        return population
    
    #@paralelizar
    def run_generation(self, pm, pc, g, e, gen_upd):
        #if g_surr >= 0:
        #    print("Dataset:", self.ds.dataset_name(),"- Ejecución:", e,"- Generación Model", g_surr+1,"- Generación NSGA2:", g+1)
        #else:
        print("Dataset:", self.ds.dataset_name(),"- Ejecución:", e,"- Generación NSGA2:", g+1)
        self.FastNonDominatedSort()
        self.get_crowding_distance()
        parents = self.population.tournament_selection()
        offsprings = parents.crossover(pc)
        offsprings.mutate(pm)
        self.no_evaluations += offsprings.evaluate(self.model)
        self.population.join(offsprings)
        self.FastNonDominatedSort()
        self.get_crowding_distance()
        self.get_new_population()
        if self.model.model_type > 0:
            if (g+1)%gen_upd==0:
                print("entro")
                for f in self.fronts[0]:
                    self.no_evaluations += f.evaluate()
                    self.model.instances.population.add_individual(f)
                self.model.update()
                if self.model.instances.population.size != self.pop_size*2:
                    print(self.model.instances.population.size)
                    #data, ff = f.to_vector(self.model.needsFilled)
                    #self.model.train = np.concatenate((self.model.train, np.array(data, dtype=float)), axis=0)
                    #self.model.classes = np.vstack((self.model.classes, np.array(ff, dtype=float)))
    
    #def execute(self, pm, pc, G_NSGA2, e, g_surr = -1):
    def execute(self, pm, pc, NG, e, no_upd = 1):
        self.population.create(self.pop_size)
        self.no_evaluations += self.population.evaluate(self.model)
        
        #Parallel(n_jobs=mp.cpu_count())(delayed(self.run_generation)(pm, pc, g, e, g_surr) for g in range(0,G_NSGA2))
        
        #pool = mp.Pool(processes=mp.cpu_count())
        #pool.starmap(self.run_generation, [(pm, pc, g, e, g_surr) for g in range(0,G_NSGA2)])
        gen_upd = math.floor(NG/no_upd)
        #print(gen_upd)
        for g in range(0,NG):
            #self.run_generation(pm, pc, g, e, g_surr)
            self.run_generation(pm, pc, g, e, gen_upd)
        return self.fronts[0]
