import sys
import os, time
import eMODiTS.Population as pop
import Surrogate.Model as model
from scipy.io import savemat
from dataset import Dataset
from EvolutionaryMethods.nsga2 import NSGA2
from utils import str_to_list

from multiprocessing import Pool, cpu_count

#import multiprocessing
from joblib import Parallel, delayed

''' pm = 0.2
    pc = 0.8
    G_NSGA2 = 10
    G_Surr = 1
    pop_size = 10
    NE = 1
    idfunctionsconf = 0
    model_type = 2 # 0=Real evaluation, 1=Subrogate by RBFNN, 2=KNNDTW
    n_neighbors = 1 '''

class Surrogate_eMODiTS:
    def __init__(self, pm=0.2, pc=0.8, NG=10, pop_size=10, NE=1, idfunctionsconf=0, model_type=2, no_updates = 5): #, n_neighbors=1):
        self.no_evaluations = 0
        self.pm = pm
        self.pc = pc
        self.NG = NG
        self.pop_size = pop_size
        self.NE = NE
        self.idfunctionsconf = idfunctionsconf
        self.model_type = model_type
        #self.n_neighbors = n_neighbors
        self.surrogate_model = None
        self.train_size = pop_size * 2
        self.no_upd = no_updates
        #self.surrogate_model = model.Model(model_type=model_type, n_neighbors=n_neighbors)
        
    def execute_surrogate(self, iDS):
        ds = Dataset(iDS, '_TRAIN', False)
        AccumulatedFront = NSGA2(ds)
        #results_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))+"/java/Postdoctorado/Results/e"+str(self.NE)+"p"+str(self.pop_size)+"g"+str(self.NG)+"_surr"
        results_dir = os.path.dirname(os.path.realpath(__file__))+"/Results/e"+str(self.NE)+"p"+str(self.pop_size)+"g"+str(self.NG)+"_u"+str(self.no_upd)+"/MODiTS/"+ds.dataset_name()
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        start_time = time.time()
        
        for e in range(0,self.NE):
            population = pop.Population(ds, pop_size=self.train_size, idfunctionsconf=self.idfunctionsconf)
            self.surrogate_model = model.Model(ds, model_type=self.model_type) #, n_neighbors=self.n_neighbors)
            self.no_evaluations += population.evaluate()
            #surrogate = model.Model(model_type=model_type, n_neighbors=n_neighbors)
            #surrogate.create(population)
            self.surrogate_model.create(population)
            #for g in range(0,G_Surr):
            nsga2 = NSGA2(ds,pop_size=self.pop_size, idfunctionsconf=self.idfunctionsconf, model=self.surrogate_model)
            front = nsga2.execute(self.pm, self.pc, self.NG, e+1, self.no_upd)
            self.no_evaluations += nsga2.no_evaluations
            for f in front:
                AccumulatedFront.population.add_individual(f)
            #        if model_type > 0:
            #            f.evaluate()
            #            data, ff = f.to_vector(surrogate.needsFilled)
            #            surrogate.train = np.concatenate((surrogate.train, np.array(data, dtype=float)), axis=0)
            #            surrogate.classes = np.vstack((surrogate.classes, np.array(ff, dtype=float)))

        AccumulatedFront.FastNonDominatedSort()
        mat = AccumulatedFront.get_first_front_as_population().export_matlab()
        mat["time"] = int(time.time() - start_time) * 1000 #Milisegundos
        mat["evaluations"] = self.no_evaluations
        savemat(results_dir+"/"+ds.dataset_name()+"_MODiTS.mat", mat)
    
if __name__ == "__main__":
    gens = int(sys.argv[1])
    popsize = int(sys.argv[2])
    exes = int(sys.argv[3])
    ids = str_to_list(sys.argv[4])
    no_updates = int(sys.argv[5])
    #n_jobs = multiprocessing.cpu_count()
    surrogate = Surrogate_eMODiTS(pm=0.2,pc=0.8,NG=gens,pop_size=popsize,NE=exes,idfunctionsconf=0,model_type=2, no_updates = no_updates) #,n_neighbors=1)
    Parallel(n_jobs=cpu_count())(delayed(surrogate.execute_surrogate)(i) for i in ids)
    #for i in ids:
    #    surrogate.execute_surrogate(i)