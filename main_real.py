import sys, os, time
#sys.path.append(os.path.abspath(os.path.join('..', 'python')))
import eMODiTS.Scheme as sc
import eMODiTS.Population as pop
from Surrogate.Model import Model
import numpy as np
from scipy.io import savemat
from dataset import Dataset
from Functions.confusion_matrix import ConfusionMatrix
from Functions.fitness_functions import FitnessFunction
from EvolutionaryMethods.nsga2 import NSGA2

import multiprocessing
from joblib import Parallel, delayed

''' import cProfile, pstats, io
from pstats import SortKey '''

''' pm = 0.2
        pc = 0.8
        G = 10
        pop_size = 10
        NE = 1
        idfunctionsconf = 0 '''

class eMODiTS:
    def __init__(self, pm=0.2, pc=0.8, NG=10, pop_size=10, NE=1, idfunctionsconf=0):
        self.no_evaluations = 0
        self.pm = pm
        self.pc = pc
        self.NG = NG
        self.pop_size = pop_size
        self.NE = NE
        self.idfunctionsconf = idfunctionsconf

    def execute_real(self, iDS):
        ds = Dataset(iDS, '_TRAIN', False)
        AccumulatedFront = NSGA2(ds)
        results_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))+"/java/Postdoctorado/Results/e"+str(self.NE)+"p"+str(self.pop_size)+"g"+str(self.NG)
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        start_time = time.time()

        """ pr = cProfile.Profile()
        pr.enable() """

        for e in range(0,self.NE):
            model = Model(model_type=0, n_neighbors=0)
            nsga2 = NSGA2(ds,pop_size=self.pop_size, idfunctionsconf=self.idfunctionsconf, model=model)
            front = nsga2.execute(self.pm, self.pc, self.NG, e+1)
            self.no_evaluations += nsga2.no_evaluations
            for f in front:
                AccumulatedFront.population.add_individual(f)

        AccumulatedFront.FastNonDominatedSort()
        mat = AccumulatedFront.get_first_front_as_population().export_matlab()
        mat["time"] = int(time.time() - start_time) * 1000 #Milisegundos
        mat["evaluations"] = self.no_evaluations
        savemat(results_dir+"/"+ds.dataset_name()+"_MODiTS.mat", mat)

#ds = [1,3,7,10,20,25,27,28,31,40,41,51]
ds = [10]

emodits = eMODiTS(pm=0.2, pc=0.8, NG=300, pop_size=100, NE=15, idfunctionsconf=0)

n_jobs = multiprocessing.cpu_count()
Parallel(n_jobs=n_jobs)(delayed(emodits.execute_real)(i) for i in ds)

#for i in ds:
#    execute_real(i, pm=0.2, pc=0.8, G=300, pop_size=100, NE=15, idfunctionsconf=0)