import itertools
import os
import random
import sys
import math
import collections
import numpy as np
import scipy.io
import pandas as pd

sys.path.append(os.path.abspath(os.path.join('.', 'src')))

import Settings as conf

sys.path.append(os.path.abspath(os.path.join('.', 'src/Functions')))
import Functions.confusion_matrix as cm
import Functions.fitness_functions as ff
from utils import normalize_matrix, to_paired_value

class Scheme:
    def __init__(self, ds, cuts = {}, filename="", idfunctionsconf=0):
        self.cuts = cuts.copy()
        self.error_rate = 1000000000000.0
        
        self.rank = -1
        self.crowding_distance = 0
        #self.front = -1
        self.domination_count = 0
        self.dominated_set = set()
        
        self.ds = ds
        self.ds_discrete_ints = []
        self.ds_discrete_strings = []
        self.ds_discrete_reconstructed = []
        self.idfunctionsconf = idfunctionsconf
        self.fitness_functions = ff.FitnessFunction(idfunctionsconf)
        self.confusion_matrix = cm.ConfusionMatrix(str_discrete=[], bd=self.ds)
        
        self._max_array_size = int(conf.MAX_NUMBER_OF_WORD_CUTS*self.ds.dimensions[1])*(conf.MAX_NUMBER_OF_ALPHABET_CUTS-1)
        if not self.cuts:
            if filename == "":
                self.random_load()
            else:
                self.load_from_matlab(filename)
    
    def reset(self):
        self.cuts = {}
        self.error_rate = 1000000000000.0
        self.rank = -1
        self.crowding_distance = 0
        #self.front = -1
        self.domination_count = 0
        self.dominated_set = set()
        self.ds_discrete_ints = []
        self.ds_discrete_strings = []
        self.ds_discrete_reconstructed = []
        self.fitness_functions = ff.FitnessFunction(self.idfunctionsconf)
        self.confusion_matrix = cm.ConfusionMatrix(str_discrete=[], bd=self.ds)
    
    def create_alphs_cuts(self):
        num_alph = random.randint(conf.MIN_NUMBER_OF_ALPHABET_CUTS, conf.MAX_NUMBER_OF_ALPHABET_CUTS-1)
        alphs = {random.uniform(self.ds.limites[0], self.ds.limites[1]) for i in range(num_alph)}
        alphs_cuts = to_paired_value(sorted(list(alphs.union({self.ds.limites[0], self.ds.limites[1]}))))
        return alphs_cuts
        
    def random_load(self):
        num_cuts = random.randint(conf.MIN_NUMBER_OF_WORD_CUTS, int(conf.MAX_NUMBER_OF_WORD_CUTS*self.ds.dimensions[1]))
        #word_cuts = set()
        wordcuts = {random.randint(1, self.ds.dimensions[1]-1) for i in range(num_cuts)}
        #for _ in range(0, num_cuts):
        #    word_cuts.add(random.randint(1, self.ds.dimensions[1]-1))
        word_cuts = to_paired_value(sorted(list(wordcuts.union({1,self.ds.dimensions[1]-1}))))
        
        self.cuts = {str(word_cuts[i]): self.create_alphs_cuts() for i in range(0,len(word_cuts))}
        
        """ for my_key in word_cuts:
            num_alph = random.randint(conf.MIN_NUMBER_OF_ALPHABET_CUTS, conf.MAX_NUMBER_OF_ALPHABET_CUTS-1)
            #alphs = set()
            #for _ in range(0, num_alph):
            #    alphs.add(random.uniform(self.ds.limites[0], self.ds.limites[1]))
            alphs = {random.uniform(self.ds.limites[0], self.ds.limites[1]) for i in range(num_alph)}
            alphs_cuts = to_paired_value(sorted(list(alphs.union({self.ds.limites[0], self.ds.limites[1]}))))
            self.cuts[str(my_key)] = alphs_cuts """
            
    def load_from_matlab(self, filename):
        sch = scipy.io.loadmat(filename)
        wordcuts = set(sch['cuts'][:,0])
        word_cuts = to_paired_value(sorted(list(wordcuts.union({1,self.ds.dimensions[1]-1}))))
        alphs = sch['cuts'][0,1:len(sch['cuts'])]
        for i in range(0,len(word_cuts)):
            alphs_cuts = sch['cuts'][i,1:len(sch['cuts'][i])]
            alphs = set(alphs_cuts[~np.isnan(alphs_cuts)])
            alphscuts = to_paired_value(sorted(list(alphs.union({self.ds.limites[0], self.ds.limites[1]+1}))))
            self.cuts[str(word_cuts[i])] = list(alphscuts)
            
    def extract_data(self):
        inits = []
        ends = []
        alphs = []
        cuts = list(self.cuts.keys())
        for cut in cuts:
            alphs.append(self.cuts[str(cut)])
            interval = cut.replace("["," ").replace("]"," ")
            interval = interval.split(",")
            inits.append(int(float(interval[0])))
            ends.append(int(float(interval[1])))
        return inits, ends, alphs
    
    def cutdiffs(self):
        diffs = []
        for wordcuts, alph_cuts in self.cuts.items():
            interval = wordcuts.replace("["," ").replace("]"," ")
            interval = interval.split(",")
            u = int(float(interval[1]))
            l = int(float(interval[0]))
            if l == 1:
                l-=1
            diffs.append(u - l)
        return diffs
    
    def discretize(self):
        discrete = []
        inits, ends, alphs = self.extract_data()
        discrete = []
        for i in range(0,len(self.ds.data)):
            row_discr = []
            row_discr.append(int(self.ds.data[i,0]))
            for j in range(0,len(inits)):
                media = self.ds.data[i,range(inits[j],ends[j])].mean()
                #agrego = False #borrar
                for k in range(0,len(alphs[j])):
                    if alphs[j][k][0] <= media < alphs[j][k][1]:
                        row_discr.append(k+1)
                #        agrego = True #borrar
                if media >= alphs[j][len(alphs[j])-1][1]:
                    row_discr.append(len(alphs[j]))
                #    agrego = True #borrar
                if media < alphs[j][0][0]:
                    row_discr.append(1)
                #    agrego = True #borrar
                #if not agrego:
                #    print("media: ",media," k: ",k, " j: ",j," i: ",i)
                #    print("Aquí")
            discrete.append(row_discr)
        self.ds_discrete_ints = np.array(discrete)
        self.reconstruct()
        
        strings = []
        for i in range(0,len(self.ds_discrete_ints)):
            string = ""
            for j in range(1,len(self.ds_discrete_ints[i,:])):
                string += conf.LETTERS[self.ds_discrete_ints[i,j]]
            strings.append([self.ds_discrete_ints[i,0],string])
        self.ds_discrete_strings = np.array(strings)
        self.confusion_matrix.create(self.ds_discrete_strings)
    
    def reconstruct(self):
        self.ds_discrete_ints = np.array(self.ds_discrete_ints)
        diffs = self.cutdiffs()
        reconstructed = np.empty([len(self.ds.data), len(self.ds.data[0,:])], dtype='uint8')
        for i in range(0,len(self.ds_discrete_ints)):
            reconstructed_row = []
            try:
                reconstructed[i,0] = self.ds_discrete_ints[i,0]
            except IndexError:
                df = pd.DataFrame(self.ds_discrete_ints).T
                df.to_excel(excel_writer = "/Users/scoramg/Dropbox/Escolaridad/Postdoctorado/python/errors.xlsx")
                print(len(self.ds_discrete_ints),len(self.ds.ds_discrete_ints[0,:]))
            for j in range(1,len(self.ds_discrete_ints[i,:])):
                reconstructed_row = reconstructed_row + list(itertools.repeat(self.ds_discrete_ints[i,j], diffs[j-1]))
            if len(reconstructed[i,1:]) != len(np.array(reconstructed_row)):
                print("i",i)
                print("len(reconstructed[i,1:]):", len(reconstructed[i,1:]), "len(np.array(reconstructed_row)):", len(np.array(reconstructed_row)))
            np.copyto(reconstructed[i,1:],np.array(reconstructed_row))  
        
        self.ds_discrete_reconstructed = normalize_matrix(reconstructed)  
        
    def copy(self):
        mycopy = Scheme(self.ds, idfunctionsconf=self.idfunctionsconf)
        mycopy.reset()
        mycopy.cuts = self.cuts.copy()
        mycopy.error_rate = self.error_rate
        mycopy.rank = self.rank
        mycopy.crowding_distance = self.crowding_distance
        mycopy.domination_count = self.domination_count
        mycopy.dominated_set = self.dominated_set.copy()
        mycopy.ds_discrete_ints = self.ds_discrete_ints.copy()
        mycopy.ds_discrete_strings = self.ds_discrete_strings.copy()
        mycopy.ds_discrete_reconstructed = self.ds_discrete_reconstructed.copy()
        mycopy.fitness_functions = self.fitness_functions.copy()
        mycopy.confusion_matrix = self.confusion_matrix.copy()
        return mycopy
    
    def evaluate(self, model=None):
        self.fitness_functions.values = []
        no_eval = 0
        if not model:
            self.discretize()
            self.confusion_matrix.create(self.ds_discrete_strings)
            self.fitness_functions.evaluate(self)  
            no_eval = 1
        else:
            data, ff = self.to_vector(model.needsFilled)
            if len(model.models) > 3:
                print("Aqui")
            for i in range(0,len(model.models)):
                y_pred = model.models[i].predict_row(np.array(data, dtype=float))
                self.fitness_functions.values.append(y_pred)
            no_eval = 0
        return no_eval
                
    def mutate_alphs(self, alphs, pm):
        new_alphs = []
        for alph in alphs:
            alph_inits = []
            for interval in alph:
                alph_inits.append(interval[0])
            #print("before:",alph_inits)
            alphs_mut = set()
            alphs_mut.add(alph_inits[0])
            for i in range(1, len(alph_inits)):
                if random.random() > pm:
                    alphs_mut.add(random.uniform(self.ds.limites[0], self.ds.limites[1]))
                else:
                    alphs_mut.add(alph_inits[i])
            #print("after:", sorted(list(alphs_mut)))
            new_alphs.append(to_paired_value(sorted(list(alphs_mut.union({self.ds.limites[0], self.ds.limites[1]+1})))))
        return new_alphs
    
    def mutate(self, pm):
        inits, _, alphs = self.extract_data()
        for j in range(1, len(inits)):
            porc = random.random()
            if (porc <= pm) and (len(inits)<self.ds.dimensions[1]):
                try:
                    new_value = random.choice([i for i in range(conf.MIN_NUMBER_OF_WORD_CUTS, self.ds.dimensions[1]) if i not in inits])
                    inits[j] = new_value
                except IndexError:
                    print("Error: please provide a name and a repeat count to the script.")
        alphs_mutate = self.mutate_alphs(alphs, pm)
        res = {inits[i]: alphs_mutate[i] for i in range(0,len(inits))}
        new_res = collections.OrderedDict(sorted(res.items()))
        new_alphs = list(new_res.values())
        if 1 not in list(new_res.keys()):
            print("Aqui")
        new_cuts = to_paired_value(sorted(list(set(new_res.keys()).union({self.ds.dimensions[1]-1}))))
        cuts_mutate = {str(new_cuts[i]): new_alphs[i] for i in range(0,len(new_cuts))}
        self.cuts = cuts_mutate.copy()
    
    def crossover(self, parent):
        inits1, _, alphs1 = self.extract_data()
        parent1 = {inits1[i]: alphs1[i] for i in range(0,len(inits1))}
        
        inits2, _, alphs2 = parent.extract_data()
        parent2 = {inits2[i]: alphs2[i] for i in range(0,len(inits2))}

        cut1 = random.randint(1,len(parent1))
        cut2 = random.randint(1,len(parent2))
        # print(cut1, cut2)

        off1_items = collections.OrderedDict(sorted(list(parent1.items())[:cut1] + list(parent2.items())[cut2:]))
        off2_items = collections.OrderedDict(sorted(list(parent2.items())[:cut2] + list(parent1.items())[cut1:]))
        
        if 1 not in list(off1_items.keys()):
            print("Aqui")
        
        if 1 not in list(off2_items.keys()):
            print("Aqui")

        new_cuts1 = to_paired_value(sorted(list(set(off1_items.keys()).union({self.ds.dimensions[1]-1}))))
        new_alphs1 = list(off1_items.values())
        new_cuts2 = to_paired_value(sorted(list(set(off2_items.keys()).union({parent.ds.dimensions[1]-1}))))
        new_alphs2 = list(off2_items.values())
        
        if (len(new_cuts1) >= conf.MIN_NUMBER_OF_WORD_CUTS) and (len(new_cuts2) >= conf.MIN_NUMBER_OF_WORD_CUTS):
            off1_cuts = {}
            off2_cuts = {}
            
            off1_cuts = {str(new_cuts1[i]): new_alphs1[i] for i in range(0,len(new_cuts1))}
            off2_cuts = {str(new_cuts2[i]): new_alphs2[i] for i in range(0,len(new_cuts2))}
            
            off1 = Scheme(ds=self.ds, cuts=off1_cuts.copy(), idfunctionsconf=self.idfunctionsconf)
            off2 = Scheme(ds=self.ds, cuts=off2_cuts.copy(), idfunctionsconf=self.idfunctionsconf)
            
            return off1, off2
        else:
            return self.crossover(parent)
    
    def dominates(self, compared):
        dominate1 = 0 
        dominate2 = 0

        flag = 0

        for i in range(0,len(self.fitness_functions.values)):
            if self.fitness_functions.values[i] < compared.fitness_functions.values[i]:
                flag = -1
            elif self.fitness_functions.values[i] > compared.fitness_functions.values[1]:
                flag = 1
            else:
                flag = 0
                
            if flag == -1:
                dominate1 = 1
            
            if flag == 1:
                dominate2 = 1

        if dominate1 == dominate2:
            dominance = False

        if dominate1 == 1:
            dominance = True #f1 dominates f2
            
        if dominate2 == 1:
            dominance = False
        
        return dominance
    
    def matlab_format(self):
        _, ends, alphs = self.extract_data()
        list_len = [len(i) for i in list(alphs)]
        maxsize = max(list_len)+1
        mat_array = []
        for i in range(0, len(ends)):
            arr = []
            arr.append(ends[i])
            for j in range(0,len(alphs[i])):
                arr.append(alphs[i][j][0])
            diff = maxsize-len(arr)
            for k in range(0,diff):
                arr.append(float("nan"))
            mat_array.append(arr)
        return mat_array, self.fitness_functions.values
    
    def to_vector(self, filled=True):
        lista = []
        if filled:
            lista = [-1000000]*self._max_array_size
        else:
            lista = []
        _, ends, alphs = self.extract_data()
        for i in range(len(ends)):
            lista.append(ends[i])
            for j in range(1,len(alphs[i])):
                lista.append(alphs[i][j][0])
        return lista, self.fitness_functions.values
    