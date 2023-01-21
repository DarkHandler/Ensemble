############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# Modified by: Sebastian Medina
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Crow Search Algorithm

# PEREIRA, V. (2022). GitHub repository: https://github.com/Valdecy/pyMetaheuristic
#https://github.com/Valdecy/pyMetaheuristic/blob/main/pyMetaheuristic/algorithm/csa.py

############################################################################

# Required Libraries
import numpy  as np
import random
import os
from solution import solution
import time
import math

import OptModel.teamSizeModel as teamSizeModel

import metrics.metrics as mtclass

class CSA:

    def __init__(self, objf, lb, ub, dim, population_size, Max_iter):
        self.objf = objf
        self.lb = lb
        self.ub = ub
        self.dim = dim
        self.population_size= population_size
        self.Max_iter = Max_iter

    ############################################################################
    # Function: Initialize Variables
    def initial_population(self, arrayFitness, proyectSize):
        population = np.zeros((self.population_size, self.dim))
        fitness = None
        best_ind = []
        for i in range(0, self.population_size):
            if (self.objf.__name__ != "teamSizeModel"):
                for j in range(0, self.dim):
                    population[i,j] = random.uniform(self.lb[j], self.ub[j]) 
            else: 
                population = teamSizeModel.initPopTeamSizeModel(self.population_size, self.lb[0], self.ub[0], proyectSize, self.dim)
                population = np.array(population)
            arrayFitness[i] = self.objf(population[i]) #save fitness
            if fitness == None or fitness > arrayFitness[i]:
                fitness = arrayFitness[i]
                best_ind = population[i] 
        return population, best_ind, fitness

    ############################################################################

    # Function: Update Position
    def update_position(self, population, ap, fL, arrayFitness, best_ind, fitness, proyectSize):
        for i in range(0, population.shape[0]):
            idx  = [i for i in range(0, population.shape[0])]
            idx.remove(i)
            idx  = random.choice(idx)
            rand = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
            for j in range(0, self.dim):
                if (rand >= ap):
                    rand_i = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
                    if (self.objf.__name__ == "teamSizeModel"):
                        newValue = population[i, j] + rand_i*fL*( population[idx, j] - population[i, j])
                        population[i, j] = np.clip(newValue, self.lb[0], self.ub[0])
                        population = list(population)
                        teamSizeModel.checkBoundaries(population[i], proyectSize, self.lb[0], self.ub[0])
                        population = np.array(population)
                    else:
                        population[i, j] = np.clip(population[i, j] + rand_i*fL*( population[idx, j] - population[i, j]), self.lb[j], self.ub[j]) #check boundaries
                else:
                    population[i,j] = random.uniform(self.lb[j], self.ub[j]) 
                    if (self.objf.__name__ == "teamSizeModel"):
                        population = list(population)
                        teamSizeModel.checkBoundaries(population[i], proyectSize, self.lb[0], self.ub[0])
                        population = np.array(population)
            
            arrayFitness[i] = self.objf(population[i]) #save fitness
            if fitness == None or fitness > arrayFitness[i]:
                fitness = arrayFitness[i]
                best_ind = population[i] 
        return population, best_ind, fitness

    ############################################################################

    # Function: Crow Search Algorithm
    def optimize(self):

        metrics = mtclass.Metrics() #objeto de metricas
        
        #initial values
        ap = 0.02
        fL = 0.02
        count = 0
        proyectSize = -1

        if not isinstance(self.lb, list):
            self.lb = [self.lb] * self.dim
        if not isinstance(self.ub, list):
            self.ub = [self.ub] * self.dim

        if self.objf.__name__ == "teamSizeModel": #si es el problema que estamos resolviendo de team size model
            proyectSize = self.dim
            self.dim = math.floor(self.dim/3) #no es necesario debido a que si no se estaran creando obligatoriamente variables con rango 3 a 17 
        
        arrayFitness = np.zeros(self.population_size)

        #Initialization population
        population, best_ind, fitness = self.initial_population(arrayFitness, proyectSize)

        # Initialize convergence
        convergence_curve = np.zeros(self.Max_iter)
        Percent_explorations = np.zeros(self.Max_iter)

        ############################
        s = solution()

        print('CSA is optimizing  "' + self.objf.__name__ + '"')

        timerStart = time.time()
        s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
        ############################

        #Main loop
        while (count < self.Max_iter):  
            if (False):
                print('Iteration = ', count, ' f(x) = ', fitness)  
            ## --------- DIVERSITY ZONE ----------
            metrics.calculateDiversity(population, self.population_size, self.dim, self.objf)
            Percent_explorations[count] = metrics.percent_exploration

            population, best_ind, fitness = self.update_position(population, ap, fL, arrayFitness, best_ind, fitness, proyectSize)

            convergence_curve[count] = fitness
            count = count + 1 

        print(fitness)

        timerEnd = time.time()
        s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
        s.executionTime = timerEnd - timerStart
        s.convergence = convergence_curve
        s.percent_explorations = Percent_explorations
        s.optimizer = "CSA"
        s.objfname = self.objf.__name__
        s.best = fitness
        s.bestIndividual = best_ind

        return s
    ############################################################################




