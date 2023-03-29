import random
import time
from solution import solution
import math

import numpy as np
import OptModel.teamSizeModel as teamSizeModel

import metrics.metrics as mtclass

"""
Created on Thu May 26 02:00:55 2016

@modified by: Sebastian Medina
"""

# Differential Evolution (DE)
# mutation factor = [0.5, 2]
# crossover_ratio = [0,1]

class DE:

    def __init__(self, objf, lb, ub, dim, PopSize, iters):
        self.objf = objf
        self.lb = lb
        self.ub = ub
        self.dim = dim
        self.PopSize= PopSize
        self.iters = iters


    def optimize(self):

        metrics = mtclass.Metrics() #objeto de metricas

        mutation_factor = 0.5
        crossover_ratio = 0.7
        stopping_func = None

        if self.objf.__name__ == "teamSizeModel": #si es el problema que estamos resolviendo de team size model
            proyectSize = self.dim
            self.dim = math.floor(self.dim/3) #no es necesario debido a que si no se estaran creando obligatoriamente variables con rango 3 a 17 

        # convert lb, ub to array
        if not isinstance(self.lb, list):
            self.lb = [self.lb for _ in range(self.dim)]
            self.ub = [self.ub for _ in range(self.dim)]

        # solution
        s = solution()

        s.best = float("inf")

        # initialize population
        population = []

        population_fitness = np.array([float("inf") for _ in range(self.PopSize)])
        
        if (self.objf.__name__ != "teamSizeModel"):
            for p in range(self.PopSize):
                sol = []
                for d in range(self.dim):
                    d_val = random.uniform(self.lb[d], self.ub[d])
                    sol.append(d_val)

                population.append(sol)
        else: 
            population = teamSizeModel.initPopTeamSizeModel(self.PopSize, self.lb[0], self.ub[0], proyectSize, self.dim)
        population = np.array(population)
        
        # calculate fitness for all the population
        for i in range(self.PopSize):
            fitness = self.objf(population[i, :])
            population_fitness[i] = fitness
            # s.func_evals += 1

            # is leader ?
            if fitness < s.best:
                s.best = fitness
                s.leader_solution = population[i, :]

        convergence_curve = np.zeros(self.iters)
        Percent_explorations = np.zeros(self.iters)

        # start work
        print('DE is optimizing  "' + self.objf.__name__ + '"')

        timerStart = time.time()
        s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

        t = 0
        while t < self.iters:
            # should i stop
            if stopping_func is not None and stopping_func(s.best, s.leader_solution, t):
                break

            # loop through population
            for i in range(self.PopSize):
                # 1. Mutation

                # select 3 random solution except current solution
                ids_except_current = [_ for _ in range(self.PopSize) if _ != i]
                id_1, id_2, id_3 = random.sample(ids_except_current, 3)

                mutant_sol = []
                for d in range(self.dim):
                    d_val = population[id_1, d] + mutation_factor * (
                        population[id_2, d] - population[id_3, d]
                    )

                    # 2. Recombination
                    rn = random.uniform(0, 1)
                    if rn > crossover_ratio:
                        d_val = population[i, d]

                    # add dimension value to the mutant solution
                    mutant_sol.append(d_val)

                # 3. Replacement / Evaluation

                # clip new solution (mutant)
                if (self.objf.__name__ == "teamSizeModel"):
                    teamSizeModel.checkBoundaries(mutant_sol, proyectSize, self.lb[0], self.ub[0])
                else:
                    mutant_sol = np.clip(mutant_sol, self.lb, self.ub)

                # calc fitness
                mutant_fitness = self.objf(mutant_sol)
                # s.func_evals += 1

                # replace if mutant_fitness is better
                if mutant_fitness < population_fitness[i]:
                    population[i, :] = mutant_sol
                    population_fitness[i] = mutant_fitness

                    # update leader
                    if mutant_fitness < s.best:
                        s.best = mutant_fitness
                        s.leader_solution = mutant_sol

            ## --------- DIVERSITY ZONE ----------
            metrics.calculateDiversity(population, self.PopSize, self.dim, self.objf)
            Percent_explorations[t] = metrics.percent_exploration

            convergence_curve[t] = s.best
            #if t % 1 == 0:
            #    print(["At iteration " + str(t + 1) + " the best fitness is " + str(s.best)])

            # increase iterations
            t = t + 1

        print(s.best)
        timerEnd = time.time()
        s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
        s.executionTime = timerEnd - timerStart
        s.convergence = convergence_curve
        s.percent_explorations = Percent_explorations
        s.optimizer = "DE"
        s.objfname = self.objf.__name__
        s.bestIndividual = s.leader_solution 

        # return solution
        return s
