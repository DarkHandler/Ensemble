# -*- coding: utf-8 -*-
"""
Created on Thu May 26 02:00:55 2016

@author: hossam
@modified by: Sebastian Medina
"""
import math
import random
import time
from solution import solution

import numpy as np
import OptModel.teamSizeModel as teamSizeModel

import metrics.metrics as mtclass

class BA:

    def __init__(self, objf, lb, ub, dim, N, Max_iteration):
        self.objf = objf
        self.lb = lb
        self.ub = ub
        self.dim = dim
        self.N= N
        self.Max_iteration = Max_iteration


    def optimize(self):

        metrics = mtclass.Metrics() #objeto de metricas

        n = self.N
        # Population size

        if not isinstance(self.lb, list):
            self.lb = [self.lb] * self.dim
        if not isinstance(self.ub, list):
            self.ub = [self.ub] * self.dim
        N_gen = self.Max_iteration  # Number of generations

        A = 0.5
        # Loudness  (constant or decreasing)
        r = 0.5
        # Pulse rate (constant or decreasing)

        Qmin = 0  # Frequency minimum
        Qmax = 2  # Frequency maximum


        if self.objf.__name__ == "teamSizeModel": #si es el problema que estamos resolviendo de team size model
            proyectSize = self.dim
            self.dim = math.floor(self.dim/3) #no es necesario debido a que si no se estaran creando obligatoriamente variables con rango 3 a 17     
        d = self.dim  # Number of dimensions

        # Initializing arrays
        Q = np.zeros(n)  # Frequency
        v = np.zeros((n, d))  # Velocities

        Convergence_curve = []
        Percent_explorations = np.zeros(self.Max_iteration)

        # Initialize the population/solutions
        if (self.objf.__name__ != "teamSizeModel"):
            Sol = np.zeros((n, d))
            for i in range(self.dim):
                Sol[:, i] = np.random.rand(n) * (self.ub[i] - self.lb[i]) + self.lb[i]
        else:
            Sol = teamSizeModel.initPopTeamSizeModel(n, self.lb[0], self.ub[0], proyectSize, self.dim)
            Sol = np.array(Sol)

        S = np.zeros((n, d))
        S = np.copy(Sol)
        Fitness = np.zeros(n)

        # initialize solution for the final results
        s = solution()
        print('BAT is optimizing  "' + self.objf.__name__ + '"')

        # Initialize timer for the experiment
        timerStart = time.time()
        s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

        # Evaluate initial random solutions
        for i in range(0, n):
            Fitness[i] = self.objf(Sol[i])

        # Find the initial best solution and minimum fitness
        I = np.argmin(Fitness)
        best = Sol[I]
        fmin = min(Fitness)
        
        # Main loop
        for t in range(0, N_gen):

            # Loop over all bats(solutions)
            for i in range(0, n):
                Q[i] = Qmin + (Qmin - Qmax) * random.random()
                if (self.objf.__name__ == "teamSizeModel"):
                    v[i, :] = np.absolute(v[i, :] + (Sol[i, :] - best) * Q[i])
                else:
                    v[i, :] = v[i, :] + (Sol[i, :] - best) * Q[i]
                S[i, :] = Sol[i, :] + v[i, :] #en este momento se obtienen numeros negativos que hay que revisa :/

                
                # Check boundaries
                if (self.objf.__name__ == "teamSizeModel"):
                    Sol = list(Sol)
                    teamSizeModel.checkBoundaries(Sol[i], proyectSize, self.lb[0], self.ub[0])
                    Sol = np.array(Sol)
                else:
                    for j in range(d):
                        Sol[i, j] = np.clip(Sol[i, j], self.lb[j], self.ub[j])

                # Pulse rate
                if random.random() > r:
                    if (self.objf.__name__ == "teamSizeModel"):
                        S[i, :] = np.absolute(best + 0.001 * np.random.randn(d))
                        S = list(S)
                        teamSizeModel.checkBoundaries(S[i], proyectSize, self.lb[0], self.ub[0])
                        S = np.array(S)
                    else:
                        S[i, :] = best + 0.001 * np.random.randn(d)


                # Evaluate new solutions
                Fnew = self.objf(S[i, :])

                # Update if the solution improves
                if (Fnew <= Fitness[i]) and (random.random() < A):
                    Sol[i, :] = np.copy(S[i, :])
                    Fitness[i] = Fnew

                # Update the current best solution
                if Fnew <= fmin:
                    best = np.copy(S[i, :])
                    fmin = Fnew
            
            ## --------- DIVERSITY ZONE ----------
            metrics.calculateDiversity(S, n, self.dim, self.objf)
            Percent_explorations[t] = metrics.percent_exploration

            # update convergence curve
            Convergence_curve.append(fmin)

            #if t % 1 == 0:
            #    print(["At iteration " + str(t) + " the best fitness is " + str(fmin)])
        print(fmin)
        timerEnd = time.time()
        s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
        s.executionTime = timerEnd - timerStart
        s.convergence = Convergence_curve
        s.percent_explorations = Percent_explorations
        s.optimizer = "BA"
        s.objfname = self.objf.__name__

        return s
