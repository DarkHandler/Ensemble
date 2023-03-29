# -*- coding: utf-8 -*-
"""
Created on November 2022

@author: Hossam Faris
@modified by: Sebastian Medina
"""

import numpy as np
import math
from solution import solution
import time
import methods.GAoperators as GAoperators
import methods.adaptativeParam as adaptativeParam
import OptModel.teamSizeModel as teamSizeModel

import metrics.metrics as mtclass


class GWO:
    def __init__(self, objf, lb, ub, dim, SearchAgents_no, Max_iter, method):
        self.objf = objf
        self.lb = lb
        self.ub = ub
        self.dim = dim
        self.SearchAgents_no = SearchAgents_no 
        self.Max_iter = Max_iter
        self.method = method

    def optimize(self, fileNameMetrics = False, minPercentExT = -1): #enfocado en la minimizacion por defecto

        if (type(fileNameMetrics) == str) and (minPercentExT < 0 or minPercentExT > 100):
            raise Exception("minPercentExt must be a number between [0 - 100]")

        metrics = mtclass.Metrics() #objeto de metricas

        proyectSize = 0
        if self.objf.__name__ == "teamSizeModel": #si es el problema que estamos resolviendo de team size model
            proyectSize = self.dim
            self.dim = math.floor(self.dim/3) #no es necesario debido a que si no se estaran creando obligatoriamente variables con rango 3 a 17 

        #array of fitness, size of search agents
        arrayPopFitness = np.zeros(self.SearchAgents_no)

        # initialize alpha, beta, and delta_pos
        Alpha_pos = np.zeros(self.dim)
        Alpha_score = float("inf")

        Beta_pos = np.zeros(self.dim)
        Beta_score = float("inf")

        Delta_pos = np.zeros(self.dim)
        Delta_score = float("inf")

        if not isinstance(self.lb, list):
            self.lb = [self.lb] * self.dim
        if not isinstance(self.ub, list):
            self.ub = [self.ub] * self.dim

        # Initialize the positions of search agents

        if (self.objf.__name__ != "teamSizeModel"):
            Positions = np.zeros((self.SearchAgents_no, self.dim))
            for i in range(self.dim):
                Positions[:, i] = (
                    (np.random.uniform(0, 1, self.SearchAgents_no) * (self.ub[i] - self.lb[i]) + self.lb[i])
                )
        else:
            Positions = []

            for i in range(0, self.SearchAgents_no):
                wolfDimensionPosition = []
                teamSize = np.random.randint(self.lb[0], self.ub[0])

                if teamSize < proyectSize:
                    wolfDimensionPosition.append(teamSize)
                    flag = 0
                    sumOfWolfDimenPos = sum(wolfDimensionPosition)
                    while flag != 1:
                            teamSize = np.random.randint(self.lb[0], self.ub[0])
                            summatory = sumOfWolfDimenPos + teamSize
                            if summatory > proyectSize:
                                newValue = proyectSize - sumOfWolfDimenPos
                                if newValue >= self.lb[0] and newValue <= (self.ub[0]-1):
                                    wolfDimensionPosition.append(newValue)
                                flag = 1
                            elif sumOfWolfDimenPos == proyectSize:
                                flag = 1
                            else:
                                wolfDimensionPosition.append(teamSize)
                            sumOfWolfDimenPos = sum(wolfDimensionPosition)
                    
                    if (len(wolfDimensionPosition) != self.dim): #si falta rellenar espacio para alcanzar el tamanio de la dimension, agregar zeros
                        rest = self.dim - len(wolfDimensionPosition)
                        for n in range(0, rest):
                            wolfDimensionPosition.append(0)
                #print(wolfDimensionPosition)
                Positions.append(wolfDimensionPosition) 
                #print("Population: ", Positions[i], len(Positions[i]))

        Convergence_curve = np.zeros(self.Max_iter)
        Percent_explorations = np.zeros(self.Max_iter)
        s = solution()

        # Loop counter
        print('GWO is optimizing  "' + self.objf.__name__ + '"')

        timerStart = time.time()
        s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
        # Main loop
        for l in range(0, self.Max_iter):
            for i in range(0, self.SearchAgents_no):

                # Return back the search agents that go beyond the boundaries of the search space and restricction
                if (self.objf.__name__ == "teamSizeModel"):
                    teamSizeModel.checkBoundaries(Positions[i], proyectSize, self.lb[0], self.ub[0])
                else:
                    for j in range(self.dim):
                        Positions[i, j] = np.clip(Positions[i, j], self.lb[j], self.ub[j])   #esto devolvera las variables que hayan pasado de su rango al valor normal que debiesen tener                     


                # Calculate objective function for each search agent and update the fitness wolf
                fitness = self.objf(Positions[i])
                arrayPopFitness[i] = fitness

                # Update Alpha, Beta, and Delta
                if fitness < Alpha_score:
                    Delta_score = Beta_score  # Update delte
                    Delta_pos = Beta_pos.copy()
                    Beta_score = Alpha_score  # Update beta
                    Beta_pos = Alpha_pos.copy()
                    Alpha_score = fitness
                    # Update alpha
                    Alpha_pos = Positions[i].copy()

                if fitness > Alpha_score and fitness < Beta_score:
                    Delta_score = Beta_score  # Update delte
                    Delta_pos = Beta_pos.copy()
                    Beta_score = fitness  # Update beta
                    Beta_pos = Positions[i].copy()

                if fitness > Alpha_score and fitness > Beta_score and fitness < Delta_score:
                    Delta_score = fitness  # Update delta
                    Delta_pos = Positions[i].copy()

            

            ## --------- DIVERSITY ZONE ----------
            metrics.calculateDiversity(Positions, self.SearchAgents_no, self.dim, self.objf)
            Percent_explorations[l] = metrics.percent_exploration
            
            #metrics.showMetrics() #---------- MOSTRAR METRICASS ---------

            #"Adaptative Parameter", "GAOperators"
            if type(fileNameMetrics) == str: #if true
                metrics.storeMetricsIn(fileNameMetrics, l, fitness, self.SearchAgents_no, proyectSize, minPercentExT)

            ## POSITIONS UPDATE METHODS OF WOLFS
            if self.method == "GAOperators":
                GAoperators.updatePopWithGAMethod(self.lb, self.ub, Positions, arrayPopFitness, self.SearchAgents_no) #funcionando explotacion
            elif self.method == "Adaptative Parameter":
                adaptativeParam.adaptativeControlParameter_a_UpdateMethod(l, self.Max_iter, self.SearchAgents_no, Positions, Alpha_pos, Delta_pos, Beta_pos, self.objf.__name__) #funcionando exploracion
            else:
                adaptativeParam.linealUpdateMethod(self.SearchAgents_no, self.Max_iter, Positions, l, Alpha_pos, Delta_pos, Beta_pos, self.objf.__name__) #funcionando este es el clasico que hay que eliminar
            
            

            Convergence_curve[l] = Alpha_score

            #if l % 1 == 0:
            #print(["At iteration " + str(l) + " the best fitness is " + str(Alpha_score)])
            #print("Alfa position:",Alpha_pos, sum(Alpha_pos))
            #print("best:",objf([8,10,7,8,8,8,8,7,9,9,10,9,9,9,7,11,10,8,10,9,8,8,10]))

        print(Alpha_score)
        #print(Alpha_pos)

        timerEnd = time.time()
        s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
        s.executionTime = timerEnd - timerStart
        s.convergence = Convergence_curve
        s.percent_explorations = Percent_explorations
        s.optimizer = "GWO"
        s.objfname = self.objf.__name__
        s.best = Alpha_score
        s.bestIndividual = Alpha_pos

        return s
