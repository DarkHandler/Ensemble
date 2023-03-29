# -*- coding: utf-8 -*-
"""
Created on November 2022

@author: Sebastian Medina
"""

import random
import numpy as np
import pandas as pd
import math
from solution import solution
import time
import methods.adaptativeParam as adaptativeParam
import methods.GAoperators as GAoperators

import OptModel.teamSizeModel as teamSizeModel

import metrics.metrics as mtclass


class GWOEL:
    
    def __init__(self, objf, lb, ub, dim, SearchAgents_no, Max_iter, modelEL):
        self.objf = objf
        self.lb = lb    
        self.ub = ub    
        self.dim = dim  
        self.SearchAgents_no = SearchAgents_no
        self.Max_iter = Max_iter              
        self.modelEL = modelEL                  

    def optimize(self, fileNameMetrics = False, minPercentExT = -1): #enfocado en la minimizacion por defecto

        #Save metrics condition
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
            Positions = teamSizeModel.initPopTeamSizeModel(self.SearchAgents_no, self.lb[0], self.ub[0], proyectSize, self.dim)

        Convergence_curve = np.zeros(self.Max_iter)
        Percent_explorations = np.zeros(self.Max_iter)
        s = solution()

        # Loop counter
        print('GWOEL is optimizing  "' + self.objf.__name__ + '"')

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

                        

                # Calculate objective function for each search agent
                fitness = self.objf(Positions[i])
                arrayPopFitness[i] = fitness    #update the fitness of wolf

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

            ## --- STORE METRICS ---
            if type(fileNameMetrics) == str: #if true
                metrics.storeMetricsIn(fileNameMetrics, l, fitness, self.SearchAgents_no, proyectSize, minPercentExT)

            ## POSITIONS UPDATE METHODS OF AGENTS WITH Ensemble Learning MODEL
            namesFeatu = "iteration,fitness,searchAgents_no,proyectSize,percent_exploration,percent_exploitation,diversidadHamming,diversidad_Dice,diversidad_Jaccard,diversidad_Kulsinski,diversidad_Rogerstanimoto,diversidad_Russellrao,diversidad_Sokalmichener,diversidad_Yule,diversidad_Sokalsneath,diversidadDimensionWise".split(',')
            metricsResults = [l, fitness, self.SearchAgents_no, proyectSize, metrics.percent_exploration, metrics.percent_exploitation, metrics.diversidadHamming, metrics.diversidad_Dice, metrics.diversidad_Jaccard, metrics.diversidad_Kulsinski, metrics.diversidad_Rogerstanimoto, metrics.diversidad_Russellrao, metrics.diversidad_Sokalmichener, metrics.diversidad_Yule, metrics.diversidad_Sokalsneath, metrics.diversidadDimensionWise]

            pdMetricResult = pd.DataFrame(data=np.array([metricsResults]), columns=namesFeatu)
            prediction = self.modelEL.predict(pdMetricResult)
            
            if prediction[0] == "exploitation": #adaParam mechanism to exploTation
                #print("xpT AdaptativeParam")
                adaptativeParam.adaptativeControlParameter_a_UpdateMethod(l, self.Max_iter, self.SearchAgents_no, Positions, Alpha_pos, Delta_pos, Beta_pos, self.objf.__name__) #funcionando exploracion
            else: #GAoperators mechanism to exploRation
                #print("xpL GAOPERATORS")
                GAoperators.updatePopWithGAMethod(self.lb, self.ub, Positions, arrayPopFitness, self.SearchAgents_no) #funcionando explotacion

            #adaptativeParam.linealUpdateMethod(SearchAgents_no, Max_iter, Positions, l, Alpha_pos, Delta_pos, Beta_pos, objf.__name__) #funcionando este es el clasico que hay que eliminar
            

            Convergence_curve[l] = Alpha_score

            #if l % 1 == 0:
            #print(["At iteration " + str(l) + " the best fitness is " + str(Alpha_score)])
            #print("Alfa position:",Alpha_pos, sum(Alpha_pos))
            #print("best:",objf([8,10,7,8,8,8,8,7,9,9,10,9,9,9,7,11,10,8,10,9,8,8,10]))

        print(Alpha_score)
    
        timerEnd = time.time()
        s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
        s.executionTime = timerEnd - timerStart
        s.convergence = Convergence_curve
        s.percent_explorations = Percent_explorations
        s.optimizer = "GWOEL"
        s.objfname = self.objf.__name__
        s.best = Alpha_score
        s.bestIndividual = Alpha_pos

        return s
