# -*- coding: utf-8 -*-
"""
Created on Sun May 15 22:37:00 2016

@author: Hossam Faris
@modified by: Sebastian Medina
"""

import random
import numpy as np
from solution import solution
import math
import time
import pandas as pd

import methods.GAoperators as GAoperators

import OptModel.teamSizeModel as teamSizeModel

import metrics.metrics as mtclass

class PSOEL:
    def __init__(self, objf, lb, ub, dim, PopSize, iters, modelEL):
        self.objf = objf
        self.lb = lb
        self.ub = ub
        self.dim = dim
        self.PopSize= PopSize
        self.iters = iters
        self.modelEL = modelEL  

    def optimize(self, fileNameMetrics = False, minPercentExT = -1):

        #Save metrics condition
        if (type(fileNameMetrics) == str) and (minPercentExT < 0 or minPercentExT > 100):
            raise Exception("minPercentExt must be a number between [0 - 100]")

        metrics = mtclass.Metrics() #objeto de metricas
        
        # PSO parameters
        Vmax = 6
        wMax = 0.9
        wMin = 0.2
        c1 = 2
        c2 = 2

        s = solution()
        if not isinstance(self.lb, list):
            self.lb = [self.lb] * self.dim
        if not isinstance(self.ub, list):
            self.ub = [self.ub] * self.dim

        if self.objf.__name__ == "teamSizeModel": #si es el problema que estamos resolviendo de team size model
            proyectSize = self.dim
            self.dim = math.floor(self.dim/3) #no es necesario debido a que si no se estaran creando obligatoriamente variables con rango 3 a 17 

        ######################## Initializations

        #array of fitness, size of search agents
        arrayPopFitness = np.zeros(self.PopSize)

        vel = np.zeros((self.PopSize, self.dim))

        pBestScore = np.zeros(self.PopSize)
        pBestScore.fill(float("inf"))

        pBest = np.zeros((self.PopSize, self.dim))
        gBest = np.zeros(self.dim)

        gBestScore = float("inf")

        if (self.objf.__name__ != "teamSizeModel"):
            pos = np.zeros((self.PopSize, self.dim))
            for i in range(self.dim):
                pos[:, i] = np.random.uniform(0, 1, self.PopSize) * (self.ub[i] - self.lb[i]) + self.lb[i]
        else:
            pos = teamSizeModel.initPopTeamSizeModel(self.PopSize, self.lb[0], self.ub[0], proyectSize, self.dim)
            pos = np.array(pos)

        convergence_curve = np.zeros(self.iters)
        Percent_explorations = np.zeros(self.iters)

        ############################################
        print('PSOEL is optimizing  "' + self.objf.__name__ + '"')

        timerStart = time.time()
        s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

        for l in range(0, self.iters):
            for i in range(0, self.PopSize):
                # Return back the search agents that go beyond the boundaries of the search space
                # pos[i,:]=checkBounds(pos[i,:],lb,ub)
                if (self.objf.__name__ == "teamSizeModel"):
                    pos = list(pos)
                    teamSizeModel.checkBoundaries(pos[i], proyectSize, self.lb[0], self.ub[0])
                    pos = np.array(pos)
                else:
                    for j in range(self.dim):
                        pos[i, j] = np.clip(pos[i, j], self.lb[j], self.ub[j])
                
                # Calculate objective function for each particle
                fitness = self.objf(pos[i, :])
                arrayPopFitness[i] = fitness

                if pBestScore[i] > fitness:
                    pBestScore[i] = fitness
                    pBest[i, :] = pos[i, :].copy()

                if gBestScore > fitness:
                    gBestScore = fitness
                    gBest = pos[i, :].copy()

            ## --------- DIVERSITY ZONE ----------
            metrics.calculateDiversity(pos, self.PopSize, self.dim, self.objf)
            Percent_explorations[l] = metrics.percent_exploration

            ## --- STORE METRICS ---
            if type(fileNameMetrics) == str: #if true
                metrics.storeMetricsIn(fileNameMetrics, l, gBestScore, self.PopSize, proyectSize, minPercentExT)

            ## POSITIONS UPDATE METHODS OF AGENTS WITH Ensemble Learning MODEL
            namesFeatu = "iteration,fitness,searchAgents_no,proyectSize,percent_exploration,percent_exploitation,diversidadHamming,diversidad_Dice,diversidad_Jaccard,diversidad_Kulsinski,diversidad_Rogerstanimoto,diversidad_Russellrao,diversidad_Sokalmichener,diversidad_Yule,diversidad_Sokalsneath,diversidadDimensionWise".split(',')
            metricsResults = [l, gBestScore, self.PopSize, proyectSize, metrics.percent_exploration, metrics.percent_exploitation, metrics.diversidadHamming, metrics.diversidad_Dice, metrics.diversidad_Jaccard, metrics.diversidad_Kulsinski, metrics.diversidad_Rogerstanimoto, metrics.diversidad_Russellrao, metrics.diversidad_Sokalmichener, metrics.diversidad_Yule, metrics.diversidad_Sokalsneath, metrics.diversidadDimensionWise]

            pdMetricResult = pd.DataFrame(data=np.array([metricsResults]), columns=namesFeatu)
            prediction = self.modelEL.predict(pdMetricResult)
            
            if prediction[0] == "exploitation": #mechanism to exploTation
                # Update the W of PSO
                w = wMax - l * ((wMax - wMin) / self.iters)

                for i in range(0, self.PopSize):
                    for j in range(0, self.dim):
                        r1 = random.random()
                        r2 = random.random()
                        vel[i, j] = (
                            w * vel[i, j]
                            + c1 * r1 * (pBest[i, j] - pos[i, j])
                            + c2 * r2 * (gBest[j] - pos[i, j])
                        )

                        if vel[i, j] > Vmax:
                            vel[i, j] = Vmax

                        if vel[i, j] < -Vmax:
                            vel[i, j] = -Vmax

                        pos[i, j] = pos[i, j] + vel[i, j]

            else: #GAoperators mechanism to exploRation
                #print("xpL GAOPERATORS")
                GAoperators.updatePopWithGAMethod(self.lb, self.ub, pos, arrayPopFitness, self.PopSize) #funcionando explotacion


            convergence_curve[l] = gBestScore

            #if l % 1 == 0:
                #print(["At iteration "+ str(l + 1)+ " the best fitness is "+ str(gBestScore)])

        print(gBestScore)    
        #print(gBest)
        #print(self.objf(gBest))
        
        timerEnd = time.time()
        s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
        s.executionTime = timerEnd - timerStart
        s.convergence = convergence_curve
        s.percent_explorations = Percent_explorations
        s.optimizer = "PSOEL"
        s.objfname = self.objf.__name__
        s.best = gBestScore
        s.bestIndividual = gBest

        return s
