# -*- coding: utf-8 -*-
"""
Created on Mon May 16 14:19:49 2016

@author: hossam

@modified by: Sebastian Medina
"""
import random
import numpy as np
import math
from solution import solution
import time
import pandas as pd

import methods.GAoperators as GAoperators

import OptModel.teamSizeModel as teamSizeModel

import metrics.metrics as mtclass

class WOAEL:
    def __init__(self, objf, lb, ub, dim, SearchAgents_no, Max_iter, modelEL):
        self.objf = objf
        self.lb = lb
        self.ub = ub
        self.dim = dim
        self.SearchAgents_no= SearchAgents_no
        self.Max_iter = Max_iter
        self.modelEL = modelEL    

    def optimize(self, fileNameMetrics = False, minPercentExT = -1):

        #Save metrics condition
        if (type(fileNameMetrics) == str) and (minPercentExT < 0 or minPercentExT > 100):
            raise Exception("minPercentExt must be a number between [0 - 100]")

        metrics = mtclass.Metrics() #objeto de metricas

        # dim=30
        # SearchAgents_no=50
        # lb=-100
        # ub=100
        # Max_iter=500
        if not isinstance(self.lb, list):
            self.lb = [self.lb] * self.dim
        if not isinstance(self.ub, list):
            self.ub = [self.ub] * self.dim

        if self.objf.__name__ == "teamSizeModel": #si es el problema que estamos resolviendo de team size model
            proyectSize = self.dim
            self.dim = math.floor(self.dim/3) #no es necesario debido a que si no se estaran creando obligatoriamente variables con rango 3 a 17 

        #array of fitness, size of search agents
        arrayPopFitness = np.zeros(self.SearchAgents_no)

        # initialize position vector and score for the leader
        Leader_pos = np.zeros(self.dim)
        Leader_score = float("inf")  # change this to -inf for maximization problems

        # Initialize the positions of search agents
        if (self.objf.__name__ != "teamSizeModel"):
            Positions = np.zeros((self.SearchAgents_no, self.dim))
            for i in range(self.dim):
                Positions[:, i] = (
                    np.random.uniform(0, 1, self.SearchAgents_no) * (self.ub[i] - self.lb[i]) + self.lb[i]
                )
        else:
            Positions = teamSizeModel.initPopTeamSizeModel(self.SearchAgents_no, self.lb[0], self.ub[0], proyectSize, self.dim)
            Positions = np.array(Positions)

        # Initialize convergence
        convergence_curve = np.zeros(self.Max_iter)
        Percent_explorations = np.zeros(self.Max_iter)

        ############################
        s = solution()

        print('WOAEL is optimizing  "' + self.objf.__name__ + '"')

        timerStart = time.time()
        s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
        ############################

        t = 0  # Loop counter

        # Main loop
        while t < self.Max_iter:
            for i in range(0, self.SearchAgents_no):

                # Return back the search agents that go beyond the boundaries of the search space
                if (self.objf.__name__ == "teamSizeModel"):
                    Positions = list(Positions)
                    teamSizeModel.checkBoundaries(Positions[i], proyectSize, self.lb[0], self.ub[0])
                    Positions = np.array(Positions)
                else:
                    # Positions[i,:]=checkBounds(Positions[i,:],lb,ub)
                    for j in range(self.dim):
                        Positions[i, j] = np.clip(Positions[i, j], self.lb[j], self.ub[j])

                # Calculate objective function for each search agent
                fitness = self.objf(Positions[i, :])
                arrayPopFitness[i] = fitness

                # Update the leader
                if fitness < Leader_score:  # Change this to > for maximization problem
                    Leader_score = fitness
                    # Update alpha
                    Leader_pos = Positions[
                        i, :
                    ].copy()  # copy current whale position into the leader position

            a = 2 - t * ((2) / self.Max_iter)
            # a decreases linearly fron 2 to 0 in Eq. (2.3)

            # a2 linearly decreases from -1 to -2 to calculate t in Eq. (3.12)
            a2 = -1 + t * ((-1) / self.Max_iter)

            ## --------- DIVERSITY ZONE ----------
            metrics.calculateDiversity(Positions, self.SearchAgents_no, self.dim, self.objf)
            Percent_explorations[t] = metrics.percent_exploration

            ## --- STORE METRICS ---
            if type(fileNameMetrics) == str: #if true
                metrics.storeMetricsIn(fileNameMetrics, t, Leader_score, self.SearchAgents_no, proyectSize, minPercentExT)

            ## POSITIONS UPDATE METHODS OF AGENTS WITH Ensemble Learning MODEL
            namesFeatu = "iteration,fitness,searchAgents_no,proyectSize,percent_exploration,percent_exploitation,diversidadHamming,diversidad_Dice,diversidad_Jaccard,diversidad_Kulsinski,diversidad_Rogerstanimoto,diversidad_Russellrao,diversidad_Sokalmichener,diversidad_Yule,diversidad_Sokalsneath,diversidadDimensionWise".split(',')
            metricsResults = [t, fitness, self.SearchAgents_no, proyectSize, metrics.percent_exploration, metrics.percent_exploitation, metrics.diversidadHamming, metrics.diversidad_Dice, metrics.diversidad_Jaccard, metrics.diversidad_Kulsinski, metrics.diversidad_Rogerstanimoto, metrics.diversidad_Russellrao, metrics.diversidad_Sokalmichener, metrics.diversidad_Yule, metrics.diversidad_Sokalsneath, metrics.diversidadDimensionWise]

            pdMetricResult = pd.DataFrame(data=np.array([metricsResults]), columns=namesFeatu)
            prediction = self.modelEL.predict(pdMetricResult)
            
            if prediction[0] == "exploitation": #mechanism to exploTation
                # Update the Position of search agents
                for i in range(0, self.SearchAgents_no):
                    r1 = random.random()  # r1 is a random number in [0,1]
                    r2 = random.random()  # r2 is a random number in [0,1]

                    A = 2 * a * r1 - a  # Eq. (2.3) in the paper
                    C = 2 * r2  # Eq. (2.4) in the paper

                    b = 1
                    #  parameters in Eq. (2.5)
                    l = (a2 - 1) * random.random() + 1  #  parameters in Eq. (2.5)

                    p = random.random()  # p in Eq. (2.6)

                    for j in range(0, self.dim):

                        if p < 0.5:
                            if abs(A) >= 1:
                                rand_leader_index = math.floor(
                                    self.SearchAgents_no * random.random()
                                )
                                X_rand = Positions[rand_leader_index, :]
                                D_X_rand = abs(C * X_rand[j] - Positions[i, j])
                                Positions[i, j] = X_rand[j] - A * D_X_rand

                            elif abs(A) < 1:
                                D_Leader = abs(C * Leader_pos[j] - Positions[i, j])
                                Positions[i, j] = Leader_pos[j] - A * D_Leader

                        elif p >= 0.5:

                            distance2Leader = abs(Leader_pos[j] - Positions[i, j])
                            # Eq. (2.5)
                            Positions[i, j] = (
                                distance2Leader * math.exp(b * l) * math.cos(l * 2 * math.pi)
                                + Leader_pos[j]
                            )
            else: #GAoperators mechanism to exploRation
                #print("xpL GAOPERATORS")
                GAoperators.updatePopWithGAMethod(self.lb, self.ub, Positions, arrayPopFitness, self.SearchAgents_no) #funcionando explotacion

            

            convergence_curve[t] = Leader_score
            #if t % 1 == 0:
            #    print(["At iteration " + str(t) + " the best fitness is " + str(Leader_score)])
            t = t + 1

        print(Leader_score)

        timerEnd = time.time()
        s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
        s.executionTime = timerEnd - timerStart
        s.convergence = convergence_curve
        s.percent_explorations = Percent_explorations
        s.optimizer = "WOAEL"
        s.objfname = self.objf.__name__
        s.best = Leader_score
        s.bestIndividual = Leader_pos

        return s
