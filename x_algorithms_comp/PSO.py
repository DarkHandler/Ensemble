# -*- coding: utf-8 -*-
"""
Created on Sun May 15 22:37:00 2016

@author: Hossam Faris
@modified by: Sebastian Medina
"""

import random
import numpy
from solution import solution
import math
import time

import OptModel.teamSizeModel as teamSizeModel

class PSO:
    def __init__(self, objf, lb, ub, dim, PopSize, iters):
        self.objf = objf
        self.lb = lb
        self.ub = ub
        self.dim = dim
        self.PopSize= PopSize
        self.iters = iters

    def optimize(self):
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

        vel = numpy.zeros((self.PopSize, self.dim))

        pBestScore = numpy.zeros(self.PopSize)
        pBestScore.fill(float("inf"))

        pBest = numpy.zeros((self.PopSize, self.dim))
        gBest = numpy.zeros(self.dim)

        gBestScore = float("inf")

        if (self.objf.__name__ != "teamSizeModel"):
            pos = numpy.zeros((self.PopSize, self.dim))
            for i in range(self.dim):
                pos[:, i] = numpy.random.uniform(0, 1, self.PopSize) * (self.ub[i] - self.lb[i]) + self.lb[i]
        else:
            pos = teamSizeModel.initPopTeamSizeModel(self.PopSize, self.lb[0], self.ub[0], proyectSize, self.dim)
            pos = numpy.array(pos)

        convergence_curve = numpy.zeros(self.iters)

        ############################################
        print('PSO is optimizing  "' + self.objf.__name__ + '"')

        timerStart = time.time()
        s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

        for l in range(0, self.iters):
            for i in range(0, self.PopSize):
                # Return back the search agents that go beyond the boundaries of the search space
                # pos[i,:]=checkBounds(pos[i,:],lb,ub)
                if (self.objf.__name__ == "teamSizeModel"):
                    pos = list(pos)
                    teamSizeModel.checkBoundaries(pos[i], proyectSize, self.lb[0], self.ub[0])
                    pos = numpy.array(pos)
                else:
                    for j in range(self.dim):
                        pos[i, j] = numpy.clip(pos[i, j], self.lb[j], self.ub[j])
                
                # Calculate objective function for each particle
                fitness = self.objf(pos[i, :])

                if pBestScore[i] > fitness:
                    pBestScore[i] = fitness
                    pBest[i, :] = pos[i, :].copy()

                if gBestScore > fitness:
                    gBestScore = fitness
                    gBest = pos[i, :].copy()

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
        s.optimizer = "PSO"
        s.objfname = self.objf.__name__

        return s
