# -*- coding: utf-8 -*-
"""
Created on Sun May 29 00:49:35 2016

@author: hossam
@modified by: Sebastian Medina
"""

#% ======================================================== %
#% Files of the Matlab programs included in the book:       %
#% Xin-She Yang, Nature-Inspired Metaheuristic Algorithms,  %
#% Second Edition, Luniver Press, (2010).   www.luniver.com %
#% ======================================================== %
#
#% -------------------------------------------------------- %
#% Firefly Algorithm for constrained optimization using     %
#% for the design of a spring (benchmark)                   %
#% by Xin-She Yang (Cambridge University) Copyright @2009   %
#% -------------------------------------------------------- %

import numpy
import math
import time
from solution import solution

import numpy as np
import OptModel.teamSizeModel as teamSizeModel

class FA:
    def __init__(self, objf, lb, ub, dim, n, MaxGeneration):
        self.objf = objf
        self.lb = lb
        self.ub = ub
        self.dim = dim
        self.n= n
        self.MaxGeneration = MaxGeneration

    def alpha_new(self, alpha, NGen):
        #% alpha_n=alpha_0(1-delta)^NGen=10^(-4);
        #% alpha_0=0.9
        delta = 1 - (10 ** (-4) / 0.9) ** (1 / NGen)
        alpha = (1 - delta) * alpha
        return alpha


    def optimize(self):

        # General parameters

        # n=50 #number of fireflies
        # dim=30 #dim
        # lb=-50
        # ub=50
        # MaxGeneration=500

        if self.objf.__name__ == "teamSizeModel": #si es el problema que estamos resolviendo de team size model
            proyectSize = self.dim
            self.dim = math.floor(self.dim/3) #no es necesario debido a que si no se estaran creando obligatoriamente variables con rango 3 a 17  

        # FFA parameters
        alpha = 0.5  # Randomness 0--1 (highly random)
        betamin = 0.20  # minimum value of beta
        gamma = 1  # Absorption coefficient
        if not isinstance(self.lb, list):
            self.lb = [self.lb] * self.dim
        if not isinstance(self.ub, list):
            self.ub = [self.ub] * self.dim

        zn = numpy.ones(self.n)
        zn.fill(float("inf"))

        # ns(i,:)=Lb+(Ub-Lb).*rand(1,d);

        # Initialize the population/solutions
        if (self.objf.__name__ != "teamSizeModel"):
            ns = numpy.zeros((self.n, self.dim))
            for i in range(self.dim):
                ns[:, i] = numpy.random.uniform(0, 1, self.n) * (self.ub[i] - self.lb[i]) + self.lb[i]
        else:
            ns = teamSizeModel.initPopTeamSizeModel(self.n, self.lb[0], self.ub[0], proyectSize, self.dim)
            ns = np.array(ns)
        
        Lightn = numpy.ones(self.n)
        Lightn.fill(float("inf"))

        # [ns,Lightn]=init_ffa(n,d,Lb,Ub,u0)

        convergence = []
        s = solution()

        print('FA is optimizing  "' + self.objf.__name__ + '"')

        timerStart = time.time()
        s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

        # Main loop
        for k in range(0, self.MaxGeneration):  # start iterations

            #% This line of reducing alpha is optional
            alpha = self.alpha_new(alpha, self.MaxGeneration)

            #% Evaluate new solutions (for all n fireflies)
            for i in range(0, self.n):
                if (self.objf.__name__ == "teamSizeModel"): #line added - CHECK BOUNDARIES of team size model
                    ns = list(ns)
                    teamSizeModel.checkBoundaries(ns[i], proyectSize, self.lb[0], self.ub[0])
                    ns = np.array(ns)
                zn[i] = self.objf(ns[i, :])
                Lightn[i] = zn[i]

            # Ranking fireflies by their light intensity/objectives

            Lightn = numpy.sort(zn)
            Index = numpy.argsort(zn)
            ns = ns[Index, :]

            # Find the current best
            nso = ns
            Lighto = Lightn
            nbest = ns[0, :]
            Lightbest = Lightn[0]

            #% For output only
            fbest = Lightbest

            #% Move all fireflies to the better locations
            #    [ns]=ffa_move(n,d,ns,Lightn,nso,Lighto,nbest,...
            #          Lightbest,alpha,betamin,gamma,Lb,Ub);
            scale = []
            for b in range(self.dim):
                scale.append(abs(self.ub[b] - self.lb[b]))
            scale = numpy.array(scale)
            for i in range(0, self.n):
                # The attractiveness parameter beta=exp(-gamma*r)
                for j in range(0, self.n):
                    r = numpy.sqrt(numpy.sum((ns[i, :] - ns[j, :]) ** 2))
                    # r=1
                    # Update moves
                    if Lightn[i] > Lighto[j]:  # Brighter and more attractive
                        beta0 = 1
                        beta = (beta0 - betamin) * math.exp(-gamma * r ** 2) + betamin
                        tmpf = alpha * (numpy.random.rand(self.dim) - 0.5) * scale
                        ns[i, :] = ns[i, :] * (1 - beta) + nso[j, :] * beta + tmpf

            # ns=numpy.clip(ns, lb, ub)

            convergence.append(fbest)

            BestQuality = fbest

            #if k % 1 == 0:
            #    print(["At iteration " + str(k) + " the best fitness is " + str(BestQuality)])
        
        print(BestQuality)
        print()
        #
        ####################### End main loop
        timerEnd = time.time()
        s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
        s.executionTime = timerEnd - timerStart
        s.convergence = convergence
        s.optimizer = "FA"
        s.objfname = self.objf.__name__

        return s
