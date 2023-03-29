
import csv
import numpy
import time
import os
import graphers.plot_convergence as conv_plot
import graphers.plot_boxplot as box_plot
import graphers.plot_Exr_Ext as percntexRexT_plot
from pathlib import Path

#INTERNAL LIBRARIES
import x_algorithms_comp.GWO as moduleGWO
import x_algorithms_comp.BA as moduleBA
import x_algorithms_comp.DE as moduleDE
import x_algorithms_comp.FA as moduleFA
import x_algorithms_comp.PSO as modulePSO
import x_algorithms_comp.WOA as moduleWOA
import x_algorithms_comp.CSA as moduleCSA

#WITH Ensemble Learning
import MHEL.GWOEL as moduleGWOEL
import MHEL.BAEL as moduleBAEL
import MHEL.CSAEL as moduleCSAEL
import MHEL.DEEL as moduleDEEL
import MHEL.FAEL as moduleFAEL
import MHEL.PSOEL as modulePSOEL
import MHEL.WOAEL as moduleWOAEL


import MachineLearning.ML_classifiers as classifiers

import OptModel.teamSizeModel as moduleTSM
import OptModel.benchmark_functions as bf

class SingletonMeta(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class RunnerSingleton(metaclass=SingletonMeta):
    
    def createDatasetMetrics(self): #Paso 1 para realizar experimento: GENERAR DATASET CON METRICAS DE GWO 
        adapParam = "Adaptative Parameter"
        GAop = "GAOperators"
        minPercentExT = 70 #minimun percent of exploitation
        fileNameMetrics = "metrics_results_" + str(minPercentExT) + ".txt"

        print("-----INICIO-----")
        #6 lobos y 250
        moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 50, 6, 250, adapParam).optimize(fileNameMetrics, minPercentExT)
        moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 100, 6, 250, adapParam).optimize(fileNameMetrics, minPercentExT)
        moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 150, 6, 250, adapParam).optimize(fileNameMetrics, minPercentExT)
        moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 200, 6, 250, adapParam).optimize(fileNameMetrics, minPercentExT)

        moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 50, 6, 250, GAop).optimize(fileNameMetrics, minPercentExT)
        moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 100, 6, 250, GAop).optimize(fileNameMetrics, minPercentExT)
        moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 150, 6, 250, GAop).optimize(fileNameMetrics, minPercentExT)
        moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 200, 6, 250, GAop).optimize(fileNameMetrics, minPercentExT)

        #12 lobos y 250
        moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 50, 12, 250, adapParam).optimize(fileNameMetrics, minPercentExT)
        moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 100, 12, 250, adapParam).optimize(fileNameMetrics, minPercentExT)
        moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 150, 12, 250, adapParam).optimize(fileNameMetrics, minPercentExT)
        moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 200, 12, 250, adapParam).optimize(fileNameMetrics, minPercentExT)

        moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 50, 12, 250, GAop).optimize(fileNameMetrics, minPercentExT)
        moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 100, 12, 250, GAop).optimize(fileNameMetrics, minPercentExT)
        moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 150, 12, 250, GAop).optimize(fileNameMetrics, minPercentExT)
        moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 200, 12, 250, GAop).optimize(fileNameMetrics, minPercentExT)


        #6 lobos y 500
        moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 50, 6, 500, adapParam).optimize(fileNameMetrics, minPercentExT)
        moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 100, 6, 500, adapParam).optimize(fileNameMetrics, minPercentExT)
        moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 150, 6, 500, adapParam).optimize(fileNameMetrics, minPercentExT)
        moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 200, 6, 500, adapParam).optimize(fileNameMetrics, minPercentExT)

        moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 50, 6, 500, GAop).optimize(fileNameMetrics, minPercentExT)
        moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 100, 6, 500, GAop).optimize(fileNameMetrics, minPercentExT)
        moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 150, 6, 500, GAop).optimize(fileNameMetrics, minPercentExT)
        moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 200, 6, 500, GAop).optimize(fileNameMetrics, minPercentExT)

        #12 lobos y 500
        moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 50, 12, 500, adapParam).optimize(fileNameMetrics, minPercentExT)
        moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 100, 12, 500, adapParam).optimize(fileNameMetrics, minPercentExT)
        moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 150, 12, 500, adapParam).optimize(fileNameMetrics, minPercentExT)
        moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 200, 12, 500, adapParam).optimize(fileNameMetrics, minPercentExT)

        moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 50, 12, 500, GAop).optimize(fileNameMetrics, minPercentExT)
        moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 100, 12, 500, GAop).optimize(fileNameMetrics, minPercentExT)
        moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 150, 12, 500, GAop).optimize(fileNameMetrics, minPercentExT)
        moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 200, 12, 500, GAop).optimize(fileNameMetrics, minPercentExT)
        print("-----FIN-----")



    def selector(self, algo, func_details, popSize, Iter, modelEL):
        function_name = func_details[0]
        lb = func_details[1]
        ub = func_details[2]
        dim = func_details[3]
        if function_name == "teamSizeModel":
            optModel = moduleTSM #assign Team Size Model
        else:
            optModel = bf #assign the benchmark functions

        if algo == "PSO":
            x = modulePSO.PSO(getattr(optModel, function_name), lb, ub, dim, popSize, Iter).optimize()
        elif algo == "BA":
            x = moduleBA.BA(getattr(optModel, function_name), lb, ub, dim, popSize, Iter).optimize()
        elif algo == "FA":
            x = moduleFA.FA(getattr(optModel, function_name), lb, ub, dim, popSize, Iter).optimize()
        elif algo == "GWO":
            #ori = "Adaptative Parameter"
            #ori = "GAOperators"
            ori = "original"
            x = moduleGWO.GWO(getattr(optModel, function_name), lb, ub, dim, popSize, Iter, ori).optimize()
        elif algo == "WOA":
            x = moduleWOA.WOA(getattr(optModel, function_name), lb, ub, dim, popSize, Iter).optimize()
        elif algo == "CSA":
            x = moduleCSA.CSA(getattr(optModel, function_name), lb, ub, dim, popSize, Iter).optimize()
        elif algo == "DE":
            x = moduleDE.DE(getattr(optModel, function_name), lb, ub, dim, popSize, Iter).optimize()
        elif algo == "GWOEL":
            if type(modelEL) == int: # if isn't a classifiers. The ModuleEL class is initialized just one time
                modelEL = classifiers.ModuleEL("metrics_results_70.txt", "modelEL_70").getELModel()    #load Ensemble Learning Model
            x = moduleGWOEL.GWOEL(getattr(optModel, function_name), lb, ub, dim, popSize, Iter, modelEL).optimize()
        elif algo == "BAEL":
            if type(modelEL) == int: # if isn't a classifiers. The ModuleEL class is initialized just one time
                modelEL = classifiers.ModuleEL("metrics_results_70.txt", "modelEL_70").getELModel()    #load Ensemble Learning Model
            x = moduleBAEL.BAEL(getattr(optModel, function_name), lb, ub, dim, popSize, Iter, modelEL).optimize()   
        elif algo == "CSAEL":
            if type(modelEL) == int: # if isn't a classifiers. The ModuleEL class is initialized just one time
                modelEL = classifiers.ModuleEL("metrics_results_70.txt", "modelEL_70").getELModel()    #load Ensemble Learning Model
            x = moduleCSAEL.CSAEL(getattr(optModel, function_name), lb, ub, dim, popSize, Iter, modelEL).optimize()   
        elif algo == "DEEL":
            if type(modelEL) == int: # if isn't a classifiers. The ModuleEL class is initialized just one time
                modelEL = classifiers.ModuleEL("metrics_results_70.txt", "modelEL_70").getELModel()    #load Ensemble Learning Model
            x = moduleDEEL.DEEL(getattr(optModel, function_name), lb, ub, dim, popSize, Iter, modelEL).optimize()  
        elif algo == "FAEL":
            if type(modelEL) == int: # if isn't a classifiers. The ModuleEL class is initialized just one time
                modelEL = classifiers.ModuleEL("metrics_results_70.txt", "modelEL_70").getELModel()    #load Ensemble Learning Model
            x = moduleFAEL.FAEL(getattr(optModel, function_name), lb, ub, dim, popSize, Iter, modelEL).optimize()    
        elif algo == "PSOEL":
            if type(modelEL) == int: # if isn't a classifiers. The ModuleEL class is initialized just one time
                modelEL = classifiers.ModuleEL("metrics_results_70.txt", "modelEL_70").getELModel()    #load Ensemble Learning Model
            x = modulePSOEL.PSOEL(getattr(optModel, function_name), lb, ub, dim, popSize, Iter, modelEL).optimize()    
        elif algo == "WOAEL":
            if type(modelEL) == int: # if isn't a classifiers. The ModuleEL class is initialized just one time
                modelEL = classifiers.ModuleEL("metrics_results_70.txt", "modelEL_70").getELModel()    #load Ensemble Learning Model
            x = moduleWOAEL.WOAEL(getattr(optModel, function_name), lb, ub, dim, popSize, Iter, modelEL).optimize()    
        else:
            return None
        return x





    def run(self, optimizer, objectivefunc, NumOfRuns, params):

        """
        It serves as the main interface of the framework for running the experiments.

        Parameters
        ----------
        optimizer : list
            The list of optimizers names
        objectivefunc : list
            The list of benchmark functions
        NumOfRuns : int
            The number of independent runs
        params  : set
            The set of parameters which are:
            1. Size of population (PopulationSize)
            2. The number of iterations (Iterations)
        export_flags : set
            The set of Boolean flags which are:
            1. Export (Exporting the results in a file)
            2. Export_details (Exporting the detailed results in files)
            3. Export_convergence (Exporting the covergence plots)
            4. Export_boxplot (Exporting the box plots)

        Returns
        -----------
        N/A
        """

        # Select general parameters for all optimizers (population size, number of iterations) ....
        PopulationSize = params["PopulationSize"]
        Iterations = params["Iterations"]

        # Export results ?
        Export = True
        Export_details = True
        Export_convergence = True
        Export_boxplot = True

        Flag = False
        Flag_details = False
        FlagPer = False #plotPercent
        Flag_details_per = False #plotPercent

        # CSV Header for for the convergence
        CnvgHeader = []

        results_directory = time.strftime("%Y-%m-%d-%H-%M-%S") + "/"
        Path(results_directory).mkdir(parents=True, exist_ok=True)

        #Variable to save permanentebly and used only when GWOEL is setted
        modelEL = -1
        bestVector = None #save best vector
        bestFitness = None #save best fitness

        for l in range(0, Iterations):
            CnvgHeader.append("Iter" + str(l + 1))

        for i in range(0, len(optimizer)):
            for j in range(0, len(objectivefunc)):
                convergence = [0] * NumOfRuns
                percent_explorations = [0] * NumOfRuns
                executionTime = [0] * NumOfRuns
                for k in range(0, NumOfRuns):
                    func_details = bf.getFunctionDetails(objectivefunc[j])
                    x = self.selector(optimizer[i], func_details, PopulationSize, Iterations, modelEL)
                    
                    #DELETE AFTER USE
                    if( bestFitness == None or x.best < bestFitness):
                        bestFitness = x.best
                        bestVector = x.bestIndividual
                    convergence[k] = x.convergence
                    

                    optimizerName = x.optimizer
                    objfname = x.objfname
                    if Export_details == True:
                        ExportToFile = results_directory + "experiment_details.csv"
                        with open(ExportToFile, "a", newline="\n") as out:
                            writer = csv.writer(out, delimiter=",")
                            if (
                                Flag_details == False
                            ):  # just one time to write the header of the CSV file
                                header = numpy.concatenate(
                                    [["Optimizer", "objfname", "ExecutionTime"], CnvgHeader]
                                )
                                writer.writerow(header)
                                Flag_details = True  # at least one experiment
                            executionTime[k] = x.executionTime
                            a = numpy.concatenate(
                                [[x.optimizer, x.objfname, x.executionTime], x.convergence]
                            )
                            writer.writerow(a)
                        out.close()

                        #SAVE PERCENTEGE EXPLORATION exploitation
                        percent_explorations[k] = x.percent_explorations
                        
                        ExportToFile = results_directory + "experiment_eReT_det.csv"
                        with open(ExportToFile, "a", newline="\n") as out:
                            writer = csv.writer(out, delimiter=",")
                            if (
                                Flag_details_per == False
                            ):  # just one time to write the header of the CSV file
                                header = numpy.concatenate(
                                    [["Optimizer", "objfname"], CnvgHeader]
                                )
                                writer.writerow(header)
                                Flag_details_per = True  # at least one experiment
                            a = numpy.concatenate(
                                [[x.optimizer, x.objfname], x.percent_explorations]
                            )
                            writer.writerow(a)
                        out.close()


                if Export == True:
                    ExportToFile = results_directory + "experiment.csv"

                    with open(ExportToFile, "a", newline="\n") as out:
                        writer = csv.writer(out, delimiter=",")
                        if (
                            Flag == False
                        ):  # just one time to write the header of the CSV file
                            header = numpy.concatenate(
                                [["Optimizer", "objfname", "ExecutionTime"], CnvgHeader]
                            )
                            writer.writerow(header)
                            Flag = True

                        avgExecutionTime = float("%0.2f" % (sum(executionTime) / NumOfRuns))
                        avgConvergence = numpy.around(
                            numpy.mean(convergence, axis=0, dtype=numpy.float64), decimals=2
                        ).tolist()
                        a = numpy.concatenate(
                            [[optimizerName, objfname, avgExecutionTime], avgConvergence]
                        )
                        writer.writerow(a)
                    out.close()

                    #SAVE SUMMARY OF ALL RUNS: PERCENTEGE EXPLORATION exploitation
                    ExportToFile = results_directory + "experiment_sumry_eReT.csv"

                    with open(ExportToFile, "a", newline="\n") as out:
                        writer = csv.writer(out, delimiter=",")
                        if (
                            FlagPer == False
                        ):  # just one time to write the header of the CSV file
                            header = numpy.concatenate(
                                [["Optimizer", "objfname"], CnvgHeader]
                            )
                            writer.writerow(header)
                            FlagPer = True

                        avgPercenteReT = numpy.around(
                            numpy.mean(percent_explorations, axis=0, dtype=numpy.float64), decimals=2
                        ).tolist()
                        a = numpy.concatenate(
                            [[optimizerName, objfname], avgPercenteReT]
                        )
                        writer.writerow(a)
                    out.close()
                

        if Export_convergence == True:
            conv_plot.run(results_directory, optimizer, objectivefunc, Iterations)

        if Export_boxplot == True:
            box_plot.run(results_directory, optimizer, objectivefunc, Iterations)

        #export plot
        percntexRexT_plot.run(results_directory, optimizer, objectivefunc, Iterations)

        print(f"Vector best solution: {bestVector}, Score fitness: {bestFitness}") #DELETE IF YOU DONT WANT TO SHOW THE RESULT

        if Flag == False:  # Faild to run at least one experiment
            print(
                "No Optomizer or Cost function is selected. Check lists of available optimizers and cost functions"
            )

        print("Execution completed")
