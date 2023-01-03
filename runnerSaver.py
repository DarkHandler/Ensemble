
import csv
import numpy
import time
import os
import graphers.plot_convergence as conv_plot
import graphers.plot_boxplot as box_plot
from pathlib import Path

#INTERNAL LIBRARIES
import GWOEL.GWO as moduleGWO
import GWOEL.GWOEL as moduleGWOEL

import x_algorithms_comp.BA as moduleBA
import x_algorithms_comp.DE as moduleDE
import x_algorithms_comp.FA as moduleFA
import x_algorithms_comp.PSO as modulePSO
import x_algorithms_comp.WOA as moduleWOA
import x_algorithms_comp.CSA as moduleCSA

import GWOEL.ML_model.ML_classifiers as modML

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

    def selector(self, algo, func_details, popSize, Iter):
        function_name = func_details[0]
        lb = func_details[1]
        ub = func_details[2]
        dim = func_details[3]
        if function_name == "teamSizeModel":
            optModel = moduleTSM
        else:
            optModel = bf

        if algo == "PSO":
            x = modulePSO.PSO(getattr(optModel, function_name), lb, ub, dim, popSize, Iter).optimize()
        elif algo == "BA":
            x = moduleBA.BA(getattr(optModel, function_name), lb, ub, dim, popSize, Iter).optimize()
        elif algo == "FA":
            x = moduleFA.FA(getattr(optModel, function_name), lb, ub, dim, popSize, Iter).optimize()
        elif algo == "GWO":
            x = moduleGWO.GWO(getattr(optModel, function_name), lb, ub, dim, popSize, Iter, "original").optimize()
        elif algo == "WOA":
            x = moduleWOA.WOA(getattr(optModel, function_name), lb, ub, dim, popSize, Iter).optimize()
        elif algo == "CSA":
            x = moduleCSA.CSA(getattr(optModel, function_name), lb, ub, dim, popSize, Iter).optimize()
        elif algo == "GWOEL":
            modelEL = modML.ModuleEL().loadELModel()    #cargar el Ensemble Learning model
            x = moduleGWOEL.GWOEL(getattr(optModel, function_name), lb, ub, dim, popSize, Iter, modelEL).optimize()
        elif algo == "DE":
            x = moduleDE.DE(getattr(optModel, function_name), lb, ub, dim, popSize, Iter).optimize()
        else:
            return null
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

        # CSV Header for for the cinvergence
        CnvgHeader = []

        results_directory = time.strftime("%Y-%m-%d-%H-%M-%S") + "/"
        Path(results_directory).mkdir(parents=True, exist_ok=True)

        for l in range(0, Iterations):
            CnvgHeader.append("Iter" + str(l + 1))

        for i in range(0, len(optimizer)):
            for j in range(0, len(objectivefunc)):
                convergence = [0] * NumOfRuns
                executionTime = [0] * NumOfRuns
                for k in range(0, NumOfRuns):
                    func_details = bf.getFunctionDetails(objectivefunc[j])
                    x = self.selector(optimizer[i], func_details, PopulationSize, Iterations)
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

        if Export_convergence == True:
            conv_plot.run(results_directory, optimizer, objectivefunc, Iterations)

        if Export_boxplot == True:
            box_plot.run(results_directory, optimizer, objectivefunc, Iterations)

        if Flag == False:  # Faild to run at least one experiment
            print(
                "No Optomizer or Cost function is selected. Check lists of available optimizers and cost functions"
            )

        print("Execution completed")
