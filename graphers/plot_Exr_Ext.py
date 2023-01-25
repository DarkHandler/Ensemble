import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#for each optimizer create a plot of iterations and exploration-exploitation balance
def run(results_directory, optimizer, objectivefunc, Iterations):
    plt.ioff()
    fileResultsData = pd.read_csv(results_directory + "/experiment_sumry_eReT.csv")

    for j in range(0, len(objectivefunc)):
        objective_name = objectivefunc[j]

        startIteration = 0
        if "SSA" in optimizer:
            startIteration = 1
        allGenerations = [x + 1 for x in range(startIteration, Iterations)]

        for i in range(len(optimizer)): #debo crear un plot por cada optimizer
            optimizer_name = optimizer[i]

            row = fileResultsData[
                (fileResultsData["Optimizer"] == optimizer_name)
                & (fileResultsData["objfname"] == objective_name)
            ]
            row = row.iloc[:, 2 + startIteration :]
            plt.plot(allGenerations, row.values.tolist()[0], label = "Exploration(Avg " + str(np.average(row.values.tolist()[0])) + ")")
            plt.plot(allGenerations, (100 - np.array(row.values.tolist()[0])), label = "Exploitation (Avg " + str(np.average((100 - np.array(row.values.tolist()[0])))) + ")")
            plt.xlabel("Iterations")
            plt.ylabel("Percentage")
            plt.legend(loc="upper right", bbox_to_anchor = (1.2, 1.02))
            plt.grid()
            fig_name = results_directory + "/percentexPexT-" + optimizer_name + "-" + objective_name + ".png"
            plt.savefig(fig_name, bbox_inches="tight")
            plt.clf()
        # plt.show()
