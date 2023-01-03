
#INTERNAL LIBRARIES
#from runnerSaver import run #RUNNER
import runnerSaver as rs

# Parametros de los modelos o funciones de optimizacion son:
    #limite inferior, limite superior, dimension, tamano poblacion, numero de iteraciones

# Optimizers
optimizer = ["GWO","FA","GWOEL","PSO","BA","WOA","CSA","DE"] # "PSO","BA","FA","GWO","GWOEL","WOA","CSA","DE"

# Benchmark function"
# "F1","F2","F3","F4","F5","F6","F7","F8","F9","F10","F11","F12","F13","F14","F15","F16","F17","F18","F19"
# "teamSizeModel"
objectivefunc = ["teamSizeModel"]

# Select number of repetitions for each experiment.
# To obtain meaningful statistical results, usually 30 independent runs are executed for each algorithm.
NumOfRuns = 30

# General parameters for all optimizers
params = {"PopulationSize": 6, "Iterations": 100}


#Experimentacion
#run(optimizer, objectivefunc, NumOfRuns, params)
runner = rs.RunnerSingleton()
runner.run(optimizer, objectivefunc, NumOfRuns, params)












#--------- EXPERIMENTACION PRINCIPAL PARA GENERAR LOS DATASET CON METRICAS DE GWO ----------------

"""
x = moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 50, 6, 250, "Adaptative Parameter")
x = moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 100, 6, 250, "Adaptative Parameter") 
x = moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 150, 6, 250, "Adaptative Parameter") 
x = moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 200, 6, 250, "Adaptative Parameter") 

x = moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 50, 6, 250, "GAOperators") 
x = moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 100, 6, 250, "GAOperators") 
x = moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 150, 6, 250, "GAOperators") 
x = moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 200, 6, 250, "GAOperators") 

x = moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 50, 6, 500, "Adaptative Parameter")
x = moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 100, 6, 500, "Adaptative Parameter") 
x = moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 150, 6, 500, "Adaptative Parameter") 
x = moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 200, 6, 500, "Adaptative Parameter") 

x = moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 50, 6, 500, "GAOperators")
x = moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 100, 6, 500, "GAOperators") 
x = moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 150, 6, 500, "GAOperators") 
x = moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 200, 6, 500, "GAOperators") 


#con 12 lobos

x = moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 50, 12, 250, "Adaptative Parameter") 
x = moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 100, 12, 250, "Adaptative Parameter") 
x = moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 150, 12, 250, "Adaptative Parameter") 
x = moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 200, 12, 250, "Adaptative Parameter") 

x = moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 50, 12, 250, "GAOperators") 
x = moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 100, 12, 250, "GAOperators") 
x = moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 150, 12, 250, "GAOperators") 
x = moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 200, 12, 250, "GAOperators") 


x = moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 50, 12, 500, "Adaptative Parameter")
x = moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 100, 12, 500, "Adaptative Parameter") 
x = moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 150, 12, 500, "Adaptative Parameter") 
x = moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 200, 12, 500, "Adaptative Parameter") 

x = moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 50, 12, 500, "GAOperators")
x = moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 100, 12, 500, "GAOperators") 
x = moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 150, 12, 500, "GAOperators") 
x = moduleGWO.GWO(moduleTSM.teamSizeModel, 3, 18, 200, 12, 500, "GAOperators") 

"""





